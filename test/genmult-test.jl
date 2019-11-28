module GenMultTests

using Test
using Test: print_test_results

#Test.TESTSET_PRINT_ENABLE[] = true

using LazyAlgebra
import LazyAlgebra: GenMult, convert_multiplier
import .GenMult: Reals, Complexes, Floats, BlasFloat

for (pfx, P) in ((:generic, GenMult.Generic()),
                 (:blas,    GenMult.Blas()),
                 (:linear,  GenMult.Linear()),
                 (:basic,   GenMult.Basic()))
    @eval begin
        $(Symbol(pfx, "_lgemv"))(α, trans, A, x) =
            GenMult._lgemv($P, α, trans, A, x)
        $(Symbol(pfx, "_lgemv!"))(α, trans, A, x, β, y) =
            GenMult._lgemv!($P, α, trans, A, x, β, y)
        $(Symbol(pfx, "_lgemm"))(α, transA, A, transB, B, Nc::Integer=2) =
            GenMult._lgemm($P, α, transA, A, transB, B, Int(Nc))
        $(Symbol(pfx, "_lgemm!"))(α, transA, A, transB, B, β, C) =
            GenMult._lgemm!($P, α, transA, A, transB, B, β, C)
    end
end

_transp(t::Char, A::AbstractMatrix) =
    t == 'N' ? A :
    t == 'T' ? transpose(A) :
    t == 'C' ? A' : error("invalid transpose character")

# in Julia < 0.7 randn() does not generate complex numbers.
_randn(::Type{T}, ::Tuple{}) where {T} = _randn(T)
_randn(::Type{T}, dims::Integer...) where {T} = _randn(T, map(Int, dims))
_randn(::Type{T}) where {T<:AbstractFloat} = randn(T)
_randn(::Type{<:Complex{T}}) where {T<:AbstractFloat} =
    Complex(randn(T), randn(T))
function _randn(::Type{T}, dims::NTuple{N,Int}) where {T,N}
    A = Array{T,N}(undef, dims)
    @inbounds for i in eachindex(A)
        A[i] =  _randn(T)
    end
    return A
end

function ref_lgemv(α::Number,
                   trans::Char,
                   A::DenseArray{Ta,Na},
                   x::DenseArray{Tx,Nx}) where {Ta,Na,Tx,Nx}
    @assert 1 ≤ Nx < Na
    Ny = Na - Nx
    dims = size(A)
    if trans == 'N'
        rows, cols = dims[1:Ny], dims[Ny+1:end]
        @assert size(x) == cols
    else
        rows, cols = dims[1:Nx], dims[Nx+1:end]
        @assert size(x) == rows
    end
    M = reshape(A, (prod(rows), prod(cols)))
    v = reshape(x, length(x))
    if trans == 'N'
        return α*reshape(M*v, rows)
    elseif trans == 'T'
        return α*reshape(transpose(M)*v, cols)
    elseif trans == 'C'
        return α*reshape(M'*v, cols)
    else
        error("invalid transpose character")
    end
end

function ref_lgemm(α::Number,
                   transA::Char,
                   A::DenseArray{Ta,Na},
                   transB::Char,
                   B::DenseArray{Tb,Nb},
                   Nc::Integer=2) where {Ta,Na,Tb,Nb}
    Ni, Nj, Nk = GenMult._lgemm_ndims(Na, Nb, Int(Nc))
    @assert Na == Ni + Nk
    @assert Nb == Nk + Nj
    @assert Nc == Ni + Nj
    Adims = size(A)
    Bdims = size(B)
    if transA == 'N'
        I, K = Adims[1:Ni], Adims[Ni+1:Ni+Nk]
    else
        K, I = Adims[1:Nk], Adims[Nk+1:Nk+Ni]
    end
    if transB == 'N'
        @assert Bdims[1:Nk] == K
        J = Bdims[Nk+1:Nk+Nj]
    else
        @assert Bdims[Nj+1:Nj+Nk] == K
        J = Bdims[1:Nj]
    end
    m = prod(I)
    n = prod(J)
    p = prod(K)
    return α*reshape(_transp(transA, reshape(A, transA == 'N' ? (m, p) : (p, m)))*
                     _transp(transB, reshape(B, transB == 'N' ? (p, n) : (n, p))),
                     (I..., J...))
end

cnv(::Type{T}, x::T) where {T<:AbstractFloat} = x
cnv(::Type{T}, x::Real) where {T<:AbstractFloat} = convert(T, x)
cnv(::Type{T}, x::Complex{<:Real}) where {T<:AbstractFloat} =
    convert(Complex{T}, x)

function similar_values(::Type{T},
                        A::AbstractArray,
                        B::AbstractArray;
                        atol::Real=zero(T),
                        rtol::Real=sqrt(eps(T))) where {T<:AbstractFloat}
    if axes(A) != axes(B)
        return false
    end
    local anrm2::T = 0
    local bnrm2::T = 0
    local cnrm2::T = 0
    n = 0
    for i in eachindex(A, B)
        a, b = cnv(T, A[i]), cnv(T, B[i])
        anrm2 += abs2(a)
        bnrm2 += abs2(b)
        cnrm2 += abs2(a - b)
        n += 1
    end
    return sqrt(cnrm2) ≤ n*atol + rtol*sqrt(max(anrm2, bnrm2))
end

function similar_values(A::AbstractArray{Ta},
                        B::AbstractArray{Tb};
                        kwds...) where {Ta,Tb}
    similar_values(float(real(promote_type(Ta,Tb))), A, B; kwds...)
end

"""

`worst_type(T1,T2)` yields the smallest floating-point real type of
real/complex types `T1` and `T2`.

"""
function worst_type(::Type{Complex{T1}},
                    ::Type{T2}) where {T1<:AbstractFloat,
                                       T2<:AbstractFloat}
    worst_type(T1, T2)
end

function worst_type(::Type{T1},
                    ::Type{Complex{T2}}) where {T1<:AbstractFloat,
                                                T2<:AbstractFloat}
    worst_type(T1, T2)
end

function worst_type(::Type{Complex{T1}},
                    ::Type{Complex{T2}}) where {T1<:AbstractFloat,
                                                T2<:AbstractFloat}
    worst_type(T1, T2)
end

function worst_type(::Type{T1},
                    ::Type{T2}) where {T1<:AbstractFloat,
                                       T2<:AbstractFloat}
    sizeof(T1) ≤ sizeof(T2) ? T1 : T2
end

function test_lgemv(reduced::Bool=false)
    # Notes:
    #  - π is good for testing special (non floating-point) values.
    #  - Use prime numbers for dimensions so that they cannot be split.
    if reduced
        MULTIPLIERS = (0, 1, π)
        DIMENSIONS = (((3,  ), (5,  )),
                      ((2, 3), (5, 7)))
        TRANS = ('N', 'T', 'C')
        TYPES = (Float32, Float64, ComplexF64)
    else
        MULTIPLIERS = (0, 1, π)
        DIMENSIONS = (((3,  ), (5,  )),
                      ((2, 3), (5,  )),
                      ((3,  ), (2, 5)),
                      ((2, 3), (5, 7)))
        TRANS = ('N', 'T', 'C')
        TYPES = (Float32, Float64, ComplexF32, ComplexF64)
    end
    @testset "LGEMV" begin
        @testset for dims in DIMENSIONS,
            Ta in TYPES, transA in TRANS,
            Tx in TYPES,
            α in MULTIPLIERS, β in MULTIPLIERS
            Ta <: Real && transA == 'C' && continue
            m, n = dims
            Ty = promote_type(Ta, Tx)
            A = _randn(Ta, (m..., n...))
            x = _randn(Tx, transA == 'N' ? n : m)
            y = _randn(Ty, transA == 'N' ? m : n)
            Tw = worst_type(Ta, Tx)
            ref = ref_lgemv(α, transA, A, x) + β*y
            if β == 0
                @test similar_values(Tw, ref,   generic_lgemv(α, transA, A, x))
                @test similar_values(Tw, ref,    linear_lgemv(α, transA, A, x))
                @test similar_values(Tw, ref,           lgemv(α, transA, A, x))
                if ndims(A) == 2 && ndims(x) == 1
                    @test similar_values(Tw, ref, basic_lgemv(α, transA, A, x))
                end
                if Ta == Tx == Ty && Ta <: BlasFloat
                    α′ = convert(Ta, α)
                    @test similar_values(Tw, ref,  blas_lgemv(α′,transA, A, x))
                end
            end
            @test similar_values(Tw, ref,    generic_lgemv!(α, transA, A, x, β, deepcopy(y)))
            @test similar_values(Tw, ref,     linear_lgemv!(α, transA, A, x, β, deepcopy(y)))
            @test similar_values(Tw, ref,            lgemv!(α, transA, A, x, β, deepcopy(y)))
            if ndims(A) == 2 && ndims(x) == 1
                @test similar_values(Tw, ref,  basic_lgemv!(α, transA, A, x, β, deepcopy(y)))
            end
            if Ta == Tx == Ty && Ta <: BlasFloat
                α′ = convert(Ta, α)
                β′ = convert(Ta, β)
                @test similar_values(Tw, ref,   blas_lgemv!(α′,transA, A, x, β′,deepcopy(y)))
            end
        end
    end
end

function test_lgemm(reduced::Bool=false)
    if reduced
        MULTIPLIERS = (0, 1, π)
        DIMENSIONS = (((5,  ), (3,  ), (2,  )),
                      ((5, 2), (4,  ), (3,  )),
                      ((4, 5), (3, 2), (2, 3)))
        TRANS = ('N', 'T', 'C')
        TYPES = (Float32, Float64, ComplexF64)
    else
        MULTIPLIERS = (0, 1, π)
        DIMENSIONS = (((5,  ), (3,     ), (2,  )),
                      ((5, 2), (3,     ), (2,  )),
                      ((5, 2), (4, 3   ), (3, 2)),
                      ((3, 4), (2, 2, 3), (2, 3)))
        TRANS = ('N', 'T', 'C')
        TYPES = (Float32, Float64, ComplexF32, ComplexF64)
    end
    @testset "LGEMM" begin
        @testset for dims in DIMENSIONS,
            α in MULTIPLIERS, β in MULTIPLIERS,
            Ta in TYPES, transA in TRANS,
            Tb in TYPES, transB in TRANS
            Ta <: Real && transA == 'C' && continue
            Tb <: Real && transB == 'C' && continue
            m, n, p = dims
            Tc = promote_type(Ta, Tb)
            C = _randn(Tc, (m..., n...))
            Nc = ndims(C)
            A = _randn(Ta, transA == 'N' ? (m..., p...) : (p..., m...))
            B = _randn(Tb, transB == 'N' ? (p..., n...) : (n..., p...))
            Tw = worst_type(Ta, Tb)
            ref = ref_lgemm(α, transA, A, transB, B, Nc) + β*C
            #ref = generic_lgemm(α, transA, A, transB, B) + β*C
            if β == 0
                @test similar_values(Tw, ref,   generic_lgemm(α, transA, A, transB, B, Nc))
                @test similar_values(Tw, ref,    linear_lgemm(α, transA, A, transB, B, Nc))
                @test similar_values(Tw, ref,           lgemm(α, transA, A, transB, B, Nc))
                if ndims(A) == ndims(B) == 2
                    @test similar_values(Tw, ref, basic_lgemm(α, transA, A, transB, B, Nc))
                end
                if Ta == Tb == Tc && Ta <: BlasFloat
                    α′ = convert(Ta, α)
                    @test similar_values(Tw, ref,  blas_lgemm(α′,transA, A, transB, B, Nc))
                end
            end
            @test similar_values(Tw, ref,   generic_lgemm!(α, transA, A, transB, B, β, deepcopy(C)))
            @test similar_values(Tw, ref,    linear_lgemm!(α, transA, A, transB, B, β, deepcopy(C)))
            @test similar_values(Tw, ref,           lgemm!(α, transA, A, transB, B, β, deepcopy(C)))
            if ndims(A) == ndims(B) == 2
                @test similar_values(Tw, ref, basic_lgemm!(α, transA, A, transB, B, β, deepcopy(C)))
            end
            if Ta == Tb == Tc && Ta <: BlasFloat
                α′ = convert(Ta, α)
                β′ = convert(Ta, β)
                @test similar_values(Tw, ref,  blas_lgemm!(α′,transA, A, transB, B, β′,deepcopy(C)))
            end
        end
    end
end

function test_all(reduced::Bool=false)
    print_test_results(test_lgemv(reduced))
    print_test_results(test_lgemm(reduced))
    nothing
end

end # module
