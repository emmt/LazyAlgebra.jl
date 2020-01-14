#
# vect-tests.jl -
#
# Tests for vectorized operations.
#

#isdefined(:LazyAlgebra) || include("../src/LazyAlgebra.jl")

module LazyAlgebraVectorsTests

using LazyAlgebra
using Test

# Deal with compatibility issues.
@static if isdefined(Base, :MathConstants)
    import Base.MathConstants: φ
end

distance(a::Real, b::Real) = abs(a - b)

distance(a::NTuple{2,Real}, b::NTuple{2,Real}) =
    hypot(a[1] - b[1], a[2] - b[2])

distance(A::AbstractArray{Ta,N}, B::AbstractArray{Tb,N}) where {Ta,Tb,N} =
    maximum(abs.(A - B))

function makeselection(n::Integer)
    sel = Array{Int}(undef, 0)
    j = [2,3,5]
    k = 1
    while k ≤ n
        push!(sel, k)
        k += rand(j)
    end
    return sel
end

@testset "Vectors" begin
    types = (Float32, Float64)
    dims = (3,4,5)
    @testset "vnorm ($T)" for T in types
        S = (T == Float32 ? Float64 : Float32)
        v = randn(T, dims)
        @test vnorminf(v) == maximum(abs.(v))
        @test vnorminf(S, v) == S(maximum(abs.(v)))
        @test vnorm1(v) ≈ sum(abs.(v))
        @test vnorm2(v) ≈ sqrt(sum(v.*v))
        z = complex.(randn(T, dims), randn(T, dims))
        @test vnorminf(z) ≈ maximum(abs.(z))
        @test vnorm1(z) ≈ sum(abs.(real.(z)) + abs.(imag.(z)))
        @test vnorm2(z) ≈ sqrt(sum(abs2.(z)))
    end
    @testset "vcopy, vswap ($T)" for T in types
        u = randn(T, dims)
        uc = vcopy(u)
        @test distance(u, uc) == 0
        v = randn(T, dims)
        vc = vcopy!(vcreate(v), v)
        @test distance(v, vc) == 0
        vswap!(u, v)
        @test distance(u, vc) == distance(v, uc) == 0
        @test_throws DimensionMismatch  vcopy!(Array{T}(undef, dims .+ 1), u)
    end
    @testset "vfill ($T)" for T in types
       a = randn(T, dims)
        @test distance(vfill!(a,0), zeros(T,dims)) == 0
        a = randn(T, dims)
        @test distance(vfill!(a,0), vzero!(a)) == 0
        a = randn(T, dims)
        @test distance(vfill!(a,1), ones(T,dims)) == 0
        a = randn(T, dims)
        @test distance(vfill!(a,π), fill!(similar(a), π)) == 0
        ac = vcopy(a)
        @test distance(vzeros(a), zeros(T,dims)) == 0
        @test distance(a, ac) == 0
        @test distance(vones(a), ones(T,dims)) == 0
        @test distance(a, ac) == 0
    end
    @testset "vscale" begin
        for T in types
            a = randn(T, dims)
            b = vcreate(a)
            for α in (0, -1, 1, π, 2.71)
                d = T(α)*a
                @test distance(vscale(α,a), d) == 0
                @test distance(vscale(a,α), d) == 0
                @test distance(vscale!(b,α,a), d) == 0
                @test distance(vscale!(b,a,α), d) == 0
                c = vcopy(a)
                @test distance(vscale!(c,α), d) == 0
                vcopy!(c, a)
                @test distance(vscale!(α,c), d) == 0
            end
        end
        v = ones(dims)
        @test_throws ErrorException vscale!(1,2)
        @test_throws ErrorException vscale!(v,2,3)
        @test_throws ErrorException vscale!(1,v,3)
        @test_throws ErrorException vscale!(1,2,v)
        @test_throws ErrorException vscale!(1,2,3)
        for Ta in types, Tb in types
            a = randn(Ta, dims)
            ac = vcopy(a)
            b = Array{Tb}(undef, dims)
            e = max(eps(Ta), eps(Tb))
            for α in (0, -1, 1, π, 2.71)
                d = α*a
                @test distance(vscale!(b,α,a), d) ≤ 8e
                @test distance(vscale(α,a), d) ≤ 8e
                @test distance(a, ac) == 0
            end
        end
    end
    @testset "vupdate ($Ta,$Tb)" for Ta in types, Tb in types
        Tmin = sizeof(Ta) ≤ sizeof(Tb) ? Ta : Tb
        Tmax = sizeof(Ta) ≥ sizeof(Tb) ? Ta : Tb
        a = randn(Ta, dims)
        b = randn(Tb, dims)
        sel = makeselection(length(a))
        atol, rtol = zero(Tmin), sqrt(eps(Tmin))
        for α in (0, -1, 1, π, 2.71)
            @test vupdate!(vcopy(a),α,b) ≈
                a + Tmax(α)*b atol=atol rtol=rtol norm=vnorm2
            c = vcopy(a)
            c[sel] .+= Tmax(α)*b[sel]
            @test vupdate!(vcopy(a),sel,α,b) ≈ c atol=atol rtol=rtol norm=vnorm2
        end
    end
    @testset "vproduct ($Ta,$Tb)" for Ta in types, Tb in types
        a = randn(Ta, dims)
        b = randn(Tb, dims)
        sel = makeselection(length(a))
        c = vcreate(a)
        e = max(eps(Ta), eps(Tb))
        @test distance(vproduct!(c,a,b), (a .* b)) ≤ 2e
        c = randn(Tb, dims)
        d = vcopy(c)
        d[sel] = a[sel] .* b[sel]
        @test distance(vproduct!(c,sel,a,b), d) ≤ 2e
    end
    @testset "vcombine ($T)" for T in types
        a = randn(T, dims)
        b = randn(T, dims)
        d = vcreate(a)
        for α in (0, -1, 1, π,  2.71),
            β in (0, -1, 1, φ, -1.33)
            @test distance(vcombine!(d,α,a,β,b), (T(α)*a + T(β)*b)) == 0
        end
    end
    @testset "vdot ($Ta,$Tb)" for Ta in types, Tb in types
        Tmin = sizeof(Ta) ≤ sizeof(Tb) ? Ta : Tb
        Tmax = sizeof(Ta) ≥ sizeof(Tb) ? Ta : Tb
        a = randn(Ta, dims)
        b = randn(Tb, dims)
        sel = makeselection(length(a))
        w = zeros(Tmax, dims)
        w[sel] = rand(length(sel))
        # check implementation for real-valued vectors
        @test vdot(a,b) ≈ sum(a.*b)
        @test vdot(w,a,b) ≈ sum(w.*a.*b)
        @test vdot(sel,a,b) ≈ sum(a[sel].*b[sel])
        # check ⟨a,b⟩ = ⟨b,a⟩ for real-valued vectors
        @test vdot(a,b) == vdot(b,a)
        @test vdot(w,a,b) ≈ vdot(w,b,a)
        @test vdot(sel,a,b) == vdot(sel,b,a)
        @test sqrt(vdot(a,a)) ≈ vnorm2(a)
    end
    @testset "vdot (Complex{$Ta},Complex{$Tb})" for Ta in types, Tb in types
        Tmin = sizeof(Ta) ≤ sizeof(Tb) ? Ta : Tb
        Tmax = sizeof(Ta) ≥ sizeof(Tb) ? Ta : Tb
        a = complex.(randn(Ta, dims), randn(Ta, dims))
        b = complex.(randn(Tb, dims), randn(Tb, dims))
        sel = makeselection(length(a))
        w = zeros(Tmax, dims)
        w[sel] = rand(length(sel))
        # check implementation for complex-valued vectors
        @test vdot(a,b) ≈ sum(conj.(a).*b)
        @test vdot(Tmax,a,b) ≈ real(sum(conj.(a).*b))
        @test vdot(w,a,b) ≈ sum(w.*conj.(a).*b)
        @test vdot(Tmax,w,a,b) ≈ real(sum(w.*conj.(a).*b))
        @test vdot(sel,a,b) ≈ sum(conj.(a[sel]).*b[sel])
        @test vdot(Tmax,sel,a,b) ≈ real(sum(conj.(a[sel]).*b[sel]))
        # check ⟨a,b⟩ = conj(⟨b,a⟩) for complex-valued vectors
        @test vdot(a,b) == conj(vdot(b,a))
        @test vdot(w,a,b) ≈ conj(vdot(w,b,a))
        @test vdot(sel,a,b) == conj(vdot(sel,b,a))
    end
end

end # module
