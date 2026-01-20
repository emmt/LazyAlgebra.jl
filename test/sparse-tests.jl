#
# sparse-tests.jl -
#
# Testing sparse operators.
#
module TestingLazyAlgebraSparseOperators

using Neutrals
using Random
using SparseArrays
using StructuredArrays
using Test
using TypeUtils
using Unitful

using Base:
    IteratorEltype, HasEltype, EltypeUnknown

using LazyAlgebra
using LazyAlgebra:
    # Wrappers
    Adjoint, Conjugate, Transpose, Inverse,
    # input element type and shape
    InputEltype, HasInputEltype, InputEltypeUnknown, input_eltype,
    InputShape, HasInputShape, InputShapeUnknown,
    input_shape, input_axes, input_size, input_length, input_ndims,
    # output element type and shape
    OutputEltype, HasOutputEltype, OutputEltypeUnknown, output_eltype,
    OutputShape, HasOutputShape, OutputShapeUnknown,
    output_shape, output_axes, output_size, output_length, output_ndims,
    # other methods
    check_structure, sparse_compressed_offsets,
    row_indices, col_indices, offsets,
    each_nz_index, each_row_index, each_col_index,
    row_index, col_index

is_csc(::Any) = false
is_csc(::SparseOperator{CSC}) = true
is_csc(::Adjoint{<:SparseOperator{CSR}}) = true

is_csr(::Any) = false
is_csr(::SparseOperator{CSR}) = true
is_csr(::Adjoint{<:SparseOperator{CSC}}) = true

is_coo(::Any) = false
is_coo(::SparseOperator{COO}) = true
is_coo(::Adjoint{<:SparseOperator{COO}}) = true

# Generate a possibly sparse array of random values. Value are small signed integers so that
# all computations should be exact (except with non-integer multipliers).
genarr(T::Type, dims::Integer...; kwds...) = genarr(T, dims; kwds...)
function genarr(::Type{T}, dims::Tuple{Vararg{Integer}};
                sparsity::Real = 0,
                range::AbstractUnitRange{<:Integer} = -17:17) where {T}
    @assert 0 ≤ sparsity ≤ 1
    A = Array{T}(undef, dims)
    sparse = (sparsity > 𝟘)
    for i in eachindex(A)
        if sparse && rand() ≤ sparsity
            A[i] = 𝟘
        elseif T <: Complex
            A[i] = complex(rand(range), rand(range))
        else
            A[i] = rand(range)
        end
    end
    return A
end

# Unpack a sparse operator into a regular array using simplest iterator. There may be
# duplicates.
function unpack_with_iterator!(dest::Array{T},
                               A::SparseOperator{F,E},
                               op = (E === Bool ? (|) : (+))) where {T,F,E}
    C = fill!(reshape(dest, (output_length(A), input_length(A))), zero(T))
    for (Aij, i, j) in zip(findnz(A)...,)
        C[i,j] = op(C[i,j], Aij)
    end
    return dest
end

brief(::Type{COO}) = "COO"
brief(::Type{CSC}) = "CSC"
brief(::Type{CSR}) = "CSR"

sparse_constructor(::Type{COO}) = SparseOperatorCOO
sparse_constructor(::Type{CSC}) = SparseOperatorCSC
sparse_constructor(::Type{CSR}) = SparseOperatorCSR

other_type(::Type{Float32}) = Float64
other_type(::Type{<:Real}) = Float32
other_type(::Type{Complex{T}}) where {T} = Complex{other_type(T)}

function check(f, A, B, I = eachindex(A, B))
    flag = true
    for i in I
        flag &= f(A[i], B[i])
    end
    return flag
end

tweak_value(::Type{T}, i::Integer) where {T<:Real} = convert(T, ifelse(isodd(i), 2*i, 2*i + 1))
tweak_value(::Type{Complex{T}}, i::Integer) where {T<:Real} =
    conj(Complex{T}(complex(2*i, 2*i + 1)))
tweak_value(::Type{T}) where {T} = Base.Fix1(tweak_value, T)

# TODO predicate, other constructors/convertors, vmul!
function runtests(::Type{F}, A::AbstractArray{T},
                  x::AbstractArray{<:Any,N},
                  y::AbstractArray{<:Any,M};
                  alphas=(-1,0,1,2), betas=(-1,0,1,-2)) where {F<:SparseFormat,T,M,N}
    rowsiz = size(y)
    colsiz = size(x)
    size(A) == (rowsiz..., colsiz...) || error("incompatible array sizes")
    nrows = prod(rowsiz)
    ncols = prod(colsiz)
    n = count(!iszero, A)
    Tp = other_type(T)
    constructor = sparse_constructor(F)
    @testset "Sparse operators in $(brief(F)) format with `T=$T`, `M=$M`, and `N=$N`" begin
        # Type hierarchy.
        @test constructor <: SparseOperator
        @test constructor <: SparseOperator{F}
        @test constructor{T} <: SparseOperator{F,T}
        @test constructor{T,M} <: SparseOperator{F,T,M}
        @test constructor{T,M,N} <: SparseOperator{F,T,M,N}

        # Constructors from the given array `A`.
        #
        # There are many different ways to build the same sparse operator from a given
        # array. We check that they all yield the same result.
        #
        # Unless `A` is a matrix, at least `M` must be specified. This can be done by
        # wrapping `A` in a pseudo-matrix.
        #
        # Check concrete constructor.
        B = @inferred(constructor(PseudoMatrix(A, Dims{M})))
        @test B isa constructor{T,M,N}
        _B = @inferred(constructor{T}(PseudoMatrix(A, Dims{M})))
        @test typeof(_B) === typeof(B)
        @test _B == B
        @test isequal(_B, B)
        if (M,N) == (1,1)
            _B = @inferred(constructor(A))
            @test typeof(_B) === typeof(B)
            @test _B == B
            @test isequal(_B, B)
            _B = @inferred(constructor{T}(A))
            @test typeof(_B) === typeof(B)
            @test _B == B
            @test isequal(_B, B)
        end
        _B = @inferred(constructor{T,M}(A))
        @test typeof(_B) === typeof(B)
        @test _B == B
        @test isequal(_B, B)
        _B = @inferred(constructor{T,M,N}(A))
        @test typeof(_B) === typeof(B)
        @test _B == B
        @test isequal(_B, B)
        #
        # Same tests but with abstract constructor.
        _B = @inferred(SparseOperator{F}(PseudoMatrix(A, Dims{M})))
        @test typeof(_B) === typeof(B)
        @test _B == B
        @test isequal(_B, B)
        _B = @inferred(SparseOperator{F,T}(PseudoMatrix(A, Dims{M})))
        @test typeof(_B) === typeof(B)
        @test _B == B
        @test isequal(_B, B)
        if (M,N) == (1,1)
            _B = @inferred(SparseOperator{F}(A))
            @test typeof(_B) === typeof(B)
            @test _B == B
            @test isequal(_B, B)
            _B = @inferred(SparseOperator{F,T}(A))
            @test typeof(_B) === typeof(B)
            @test _B == B
            @test isequal(_B, B)
        end
        _B = @inferred(SparseOperator{F,T,M}(A))
        @test typeof(_B) === typeof(B)
        @test _B == B
        @test isequal(_B, B)
        _B = @inferred(SparseOperator{F,T,M,N}(A))
        @test typeof(_B) === typeof(B)
        @test _B == B
        @test isequal(_B, B)
        #
        # `convert` calls constructor.
        _B = @inferred(convert(constructor, PseudoMatrix(A, Dims{M})))
        @test typeof(_B) === typeof(B)
        @test _B == B
        @test isequal(_B, B)
        _B = @inferred(convert(constructor{T}, PseudoMatrix(A, Dims{M})))
        @test typeof(_B) === typeof(B)
        @test _B == B
        @test isequal(_B, B)
        if (M,N) == (1,1)
            _B = @inferred(convert(constructor, A))
            @test typeof(_B) === typeof(B)
            @test _B == B
            @test isequal(_B, B)
            _B = @inferred(convert(constructor{T}, A))
            @test typeof(_B) === typeof(B)
            @test _B == B
            @test isequal(_B, B)
        end
        _B = @inferred(convert(constructor{T,M}, A))
        @test typeof(_B) === typeof(B)
        @test _B == B
        @test isequal(_B, B)
        _B = @inferred(convert(constructor{T,M,N}, A))
        @test typeof(_B) === typeof(B)
        @test _B == B
        @test isequal(_B, B)
        #
        # Same tests but with abstract constructor.
        _B = @inferred(convert(SparseOperator{F}, PseudoMatrix(A, Dims{M})))
        @test typeof(_B) === typeof(B)
        @test _B == B
        @test isequal(_B, B)
        _B = @inferred(convert(SparseOperator{F,T}, PseudoMatrix(A, Dims{M})))
        @test typeof(_B) === typeof(B)
        @test _B == B
        @test isequal(_B, B)
        if (M,N) == (1,1)
            _B = @inferred(convert(SparseOperator{F}, A))
            @test typeof(_B) === typeof(B)
            @test _B == B
            @test isequal(_B, B)
            _B = @inferred(convert(SparseOperator{F,T}, A))
            @test typeof(_B) === typeof(B)
            @test _B == B
            @test isequal(_B, B)
        end
        _B = @inferred(convert(SparseOperator{F,T,M}, A))
        @test typeof(_B) === typeof(B)
        @test _B == B
        @test isequal(_B, B)
        _B = @inferred(convert(SparseOperator{F,T,M,N}, A))
        @test typeof(_B) === typeof(B)
        @test _B == B
        @test isequal(_B, B)

        # Conversion constructors that just return their argument unchanged.
        @test @inferred(constructor(B)) === B
        @test @inferred(constructor{T}(B)) === B
        @test @inferred(constructor{T,M}(B)) === B
        @test @inferred(constructor{T,M,N}(B)) === B
        @test @inferred(SparseOperator(B)) === B
        @test @inferred(SparseOperator{F}(B)) === B
        @test @inferred(SparseOperator{F,T}(B)) === B
        @test @inferred(SparseOperator{F,T,M}(B)) === B
        @test @inferred(SparseOperator{F,T,M,N}(B)) === B

        # Conversions that do nothing.
        @test @inferred(convert(constructor, B)) === B
        @test @inferred(convert(constructor{T}, B)) === B
        @test @inferred(convert(constructor{T,M}, B)) === B
        @test @inferred(convert(constructor{T,M,N}, B)) === B
        @test @inferred(convert(SparseOperator{F}, B)) === B
        @test @inferred(convert(SparseOperator{F,T}, B)) === B
        @test @inferred(convert(SparseOperator{F,T,M}, B)) === B
        @test @inferred(convert(SparseOperator{F,T,M,N}, B)) === B

        # Conversion of element-type by the constructors.
        Bp = @inferred(constructor{Tp}(PseudoMatrix(A, Dims{M})))
        @test Bp isa constructor{Tp,M,N}

        # Convert back to regular array.
        C = @inferred(Array(B))
        @test C isa Array{T,M+N}
        @test C == A

        # Check structures.
        @test @inferred(check_structure(B)) === B

        # Basic methods.
        @test B == B
        @test isequal(B, B)
        @test @inferred(nnz(B)) == n
        @test @inferred(nnz(adjoint(B))) == n
        @test @inferred(nnz(transpose(B))) == n
        @test @inferred(nnz(conj(B))) == n
        @test @inferred(nonzeros(B)) isa AbstractVector{T}
        @test length(@inferred(nonzeros(B))) == n
        I, J, V = @inferred(findnz(B))
        @test I === @inferred(row_indices(B))
        @test J === @inferred(col_indices(B))
        @test V === @inferred(nonzeros(B))
        # TODO nonzeros for adjoint, conjugate, and transpose

        # Element type trait.
        @test @inferred(IteratorEltype(B)) == HasEltype()
        @test @inferred(eltype(typeof(B))) === T
        @test @inferred(IteratorEltype(adjoint(B))) == HasEltype()
        @test @inferred(eltype(adjoint(B))) === T
        @test @inferred(eltype(typeof(adjoint(B)))) === T
        @test @inferred(IteratorEltype(conj(B))) == HasEltype()
        @test @inferred(eltype(conj(B))) === T
        @test @inferred(eltype(eltype(conj(B)))) === T
        @test @inferred(IteratorEltype(transpose(B))) == HasEltype()
        @test @inferred(eltype(transpose(B))) === T
        @test @inferred(eltype(typeof(transpose(B)))) === T

        # Abstract vector API.
        @test @inferred(length(B)) == n
        @test @inferred(eltype(B)) === T
        vect = @inferred(collect(B))
        @test vect isa Vector{T}
        @test vect == @inferred(nonzeros(B))
        @test vect !== @inferred(nonzeros(B))
        #
        C = @inferred(adjoint(B))
        @test C isa Adjoint{typeof(B)}
        @test @inferred(SparseFormat(C)) === @inferred(transpose(SparseFormat(B)))
        @test @inferred(SparseFormat(typeof(C))) === @inferred(transpose(SparseFormat(B)))
        @test @inferred(eltype(C)) === T
        @test @inferred(length(C)) == n
        @test @inferred(eachindex(C)) === @inferred(eachindex(B))
        @test @inferred(nonzeros(C)) == conj.(nonzeros(B))
        #
        C = @inferred(conj(B))
        @test C isa Conjugate{typeof(B)}
        @test @inferred(SparseFormat(C)) === @inferred(SparseFormat(B))
        @test @inferred(SparseFormat(typeof(C))) === @inferred(SparseFormat(B))
        @test @inferred(eltype(C)) === T
        @test @inferred(length(C)) == n
        @test @inferred(eachindex(C)) === @inferred(eachindex(B))
        @test @inferred(nonzeros(C)) == conj.(nonzeros(B))
        #
        C = @inferred(transpose(B))
        @test C isa Transpose{typeof(B)}
        @test @inferred(SparseFormat(C)) === @inferred(transpose(SparseFormat(B)))
        @test @inferred(SparseFormat(typeof(C))) === @inferred(transpose(SparseFormat(B)))
        @test @inferred(eltype(C)) === T
        @test @inferred(length(C)) == n
        @test @inferred(eachindex(C)) === @inferred(eachindex(B))
        @test @inferred(nonzeros(C)) === nonzeros(B)

        # getindex and setindex!
        I = @inferred(collect(eachindex(B)))
        C = @inferred(copy(B)) # copy to not disturb B
        vals = @inferred(nonzeros(B))
        @test check(isequal, C, vals, I)
        @test check(isequal, transpose(C), vals, I)
        conj_vals = conj.(vals)
        @test check(isequal, adjoint(C), conj_vals, I)
        @test check(isequal, conj(C), conj_vals, I)
        #
        vals = @inferred(copy(nonzeros(B))) # copy to not disturb B
        map!(tweak_value(T), vals, I) # change all values
        conj_vals = conj.(vals)
        #
        fill!(nonzeros(C), zero(T))
        for i in I; C[i] = tweak_value(T, i); end
        @test check(isequal, C, vals, I)
        #
        fill!(nonzeros(C), zero(T))
        for i in I; transpose(C)[i] = tweak_value(T, i); end
        @test check(isequal, C, vals, I)
        #
        fill!(nonzeros(C), zero(T))
        for i in I; conj(C)[i] = tweak_value(T, i); end
        @test check(isequal, C, conj_vals, I)
        #
        fill!(nonzeros(C), zero(T))
        for i in I; adjoint(C)[i] = tweak_value(T, i); end
        @test check(isequal, C, conj_vals, I)

        # Format trait.
        @test @inferred(SparseFormat(B)) === F()
        @test @inferred(SparseFormat(typeof(B))) === F()
        @test @inferred(SparseFormat(Adjoint{typeof(B)})) === @inferred(transpose(F()))
        @test @inferred(SparseFormat(Transpose{typeof(B)})) === @inferred(transpose(F()))
        @test @inferred(SparseFormat(Conjugate{typeof(B)})) === F()

        # Input/output shapes.
        @test @inferred(OutputShape(B)) === HasOutputShape{M}()
        @test @inferred(OutputShape(typeof(B))) === HasOutputShape{M}()
        @test as_array_axes(@inferred(output_shape(B))) === as_array_axes(rowsiz)
        @test @inferred(output_length(B)) === prod(rowsiz)
        @test @inferred(InputShape(B)) === HasInputShape{N}()
        @test @inferred(InputShape(typeof(B))) === HasInputShape{N}()
        @test as_array_axes(@inferred(input_shape(B))) === as_array_axes(colsiz)
        @test @inferred(input_length(B)) === prod(colsiz)

        # Input/output element types.
        @test @inferred(OutputEltype(B)) === OutputEltypeUnknown()
        @test @inferred(OutputEltype(typeof(B))) === OutputEltypeUnknown()
        @test @inferred(InputEltype(B)) === InputEltypeUnknown()
        @test @inferred(InputEltype(typeof(B))) === InputEltypeUnknown()

        # Convert element type.
        @test @inferred(convert_eltype(T, B)) === B
        Bp = @inferred(convert_eltype(Tp, B))
        @test Bp isa constructor{Tp,M,N}
        @test @inferred(nonzeros(Bp)) isa AbstractVector{Tp}
        @test @inferred(nonzeros(Bp)) == convert_eltype(Tp, nonzeros(B))

        # Precision.
        @test @inferred(get_precision(B)) === get_precision(T)
        @test @inferred(get_precision(typeof(B))) === get_precision(T)
        Bp = @inferred(adapt_precision(get_precision(Tp), B))
        @test Bp isa constructor{Tp,M,N}
        @test @inferred(nonzeros(Bp)) isa AbstractVector{Tp}
        @test @inferred(nonzeros(Bp)) == adapt_precision(get_precision(Tp), nonzeros(B))

        # Copy.
        _B = @inferred(copy(B))
        @test _B isa constructor{T,M,N}
        @test _B == B
        @test isequal(_B, B)
        @test @inferred(nonzeros(_B)) isa AbstractVector{T}
        @test @inferred(nonzeros(_B)) == @inferred(nonzeros(B))
        @test @inferred(nonzeros(_B)) !== @inferred(nonzeros(B))
        if F !== COO
            @test @inferred(offsets(_B)) === @inferred(offsets(B))
        end
        if F === CSR
            @test @inferred(row_indices(_B)) == @inferred(row_indices(B))
        else
            @test @inferred(row_indices(_B)) === @inferred(row_indices(B))
        end
        if F === CSC
            @test @inferred(col_indices(_B)) == @inferred(col_indices(B))
        else
            @test @inferred(col_indices(_B)) === @inferred(col_indices(B))
        end

        # Deep copy.
        _B = @inferred(deepcopy(B))
        @test _B isa constructor{T,M,N}
        @test _B == B
        @test isequal(_B, B)
        @test @inferred(nonzeros(_B)) isa AbstractVector{T}
        @test @inferred(nonzeros(_B)) == @inferred(nonzeros(B))
        @test @inferred(nonzeros(_B)) !== @inferred(nonzeros(B))
        if F !== COO
            @test @inferred(offsets(_B)) == @inferred(offsets(B))
            @test @inferred(offsets(_B)) !== @inferred(offsets(B))
        end
        @test @inferred(row_indices(_B)) == @inferred(row_indices(B))
        @test @inferred(row_indices(_B)) !== @inferred(row_indices(B))
        @test @inferred(col_indices(_B)) == @inferred(col_indices(B))
        @test @inferred(col_indices(_B)) !== @inferred(col_indices(B))

        # Conversion to an other format.
        if F !== COO
            C = @inferred(convert(SparseOperator{COO}, B))
            @test C isa SparseOperator{COO,T,M,N}
            @test @inferred(SparseOperator{COO}(B)) == C
        end
        if F !== CSC
            C = @inferred(convert(SparseOperator{CSC}, B))
            @test C isa SparseOperator{CSC,T,M,N}
            @test @inferred(SparseOperator{CSC}(B)) == C
        end
        if F !== CSR
            C = @inferred(convert(SparseOperator{CSR}, B))
            @test C isa SparseOperator{CSR,T,M,N}
            @test @inferred(SparseOperator{CSR}(B)) == C
        end

        # Apply operator.
        if (M,N) == (1,1)
            A_x = A*x
            Ac_x = conj.(A)*x
            Ap_y = A'*y
            At_y = transpose(A)*y
        else
            _A = reshape(A, (nrows, ncols))
            _x = view(x, :)
            _y = view(y, :)
            A_x = reshape(_A*_x, rowsiz)
            Ac_x = reshape(conj.(_A)*_x, rowsiz)
            Ap_y = reshape(_A'*_y, colsiz)
            At_y = reshape(transpose(_A)*_y, colsiz)
        end
        @test @inferred(B*x) == A_x
        @test @inferred(conj(B)*x) == Ac_x
        @test @inferred(B'*y) == Ap_y
        @test @inferred(transpose(B)*y) == At_y
        x_cpy = copy(x)
        y_cpy = copy(y)
        @testset "vmul!, α=$α, β=$β" for α in alphas, β in betas
            Tz = typeof(unit(α)*zero(eltype(A))*zero(eltype(x)) + unit(β)*zero(eltype(y)))
            _α = adapt_precision(get_precision(zero(eltype(A))*zero(eltype(x))), α)
            _β = adapt_precision(get_precision(eltype(y)), β)
            z = similar(y, Tz)
            @test @inferred(vmul!(z, α, B, x, β, y)) === z
            @test x == x_cpy
            @test y == y_cpy
            @test z == _α*A_x + _β*y
            @test @inferred(vmul!(z, α, conj(B), x, β, y)) === z
            @test x == x_cpy
            @test y == y_cpy
            @test z == _α*Ac_x + _β*y
            z = similar(x, Tz)
            @test @inferred(vmul!(z, α, B', y, β, x)) === z
            @test x == x_cpy
            @test y == y_cpy
            @test z == _α*Ap_y + _β*x
            @test @inferred(vmul!(z, α, transpose(B), y, β, x)) === z
            @test x == x_cpy
            @test y == y_cpy
            @test z == _α*At_y + _β*x
        end

        # Additional tests for COO.
        if F === COO
            # Make a COO version with randomly permuted entries.
            Ip = randperm(nnz(B))
            Bp = @inferred(SparseOperatorCOO(nonzeros(B)[Ip],
                                             row_indices(B)[Ip],
                                             col_indices(B)[Ip],
                                             output_size(B),
                                             input_size(B)))
            @test Bp isa SparseOperatorCOO{T,M,N}

            # Apply operator.
            @test @inferred(Bp*x) == A_x
            @test @inferred(conj(Bp)*x) == Ac_x
            @test @inferred(Bp'*y) == Ap_y
            @test @inferred(transpose(Bp)*y) == At_y

            # Convert back to regular array.
            Ap = @inferred(Array(Bp))
            @test Ap isa Array{T,M+N}
            @test Ap == A

            # Convert to CSR.
            Cp = @inferred(SparseOperatorCSR(Bp))
            @test Cp isa SparseOperatorCSR{T,M,N}
            @test Cp == @inferred(SparseOperatorCSR(PseudoMatrix(A, Dims{M})))

            # Convert to CSC.
            Cp = @inferred(SparseOperatorCSC(Bp))
            @test Cp isa SparseOperatorCSC{T,M,N}
            @test Cp == @inferred(SparseOperatorCSC(PseudoMatrix(A, Dims{M})))

            # Make a COO version with randomly permuted entries and some duplicates. Use
            # fractions 1/3 and 3/4 for duplicating so that there is no loss of precision.
            l = 7
            I = zeros(Int, length(Ip) + l)
            w = ones(float(real(T)), length(I))
            I[1:length(Ip)] = Ip
            for i in 1:l
                j1 = length(Ip) - i + 1
                j2 = length(Ip) + i
                w[j1] *= 1/4
                w[j2] *= 3/4
                I[j2] = I[j1]
            end
            Bp = @inferred(SparseOperatorCOO(nonzeros(B)[I] .* w,
                                             row_indices(B)[I],
                                             col_indices(B)[I],
                                             output_size(B),
                                             input_size(B)))
            @test Bp isa SparseOperatorCOO{T,M,N}

            # Apply operator.
            @test @inferred(Bp*x) == A_x
            @test @inferred(conj(Bp)*x) == Ac_x
            @test @inferred(Bp'*y) == Ap_y
            @test @inferred(transpose(Bp)*y) == At_y

            # Convert back to regular array.
            Ap = @inferred(Array(Bp))
            @test Ap isa Array{T,M+N}
            @test Ap == A

            # Convert to CSR.
            Cp = @inferred(SparseOperatorCSR(Bp))
            @test Cp isa SparseOperatorCSR{T,M,N}
            @test Cp == @inferred(SparseOperatorCSR(PseudoMatrix(A, Dims{M})))

            # Convert to CSC.
            Cp = @inferred(SparseOperatorCSC(Bp))
            @test Cp isa SparseOperatorCSC{T,M,N}
            @test Cp == @inferred(SparseOperatorCSC(PseudoMatrix(A, Dims{M})))
       end
    end
end

function runtests()
    @testset "Sparse operators" begin
        @testset "Low level sparse utilities" begin
            @test sparse_compressed_offsets(2, Int[]) == [0,0,0]
            @test sparse_compressed_offsets(5, [2,2,3,5]) == [0,0,2,3,3,4]
            @test sparse_compressed_offsets(5, [1,3,3]) == [0,1,1,3,3,3]
            # Check for non-decreasing order.
            @test_throws AssertionError sparse_compressed_offsets(5, [1,3,2])
            # Check for out-of-bounds.
            @test_throws AssertionError sparse_compressed_offsets(5, [0,3,3])
            @test_throws AssertionError sparse_compressed_offsets(5, [1,3,7])
        end
        @testset "Compressed sparse formats" begin
            @test COO === CompressedSparseCoordinate
            @test CSC === CompressedSparseColumn
            @test CSR === CompressedSparseRow
            @test_throws ArgumentError SparseFormat(:COO)
            @testset "... $F" for (F, Ft) in (COO => COO, CSR => CSC, CSC => CSR)
                @test @inferred(SparseFormat(F)) === F()
                @test @inferred(SparseFormat(F())) === F()
                @test @inferred(transpose(F())) === Ft()
                io = IOBuffer()
                str = @inferred(summary(F))
                @test str isa String
                @test startswith(str, "Compressed Sparse ")
                @test endswith(str, " format")
                @test @inferred(summary(F())) == str
                @test @inferred(summary(io, F)) === nothing
                @test String(take!(io)) == str
                @test @inferred(summary(io, F())) === nothing
                @test String(take!(io)) == str
            end
        end

        # Trial array with 60 (= 2*2*3*5) entries and predefined mask with many successive
        # zeros to have some rows or columns of zeros.
        T = Complex{Float32}
        mask = Bool[1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0,
                    1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1,
                    0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0]
        vals = zeros(T, size(mask))

        # 12×5
        rowsiz = (12,)
        colsiz = (5,)
        vals[mask] = genarr(T, count(mask))
        A = reshape(vals, (rowsiz..., colsiz...))
        x = genarr(T, colsiz)
        y = genarr(T, rowsiz)
        runtests(CSR, A, x, y)
        runtests(CSC, A, x, y)
        runtests(COO, A, x, y)

        # 5×(2,3,2)
        rowsiz = (5,)
        colsiz = (2,3,2,)
        vals[mask] = genarr(T, count(mask))
        A = reshape(vals, (rowsiz..., colsiz...))
        x = genarr(T, colsiz)
        y = genarr(T, rowsiz)
        runtests(CSR, A, x, y)
        runtests(CSC, A, x, y)
        runtests(COO, A, x, y)

        # (2,3)×(5,2)
        rowsiz = (2,3,)
        colsiz = (5,2,)
        vals[mask] = genarr(T, count(mask))
        A = reshape(vals, (rowsiz..., colsiz...))
        x = genarr(T, colsiz)
        y = genarr(T, rowsiz)
        runtests(CSR, A, x, y)
        runtests(CSC, A, x, y)
        runtests(COO, A, x, y)

        # (5,2)×6
        rowsiz = (5,2,)
        colsiz = (6,)
        vals[mask] = genarr(T, count(mask))
        A = reshape(vals, (rowsiz..., colsiz...))
        x = genarr(T, colsiz)
        y = genarr(T, rowsiz)
        runtests(CSR, A, x, y)
        runtests(CSC, A, x, y)
        runtests(COO, A, x, y)

    end
end # function

end # module
