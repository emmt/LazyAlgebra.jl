#
# sparse.jl -
#
# Implement sparse linear mappings.
#
#-------------------------------------------------------------------------------
#
# This file is part of LazyAlgebra (https://github.com/emmt/LazyAlgebra.jl)
# released under the MIT "Expat" license.
#
# Copyright (c) 2017-2020 Éric Thiébaut.
#

module SparseOperators

export
    SparseOperator,
    sparse,
    unpack!

using ..LazyAlgebra
using ..LazyAlgebra: @callable, promote_multiplier, Endomorphism, Morphism
import ..LazyAlgebra: vcreate, apply!, are_same_mappings, coefficients, check,
    MorphismType, input_size, output_size, input_ndims, output_ndims

using ArrayTools
using SparseArrays # for SparseMatrixCSC
import SparseArrays: sparse
import Base: eltype, ndims, reshape, Array, Matrix, *

"""
```julia
SparseOperator(I, J, C, rowdims, coldims)
```

yields a sparse linear mapping with `I` and `J` the row and column indices of
the non-zero coefficients whose values are specified by `C` and with the
dimensions of the *rows* and of the *columns* given by `rowdims` and `coldims`.

```julia
SparseOperator(A, n=1)
```

yields an instance of `SparseOperator` whose coefficients are the non-zero
coefficients of array `A` and which implements generalized matrix
multiplication by `A` in such a way that the result of applying the operator
are arrays whose dimensions are the `n` leading dimensions of `A` for input
arrays whose dimensions are the remaining trailing dimensions of `A`.

```julia
SparseOperator(A)
```

yields an instance of `SparseOperator` built from sparse matrix
(`SparseMatrixCSC`) or sparse operator (`SparseOperator`) `A`.  The `sparse`
method can be used to convert a sparse operator into a sparse matrix.

Call:

```julia
SparseOperator{T}(args...)
```

to build an instance of `SparseOperator` whose coefficients have type `T`.

!!! note
    For efficiency reasons, sparse operators are currently limited to *fast*
    arrays because they can be indexed linearly with no loss of performances.
    If `C`, `I` and/or `J` are not fast arrays, they will be automatically
    converted to linearly indexed arrays.

See also [`isfastarray`](@ref), [`GeneralMatrix`](@ref) and [`lgemv`](@ref).

"""
struct SparseOperator{T,M,N,
                      Ti<:DenseVector{Int},
                      Tj<:DenseVector{Int},
                      Tc<:DenseVector{T}} <: LinearMapping
    I::Ti                  # Row indices
    J::Tj                  # Column indices
    C::Tc                  # Non-zero coefficients
    rowdims::NTuple{M,Int} # Dimensions of rows
    coldims::NTuple{N,Int} # Dimensions of columns
    samedims::Bool         # Rows and columns have same dimensions

    # The inner constructor checks whether arguments are indeed fast arrays,
    # it is not meant to be directly called.
    function SparseOperator{T,M,N,Ti,Tj,Tc}(
        I::Ti, J::Tj, C::Tc,
        rowdims::NTuple{M,Int},
        coldims::NTuple{N,Int}) where {T,M,N,
                                       Ti<:AbstractVector{Int},
                                       Tj<:AbstractVector{Int},
                                       Tc<:AbstractVector{T}}
        check(new{T,M,N,Ti,Tj,Tc}(I, J, C, rowdims, coldims,
                                  (rowdims == coldims)))
    end
end

@callable SparseOperator

function SparseOperator{T}(I::Ti, J::Tj, C::Tc,
                           rowdims::NTuple{M,Int},
                           coldims::NTuple{N,Int}) where {T,M,N,
                                                          Ti<:DenseVector{Int},
                                                          Tj<:DenseVector{Int},
                                                          Tc<:DenseVector{T}}
    SparseOperator{T,M,N,Ti,Tj,Tc}(I, J, C, rowdims, coldims)
end

function SparseOperator{T}(I::AbstractVector{<:Integer},
                           J::AbstractVector{<:Integer},
                           C::AbstractVector,
                           rowdims::Dimensions,
                           coldims::Dimensions) where {T}
    # Convert indices.
    convert_indices(I::DenseVector{Int}) = I
    convert_indices(I::AbstractVector{<:Integer}) =
        convert(Vector{Int}, I)::Vector{Int}

    # Convert coefficients.
    convert_coefficients(::Type{T}, C::DenseVector{T}) where {T} = C
    convert_coefficients(::Type{T}, C::AbstractVector) where {T} =
        convert(Vector{T}, C)::Vector{T}

    SparseOperator{T}(convert_indices(I), convert_indices(J),
                      convert_coefficients(T, C),
                      dimensions(rowdims), dimensions(coldims))
end

function SparseOperator(I::AbstractVector{<:Integer},
                        J::AbstractVector{<:Integer},
                        C::AbstractVector{T},
                        rowdims::Dimensions,
                        coldims::Dimensions) where {T}
    SparseOperator{T}(I, J, C, rowdims, coldims)
end

SparseOperator(A::SparseOperator) = A
SparseOperator{T}(A::SparseOperator{T}) where {T} = A
SparseOperator{T}(A::SparseOperator) where {T} =
    SparseOperator(rows(A), cols(A), convert(Vector{T}, coefficients(A)),
                   output_size(A), input_size(A))

SparseOperator(A::AbstractArray{T}, n::Integer = 1) where {T} =
    SparseOperator{T}(A, n)

function SparseOperator{T}(A::AbstractArray{S,N}, n::Integer = 1) where {T,S,N}
    1 ≤ n < N || throw(ArgumentError("bad number of of leading dimensions"))
    has_standard_indexing(A) || _bad_indexing()
    dims = size(A)
    rowdims = dims[1:n]
    coldims = dims[n+1:end]
    nrows = prod(rowdims)
    ncols = prod(coldims)
    nz = 0
    @inbounds for l in eachindex(A)
        if A[l] != zero(S)
            nz += 1
        end
    end
    C = Array{T}(undef, nz)
    I = Array{Int}(undef, nz)
    J = Array{Int}(undef, nz)
    i, j, k = 0, 1, 0
    @inbounds for l in eachindex(A)
        if i < nrows
            i += 1
        else
            i = 1
            j += 1
        end
        if (a = A[l]) != zero(S)
            k += 1
            C[k] = a
            I[k] = i
            J[k] = j
        end
    end
    @assert k == nz
    return SparseOperator(I, J, C, rowdims, coldims)
end

SparseOperator(A::SparseMatrixCSC{T,<:Integer}; kwds...) where {T} =
    SparseOperator{T}(A; kwds...)

function SparseOperator{T}(A::SparseMatrixCSC{Tv,Ti}) where {T,Tv,Ti<:Integer}
    nz = length(A.nzval)
    @assert length(A.rowval) == nz
    nrows, ncols = A.m, A.n
    colptr = A.colptr
    @assert length(colptr) == ncols + 1
    @assert colptr[end] == nz + 1
    J = Vector{Int}(undef, nz)
    k2 = colptr[1]
    for j in 1:ncols
        k1, k2 = k2, colptr[j+1]
        @assert k1 ≤ k2 ≤ nz + 1
        @inbounds for k in k1:(k2-1)
            J[k] = j
        end
    end
    return SparseOperator(convert(Vector{Int}, A.rowval)::Vector{Int},
                          J, A.nzval, nrows, ncols)
end

#convert(::Type{T}, A::SparseOperator) where {T<:SparseOperator} = T(A)
#convert(::Type{T}, A::SparseMatrixCSC) where {T<:SparseOperator} = T(A)

eltype(S::SparseOperator{T,M,N}) where {T,M,N} = T
ndims(S::SparseOperator{T,M,N}) where {T,M,N} = N+M

coefficients(S::SparseOperator) = S.C
rows(S::SparseOperator) = S.I
cols(S::SparseOperator) = S.J
output_size(S::SparseOperator) = S.rowdims
input_size(S::SparseOperator) = S.coldims
samedims(S::SparseOperator) = S.samedims
nrows(S::SparseOperator) = prod(output_size(S))
ncols(S::SparseOperator) = prod(input_size(S))

function check(S::SparseOperator{T,M,N}) where {T,M,N}
    @assert isfastarray(coefficients(S), rows(S), cols(S))
    @assert length(coefficients(S)) == length(rows(S)) == length(cols(S))
    @assert samedims(S) == (output_size(S) == input_size(S))
    imin, imax = 1, nrows(S)
    @inbounds for i in rows(S)
        imin ≤ i ≤ imax || throw(AssertionError("out of range row index"))
    end
    jmin, jmax = 1, ncols(S)
    @inbounds for j in cols(S)
        jmin ≤ j ≤ jmax || throw(AssertionError("out of range column index"))
    end
    return S
end

are_same_mappings(A::T, B::T) where {T<:SparseOperator} =
    (coefficients(A) === coefficients(B) && rows(A) === rows(B) && cols(A) === cols(B) &&
     output_size(A) == output_size(B) && input_size(A) == input_size(B))

# FIXME: This cannot be considered as a *pure* trait as it does not only
#        depend on the type of the object.
MorphismType(S::SparseOperator) =
    (samedims(S) ? Endomorphism() : Morphism())

# Convert to a sparse matrix.
sparse(A::SparseOperator) =
    sparse(rows(A), cols(A), coefficients(A), nrows(A), ncols(A))

Array(S::SparseOperator{T}) where {T} =
    unpack!(zeros(T, (output_size(S)..., input_size(S)...,)), S)

Matrix(S::SparseOperator{T}) where {T} =
    unpack!(zeros(T, (nrows(S), ncols(S))), S)

"""
```julia
unpack!(A, S) -> A
```

unpacks the non-zero coefficients of the sparse operator `S` into the array `A`
and returns `A`.

Here `A` must have the same element type as the coefficients of `S` and the
same number of elements as the products of the row and of the column dimensions
of `S`.  Unpacking is perfomed by adding the non-zero coefficients of `S` to
the correponding element of `A` (or using the `|` operator for boolean
elements).

"""
function unpack!(A::Array{T}, S::SparseOperator{T}) where {T}
    @assert length(A) == nrows(S)*ncols(S)
    I, J, C = rows(S), cols(S), coefficients(S)
    len = length(C)
    @assert length(I) == length(J) == len
    stride = nrows(S)
    @inbounds for k in 1:len
        l = (J[k] - 1)*stride + I[k]
        A[l] += C[k]
    end
    return A
end

function unpack!(A::Array{Bool}, S::SparseOperator{Bool})
    @assert length(A) == nrows(S)*ncols(S)
    I, J, C = rows(S), cols(S), coefficients(S)
    len = length(C)
    @assert length(I) == length(J) == len
    stride = nrows(S)
    @inbounds for k in 1:len
        l = (J[k] - 1)*stride + I[k]
        A[l] |= C[k]
    end
    return A
end

reshape(S::SparseOperator, rowdims::Dimensions, coldims::Dimensions) =
    reshape(S, dimensions(rowdims), dimensions(coldims))

function reshape(S::SparseOperator,
                 rowdims::Tuple{Vararg{Int}},
                 coldims::Tuple{Vararg{Int}})
    prod(rowdims) == nrows(S) ||
        throw(DimensionMismatch("products of row dimensions must be equal"))
    prod(coldims) == ncols(S) ||
        throw(DimensionMismatch("products of column dimensions must be equal"))
    return SparseOperator(rows(S), cols(S), coefficients(S), rowdims, coldims)
end

# Extend left multiplication (and division) by a scalar.
function *(α::Number, A::SparseOperator{T})::SparseOperator where {T}
    if α == one(α)
        return A
    elseif α == zero(α)
        nil = Vector{Int}(undef, 0)
        return SparseOperator(nil, nil, Vector{T}(undef, 0),
                              output_size(A), input_size(A))
    else
        return SparseOperator(rows(A), cols(A), vscale(α, coefficients(A)),
                              output_size(A), input_size(A))
    end
end

# Extend left and right composition by a diagonal operator.
function *(W::NonuniformScalingOperator, S::SparseOperator)::SparseOperator
    D = coefficients(W)
    @assert has_standard_indexing(D)
    size(D) == output_size(S) ||
        throw(DimensionMismatch("the non-uniform scaling array and the rows of the sparse operator must have the same dimensions"))
    I, J, C = rows(S), cols(S), coefficients(S)
    T = promote_eltype(D, C)
    return SparseOperator(I, J, _scaleleft(T, D, I, C),
                          output_size(S), input_size(S))
end

function *(S::SparseOperator, W::NonuniformScalingOperator)::SparseOperator
    D = coefficients(W)
    @assert has_standard_indexing(D)
    size(D) == input_size(S) ||
        throw(DimensionMismatch("the non-uniform scaling array and the columns of the sparse operator must have the same dimensions"))
    I, J, C = rows(S), cols(S), coefficients(S)
    T = promote_eltype(D, C)
    return SparseOperator(I, J, _scaleright(T, C, D, J),
                          output_size(S), input_size(S))
end

function _scaleleft(::Type{T},
                    D::AbstractArray,
                    I::AbstractVector{Int},
                    C::AbstractVector) where {T}
    # FIXME: If the sparse operator is "optimized", Q can be undefined and
    #        set instead of incremented.
    len = length(C)
    @assert length(I) == len
    Q = zeros(T, len)
    @inbounds for k in 1:len
        Q[k] += D[I[k]]*C[k]
    end
    return Q
end

function _scaleright(::Type{T},
                     C::AbstractVector,
                     D::AbstractArray,
                     J::AbstractVector{Int}) where {T}
    len = length(C)
    @assert length(J) == len
    Q = zeros(T, len)
    @inbounds for k in 1:len
        Q[k] += C[k]*D[J[k]]
    end
    return Q
end

_bad_input_dimensions() = throw(DimensionMismatch("bad input dimensions"))
_bad_output_dimensions() = throw(DimensionMismatch("bad output dimensions"))

_bad_indexing() =
    throw(ArgumentError("array has non-standard indices"))
_bad_input_indexing() =
    throw(ArgumentError("input array has non-standard indices"))
_bad_output_indexing() =
    throw(ArgumentError("output array has non-standard indices"))

function vcreate(::Type{Direct},
                 S::SparseOperator{<:Any,M,N},
                 x::AbstractArray{<:Any,N},
                 scratch::Bool) where {M,N}
    # In-place operation is not possible so we simply ignore the scratch flag.
    size(x) == input_size(S) || _bad_input_dimensions()
    T = promote_eltype(coefficients(S),x)
    return Array{T}(undef, output_size(S))
end

function vcreate(::Type{Adjoint},
                 S::SparseOperator{<:Any,M,N},
                 x::AbstractArray{<:Any,M},
                 scratch::Bool) where {M,N}
    # In-place operation is not possible so we simply ignore the scratch flag.
    size(x) == output_size(S) || _bad_input_dimensions()
    T = promote_eltype(coefficients(S),x)
    return Array{T}(undef, input_size(S))
end

function apply!(α::Number,
                ::Type{Direct},
                S::SparseOperator{Ts,M,N},
                x::AbstractArray{Tx,N},
                scratch::Bool,
                β::Number,
                y::AbstractArray{Ty,M}) where {Ts,Tx,Ty,M,N}
    size(x) == input_size(S)  || _bad_input_dimensions()
    has_standard_indexing(x)  || _bad_input_indexing()
    size(y) == output_size(S) || _bad_output_dimensions()
    has_standard_indexing(y)  || _bad_output_indexing()
    β != 1 && vscale!(y, β)
    if α != 0
        I, J, C = rows(S), cols(S), coefficients(S)
        n = length(C)
        length(I) == length(J) == n ||
            error("corrupted sparse operator structure")
        if α == 1
            @inbounds for k in Base.OneTo(n)
                i, j, c = I[k], J[k], C[k]
                y[i] += x[j]*c
            end
        elseif α == -1
            @inbounds for k in Base.OneTo(n)
                i, j, c = I[k], J[k], C[k]
                y[i] -= x[j]*c
            end
        else
            # The ordering of operations is to minimize the number of
            # operations in case c is complex while α and x are reals.
            alpha = promote_multiplier(α, Ts, Tx)
            @inbounds for k in Base.OneTo(n)
                i, j, c = I[k], J[k], C[k]
                y[i] += (alpha*x[j])*c
            end
        end
    end
    return y
end

function apply!(α::Number,
                ::Type{Adjoint},
                S::SparseOperator{Ts,M,N},
                x::AbstractArray{Tx,M},
                scratch::Bool,
                β::Number,
                y::AbstractArray{Ty,N}) where {Ts,Tx,Ty,M,N}
    size(x) == output_size(S) || _bad_input_dimensions()
    has_standard_indexing(x)  || _bad_input_indexing()
    size(y) == input_size(S)  || _bad_output_dimensions()
    has_standard_indexing(y)  || _bad_output_indexing()
    β != 1 && vscale!(y, β)
    if α != 0
        I, J, C = rows(S), cols(S), coefficients(S)
        n = length(C)
        length(I) == length(J) == n ||
            error("corrupted sparse operator structure")
        if α == 1
            @inbounds for k in Base.OneTo(n)
                i, j, c = I[k], J[k], C[k]
                y[j] += x[i]*conj(c)
            end
        elseif α == -1
            @inbounds for k in Base.OneTo(n)
                i, j, c = I[k], J[k], C[k]
                y[j] -= x[i]*conj(c)
            end
        else
            # The ordering of operations is to minimize the number of
            # operations in case c is complex while α and x are reals.
            alpha = promote_multiplier(α, Ts, Tx)
            @inbounds for k in Base.OneTo(n)
                i, j, c = I[k], J[k], C[k]
                y[i] += (alpha*x[j])*conj(c)
            end
        end
    end
    return y
end

end # module
