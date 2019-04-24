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
# Copyright (c) 2017-2019 Éric Thiébaut.
#

"""
```julia
SparseOperator(C, I, J, rowdims, coldims)
```

yields a sparse linear map whose non-zero coefficients are given by `C` with
corresponding row and column linear indices given by `I` and `J` and whose
*rows* and *columns* have respective dimensions `rowdims` and `coldims`.

```julia
SparseOperator(A, n=1)
```

yields an instance of `SparseOperator` whose coefficients are the non-zero
coefficients of array `A` and which implements generalized matrix
multiplication by `A` in such a way that the result of applying the operator
are arrays whose dimensions are the `n` leading dimensions of `A` for input
arrays whose dimensions are the remaining trailing dimensions of `A`.

!!! note
    For efficiency reasons, sparse operators are currently limited to *flat*
    Julia arrays because they can be indexed linearly with no loss of
    performances.  If `C`, `I` and/or `J` are not flat arrays, they will be
    automatically converted to regular arrays.

See also [`isflatarray`](@ref), [`GeneralMatrix`](@ref) and [`lgemv`](@ref).

"""
struct SparseOperator{T,M,N,
                      Tc<:AbstractVector{T},
                      Ti<:AbstractVector{Int},
                      Tj<:AbstractVector{Int}} <: LinearMapping
    C::Tc                  # Non-zero coefficients
    I::Ti                  # Row indices
    J::Tj                  # Column indices
    rowdims::NTuple{M,Int} # Dimensions of rows
    coldims::NTuple{N,Int} # Dimensions of columns
    samedims::Bool         # Rows and columns have same dimensions

    # The inner constructor checks whether arguments are indeed flat arrays,
    # it is not meant to be directly called.
    function SparseOperator{T,M,N,Tc,Ti,Tj}(C::Tc,
                                            I::Ti,
                                            J::Tj,
                                            rowdims::NTuple{M,Int},
                                            coldims::NTuple{N,Int}
                                            ) where {T,M,N,
                                                     Tc<:AbstractVector{T},
                                                     Ti<:AbstractVector{Int},
                                                     Tj<:AbstractVector{Int}}
        samedims = (M == N && rowdims == coldims)
        return check(new{T,M,N,Tc,Ti,Tj}(C, I, J, rowdims, coldims, samedims))
    end
end

@callable SparseOperator

# helper to call inner constructor
function _sparseoperator(C::Tc,
                         I::Ti,
                         J::Tj,
                         rowdims::NTuple{M,Int},
                         coldims::NTuple{N,Int}) where {T,M,N,
                                                        Tc<:AbstractVector{T},
                                                        Ti<:AbstractVector{Int},
                                                        Tj<:AbstractVector{Int}}
    return SparseOperator{T,M,N,Tc,Ti,Tj}(C, I, J, rowdims, coldims)
end

function SparseOperator(C::AbstractVector{T},
                        I::AbstractVector{<:Integer},
                        J::AbstractVector{<:Integer},
                        rowdims::Union{Integer,Tuple{Vararg{Integer}}},
                        coldims::Union{Integer,Tuple{Vararg{Integer}}}) where T
    return _sparseoperator(flatvector(T,   C),
                           flatvector(Int, I),
                           flatvector(Int, J),
                           makedims(rowdims),
                           makedims(coldims))
end

function SparseOperator(A::AbstractArray{T,N}, n::Integer = 1) where {T,N}
    1 ≤ n < N || throw(ArgumentError("bad number of of leading dimensions"))
    has_standard_indexing(A) || _bad_indexing()
    dims = size(A)
    rowdims = dims[1:n]
    coldims = dims[n+1:end]
    nrows = prod(rowdims)
    ncols = prod(coldims)
    nz = 0
    @inbounds for k in eachindex(A)
        if A[k] != zero(T)
            nz += 1
        end
    end
    C = Array{T}(undef, nz)
    I = Array{Int}(undef, nz)
    J = Array{Int}(undef, nz)
    i, j, l = 0, 1, 0
    @inbounds for k in eachindex(A)
        if i < nrows
            i += 1
        else
            i = 1
            j += 1
        end
        if (a = A[k]) != zero(T)
            l += 1
            C[l] = a
            I[l] = i
            J[l] = j
        end
    end
    @assert l == nz
    return SparseOperator(C, I, J, rowdims, coldims)
end

function SparseOperator(A::SparseMatrixCSC{Tv,Ti};
                        copy::Bool=false) where {Tv,Ti<:Integer}
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
    return SparseOperator((copy ? copyto!(Vector{Tv}(undef, nz), A.nzval) :
                           flatvector(Tv, A.nzval)),
                          (copy ? copyto!(Vector{Int}(undef, nz), A.rowval) :
                           flatvector(Int, A.rowval)), J, nrows, ncols)
end

coefs(S::SparseOperator) = S.C
rows(S::SparseOperator) = S.I
cols(S::SparseOperator) = S.J
rowdims(S::SparseOperator) = S.rowdims
coldims(S::SparseOperator) = S.coldims
samedims(S::SparseOperator) = S.samedims
nrows(S::SparseOperator) = prod(rowdims(S))
ncols(S::SparseOperator) = prod(coldims(S))

input_size(S::SparseOperator) = coldims(S)
output_size(S::SparseOperator) = rowdims(S)

"""
```julia
check(A) -> A
```

checks integrity of operator `A` and returns it.

"""
function check(S::SparseOperator{T,M,N}) where {T,M,N}
    @assert isflatarray(coefs(S), rows(S), cols(S))
    @assert samedims(S) == (M == N && rowdims(S) == coldims(S))
    imin, imax = extrema(rows(S))
    jmin, jmax = extrema(cols(S))
    @assert 1 ≤ imin ≤ imax ≤ prod(rowdims(S))
    @assert 1 ≤ jmin ≤ jmax ≤ prod(coldims(S))
    return S
end

function is_same_mapping(A::SparseOperator{T,M,N,Tc,Ti,Tj},
                         B::SparseOperator{T,M,N,Tc,Ti,Tj}) where {T,M,N,
                                                                   Tc,Ti,Tj}
    return (is_same_mutable_object(coefs(A), coefs(B)) &&
            is_same_mutable_object(rows(A), rows(B)) &&
            is_same_mutable_object(cols(A), cols(B)) &&
            rowdims(A) == rowdims(B) &&
            coldims(A) == coldims(B))
end

EndomorphismType(S::SparseOperator) =
    (samedims(S) ? Endomorphism() : Morphism())

# Convert to a sparse matrix (silently "flatten" the operator if columns/rows
# were multi-dimensional).
sparse(A::SparseOperator) =
    sparse(rows(A), cols(A), coefs(A), nrows(A), ncols(A))

Base.Array(S::SparseOperator{T}) where {T} =
    unpack!(zeros(T, (rowdims(S)..., coldims(S)...)), S)

Base.Matrix(S::SparseOperator{T}) where {T} =
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
    I, J, C = rows(S), cols(S), coefs(S)
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
    I, J, C = rows(S), cols(S), coefs(S)
    len = length(C)
    @assert length(I) == length(J) == len
    stride = nrows(S)
    @inbounds for k in 1:len
        l = (J[k] - 1)*stride + I[k]
        A[l] |= C[k]
    end
    return A
end

function Base.reshape(S::SparseOperator,
                      rowdims::Union{Integer,Tuple{Vararg{Integer}}},
                      coldims::Union{Integer,Tuple{Vararg{Integer}}})
    return reshape(S, makedims(rowdims), makedims(coldims))
end


function Base.reshape(S::SparseOperator,
                      rowdims::Tuple{Vararg{Int}},
                      coldims::Tuple{Vararg{Int}})
    prod(rowdims) == nrows(S) ||
        throw(DimensionMismatch("product of row dimensions must be equal"))
    prod(coldims) == ncols(S)) ||
        throw(DimensionMismatch("product of column dimensions must be equal"))
    return SparseOperator(coefs(S), rows(S), cols(S), rowdims, coldims)
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
                 S::SparseOperator{Ts,M,N},
                 x::AbstractArray{Tx,N},
                 scratch::Bool=false) where {Ts<:Real,Tx<:Real,M,N}
    # In-place operation is not possible so we simply ignore the scratch flag.
    size(x) == coldims(S) || _bad_input_dimensions()
    return Array{promote_type(Ts,Tx)}(undef, rowdims(S))
end

function vcreate(::Type{Adjoint},
                 S::SparseOperator{Ts,M,N},
                 x::AbstractArray{Tx,M},
                 scratch::Bool=false) where {Ts<:Real,Tx<:Real,M,N}
    # In-place operation is not possible so we simply ignore the scratch flag.
    size(x) == rowdims(S) || _bad_input_dimensions()
    return Array{promote_type(Ts,Tx)}(undef, coldims(S))
end

function apply!(α::Real,
                ::Type{Direct},
                S::SparseOperator{Ts,M,N},
                x::AbstractArray{Tx,N},
                scratch::Bool,
                β::Real,
                y::AbstractArray{Ty,M}) where {Ts<:Number,Tx<:Number,
                                               Ty<:Number,M,N}
    size(x) == coldims(S)    || _bad_input_dimensions()
    has_standard_indexing(x) || _bad_input_indexing()
    size(y) == rowdims(S)    || _bad_output_dimensions()
    has_standard_indexing(y) || _bad_output_indexing()
    β == 1 || vscale!(y, β)
    α == 0 || _apply_sparse!(y,
                             convert_multiplier(α, promote_type(Ts, Tx), Ty),
                             coefs(S), rows(S), cols(S), x)
    return y
end

function apply!(α::Real,
                ::Type{Adjoint},
                S::SparseOperator{Ts,M,N},
                x::AbstractArray{Tx,M},
                scratch::Bool,
                β::Real,
                y::AbstractArray{Ty,N}) where {Ts<:Number,Tx<:Number,
                                               Ty<:Number,M,N}
    size(x) == rowdims(S)    || _bad_input_dimensions()
    has_standard_indexing(x) || _bad_input_indexing()
    size(y) == coldims(S)    || _bad_output_dimensions()
    has_standard_indexing(y) || _bad_output_indexing()
    β == 1 || vscale!(y, β)
    α == 0 || _apply_sparse!(y, convert_multiplier(α, promote_type(Ts, Tx), Ty),
                             coefs(S), cols(S), rows(S), x)
    return y
end

function _apply_sparse!(y::AbstractArray{<:Real},
                        α::Real,
                        C::AbstractVector{<:Real},
                        I::AbstractVector{Int},
                        J::AbstractVector{Int},
                        x::AbstractArray{<:Real})
    # We already known that α is non-zero.
    length(I) == length(J) == length(C) ||
        error("corrupted sparse operator structure")
    if α == 1
        @inbounds for k in 1:length(C)
            c, i, j = C[k], I[k], J[k]
            y[i] += c*x[j]
        end
    elseif α == -1
        @inbounds for k in 1:length(C)
            c, i, j = C[k], I[k], J[k]
            y[i] -= c*x[j]
        end
    else
        @inbounds for k in 1:length(C)
            c, i, j = C[k], I[k], J[k]
            y[i] += α*c*x[j]
        end
    end
end

function _apply_sparse!(y::AbstractArray{<:Complex},
                        α::Real,
                        C::AbstractVector{<:Complex},
                        I::AbstractVector{Int},
                        J::AbstractVector{Int},
                        x::AbstractArray{<:Real})
    # We already known that α is non-zero.
    if α == 1
        @inbounds for k in 1:length(C)
            c, i, j = C[k], I[k], J[k]
            y[i] += c*x[j]
        end
    elseif α == -1
        @inbounds for k in 1:length(C)
            c, i, j = C[k], I[k], J[k]
            y[i] -= c*x[j]
        end
    else
        @inbounds for k in 1:length(C)
            c, i, j = C[k], I[k], J[k]
            y[i] += c*(α*x[j])
        end
    end
end

function _apply_sparse!(y::AbstractArray{<:Complex},
                        α::Real,
                        C::AbstractVector{<:Complex},
                        I::AbstractVector{Int},
                        J::AbstractVector{Int},
                        x::AbstractArray{<:Complex})
    # We already known that α is non-zero.
    if α == 1
        @inbounds for k in 1:length(C)
            c, i, j = C[k], I[k], J[k]
            y[i] += c*x[j]
        end
    elseif α == -1
        @inbounds for k in 1:length(C)
            c, i, j = C[k], I[k], J[k]
            y[i] -= c*x[j]
        end
    else
        @inbounds for k in 1:length(C)
            c, i, j = C[k], I[k], J[k]
            y[i] += α*c*x[j]
        end
    end
end
