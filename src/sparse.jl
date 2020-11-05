#
# sparse.jl --
#
# Implement various format of compressed sparse linear operators.  The purpose
# of this module is mainly to convert between different formats and to provide
# LazyAlgebra wrappers to apply the corresponding linear mappings.  Julia's
# SparseArrays standard package only provides "Compressed Sparse Column" (CSC)
# format.
#
# See https://en.wikipedia.org/wiki/Sparse_matrix.
#
#------------------------------------------------------------------------------
#
# This file is part of LazyAlgebra (https://github.com/emmt/LazyAlgebra.jl)
# released under the MIT "Expat" license.
#
# Copyright (C) 2017-2020, Éric Thiébaut.
#
module SparseOperators

export
    CompressedSparseOperator,
    SparseOperator,
    SparseOperatorCOO,
    SparseOperatorCSC,
    SparseOperatorCSR,
    nrows,
    ncols,
    row_size,
    col_size,
    nonzeros,
    nnz

using StructuredArrays
using ZippedArrays

import LinearAlgebra

using ..LazyAlgebra
using .LazyAlgebra:
    Adjoint,
    Direct,
    Inverse,
    InverseAdjoint,
    Endomorphism,
    Morphism,
    promote_multiplier,
    axpby_yields_x,
    axpby_yields_xpy,
    axpby_yields_xmy,
    axpby_yields_xpby,
    axpby_yields_ax,
    axpby_yields_axpy,
    axpby_yields_axmy,
    axpby_yields_axpby,
    @callable

import .LazyAlgebra:
    MorphismType,
    apply!,
    vcreate,
    identical,
    coefficients,
    row_size,
    col_size,
    nrows,
    ncols,
    input_ndims,
    input_size,
    output_ndims,
    output_size

import SparseArrays
using SparseArrays: SparseMatrixCSC, nonzeros, nnz
if isdefined(SparseArrays, :AbstractSparseMatrixCSC)
    const AbstractSparseMatrixCSC{Tv,Ti} =
        SparseArrays.AbstractSparseMatrixCSC{Tv,Ti}
else
    const AbstractSparseMatrixCSC{Tv,Ti} = SparseArrays.SparseMatrixCSC{Tv,Ti}
end

import Base: getindex, setindex!, iterate
using Base: @propagate_inbounds

#------------------------------------------------------------------------------
# Convert to integer type suitable for indexing.
to_int(i::Int) = i
to_int(i::Integer) = Int(i)

# Convert to vector of indices.
to_indices(inds::AbstractVector{<:Integer}) = to_values(Int, inds)

# Convert to vector of values with given element type and make sure it is a
# fast vector.
to_values(vals::AbstractVector{T}) where {T} = to_values(T, vals)
to_values(::Type{T}, vals::Vector{T}) where {T} = vals
to_values(::Type{T}, vals::AbstractVector) where {T} = convert(Vector{T}, vals)
@inline to_values(::Type{T}, vals::AbstractVector{T}) where {T} =
    _to_values(T, vals, eachindex(vals))

@inline _to_values(::Type{T}, vals::AbstractVector, inds) where {T} =
    convert(Vector{T}, vals) # Convert because not a fast vector.

@inline function _to_values(::Type{T}, vals::AbstractVector,
                            inds::AbstractUnitRange{Int}) where {T}
    (first(inds) == 1 ? vals : convert(Vector{T}, vals))
end

# Union of types acceptable to define array size and methods to convert to
# canonic form.
const ArraySize = Union{Integer,Tuple{Vararg{Integer}}}
to_size(siz::Tuple{Vararg{Int}}) = siz
to_size(siz::Tuple{Vararg{Integer}}) = map(to_int, siz)
to_size(siz::Integer) = (to_int(siz),)

as_matrix(A::AbstractMatrix, nrows::Int, ncols::Int) =
    (@assert nrows*ncols == length(A); A)
as_matrix(A::AbstractArray, nrows::Int, ncols::Int) =
    reshape(A, (nrows, ncols))

#------------------------------------------------------------------------------

"""

`SparseOperator{T,M,N}` is the abstract type inherited by the sparse operator
types. Parameter `T` is the type of the elements.  Parameters `M` and `N` are
the number of dimensions of the *rows* and of the *columns* respectively.
Sparse operators are a generalization of sparse matrices in the sense that they
implement linear mappings which can be applied to `N`-dimensonal arguments to
produce `M`-dimensional results (as explained below).  See
[`GeneralMatrix`](@ref) for a similar generalization but for *dense* matrices.

See [`CompressedSparseOperator`](@ref) for usage of sparse operators
implementing compressed storage formats.

"""
abstract type SparseOperator{T,M,N} <: LinearMapping end

"""

`CompressedSparseOperator{F,T,M,N}` is an abstract sub-type of
`SparseOperator{T,M,N}` and is inherited by the concrete types implementing
sparse operators with compressed storage in format `F`.

Format `F` is specificed as a symbol and can be:

- `:COO` for *Compressed Sparse Coordinate* storage format.  This format is not
  the most efficient, it is mostly used as an intermediate for building a
  sparse operator in one of the following format.

- `:CSC` for *Compressed Sparse Column* storage format.  This format is very
  efficient for applying the adjoint of the sparse operator.

- `:CSR` for *Compressed Sparse Row* storage format.  This format is very
  efficient for directly applying the sparse operator.

To construct (or convert to) a sparse operator with compressed storage format
`F`, you can call:

    CompressedSparseOperator{F}(args...; kwds...)
    CompressedSparseOperator{F,T}(args...; kwds...)
    CompressedSparseOperator{F,T,M}(args...; kwds...)
    CompressedSparseOperator{F,T,M,N}(args...; kwds...)

where given parameters `T`, `M` and `N`, arguments `args...` and optional
keywords `kwds...` will be passed to the concrete constructor
[`SparseOperatorCOO`](@ref), [`SparseOperatorCSC`](@ref) or
[`SparseOperatorCSR`](@ref) corresponding to the format `F`.

It is possible to use a compressed sparse operator `A` as an iterator:

```julia
for (Aij,i,j) in A # simple but slow for CSR and CSC
    ...
end
```

to retrieve the values `Aij` and respective row `i` and column `j` indices for
all the entries stored in `A`.  It is however more efficient to access them
according to their storage order which depends on the compressed format.

- If `A` is in CSC format:

  ```julia
  using LazyAlgebra.SparseMethods
  for j in each_col(A)        # loop over column index
      for k in each_off(A, j) # loop over structural non-zeros in this column
          i   = get_row(A, k) # get row index of entry
          Aij = get_val(A, k) # get value of entry
       end
  end
  ```

- If `A` is in CSR format:

  ```julia
  using LazyAlgebra.SparseMethods
  for i in each_row(A)        # loop over row index
      for k in each_off(A, i) # loop over structural non-zeros in this row
          j   = get_col(A, k) # get column index of entry
          Aij = get_val(A, k) # get value of entry
       end
  end
  ```

- If `A` is in COO format:

  ```julia
  using LazyAlgebra.SparseMethods
  for k in each_off(A)
       i   = get_row(A, k) # get row index of entry
       j   = get_col(A, k) # get column index of entry
       Aij = get_val(A, k) # get value of entry
  end
  ```

The low-level methods `each_row`, `each_col`, `each_off`, `get_row`, `get_col`
and `get_val` are not automatically exported by `LazyAlgebra`, this is the
purpose of the statement `using LazyAlgebra.SparseMethods`.

"""
abstract type CompressedSparseOperator{F,T,M,N} <: SparseOperator{T,M,N} end

struct SparseOperatorCSR{T,M,N,
                         V<:AbstractVector{T},
                         J<:AbstractVector{Int},
                         K<:AbstractVector{Int}
                         } <: CompressedSparseOperator{:CSR,T,M,N}
    m::Int                # equivalent number of rows of the operator
    n::Int                # number of columns of the operator
    vals::V               # values of entries
    cols::J               # linear column indices of entries
    offs::K               # row offsets in arrays of entries and column indices
    rowsiz::NTuple{M,Int} # dimensions of rows
    colsiz::NTuple{N,Int} # dimensions of columns

    # An inner constructor is defined to prevent Julia from providing a simple
    # outer constructor, it is not meant to be called directly as it does not
    # check whether arguments are correct.
    function SparseOperatorCSR{T,M,N,V,J,K}(m::Int,
                                            n::Int,
                                            vals::V,
                                            cols::J,
                                            offs::K,
                                            rowsiz::NTuple{M,Int},
                                            colsiz::NTuple{N,Int}) where {
                                                T,M,N,
                                                V<:AbstractVector{T},
                                                J<:AbstractVector{Int},
                                                K<:AbstractVector{Int}}
        new{T,M,N,V,J,K}(m, n, vals, cols, offs, rowsiz, colsiz)
    end
end

struct SparseOperatorCSC{T,M,N,
                         V<:AbstractVector{T},
                         I<:AbstractVector{Int},
                         K<:AbstractVector{Int}
                         } <: CompressedSparseOperator{:CSC,T,M,N}
    m::Int                # equivalent number of rows of the operator
    n::Int                # number of columns of the operator
    vals::V               # values of entries
    rows::I               # linear row indices of entries
    offs::K               # columns offsets in arrays of entries and row indices
    rowsiz::NTuple{M,Int} # dimensions of rows
    colsiz::NTuple{N,Int} # dimensions of columns

    # An inner constructor is defined to prevent Julia from providing a simple
    # outer constructor, it is not meant to be called directly as it does not
    # check whether arguments are correct.
    function SparseOperatorCSC{T,M,N,V,I,K}(m::Int,
                                            n::Int,
                                            vals::V,
                                            rows::I,
                                            offs::K,
                                            rowsiz::NTuple{M,Int},
                                            colsiz::NTuple{N,Int}) where {
                                                T,M,N,
                                                V<:AbstractVector{T},
                                                I<:AbstractVector{Int},
                                                K<:AbstractVector{Int}}
        new{T,M,N,V,I,K}(m, n, vals, rows, offs, rowsiz, colsiz)
    end
end

struct SparseOperatorCOO{T,M,N,
                         V<:AbstractVector{T},
                         I<:AbstractVector{Int},
                         J<:AbstractVector{Int}
                         } <: CompressedSparseOperator{:COO,T,M,N}
    m::Int                # equivalent number of rows of the operator
    n::Int                # number of columns of the operator
    vals::V               # values of entries
    rows::I               # linear row indices of entries
    cols::J               # linear column indices of entries
    rowsiz::NTuple{M,Int} # dimensions of rows
    colsiz::NTuple{N,Int} # dimensions of columns

    # An inner constructor is defined to prevent Julia from providing a simple
    # outer constructor, it is not meant to be called directly as it does not
    # check whether arguments are correct.
    function SparseOperatorCOO{T,M,N,V,I,J}(m::Int,
                                            n::Int,
                                            vals::V,
                                            rows::I,
                                            cols::J,
                                            rowsiz::NTuple{M,Int},
                                            colsiz::NTuple{N,Int}) where {
                                                T,M,N,
                                                V<:AbstractVector{T},
                                                I<:AbstractVector{Int},
                                                J<:AbstractVector{Int}}
        new{T,M,N,V,I,J}(m, n, vals, rows, cols, rowsiz, colsiz)
    end
end

# Unions of compressed sparse operators that can be considered as being in a
# given storage format.

const AnyCSR{T,M,N} = Union{CompressedSparseOperator{:CSR,T,M,N},
                            Adjoint{<:CompressedSparseOperator{:CSC,T,M,N}}}

const AnyCSC{T,M,N} = Union{CompressedSparseOperator{:CSC,T,M,N},
                            Adjoint{<:CompressedSparseOperator{:CSR,T,M,N}}}

const AnyCOO{T,M,N} = Union{CompressedSparseOperator{:COO,T,M,N},
                            Adjoint{<:CompressedSparseOperator{:COO,T,M,N}}}

#------------------------------------------------------------------------------
# Accessors and basic methods.

nrows(A::SparseOperator) = getfield(A, :m)
ncols(A::SparseOperator) = getfield(A, :n)
row_size(A::SparseOperator) = getfield(A, :rowsiz)
col_size(A::SparseOperator) = getfield(A, :colsiz)
output_size(A::SparseOperator) = row_size(A)
input_size(A::SparseOperator) = col_size(A)
output_ndims(A::SparseOperator{T,M,N}) where {T,M,N} = M
input_ndims(A::SparseOperator{T,M,N}) where {T,M,N} = N

Base.eltype(A::SparseOperator{T,M,N}) where {T,M,N} = T
Base.ndims(A::SparseOperator{T,M,N}) where {T,M,N} = M+N
Base.length(A::SparseOperator) = nrows(A)*ncols(A)
Base.size(A::SparseOperator) = (row_size(A)..., col_size(A)...)

# FIXME: This cannot be considered as a *pure* trait as it does not only
#        depend on the type of the object.
MorphismType(A::SparseOperator) =
    (row_size(A) == col_size(A) ? Endomorphism() : Morphism())

coefficients(A::SparseOperator) = get_vals(A)

identical(A::T, B::T) where {T<:CompressedSparseOperator{:CSR}} =
    (get_vals(A) === get_vals(B) && get_cols(A) === get_cols(B) &&
     get_offs(A) === get_offs(B) &&
     row_size(A) == row_size(B) && col_size(A) == col_size(B))

identical(A::T, B::T) where {T<:CompressedSparseOperator{:CSC}} =
    (get_vals(A) === get_vals(B) && get_rows(A) === get_rows(B) &&
     get_offs(A) === get_offs(B) &&
     row_size(A) == row_size(B) && col_size(A) == col_size(B))

identical(A::T, B::T) where {T<:CompressedSparseOperator{:COO}} =
    (get_vals(A) === get_vals(B) && get_rows(A) === get_rows(B) &&
     get_cols(A) === get_cols(B) &&
     row_size(A) == row_size(B) && col_size(A) == col_size(B))

# Assume that a copy of a compressed sparse linear operator is to keep the same
# structure for the structural non-zeros but possibly change the values.  So
# only duplicate the value part.

Base.copy(A::SparseOperatorCSR{T,M,N}) where {T,M,N} =
    unsafe_csr(nrows(A), ncols(A), copy_vals(A), get_cols(A), get_offs(A),
               row_size(A), col_size(A))

Base.copy(A::SparseOperatorCSC{T,M,N}) where {T,M,N} =
    unsafe_csc(nrows(A), ncols(A), copy_vals(A), get_rows(A), get_offs(A),
               row_size(A), col_size(A))

Base.copy(A::SparseOperatorCOO{T,M,N}) where {T,M,N} =
    unsafe_coo(nrows(A), ncols(A), copy_vals(A), get_rows(A), get_cols(A),
               row_size(A), col_size(A))

# `findnz(A) -> I,J,V` yields the row and column indices and the values of the
# stored values in A`.
SparseArrays.findnz(A::SparseOperator) =
    (get_rows(A), get_cols(A), get_vals(A))

# Extend some methods in SparseArrays.  The "structural" non-zeros are the
# entries stored by the sparse structure, they may or not be equal to zero
# un-stored entries are always considered as being equal to zero.
SparseArrays.nonzeros(A::SparseOperator) = get_vals(A)
SparseArrays.nnz(A::SparseOperator) = length(nonzeros(A))

"""
    get_vals(A)

yields the array storing the values of the sparse linear operator `A`.  The
returned array is shared with `A`, call `copy_vals(A)` instead if you want to
modify the contents of the returned array with no side effects on `A`.

As a convenience, argument may also be the adjoint of a sparse linear
operator:

    get_vals(A') -> get_vals(A)

which yields the **unmodified** values of `A`, hence the caller has to take the
conjugate of these values.  The method `get_val(A',k)` however takes care of
conjugating the values.

"""
get_vals(A::SparseOperator) = getfield(A, :vals)
get_vals(A::Adjoint{<:SparseOperator}) = get_vals(unveil(A))

"""
    copy_vals([T = eltype(A),] A) -> vals

yields a copy of the values of the entries in sparse linear operator `A`
converted to type `T`.  The result is a vector that is not shared by `A`, the
caller may thus modify its contents with no side effects on `A`.

"""
copy_vals(A::SparseOperator{T}) where {T} = copy_vals(T, A)
function copy_vals(::Type{T}, A::SparseOperator) where {T}
    vals = get_vals(A)
    copyto!(Vector{T}(undef, size(vals)), vals)
end

"""
    get_rows(A)

yields the row indices of the entries of the sparse linear operator `A`.  The
returned array may be shared with `A`, call `copy_rows(A)` instead if you want
to modify the contents of the returned array with no side effects on `A`.

"""
get_rows(A::SparseOperatorCSC) = getfield(A, :rows)
get_rows(A::SparseOperatorCOO) = getfield(A, :rows)
get_rows(A::CompressedSparseOperator{:CSR}) =
    copy_rows(A) # FIXME: yield an iterator
get_rows(A::Adjoint{<:SparseOperator}) = get_cols(unveil(A))

"""
    copy_rows(A) -> rows

yields a copy of the linear row indices of entries in sparse linear operator
`A`.  The result is a vector that is not shared by `A`, the caller may thus
modify its contents with no side effects on `A`.

"""
function copy_rows(A::SparseOperator)
    rows = get_rows(A)
    copyto!(Vector{Int}(undef, size(rows)), rows)
end
function copy_rows(A::CompressedSparseOperator{:CSR})
    rows = Vector{Int}(undef, length(get_vals(A)))
    @inbounds for i in each_row(A)
        @simd for k in each_off(A, i)
            rows[k] = i
        end
    end
    return rows
end

"""
    get_cols(A)

yields the column indices of the entries of the sparse linear operator `A`.
The returned array may be shared with `A`, call `copy_cols(A)` instead if you
want to modify the contents of the returned array with no side effects on `A`.

"""
get_cols(A::SparseOperatorCSR) = getfield(A, :cols)
get_cols(A::SparseOperatorCOO) = getfield(A, :cols)
get_cols(A::Union{CompressedSparseOperator{:CSC},SparseMatrixCSC}) =
    copy_cols(A) # FIXME: yield an iterator
get_cols(A::Adjoint{<:SparseOperator}) = get_rows(unveil(A))

"""
    copy_cols(A) -> cols

yields a copy of the linear column indices of entries in sparse linear operator
`A`.  The result is a vector that is not shared by `A`, the caller may thus
modify its contents with no side effects on `A`.

"""
function copy_cols(A::SparseOperator)
    cols = get_cols(A)
    copyto!(Vector{Int}(undef, size(cols)), cols)
end
function copy_cols(A::Union{CompressedSparseOperator{:CSC},SparseMatrixCSC})
    cols = Vector{Int}(undef, length(get_vals(A)))
    @inbounds for j in each_col(A)
        @simd for k in each_off(A, j)
            cols[k] = j
        end
    end
    return cols
end

"""
    get_offs(A)

yields the table of offsets of the sparse linear operator `A`.  Not all
operators extend this method.

!!! warning
    The interpretation of offsets depend on the type of `A`.  For instance,
    assuming `offs = get_offs(A)`, then the index range of the `j`-th column of
    a `SparseMatrixCSC` is `offs[j]:(offs[j+1]-1)` while the index range is
    `(offs[j]+1):offs[j+1]` for a `SparseOperatorCSC`.  For this reason,
    it is recommended to call [`each_off`](@ref) instead or to call `get_offs`
    with 2 arguments as shown below.

For a transparent usage of the offsets, the method should be called with 2
arguments.

    get_offs(A, i) -> k1, k2

yields the offsets of the first and last elements in the arrays of values and
linear column indices for the `i`-th row of the sparse linear operator `A`
stored in a *Compressed Sparse Row* (CSR) format.  If `k2 < k1`, it means that
the `i`-th row is empty.  Calling `each_off(A,i)` directly yields `k1:k2`.

    get_offs(A, j) -> k1, k2

yields the offsets of the first and last elements in the arrays of values and
linear row indices for the `j`-th column of the sparse linear operator `A`
stored in a *Compressed Sparse Column* (CSC) format.  If `k2 < k1`, it means
that the `j`-th column is empty.  Calling `each_off(A,j)` directly yields
`k1:k2`.

"""
get_offs(A::SparseOperatorCSR) = getfield(A, :offs)
get_offs(A::SparseOperatorCSC) = getfield(A, :offs)
get_offs(A::Adjoint{<:CompressedSparseOperator{:CSR}}) = get_offs(unveil(A))
get_offs(A::Adjoint{<:CompressedSparseOperator{:CSC}}) = get_offs(unveil(A))

@inline function get_offs(A::AnyCSR, i::Int)
    offs = get_offs(A)
    @boundscheck ((i < 1)|(i ≥ length(offs))) && out_of_range_row_index(A, i)
    return ((@inbounds offs[i] + 1),
            (@inbounds offs[i+1]))
end

@inline function get_offs(A::AnyCSC, j::Int)
    offs = get_offs(A)
    @boundscheck ((j < 1)|(j ≥ length(offs))) && out_of_range_column_index(A, j)
    return ((@inbounds offs[j] + 1),
            (@inbounds offs[j+1]))
end

@noinline out_of_range_row_index(A, i::Integer) =
    throw(ErrorException(string("out of range row index ", i,
                                " for sparse linear operator with ", nrows(A),
                                " rows")))

@noinline out_of_range_column_index(A, j::Integer) =
    throw(ErrorException(string("out of range column index ", j,
                                " for sparse linear operator with ", ncols(A),
                                " columns")))

"""
    each_off(A, i)

yields an iterator over the indices in the arrays of values and linear column
indices for the `i`-th row of the sparse linear operator `A` stored in a
*Compressed Sparse Row* (CSR) format.

    each_off(A, j)

yields an iterator over the indices in the arrays of values and linear row
indices for the `j`-th column of the sparse linear operator `A` stored in a
*Compressed Sparse Column* (CSC) format.

    each_off(A)

yields an iterator over the indices in the arrays of values and of linear row
and column indices for the `k`-th entry of the sparse linear operator `A`
stored in a *Compressed Sparse Coordinate* (COO) format.

"""
@inline each_off(A::CompressedSparseOperator{:COO}) = Base.OneTo(nnz(A))
@inline each_off(A::Adjoint{<:CompressedSparseOperator{:COO}}) =
    each_off(unveil(A))

@propagate_inbounds @inline function each_off(A::AnyCSR, i::Int)
    k1, k2 = get_offs(A, i)
    return k1:k2
end

@propagate_inbounds @inline function each_off(A::AnyCSC, j::Int)
    k1, k2 = get_offs(A, j)
    return k1:k2
end

"""
    each_row(A)

yields an iterator over the linear row indices of the sparse linear operator
`A` stored in a *Compressed Sparse Row* (CSR) format.

"""
each_row(A::CompressedSparseOperator{:CSR}) = Base.OneTo(nrows(A))
each_row(A::Adjoint{<:CompressedSparseOperator{:CSC}}) = each_col(unveil(A))

"""
    each_col(A)

yields an iterator over the linear column indices of the sparse linear operator
`A` stored in a *Compressed Sparse Column* (CSC) format.

"""
each_col(A::CompressedSparseOperator{:CSC}) = Base.OneTo(ncols(A))
each_col(A::Adjoint{<:CompressedSparseOperator{:CSR}}) = each_row(unveil(A))

"""
    get_row(A, k) -> i

yields the linear row index of the `k`-th entry of the sparse linear operator
`A` stored in a *Compressed Sparse Column* (CSC) or *Coordinate* (COO) format.

"""
@propagate_inbounds @inline get_row(A, k::Integer) = get_rows(A)[k]

"""
    get_col(A, k) -> j

yields the linear column index of the `k`-th entry of the sparse linear
operator `A` stored in a *Compressed Sparse Row* (CSR) or *Coordinate* (COO)
format.

"""
@propagate_inbounds @inline get_col(A, k::Integer) = get_cols(A)[k]

"""
    get_val(A, k) -> v

yields the value of the `k`-th entry of the sparse linear operator `A` stored
in a *Compressed Sparse Row* (CSR), *Compressed Sparse Column* (CSC) or
*Coordinate* (COO) format.

Argument may also be the adjoint of a sparse linear operator:

    get_val(A', k) -> conj(get_val(A, k))

"""
@propagate_inbounds @inline get_val(A, k::Integer) = get_vals(A)[k]
@propagate_inbounds @inline get_val(A::Adjoint, k::Integer) =
    conj(get_vals(A)[k])

"""
    set_val!(A, k, v) -> v

assigns `v` to the value of the `k`-th entry of the sparse linear operator `A`
stored in a *Compressed Sparse Row* (CSR), *Compressed Sparse Column* (CSC) or
*Coordinate* (COO) format.

"""
@propagate_inbounds @inline set_val!(A, k::Integer, v) = get_vals(A)[k] = v
@propagate_inbounds @inline set_val!(A::Adjoint, k::Integer, v) = begin
    get_vals(A)[k] = conj(v)
    return v
end


# Iterators to deliver (v,i,j).

@inline function Base.iterate(A::AnyCSR, state::Tuple{Int,Int,Int} = (0,0,0))
    i, k, kmax = state
    @inbounds begin
        k += 1
        while k > kmax
            if i ≥ nrows(A)
                return nothing
            end
            i += 1
            k, kmax = get_offs(A, i)
        end
        v = get_val(A, k)
        j = get_col(A, k)
        return ((v, i, j), (i, k, kmax))
    end
end

@inline function Base.iterate(A::AnyCSC, state::Tuple{Int,Int,Int} = (0,0,0))
    j, k, kmax = state
    @inbounds begin
        k += 1
        while k > kmax
            if j ≥ ncols(A)
                return nothing
            end
            j += 1
            k, kmax = get_offs(A, j)
        end
        v = get_val(A, k)
        i = get_row(A, k)
        return ((v, i, j), (j, k, kmax))
    end
end

@inline function Base.iterate(A::AnyCOO, state::Tuple{Int,Int} = (0, nnz(A)))
    k, kmax = state
    @inbounds begin
        if k < kmax
            k += 1
            return ((get_val(A, k), get_row(A, k), get_col(A, k)), (k, kmax))
        else
            return nothing
        end
    end
end

#------------------------------------------------------------------------------
# Extend methods for SparseMatrixCSC defined in SparseArrays.

nrows(A::SparseMatrixCSC) = getfield(A, :m)
ncols(A::SparseMatrixCSC) = getfield(A, :n)
get_vals(A::SparseMatrixCSC) = getfield(A, :nzval)
get_offs(A::SparseMatrixCSC) = getfield(A, :colptr)
get_rows(A::SparseMatrixCSC) = getfield(A, :rowval)
# get_cols is already done elsewhere.
row_size(A::SparseMatrixCSC) = (nrows(A),)
col_size(A::SparseMatrixCSC) = (ncols(A),)
each_col(A::SparseMatrixCSC) = Base.OneTo(ncols(A))

@propagate_inbounds @inline each_off(A::SparseMatrixCSC, j::Integer) =
    ((k1, k2) = get_offs(A, j); k1:k2)

# Provide a specific version of `get_offs(A,j)` because offsets have a slightly
# different difinition than out CSC format.
@inline function get_offs(A::SparseMatrixCSC, j::Integer)
    offs = get_offs(A)
    @boundscheck ((j < 1)|(j ≥ length(offs))) && out_of_range_column_index(A, j)
    return ((@inbounds offs[j]),
            (@inbounds offs[j+1]-1))
end

#------------------------------------------------------------------------------
# Constructors.

"""
    SparseOperatorCSR{T,M,N}(A, sel = (v,i,j) -> (v != zero(v)))

yields a sparse linear operator in *Compressed Sparse Row* (CSR) format whose
structure and values are taken from the selected entries in array `A`.
Parameter `T` is the type of the values stored by the sparse operator.
Parameters `M` and `N` are the number of leading and trailing dimensions of `A`
to group to form the equivalent *rows* and *columns* of the sparse
operator. Optional argument `sel` is a selector function which is called as
`sel(v,i,j)` with `v`, `i` and `j` the value, the row and the column linear
indices for each entries of `A` and which is assumed to yield `true` for the
entries of `A` to be selected in the sparse structure and `false` for the
entries of `A` to discard.  The default selector is such that all non-zeros of
`A` are selected.

See [`CompressedSparseOperator{:CSR}`](@ref) about the most efficient way to access
the entries of a sparse operator in CSR format.

The equality `M + N = ndims(A)` must hold, so it is sufficient to only
specifify `M`:

    SparseOperatorCSR{T,M}(A, sel)

If `A` is a simple matrix, that is a two-dimensional array, parameters `M` and
`N` must be equal to 1 and may be omitted.  In that case, the type `T` may also
be omitted and is `eltype(A)` by default.

The components of the CSR storage can also be directly provided:

    SparseOperatorCSR(vals, cols, offs, rowsiz, colsiz)

or

    SparseOperatorCSR{T}(vals, cols, offs, rowsiz, colsiz)

to force the element type of the result.  Here, `vals` is the vector of values
of the sparse entries, `cols` is an integer valued vector of the linear column
indices of the sparse entries, `offs` is a column-wise table of offsets in
these arrays, `rowsiz` and `colsiz` are the sizes of the row and column
dimensions.  The entries values and respective linear column indices of the
`i`-th row are given by `vals[k]` and `cols[k]` with `k ∈ offs[i]+1:offs[i+1]`.
The linear row index `i` is in the range `1:m` where `m = prod(rowsiz)` is the
equivalent number of rows.

For efficiency reasons, sparse operators are currently limited to *fast* arrays
because they can be indexed linearly with no loss of performances.  If `vals`,
`cols` and/or `offs` are not fast arrays, they will be automatically converted
to linearly indexed arrays.

A sparse linear operator in CSR format can be built given a sparse linear
operator `A` in another storage format:

    SparseOperatorCSR(A)

or

    SparseOperatorCSR{T}(A)

to force the element type of the result.  If `A` is in Compressed Sparse
Coordinate (COO) format, entries are sorted and duplicates merged.  See
[`SparseOperatorCSC`](@ref) and [`SparseOperatorCOO`](@ref) for other storage
formats.

The special constructor call:

    SparseOperatorCSR{Bool,M,N,UniformVector{Bool}}(args...)

yields a sparse operator which ony stores the row and column indices (in CSR
format) of the structural non-zeros (as defined by `args...`) and whose values
are an immutable uniform vector of `true` values.  This is an efficient mean to
store the *structure* of the sparse operator.

""" SparseOperatorCSR

"""
    SparseOperatorCSC{T,M,N}(A, sel = (v,i,j) -> (v != zero(v)))

yields a sparse linear operator in *Compressed Sparse Column* (CSC) format
whose structure and values are taken from the selected entries in array `A`.
Parameter `T` is the type of the values stored by the sparse operator.
Parameters `M` and `N` are the number of leading and trailing dimensions of `A`
to group to form the equivalent *rows* and *columns* of the sparse operator.
Optional argument `sel` is a selector function which is called as `sel(v,i,j)`
with `v`, `i` and `j` the value, the row and the column linear indices for each
entries of `A` and which is assumed to yield `true` for the entries of `A` to
be selected in the sparse structure and `false` for the entries of `A` to
discard.  The default selector is such that all non-zeros of `A` are selected.

See [`CompressedSparseOperator{:CSC}`](@ref) about the most efficient way to access
the entries of a sparse operator in CSC format.

The equality `M + N = ndims(A)` must hold, so it is sufficient to only
specifify `M`:

    SparseOperatorCSC{T,M}(A, sel)

If `A` is a simple matrix, that is a two-dimensional array, parameters `M` and
`N` must be equal to 1 and may be omitted.  In that case, the type `T` may also
be omitted and is `eltype(A)` by default.

The components of the CSC storage can also be directly provided:

    SparseOperatorCSC(vals, rows, offs, rowsiz, colsiz)

or

    SparseOperatorCSC{T}(vals, rows, offs, rowsiz, colsiz)

to force the element type of the result.  Here, `vals` is the vector of values
of the sparse entries, `rows` is an integer valued vector of the linear row
indices of the sparse entries, `offs` is a column-wise table of offsets in
these arrays, `rowsiz` and `colsiz` are the sizes of the row and column
dimensions.  The entries values and respective linear row indices of the `j`-th
column are given by `vals[k]` and `rows[k]` with `k ∈ offs[j]+1:offs[j+1]`.
The linear column index `j` is in the range `1:n` where `n = prod(colsiz)` is
the equivalent number of columns.

For efficiency reasons, sparse operators are currently limited to *fast* arrays
because they can be indexed linearly with no loss of performances.  If `vals`,
`rows` and/or `offs` are not fast arrays, they will be automatically converted
to linearly indexed arrays.

A sparse linear operator in CSC format can be built given a sparse linear
operator `A` in another storage format:

    SparseOperatorCSC(A)

or

    SparseOperatorCSC{T}(A)

to force the element type of the result.  If `A` is in Compressed Sparse
Coordinate (COO) format, entries are sorted and duplicates merged.  See
[`SparseOperatorCSR`](@ref) and [`SparseOperatorCOO`](@ref) for other storage
formats.

The special constructor call:

    SparseOperatorCSC{Bool,M,N,UniformVector{Bool}}(args...)

yields a sparse operator which ony stores the row and column indices (in CSC
format) of the structural non-zeros (as defined by `args...`) and whose values
are an immutable uniform vector of `true` values.  This is an efficient mean to
store the *structure* of the sparse operator.

""" SparseOperatorCSC

"""
    SparseOperatorCOO{T,M,N}(A, sel=select_non_zeros)

yields a sparse linear operator in *Compressed Sparse Coordinate* (COO) format
whose structure and values are taken from the selected entries in array `A`.
Parameter `T` is the type of the values stored by the sparse operator.
Parameters `M` and `N` are the number of leading and trailing dimensions of `A`
to group to form the equivalent *rows* and *columns* of the sparse operator.
Optional argument `sel` is a selector function which is called as `sel(v,i,j)`
with `v`, `i` and `j` the value, the row and the column linear indices for each
entries of `A` and which is assumed to yield `true` for the entries of `A` to
be selected in the sparse structure and `false` for the entries of `A` to
discard.  The default selector is such that all non-zeros of `A` are selected.

The equality `M + N = ndims(A)` must hold, so it is sufficient to only
specifify `M`:

    SparseOperatorCOO{T,M}(A, sel=select_non_zeros)

If `A` is a simple matrix, that is a two-dimensional array, parameters `M` and
`N` must be equal to 1 and may be omitted.  In that case, the type `T` may also
be omitted and is `eltype(A)` by default.

The components of the COO storage can also be directly provided:

    SparseOperatorCOO(vals, rows, cols, rowsiz, colsiz)

or

    SparseOperatorCOO{T}(vals, rows, cols, rowsiz, colsiz)

to force the element type of the result.  Here, `vals` is the vector of values
of the sparse entries, `rows` and `cols` are integer valued vectors with the
linear row and column indices of the sparse entries, `rowsiz` and `colsiz` are
the sizes of the row and column dimensions.  The entries values and respective
linear row and column indices of the `k`-th sparse entry are given by
`vals[k]`, `rows[k]` and `cols[k]`.

For efficiency reasons, sparse operators are currently limited to *fast* arrays
because they can be indexed linearly with no loss of performances.  If `vals`,
`rows` and/or `cols` are not fast arrays, they will be automatically converted
to linearly indexed arrays.

A sparse linear operator in COO format can be built given a sparse linear
operator `A` in another storage format:

    SparseOperatorCOO(A)

or

    SparseOperatorCOO{T}(A)

to force the element type of the result.  See [`SparseOperatorCSR`](@ref) and
[`SparseOperatorCSC`](@ref) for other storage formats.

The special constructor call:

    SparseOperatorCOO{Bool,M,N,UniformVector{Bool}}(args...)

yields a sparse operator which ony stores the row and column indices (in CSO
format) of the structural non-zeros (as defined by `args...`) and whose values
are an immutable uniform vector of `true` values.  This is an efficient mean to
store the *structure* of the sparse operator.

""" SparseOperatorCOO

# Make sparse operators callable.
@callable SparseOperatorCSR
@callable SparseOperatorCSC
@callable SparseOperatorCOO

SparseOperator(A::SparseOperator) = A
SparseOperator{T}(A::SparseOperator{T}) where {T} = A
SparseOperator{T,M}(A::SparseOperator{T,M}) where {T,M} = A
SparseOperator{T,M,N}(A::SparseOperator{T,M,N}) where {T,M,N} = A

# Change element type.
SparseOperator{T}(A::SparseOperator{<:Any,M,N}) where {T,M,N} =
    SparseOperator{T,M,N}(A)
SparseOperator{T,M}(A::SparseOperator{<:Any,M,N}) where {T,M,N} =
    SparseOperator{T,M,N}(A)
for F in (:SparseOperatorCSC, :SparseOperatorCSR, :SparseOperatorCOO)
    @eval begin
        SparseOperator{T,M,N}(A::$F{<:Any,M,N}) where {T,M,N} =
            $F{T,M,N}(A)
    end
end

for (fmt,func) in ((:CSC, :SparseOperatorCSC),
                   (:CSR, :SparseOperatorCSR),
                   (:COO, :SparseOperatorCOO),)
    F = Expr(:quote, Symbol(fmt))
    @eval begin
        CompressedSparseOperator{$F}(args...; kwds...) =
            $func(args...; kwds...)
        CompressedSparseOperator{$F,T}(args...; kwds...) where {T} =
            $func{T}(args...; kwds...)
        CompressedSparseOperator{$F,T,M}(args...; kwds...) where {T,M} =
            $func{T,M}(args...; kwds...)
        CompressedSparseOperator{$F,T,M,N}(args...; kwds...) where {T,M,N} =
            $func{T,M,N}(args...; kwds...)
    end
end

# Basic outer constructors return a fully checked structure.

function SparseOperatorCSR(vals::AbstractVector,
                           cols::AbstractVector{<:Integer},
                           offs::AbstractVector{<:Integer},
                           rowsiz::ArraySize,
                           colsiz::ArraySize)
    check_structure(unsafe_csr(to_values(vals),
                               to_indices(cols),
                               to_indices(offs),
                               to_size(rowsiz),
                               to_size(colsiz)))
end

function SparseOperatorCSC(vals::AbstractVector,
                           rows::AbstractVector{<:Integer},
                           offs::AbstractVector{<:Integer},
                           rowsiz::ArraySize,
                           colsiz::ArraySize)
    check_structure(unsafe_csc(to_values(vals),
                               to_indices(rows),
                               to_indices(offs),
                               to_size(rowsiz),
                               to_size(colsiz)))
end

function SparseOperatorCOO(vals::AbstractVector,
                           rows::AbstractVector{<:Integer},
                           cols::AbstractVector{<:Integer},
                           rowsiz::ArraySize,
                           colsiz::ArraySize)
    check_structure(unsafe_coo(to_values(vals),
                               to_indices(rows),
                               to_indices(cols),
                               to_size(rowsiz),
                               to_size(colsiz)))
end

@inline isnonzero(v::T, i::Integer, j::Integer) where {T} = (v != zero(T))

for CS in (:SparseOperatorCSR,
           :SparseOperatorCSC,
           :SparseOperatorCOO)
    @eval begin
        # Get rid of the M,N parameters, but keep/set T for conversion of
        # values.
        $CS{T,M,N}(A::SparseOperator{<:Any,M,N}) where {T,M,N} = $CS{T}(A)
        $CS{T,M}(A::SparseOperator{<:Any,M}) where {T,M} = $CS{T}(A)
        $CS(A::SparseOperator{T}) where {T} = $CS{T}(A)

        # Cases which do nothing (it makes sense that a constructor of an
        # immutable type be able to just return its argument if it is already
        # of the correct type).
        $CS{T}(A::$CS{T}) where {T} = A

        # Manage to call constructors of compressed sparse linear operator
        # given a regular Julia array with correct parameters and selector.
        $CS(A::AbstractMatrix{T}, args...; kwds...) where {T} =
            $CS{T,1,1}(A, args...; kwds...)
        $CS{Any}(A::AbstractMatrix{T}, args...; kwds...) where {T} =
            $CS{T,1,1}(A, args...; kwds...)
        $CS{T}(A::AbstractMatrix, args...; kwds...) where {T} =
            $CS{T,1,1}(A, args...; kwds...)
        $CS{Any,M}(A::AbstractArray{T}, args...; kwds...) where {T,M} =
            $CS{T,M}(A, args...; kwds...)
        $CS{Any,M,N}(A::AbstractArray{T}, args...; kwds...) where {T,M,N} =
            $CS{T,M,N}(A, args...; kwds...)
        $CS{Any,M,N,V}(A::AbstractArray{T}, args...; kwds...) where {T,M,N,V} =
            $CS{T,M,N,V}(A, args...; kwds...)
        function $CS{T,M}(A::AbstractArray{S,L}, args...; kwds...) where {S,T,L,M}
            @assert 1 ≤ M < L
            $CS{T,M,L-M}(A, args...; kwds...)
        end
        $CS{T,M,N}(A::AbstractArray, args...; kwds...) where {T,M,N} =
            $CS{T,M,N,Vector{T}}(A, args...; kwds...)
        $CS{T,M,N,V}(A::AbstractArray, sel::Function = isnonzero) where {T,M,N,V} =
           $CS{T,M,N,V}(A, sel)
    end
end

# Constructors that convert array of values.  Other fields have already been
# checked so do not check structure again.

SparseOperatorCSR{T}(A::SparseOperatorCSR{S,M,N}) where {S,T,M,N} =
    unsafe_csr(nrows(A), ncols(A), to_values(T, get_vals(A)),
               get_cols(A), get_offs(A), row_size(A), col_size(A))

SparseOperatorCSC{T}(A::SparseOperatorCSC{S,M,N}) where {S,T,M,N} =
    unsafe_csc(nrows(A), ncols(A), to_values(T, get_vals(A)),
               get_rows(A), get_offs(A), row_size(A), col_size(A))

SparseOperatorCOO{T}(A::SparseOperatorCOO{S,M,N}) where {S,T,M,N} =
    unsafe_coo(nrows(A), ncols(A), to_values(T, get_vals(A)),
               get_rows(A), get_cols(A), row_size(A), col_size(A))

# Constructors for CSR format similar to the basic ones but have parameters
# that may imply converting arguments.

function SparseOperatorCSR{T,M,N}(vals::AbstractVector,
                                  cols::AbstractVector{<:Integer},
                                  offs::AbstractVector{<:Integer},
                                  rowsiz::ArraySize,
                                  colsiz::ArraySize) where {T,M,N}
    @assert length(rowsiz) == M
    @assert length(colsiz) == N
    SparseOperatorCSR{T}(vals, cols, offs, rowsiz, colsiz)
end

function SparseOperatorCSR{T,M}(vals::AbstractVector,
                                cols::AbstractVector{<:Integer},
                                offs::AbstractVector{<:Integer},
                                rowsiz::ArraySize,
                                colsiz::ArraySize) where {T,M}
    @assert length(rowsiz) == M
    SparseOperatorCSR{T}(vals, cols, offs, rowsiz, colsiz)
end

function SparseOperatorCSR{T}(vals::AbstractVector,
                              cols::AbstractVector{<:Integer},
                              offs::AbstractVector{<:Integer},
                              rowsiz::ArraySize,
                              colsiz::ArraySize) where {T}
    SparseOperatorCSR(to_values(T, vals), cols, offs, rowsiz, colsiz)
end

# Idem for CSC format.

function SparseOperatorCSC{T,M,N}(vals::AbstractVector,
                                  rows::AbstractVector{<:Integer},
                                  offs::AbstractVector{<:Integer},
                                  rowsiz::ArraySize,
                                  colsiz::ArraySize) where {T,M,N}
    @assert length(rowsiz) == M
    @assert length(colsiz) == N
    SparseOperatorCSC{T}(vals, rows, offs, rowsiz, colsiz)
end

function SparseOperatorCSC{T,M}(vals::AbstractVector,
                                rows::AbstractVector{<:Integer},
                                offs::AbstractVector{<:Integer},
                                rowsiz::ArraySize,
                                colsiz::ArraySize) where {T,M}
    @assert length(rowsiz) == M
    SparseOperatorCSC{T}(vals, rows, offs, rowsiz, colsiz)
end

function SparseOperatorCSC{T}(vals::AbstractVector,
                              rows::AbstractVector{<:Integer},
                              offs::AbstractVector{<:Integer},
                              rowsiz::ArraySize,
                              colsiz::ArraySize) where {T}
    SparseOperatorCSC(to_values(T, vals), rows, offs, rowsiz, colsiz)
end

# Idem for COO format.

function SparseOperatorCOO{T,M,N}(vals::AbstractVector,
                                  rows::AbstractVector{<:Integer},
                                  cols::AbstractVector{<:Integer},
                                  rowsiz::ArraySize,
                                  colsiz::ArraySize) where {T,M,N}
    @assert length(rowsiz) == M
    @assert length(colsiz) == N
    SparseOperatorCOO{T}(vals, rows, cols, rowsiz, colsiz)
end

function SparseOperatorCOO{T,M}(vals::AbstractVector,
                                rows::AbstractVector{<:Integer},
                                cols::AbstractVector{<:Integer},
                                rowsiz::ArraySize,
                                colsiz::ArraySize) where {T,M}
    @assert length(rowsiz) == M
    SparseOperatorCOO{T}(vals, rows, cols, rowsiz, colsiz)
end

function SparseOperatorCOO{T}(vals::AbstractVector,
                              rows::AbstractVector{<:Integer},
                              cols::AbstractVector{<:Integer},
                              rowsiz::ArraySize,
                              colsiz::ArraySize) where {T}
    SparseOperatorCOO(to_values(T, vals), rows, cols, rowsiz, colsiz)
end


# Constructors of a sparse linear operator in various format given a regular
# Julia array and a selector function.  Julia arrays are usually in
# column-major order but this is not always the vase, to handle various storage
# orders when extracting selected entries, we convert the input array into a
# equivalent "matrix", that is a 2-dimensional array.

function SparseOperatorCSR{T,M,N,V}(arr::AbstractArray{S,L},
                                    sel::Function) where {S,T,L,M,N,
                                                          V<:AbstractVector{T}}
    # Get equivalent matrix dimensions.
    nrows, ncols, rowsiz, colsiz = get_equivalent_size(arr, Val(M), Val(N))

    # Convert into equivalent matrix.
    A = as_matrix(arr, nrows, ncols)

    # Count the number of selected entries.
    nvals = count_selection(A, sel)

    # Extract the selected entries and their column indices and count the
    # numver of selected entries per row.  The pseudo-matrix is walked in
    # row-major order.
    cols = Vector{Int}(undef, nvals)
    offs = Vector{Int}(undef, nrows + 1)
    k = 0
    if V <: UniformVector{Bool}
        # Just extract the structure, not the values.
        vals = V(true, nvals)
        @inbounds for i in 1:nrows
            offs[i] = k
            for j in 1:ncols
                if sel(A[i,j], i, j)
                    (k += 1) ≤ nvals || bad_selector()
                    cols[k] = j
                end
            end
        end
    else
        # Extract the structure and the values.
        vals = V(undef, nvals)
        @inbounds for i in 1:nrows
            offs[i] = k
            for j in 1:ncols
                v = A[i,j]
                if sel(v, i, j)
                    (k += 1) ≤ nvals || bad_selector()
                    vals[k] = v
                    cols[k] = j
                end
            end
        end
    end
    k == nvals || bad_selector()
    offs[end] = nvals

    # By construction, the sparse structure should be correct so just call the
    # "unsafe" constructor.
    return unsafe_csr(nrows, ncols, vals, cols, offs, rowsiz, colsiz)
end

function SparseOperatorCSC{T,M,N,V}(arr::AbstractArray{S,L},
                                    sel::Function) where {S,T,L,M,N,
                                                          V<:AbstractVector{T}}
    # Get equivalent matrix dimensions.
    nrows, ncols, rowsiz, colsiz = get_equivalent_size(arr, Val(M), Val(N))

    # Convert into equivalent matrix.
    A = as_matrix(arr, nrows, ncols)

    # Count the number of selected entries.
    nvals = count_selection(A, sel)

    # Extract the selected entries and their row indices and count the numver
    # of selected entries per column.  The pseudo-matrix is walked in
    # column-major order.
    rows = Vector{Int}(undef, nvals)
    offs = Vector{Int}(undef, ncols + 1)
    k = 0
    if V <: UniformVector{Bool}
        # Just extract the structure, not the values.
        vals = V(true, nvals)
        @inbounds for j in 1:ncols
            offs[j] = k
            for i in 1:nrows
                if sel(A[i,j], i, j)
                    (k += 1) ≤ nvals || bad_selector()
                    rows[k] = i
                end
            end
        end
    else
        # Extract the structure and the values.
        vals = V(undef, nvals)
        @inbounds for j in 1:ncols
            offs[j] = k
            for i in 1:nrows
                v = A[i,j]
                if sel(v, i, j)
                    (k += 1) ≤ nvals || bad_selector()
                    vals[k] = v
                    rows[k] = i
                end
            end
        end
    end
    k == nvals || bad_selector()
    offs[end] = nvals

    # By construction, the sparse structure should be correct so just call the
    # "unsafe" constructor.
    return unsafe_csc(nrows, ncols, vals, rows, offs, rowsiz, colsiz)
end

function SparseOperatorCOO{T,M,N,V}(arr::AbstractArray{S,L},
                                    sel::Function) where {S,T,L,M,N,
                                                          V<:AbstractVector{T}}
    # Get equivalent matrix dimensions.
    nrows, ncols, rowsiz, colsiz = get_equivalent_size(arr, Val(M), Val(N))

    # Convert into equivalent matrix.
    A = as_matrix(arr, nrows, ncols)

    # Count the number of selected entries.
    nvals = count_selection(A, sel)

    # Extract the selected entries and their row and column indices.  The
    # pseudo-matrix is walked in column-major order since most Julia arrays are
    # stored in that order.
    rows = Vector{Int}(undef, nvals)
    cols = Vector{Int}(undef, nvals)
    k = 0
    if V <: UniformVector{Bool}
        # Just extract the structure, not the values.
        vals = V(true, nvals)
        @inbounds for j in 1:ncols
            for i in 1:nrows
                if sel(A[i,j], i, j)
                    (k += 1) ≤ nvals || bad_selector()
                    vals[k] = v
                    rows[k] = i
                    cols[k] = j
                end
            end
        end
    else
        # Extract the structure and the values.
        vals = V(undef, nvals)
        @inbounds for j in 1:ncols
            for i in 1:nrows
                v = A[i,j]
                if sel(v, i, j)
                    (k += 1) ≤ nvals || bad_selector()
                    vals[k] = v
                    rows[k] = i
                    cols[k] = j
                end
            end
        end
    end
    k == nvals || bad_selector()

    # By construction, the sparse structure should be correct so just call the
    # "unsafe" constructor.
    return unsafe_coo(nrows, ncols, vals, rows, cols, rowsiz, colsiz)
end

"""
    unpack!(A, S) -> A

unpacks the non-zero coefficients of the sparse operator `S` into the array `A`
and returns `A`.

""" unpack!

# Convert to standard Julia arrays which are stored in column-major order,
# hence the stride is the equivalent number of rows.  For COO format, as
# duplicates are allowed, values must be combined by an operator.

Base.Array(A::SparseOperator{T,M,N}) where {T,M,N} = Array{T,M+N}(A)
Base.Array{T}(A::SparseOperator{<:Any,M,N}) where {T,M,N} = Array{T,M+N}(A)
function Base.Array{T,L}(A::SparseOperator{<:Any,M,N}) where {T,L,M,N}
    L == M + N || throw_incompatible_number_of_dimensions()
    return unpack!(Array{T}(undef, (row_size(A)..., col_size(A)...,)), A)
end

function unpack!(B::Array{T,L},
                 A::SparseOperatorCSR{<:Any,M,N}) where {T,L,M,N}
    size(B) == (row_size(A)..., col_size(A)...,) ||
        throw_incompatible_dimensions()
    fill!(B, zero(T))
    m = nrows(A) # used as the "stride" in B
    @inbounds for i in each_row(A)
        for k in each_off(A, i)
            j = get_col(A, k)
            v = get_val(A, k)
            B[i + m*(j - 1)] = v
        end
    end
    return B
end

function unpack!(B::Array{T,L},
                 A::SparseOperatorCSC{<:Any,M,N}) where {T,L,M,N}
    size(B) == (row_size(A)..., col_size(A)...,) ||
        throw_incompatible_dimensions()
    fill!(B, zero(T))
    m = nrows(A) # used as the "stride" in B
    @inbounds for j in each_col(A)
        for k in each_off(A, j)
            i = get_row(A, k)
            v = get_val(A, k)
            B[i + m*(j - 1)] = v
        end
    end
    return B
end

unpack!(B::Array, A::SparseOperatorCOO) = unpack!(B, A, +)
unpack!(B::Array, A::SparseOperatorCOO{Bool}) = unpack!(B, A, |)
function unpack!(B::Array{T,L},
                 A::SparseOperatorCOO{<:Any,M,N}, op) where {T,L,M,N}
    size(B) == (row_size(A)..., col_size(A)...,) ||
        throw_incompatible_dimensions()
    fill!(B, zero(T))
    m = nrows(A) # used as the "stride" in B
    @inbounds for k in each_off(A)
        i = get_row(A, k)
        j = get_col(A, k)
        v = get_val(A, k)
        l = i + m*(j - 1)
        B[l] = op(B[l], v)
    end
    return B
end

@noinline throw_incompatible_dimensions() =
    error("incompatible dimensions")

@noinline throw_incompatible_number_of_dimensions() =
    error("incompatible number of dimensions")

function check_new_shape(A::SparseOperator,
                         rowsiz::Tuple{Vararg{Int}},
                         colsiz::Tuple{Vararg{Int}})
    prod(rowsiz) == nrows(A) ||
        bad_size("products of row dimensions must be equal")
    prod(colsiz) == ncols(A) ||
        bad_size("products of column dimensions must be equal")
end

Base.reshape(A::SparseOperator, rowsiz::ArraySize, colsiz::ArraySize) =
    reshape(A, to_size(rowsiz), to_size(colsiz))

function Base.reshape(A::SparseOperatorCSR,
                      rowsiz::Tuple{Vararg{Int}},
                      colsiz::Tuple{Vararg{Int}})
    check_new_shape(A, rowsiz, colsiz)
    unsafe_csr(nrows(A), ncols(A), get_vals(A), get_cols(A), get_offs(A),
               rowsiz, colsiz)
end

function Base.reshape(A::SparseOperatorCSC,
                      rowsiz::Tuple{Vararg{Int}},
                      colsiz::Tuple{Vararg{Int}})
    check_new_shape(A, rowsiz, colsiz)
    unsafe_csc(nrows(A), ncols(A), get_vals(A), get_rows(A), get_offs(A),
               rowsiz, colsiz)
end

function Base.reshape(A::SparseOperatorCOO,
                      rowsiz::Tuple{Vararg{Int}},
                      colsiz::Tuple{Vararg{Int}})
    check_new_shape(A, rowsiz, colsiz)
    unsafe_coo(nrows(A), ncols(A), get_vals(A), get_rows(A), get_cols(A),
               rowsiz, colsiz)
end

# Convert from other compressed sparse formats.  For compressed sparse row and
# column (CSR and CSC) formats, the compressed sparse coordinate (COO) format
# is used as an intermediate representation and entries are sorted in
# row/column major order.  To avoid side-effects, they must be copied first.
# Unless values are converted, there is no needs to copy when converting to a
# compressed sparse coordinate (COO) format.

SparseOperatorCSR{T}(A::SparseOperator) where {T} =
    coo_to_csr!(copy_vals(T, A),
                copy_rows(A),
                copy_cols(A),
                row_size(A),
                col_size(A))

SparseOperatorCSC{T}(A::SparseOperator) where {T} =
    coo_to_csc!(copy_vals(T, A),
                copy_rows(A),
                copy_cols(A),
                row_size(A),
                col_size(A))

SparseOperatorCOO{T}(A::SparseOperator) where {T} =
    SparseOperatorCOO(copy_vals(T, A),
                      get_rows(A),
                      get_cols(A),
                      row_size(A),
                      col_size(A))

SparseOperatorCOO{T}(A::SparseOperator{T}) where {T} =
    SparseOperatorCOO(get_vals(A),
                      get_rows(A),
                      get_cols(A),
                      row_size(A),
                      col_size(A))

"""
    coo_to_csr!(vals, rows, cols, rowsiz, colsiz [, mrg]) -> A

yields the a compressed sparse linear operator in a CSR format given the
components `vals`, `rows` and `cols` in the COO format and the sizes `rowsiz`
and `colsiz` of the row and column dimensions.  Input arrays are modified
in-place.  Optional argument `mrg` is a function called to merge values of
entries with the same row and column indices.

Input arrays must be regular Julia vectors to ensure type stability in case of
duplicates.

"""
function coo_to_csr!(vals::Vector{T},
                     rows::Vector{Int},
                     cols::Vector{Int},
                     rowsiz::NTuple{M,Int},
                     colsiz::NTuple{N,Int},
                     mrg::Function = (T <: Bool ? (|) : (+))) where {T,M,N}
    # Check row and column sizes.
    nrows = check_size(rowsiz, "row")
    ncols = check_size(colsiz, "column")

    # Check row and column indices.
    check_rows(rows, nrows)
    check_cols(cols, ncols)

    # Sort and merge entries in row-major order, resize arrays if needed and
    # compute offsets.
    nvals = sort_and_merge!(vals, rows, cols, mrg)
    if nvals < length(vals)
        vals = vals[1:nvals]
        cols = cols[1:nvals]
    end
    offs = compute_offsets(nrows, rows, nvals)

    # Since everything will have
    # been checked, we can call the unsafe constructor.
    return unsafe_csr(nrows, ncols, vals, cols, offs, rowsiz, colsiz)
end

"""
    coo_to_csc!(vals, rows, cols, rowsiz, colsiz [, mrg]) -> A

yields the a compressed sparse linear operator in a CSC format given the
components `vals`, `rows` and `cols` in the COO format and the sizes `rowsiz`
and `colsiz` of the row and column dimensions.  Input arrays are modified
in-place.  Optional argument `mrg` is a function called to merge values of
entries with the same row and column indices.

Input arrays must be regular Julia vectors to ensure type stability in case of
duplicates.

"""
function coo_to_csc!(vals::Vector{T},
                     rows::Vector{Int},
                     cols::Vector{Int},
                     rowsiz::NTuple{M,Int},
                     colsiz::NTuple{N,Int},
                     mrg::Function = (T <: Bool ? (|) : (+))) where {T,M,N}
    # Check row and column sizes.
    nrows = check_size(rowsiz, "row")
    ncols = check_size(colsiz, "column")

    # Check row and column indices.
    check_rows(rows, nrows)
    check_cols(cols, ncols)

    # Sort and merge entries in column-major order, resize arrays if needed and
    # compute offsets.
    nvals = sort_and_merge!(vals, cols, rows, mrg)
    if nvals < length(vals)
        vals = vals[1:nvals]
        rows = rows[1:nvals]
    end
    offs = compute_offsets(ncols, cols, nvals)

    # Since everything will have
    # been checked, we can call the unsafe constructor.
    return unsafe_csc(nrows, ncols, vals, rows, offs, rowsiz, colsiz)
end

# "less-than" method for sorting entries in order, arguments are 3-tuples
# `(v,major,minor)` with `v` the entry value, `major` the major index and
# `minir` the minor index.
@inline major_order(a::Tuple{T,Int,Int},b::Tuple{T,Int,Int}) where {T} =
    ifelse(a[2] == b[2], a[3] < b[3], a[2] < b[2])

"""
    sort_and_merge!(vals, major, minor, mrg) -> nvals

sorts entries and merges duplicates in input arrays `vals`, `major` and
`minor`.  Entries consist in the 3-tuples `(vals[i],major[i],minor[i])`.  The
sorting order of the `i`-th entry is based the value of `major[i]` and, if
equal, on the value of `minor[i]`.  After sorting, duplicates entries, that is
those which have the same `(major[i],minor[i])`, are removed merging the
associated values in `vals` with the `mrg` function.  All operations are done
in-place, the number of unique entries is returned but inputs arrays are not
resized, only the `nvals` first entries are valid.

"""
function sort_and_merge!(vals::AbstractVector,
                         major::AbstractVector{Int},
                         minor::AbstractVector{Int},
                         mrg::Function)
    # Sort entries in order (this also ensures that all arrays have the same
    # dimensions).
    sort!(ZippedArray(vals, major, minor); lt=major_order)

    # Merge duplicates.
    j = 1
    @inbounds for k in 2:length(vals)
        if major[k] == major[j] && minor[k] == minor[j]
            vals[j] = mrg(vals[j], vals[k])
        elseif (j += 1) < k
            vals[j], major[j], minor[j] = vals[k], major[k], minor[k]
        end
    end
    return j
end

"""
    compute_offsets(n, inds, len=length(inds)) -> offs

yields the `n+1` offsets computed from the list of indices `inds[1:len]`.
Indices must be in non-increasing order an all in the range `1:n`.

"""
function compute_offsets(n::Int,
                         inds::AbstractVector{Int},
                         len::Int = length(inds))
    @assert len ≤ length(inds)
    @inbounds begin
        offs = Vector{Int}(undef, n + 1)
        i = 0
        for k in 1:len
            j = inds[k]
            j == i && continue
            ((j < i)|(j > n)) &&
                error(1 ≤ j ≤ n ?
                      "indices must be in non-increasing order" :
                      "out of bound indices")
            off = k - 1
            while i < j
                i += 1
                offs[i] = off
            end
        end
        while i ≤ n
            i += 1
            offs[i] = len
        end
    end
    return offs
end

# This error is due to the non-zeros selector not returning the same results in
# the two selection passes.
bad_selector() = argument_error("inconsistent selector function")

@inline select_non_zeros(v::Bool, i::Int, j::Int) = v
@inline select_non_zeros(v::T, i::Int, j::Int) where {T} = (v != zero(T))

@inline select_non_zeros_in_diagonal(v::T, i::Int, j::Int) where {T} =
    ((i == j)|(v != zero(T)))

@inline select_non_zeros_in_lower_part(v::T, i::Int, j::Int) where {T} =
    ((i ≥ j)&(v != zero(T)))

@inline select_non_zeros_in_upper_part(v::T, i::Int, j::Int) where {T} =
    ((i ≤ j)&(v != zero(T)))

"""
    get_equivalent_size(A, Val(M), Val(N)) -> nrows, ncols, rowsiz, colsiz

yields equivalent matrix dimensions of array `A` assuming the *rows* account
for the `M` leading dimensions while the *columns* account for the other `N`
dimensions.

"""
function get_equivalent_size(A::AbstractArray{T,L},
                             ::Val{M}, ::Val{N}) where {T,L,M,N}
   @assert L == M + N
    @assert M ≥ 1
    @assert N ≥ 1
    eachindex(A) == 1:length(A) ||
        argument_error("array must have standard linear indexing")
    siz = size(A)
    rowsiz = siz[1:M]
    colsiz = siz[M+1:end]
    nrows = prod(rowsiz)
    ncols = prod(colsiz)
    return nrows, ncols, rowsiz, colsiz
end

"""
    count_selection(A, sel, rowmajor=false) -> nvals

yields the number of selected entries in matrix `A` such that `sel(A[i,j],i,j)`
is `true` and with `i` and `j` the row and column indices.  if optinal argument
`rowmajor` is true, the array is walked in row-major order; otherwsie (the
default), the array is walked in column-major order.

"""
function count_selection(A::AbstractMatrix, sel::Function,
                         rowmajor::Bool = false)
    nrows, ncols = size(A)
    nvals = 0
    if rowmajor
        # Walk the coefficients in row-major order.
        @inbounds for i in 1:nrows, j in 1:ncols
            if sel(A[i,j], i, j)
                nvals += 1
            end
        end
    else
        # Walk the coefficients in column-major order.
        @inbounds for j in 1:ncols, i in 1:nrows
            if sel(A[i,j], i, j)
                nvals += 1
            end
        end
    end
    return nvals
end

"""
    check_structure(A) -> A

checks the structure of the compressed sparse linear operator `A` throwing an
exception if there are any inconsistencies.

"""
function check_structure(A::CompressedSparseOperator{:CSR})
    check_size(A)
    check_vals(A)
    check_cols(A)
    check_offs(A)
    return A
end

function check_structure(A::CompressedSparseOperator{:CSC})
    check_size(A)
    check_vals(A)
    check_rows(A)
    check_offs(A)
    return A
end

function check_structure(A::CompressedSparseOperator{:COO})
    check_size(A)
    check_vals(A)
    check_rows(A)
    check_cols(A)
    return A
end

"""
    check_size(siz, id="array") -> len

checks the array size `siz` and returns the corresponding number of elements.
An `ArgumentError` is thrown if a dimension is invalid, using `id` to identify
the argument.

    check_size(A)

checks the validity of the row and column sizes in compressed sparse linear
operator `A` throwing an exception if there are any inconsistencies.

"""
function check_size(siz::NTuple{N,Int}, id::String="array") where {N}
    len = 1
    @inbounds for i in 1:N
        (dim = siz[i]) ≥ 0 || bad_dimension(dim, i, id)
        len *= dim
    end
    return len
end

function check_size(A::SparseOperator)
    check_size(row_size(A), "row") == nrows(A) ||
        dimension_mismatch("incompatible equivalent number of rows and row size")
    check_size(col_size(A), "column") == ncols(A) ||
        dimension_mismatch("incompatible equivalent number of columns and column size")
    nothing
end

@noinline bad_dimension(dim::Integer, i::Integer, id) =
    argument_error("invalid ", i, ordinal_suffix(i), " ", id,
                   " dimension: ", dim)

"""
    check_vals(A)

checks the array of values in compressed sparse linear operator `A` throwing an
exception if there are any inconsistencies.

"""
function check_vals(A::SparseOperator)
    vals = get_vals(A)
    is_fast_array(vals) || not_fast_array("array of values")
    length(vals) == nnz(A) || argument_error("bad number of values")
    nothing
end

"""
    check_rows(A)

checks the array of linear row indices in the compressed sparse linear operator
`A` stored in a *Compressed Sparse Column* (CSC) or *Compressed Sparse
Coordinate* (COO) format.  Throws an exception in case of inconsistency.

    check_rows(rows, m)

check the array of linear row indices `rows` for being a fast vetor of values
in the range `1:m`.

"""
function check_rows(A::Union{<:CompressedSparseOperator{:CSC},
                             <:CompressedSparseOperator{:COO}})
    rows = get_rows(A)
    length(rows) == nnz(A) || argument_error("bad number of row indices")
    check_rows(rows, nrows(A))
    # FIXME: also check sorting for CompressedSparseOperator{:CSC}?
end

function check_rows(rows::AbstractVector{Int}, m::Int)
    is_fast_array(rows) || not_fast_array("array of row indices")
    anyerror = false
    @inbounds @simd for k in eachindex(rows)
        i = rows[k]
        anyerror |= ((i < 1)|(i > m))
    end
    anyerror && error("out of range row indices")
    nothing
end

"""
    check_cols(A)

checks the array of linear column indices in the compressed sparse linear
operator `A` stored in a *Compressed Sparse Row* (CSR) or *Compressed Sparse
Coordinate* (COO) format.  Throws an exception in case of inconsistency.

    check_cols(cols, n)

check the array of linear column indices `cols` for being a fast vetor of
values in the range `1:n`.

"""
function check_cols(A::Union{<:CompressedSparseOperator{:CSR},
                             <:CompressedSparseOperator{:COO}})
    cols = get_cols(A)
    length(cols) == nnz(A) || argument_error("bad number of column indices")
    check_cols(cols, ncols(A))
    # FIXME: also check sorting for CompressedSparseOperator{:CSR}?
end

function check_cols(cols::AbstractVector{Int}, n::Int)
    is_fast_array(cols) || not_fast_array("array of column indices")
    anyerror = false
    @inbounds @simd for k in eachindex(cols)
        j = cols[k]
        anyerror |= ((j < 1)|(j > n))
    end
    anyerror && error("out of range column indices")
    nothing
end

"""
    check_offs(A)

checks the array of offsets in the compressed sparse linear operator `A` stored
in a *Compressed Sparse Row* (CSR) or *Compressed Sparse Column* (CSC) format.
Throws an exception in case of inconsistency.

"""
function check_offs(A::T) where {T<:Union{CompressedSparseOperator{:CSR},
                                          CompressedSparseOperator{:CSC}}}
    offs = get_offs(A)
    is_fast_array(offs) || not_fast_array("array of offsets")
    n = (T <: CompressedSparseOperator{:CSR} ? nrows(A) : ncols(A))
    length(offs) == n + 1 || error("bad number of offsets")
    offs[1] == 0 || error("bad initial offset")
    len = 0
    anyerrors = false
    @inbounds for i in 1:n
        k1, k2 = offs[i], offs[i+1]
        anyerrors |= (k2 < k1)
        len += ifelse(k2 > k1, k2 - k1, 0)
    end
    anyerrors && error("offsets must be non-decreasing")
    len == nnz(A) ||
        error("offsets incompatible with number of structural non-zeros")
    nothing
end

"""
    unsafe_csr([m, n,] vals, cols, offs, rowsiz, colsiz)

yields a compressed sparse linear operator in *Compressed Sparse Row* (CSR)
format as an instance of `SparseOperatorCSR`.  This method assumes that
arguments are correct, it just calls the inner constructor with suitable
parameters.  This method is mostly used by converters and outer constructors.

"""
function unsafe_csr(m::Integer, n::Integer,
                    vals::V, cols::J, offs::K,
                    rowsiz::NTuple{M,Int},
                    colsiz::NTuple{N,Int}) where {T,M,N,
                                                  V<:AbstractVector{T},
                                                  J<:AbstractVector{Int},
                                                  K<:AbstractVector{Int}}
    SparseOperatorCSR{T,M,N,V,J,K}(to_int(m), to_int(n),
                                   vals, cols, offs,
                                   rowsiz, colsiz)
end

function unsafe_csr(vals::AbstractVector,
                    cols::AbstractVector{Int},
                    offs::AbstractVector{Int},
                    rowsiz::NTuple{M,Int},
                    colsiz::NTuple{N,Int}) where {M,N}
    unsafe_csr(prod(rowsiz), prod(colsiz), vals, cols, offs, rowsiz, colsiz)
end

"""
    unsafe_csc([m, n,] vals, rows, offs, rowsiz, colsiz)

yields a compressed sparse linear operator in *Compressed Sparse Column* (CSC)
format as an instance of `SparseOperatorCSC`.  This method assumes that
arguments are correct, it just calls the inner constructor with suitable
parameters.  This method is mostly used by converters and outer constructors.

"""
function unsafe_csc(m::Integer, n::Integer,
                    vals::V, rows::I, offs::K,
                    rowsiz::NTuple{M,Int},
                    colsiz::NTuple{N,Int}) where {T,M,N,
                                                  V<:AbstractVector{T},
                                                  I<:AbstractVector{Int},
                                                  K<:AbstractVector{Int}}
    SparseOperatorCSC{T,M,N,V,I,K}(to_int(m), to_int(n),
                                   vals, rows, offs,
                                   rowsiz, colsiz)
end

function unsafe_csc(vals::AbstractVector,
                    rows::AbstractVector{Int},
                    offs::AbstractVector{Int},
                    rowsiz::NTuple{M,Int},
                    colsiz::NTuple{N,Int}) where {M,N}
    unsafe_csc(prod(rowsiz), prod(colsiz), vals, rows, offs, rowsiz, colsiz)
end

"""
    unsafe_coo([m, n,] vals, rows, cols, rowsiz, colsiz)

yields a compressed sparse linear operator in *Compressed Sparse Coordinate*
(COO) format as an instance of `SparseOperatorCOO`.  This method
assumes that arguments are correct, it just calls the inner constructor with
suitable parameters.  This method is mostly used by converters and outer
constructors.

"""
function unsafe_coo(m::Integer, n::Integer,
                    vals::V, rows::I, cols::J,
                    rowsiz::NTuple{M,Int},
                    colsiz::NTuple{N,Int}) where {T,M,N,
                                                  V<:AbstractVector{T},
                                                  I<:AbstractVector{Int},
                                                  J<:AbstractVector{Int}}
    SparseOperatorCOO{T,M,N,V,I,J}(to_int(m), to_int(n),
                                   vals, rows, cols,
                                   rowsiz, colsiz)
end

function unsafe_coo(vals::AbstractVector,
                    rows::AbstractVector{Int},
                    cols::AbstractVector{Int},
                    rowsiz::NTuple{M,Int},
                    colsiz::NTuple{N,Int}) where {M,N}
    unsafe_coo(prod(rowsiz), prod(colsiz), vals, rows, cols, rowsiz, colsiz)
end

"""
    is_fast_array(A) -> bool

yields whether `A` is a *fast array* that is an array with standard linear
indexing.

"""
is_fast_array(A::AbstractArray) = is_fast_indices(eachindex(A))

"""
    is_fast_indices(inds) -> bool

yields whether `inds` is an iterator for *fast indices* that is linear indices
starting at `1`.

"""
is_fast_indices(inds::AbstractUnitRange{Int}) = (first(inds) == 1)
is_fast_indices(inds) = false

"""
    check_argument(A, siz, id="array")

checks whether array `A` has size `siz` and implements standard linear
indexing.  An exception is thrown if any of these do not hold.

"""
function check_argument(A::AbstractArray{<:Any,N},
                        siz::NTuple{N,Int},
                        id="array") where {N}
    IndexStyle(A) === IndexLinear() || non_linear_indexing(id)
    inds = axes(A)
    @inbounds for i in 1:N
        first(inds[i]) == 1 || non_standard_indexing(id)
        length(inds[i]) == siz[i] || incompatible_dimensions(id)
    end
    nothing
end

@noinline incompatible_dimensions(id) =
    dimension_mismatch(id, " has incompatible dimensions")

@noinline not_fast_array(id) =
    argument_error(id, " does not implement fast indexing")

@noinline non_linear_indexing(id) =
    argument_error(id, " does not implement linear indexing")

@noinline non_standard_indexing(id) =
    argument_error(id, " has non-standard indexing")

"""
    argument_error(args...)

throws an `ArgumentError` exception with a textual message made of `args...`.

"""
argument_error(mesg::AbstractString) = throw(ArgumentError(mesg))
@noinline argument_error(args...) = argument_error(string(args...))

"""
    dimension_mismatch(args...)

throws a `DimensionMismatch` exception with a textual message made of
`args...`.

"""
dimension_mismatch(mesg::AbstractString) = throw(DimensionMismatch(mesg))
@noinline dimension_mismatch(args...) = dimension_mismatch(string(args...))

"""
    ordinal_suffix(n) -> "st" or "nd" or "rd" or "th"

yields the ordinal suffix for integer `n`.

"""
function ordinal_suffix(n::Integer)
    if n > 0
        d = mod(n, 10)
        if d == 1
            return "st"
        elseif d == 2
            return "nd"
        elseif d == 3
            return "rd"
        end
    end
    return "th"
end

#------------------------------------------------------------------------------
# Apply operators.

"""
    dispatch_multipliers!(α, f, A, x, β, y) -> y

dispatch calls to function `f` as `f(α,A,x,β,axpy)` with `α`, `A`, `x`, `β` and
`y` the other arguments and where `axpy` is a function called with 4 scalar
arguments as `axpy(α,x,β,y)` to yield `α*x + β*y` but which is optimized
depending on the values of the multipliers `α` and `β`.  For instance, if `α=1`
and `β=0`, then `axpy(α,x,β,y)` just evaluates as `x`.

The `dispatch_multipliers!` method is a helper to apply a mapping `A` to an
argument `x` and store the result in `y`.  In pseudo-code, this amounts to
performing `y <- α*op(A)(x) + β*y` where `op(A)` denotes a variant of `A` which
usually depends on `f`.

"""
function dispatch_multipliers!(α::Number, f::Function, A, x, β::Number, y)
    if α == 0
        vscale!(y, β)
    elseif α == 1
        if β == 0
            f(1, A, x, 0, y, axpby_yields_x)
        elseif β == 1
            f(1, A, x, 1, y, axpby_yields_xpy)
        else
            b = promote_multiplier(β, eltype(y))
            f(1, A, x, b, y, axpby_yields_xpby)
        end
    else
        a = promote_multiplier(α, eltype(A), eltype(x))
        if β == 0
            f(a, A, x, 0, y, axpby_yields_ax)
        elseif β == 1
            f(a, A, x, 1, y, axpby_yields_axpy)
        else
            b = promote_multiplier(β, eltype(y))
            f(a, A, x, b, y, axpby_yields_axpby)
        end
    end
    return y
end

# Generic version of `vcreate` for most compressed sparse linear operators.
#
# We assume that in-place operation is not possible and thus simply ignore the
# `scratch` flag.  Operators which can be applied in-place shall specialize
# this method.  We do not check the dimensions and indexing of `x` as this will
# be done when `apply!` is called.

function vcreate(::Type{P},
                 A::SparseOperator{Ta,M,N},
                 x::AbstractArray{Tx,N},
                 scratch::Bool) where {Ta,Tx,M,N,P<:Union{Direct,InverseAdjoint}}
    Ty = promote_type(Ta,Tx)
    return Array{Ty}(undef, row_size(A))
end

function vcreate(::Type{P},
                 A::SparseOperator{Ta,M,N},
                 x::AbstractArray{Tx,M},
                 scratch::Bool) where {Ta,Tx,M,N,P<:Union{Adjoint,Inverse}}
    Ty = promote_type(Ta,Tx)
    return Array{Ty}(undef, col_size(A))
end

# Apply a sparse linear mapping, and its adjoint, stored in Compressed Sparse
# Row (CSR) format.

function apply!(α::Number,
                ::Type{Direct},
                A::CompressedSparseOperator{:CSR,Ta,M,N},
                x::AbstractArray{Tx,N},
                scratch::Bool,
                β::Number,
                y::AbstractArray{Ty,M}) where {Ta,Tx,Ty,M,N}
    check_argument(x, col_size(A))
    check_argument(y, row_size(A))
    dispatch_multipliers!(α, unsafe_apply_direct!, A, x, β, y)
end

function unsafe_apply_direct!(α::Number,
                              A::CompressedSparseOperator{:CSR,Ta,M,N},
                              x::AbstractArray{Tx,N},
                              β::Number,
                              y::AbstractArray{Ty,M},
                              axpby::Function) where {Ta,Tx,Ty,M,N}
    @inbounds for i in each_row(A)
        s = zero(promote_type(Ta, Tx))
        for k in each_off(A, i)
            j = get_col(A, k)
            v = get_val(A, k)
            s += v*x[j]
        end
        y[i] = axpby(α, s, β, y[i])
    end
end

function apply!(α::Number,
                ::Type{Adjoint},
                A::CompressedSparseOperator{:CSR,Ta,M,N},
                x::AbstractArray{Tx,M},
                scratch::Bool,
                β::Number,
                y::AbstractArray{Ty,N}) where {Ta,Tx,Ty,M,N}
    check_argument(x, row_size(A))
    check_argument(y, col_size(A))
    Tm = promote_type(Ta, Tx) # to promote multipliers
    β == 1 || vscale!(y, β)
    if α == 1
        @inbounds for i in each_row(A)
            q = promote_multiplier(x[i], Tm)
            if q != 0
                for k in each_off(A, i)
                    j = get_col(A, k)
                    v = get_val(A, k)
                    y[j] += q*conj(v)
                end
            end
        end
    elseif α != 0
        a = promote_multiplier(α, Tm)
        @inbounds for i in each_row(A)
            q = a*promote_multiplier(x[i], Tm)
            if q != 0
                for k in each_off(A, i)
                    j = get_col(A, k)
                    v = get_val(A, k)
                    y[j] += q*conj(v)
                end
            end
        end
    end
    return y
end

# Apply a sparse linear mapping, and its adjoint, stored in Compressed Sparse
# Column (CSC) format.

function apply!(α::Number,
                ::Type{Direct},
                A::CompressedSparseOperator{:CSC,Ta,M,N},
                x::AbstractArray{Tx,N},
                scratch::Bool,
                β::Real,
                y::AbstractArray{Ty,M}) where {Ta,Tx,Ty,M,N}
    check_argument(x, col_size(A))
    check_argument(y, row_size(A))
    Tm = promote_type(Ta, Tx) # to promote multipliers
    β == 1 || vscale!(y, β)
    if α == 1
        @inbounds for j in each_col(A)
            q = promote_multiplier(x[j], Tm)
            if q != 0
                for k in each_off(A, j)
                    i = get_row(A, k)
                    v = get_val(A, k)
                    y[i] += q*v
                end
            end
        end
    elseif α != 0
        a = promote_multiplier(α, Tm)
        @inbounds for j in each_col(A)
            q = a*promote_multiplier(x[j], Tm)
            if q != 0
                for k in each_off(A, j)
                    i = get_row(A, k)
                    v = get_val(A, k)
                    y[i] += q*v
                end
            end
        end
    end
    return y
end

function apply!(α::Real,
                ::Type{Adjoint},
                A::CompressedSparseOperator{:CSC,Ta,M,N},
                x::AbstractArray{Tx,M},
                scratch::Bool,
                β::Real,
                y::AbstractArray{Ty,N}) where {Ta,Tx,Ty,M,N}
    check_argument(x, row_size(A))
    check_argument(y, col_size(A))
    dispatch_multipliers!(α, unsafe_apply_adjoint!, A, x, β, y)
end

function unsafe_apply_adjoint!(α::Number,
                               A::CompressedSparseOperator{:CSC,Ta,M,N},
                               x::AbstractArray{Tx,M},
                               β::Number,
                               y::AbstractArray{Ty,N},
                               axpby::Function) where {Ta,Tx,Ty,M,N}
    @inbounds for j in each_col(A)
        s = zero(promote_type(Ta, Tx))
        for k in each_off(A, j)
            i = get_row(A, k)
            v = get_val(A, k)
            s += conj(v)*x[i]
        end
        y[j] = axpby(α, s, β, y[j])
    end
    return y
end

# Apply a sparse linear mapping, and its adjoint, stored in Compressed Sparse
# Coordinate (COO) format.

function apply!(α::Number,
                ::Type{Direct},
                A::CompressedSparseOperator{:COO,Ta,M,N},
                x::AbstractArray{Tx,N},
                scratch::Bool,
                β::Number,
                y::AbstractArray{Ty,M}) where {Ta,Tx,Ty,M,N}
    check_argument(x, col_size(A))
    check_argument(y, row_size(A))
    β == 1 || vscale!(y, β)
    if α != 0
        V, I, J = get_vals(A), get_rows(A), get_cols(A)
        if α == 1
            @inbounds for k in eachindex(V, I, J)
                v, i, j = V[k], I[k], J[k]
                y[i] += x[j]*v
            end
        elseif α == -1
            @inbounds for k in eachindex(V, I, J)
                v, i, j = V[k], I[k], J[k]
                y[i] -= x[j]*v
            end
        else
            # The ordering of operations is to minimize the number of
            # operations in case `v` is complex while `α` and `x` are reals.
            alpha = promote_multiplier(α, Ta, Tx)
            @inbounds for k in eachindex(V, I, J)
                v, i, j = V[k], I[k], J[k]
                y[i] += (alpha*x[j])*v
            end
        end
    end
    return y
end

function apply!(α::Number,
                ::Type{Adjoint},
                A::CompressedSparseOperator{:COO,Ta,M,N},
                x::AbstractArray{Tx,M},
                scratch::Bool,
                β::Number,
                y::AbstractArray{Ty,N}) where {Ta,Tx,Ty,M,N}
    check_argument(x, row_size(A))
    check_argument(y, col_size(A))
    β == 1 || vscale!(y, β)
    if α != 0
        V, I, J = get_vals(A), get_rows(A), get_cols(A)
        if α == 1
            @inbounds for k in eachindex(V, I, J)
                v, i, j = V[k], I[k], J[k]
                y[j] += x[i]*conj(v)
            end
        elseif α == -1
            @inbounds for k in eachindex(V, I, J)
                v, i, j = V[k], I[k], J[k]
                y[j] -= x[i]*conj(v)
            end
        else
            # The ordering of operations is to minimize the number of
            # operations in case `v` is complex while `α` and `x` are reals.
            alpha = promote_multiplier(α, Ta, Tx)
            @inbounds for k in eachindex(V, I, J)
                v, i, j = V[k], I[k], J[k]
                y[j] += (alpha*x[i])*conj(v)
            end
        end
    end
    return y
end

end # module SparseOperators

# The following module is to facilitate using compressed sparse linear
# operators at a lower level than the exported API.
module SparseMethods

export
    CompressedSparseOperator,
    SparseOperator,
    SparseOperatorCOO,
    SparseOperatorCSC,
    SparseOperatorCSR,
    col_size,
    copy_cols,
    copy_rows,
    copy_vals,
    each_col,
    each_off,
    each_row,
    get_col,
    get_cols,
    get_offs,
    get_row,
    get_rows,
    get_val,
    get_vals,
    ncols,
    nnz,
    nonzeros,
    nrows,
    row_size,
    set_val!

import ..SparseOperators:
    CompressedSparseOperator,
    SparseOperator,
    SparseOperatorCOO,
    SparseOperatorCSC,
    SparseOperatorCSR,
    col_size,
    copy_cols,
    copy_rows,
    copy_vals,
    each_col,
    each_off,
    each_row,
    get_col,
    get_cols,
    get_offs,
    get_row,
    get_rows,
    get_val,
    get_vals,
    ncols,
    nnz,
    nonzeros,
    nrows,
    row_size,
    set_val!

end # module SparseMethods
