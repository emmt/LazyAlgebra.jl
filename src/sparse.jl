"""

Module `SparseOperators` implements various format of compressed sparse linear operators,
an API to deal with sparse operators, and methods to apply sparse operators and convert
between different sparse format. This goes beyond Julia's `SparseArrays` standard package
which only provides "Compressed Sparse Column" (CSC) format.

See https://en.wikipedia.org/wiki/Sparse_matrix.

"""
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
using ..LazyAlgebra:
    @callable,
    Adjoint,
    HasInputShape,
    HasOutputShape,
    LazyMap

import .LazyAlgebra:
    #MorphismType,
    InputShape,
    OutputShape,
    unsafe_vmul!,
    dispatch_vmul!,
    vmul!,
    #identical,
    coefficients,
    #row_size,
    #col_size,
    #nrows,
    #ncols,
    #input_ndims,
    #input_size,
    #output_ndims,
    #output_size,
    output_axes,
    output_eltype

import SparseArrays
using SparseArrays: SparseMatrixCSC, nonzeros, nnz

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
to_values(::Type{Any}, vals::AbstractVector{T}) where {T} = to_values(T, vals)
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

as_matrix(A::AbstractMatrix, nrows::Int, ncols::Int) = begin
    size(A) == (nrows, ncols) || throw(DimensionMismatch(
        "argument has size $(size(A)), expecting ($nrows, $ncols)"))
    return A
end
as_matrix(A::AbstractArray, nrows::Int, ncols::Int) =
    reshape(A, (nrows, ncols))

#------------------------------------------------------------------------------

"""
    SparseOperator{T,M,N}

is the abstract type inherited by sparse operator types. Parameter `T` is the type of the
elements. Parameters `M` and `N` are the number of dimensions of the *rows* and of the
*columns* respectively. Sparse operators are a generalization of sparse matrices in the
sense that they implement linear operators which can be applied to `N`-dimensional
arguments to produce `M`-dimensional results (as explained below). See
[`PseudoMatrix`](@ref) for a similar generalization but for *dense* matrices.

See [`CompressedSparseOperator`](@ref) for usage of sparse operators implementing
compressed storage formats.

"""
abstract type SparseOperator{T,M,N} <: Operator end

"""
    CompressedSparseOperator{F,T,M,N}

is an abstract sub-type of `SparseOperator{T,M,N}` and is inherited by the concrete types
implementing sparse operators with compressed storage in format `F`.

Format `F` is specified as a symbol and can be:

- `:COO` for *Compressed Sparse Coordinate* storage format. This format is not the most
  efficient, it is mostly used as an intermediate for building a sparse operator in one of
  the other formats.

- `:CSC` for *Compressed Sparse Column* storage format. This format is very efficient for
  applying the adjoint of the sparse operator.

- `:CSR` for *Compressed Sparse Row* storage format. This format is very efficient for
  applying the sparse operator.

To construct (or convert to) a sparse operator with compressed storage format `F`, you can
call:

    CompressedSparseOperator{F}(args...; kwds...)
    CompressedSparseOperator{F,T}(args...; kwds...)
    CompressedSparseOperator{F,T,M}(args...; kwds...)
    CompressedSparseOperator{F,T,M,N}(args...; kwds...)

where given parameters `T`, `M` and `N`, arguments `args...` and optional keywords
`kwds...` will be passed to the concrete constructor [`SparseOperatorCOO`](@ref),
[`SparseOperatorCSC`](@ref) or [`SparseOperatorCSR`](@ref) corresponding to the format
`F`.

It is possible to use a compressed sparse operator `A` as an iterator:

```julia
for (Aij,i,j) in A # simple but slow for CSR and CSC
    ...
end
```

to retrieve the values `Aij` and respective row `i` and column `j` indices for all the
entries stored in `A`. It is however more efficient to access them according to their
storage order which depends on the compressed format.

- If `A` is in CSC format or is the adjoint of a sparse operator in CSR format:

  ```julia
  using LazyAlgebra.SparseMethods
  for j in each_col(A)        # loop over column index
      for k in each_nz(A, j)  # loop over structural non-zeros in this column
          i   = get_row(A, k) # get row index of entry
          Aij = get_val(A, k) # get value of entry
       end
  end
  ```

- If `A` is in CSR format or is the adjoint of a sparse operator in CSC format:

  ```julia
  using LazyAlgebra.SparseMethods
  for i in each_row(A)        # loop over row index
      for k in each_nz(A, i)  # loop over structural non-zeros in this row
          j   = get_col(A, k) # get column index of entry
          Aij = get_val(A, k) # get value of entry
       end
  end
  ```

- If `A` is in COO format:

  ```julia
  using LazyAlgebra.SparseMethods
  for k in each_nz(A)      # loop over all structural non-zeros
       i   = get_row(A, k) # get row index of entry
       j   = get_col(A, k) # get column index of entry
       Aij = get_val(A, k) # get value of entry
  end
  ```

The low-level methods `each_row`, `each_col`, `each_nz`, `get_row`, `get_col` and
`get_val` are not automatically exported by `LazyAlgebra`, this is the purpose of the
statement `using LazyAlgebra.SparseMethods`.

"""
abstract type CompressedSparseOperator{F,T,M,N} <: SparseOperator{T,M,N} end

@callable struct SparseOperatorCSR{T,M,N,
                                   V<:AbstractVector{T},
                                   J<:AbstractVector{Int},
                                   K<:AbstractVector{Int}
                                   } <: CompressedSparseOperator{:CSR,T,M,N}
    m::Int          # equivalent number of rows of the operator
    n::Int          # number of columns of the operator
    vals::V         # values of entries
    cols::J         # linear column indices of entries
    offs::K         # row offsets in arrays of entries and column indices
    rowsiz::Dims{M} # dimensions of rows
    colsiz::Dims{N} # dimensions of columns

    # A private inner constructor is defined to prevent Julia from providing a simple
    # outer constructor, it is not meant to be called directly as it does not check
    # whether arguments are correct.
    global _SparseOperatorCSR
    function _SparseOperatorCSR(m::Integer, n::Integer,
                                vals::V, cols::J, offs::K,
                                rowsiz::Dims{M},
                                colsiz::Dims{N}) where {T,M,N,
                                                        V<:AbstractVector{T},
                                                        J<:AbstractVector{Int},
                                                        K<:AbstractVector{Int}}
        new{T,M,N,V,J,K}(m, n, vals, cols, offs, rowsiz, colsiz)
    end
end

@callable struct SparseOperatorCSC{T,M,N,
                                   V<:AbstractVector{T},
                                   I<:AbstractVector{Int},
                                   K<:AbstractVector{Int}
                                   } <: CompressedSparseOperator{:CSC,T,M,N}
    m::Int          # equivalent number of rows of the operator
    n::Int          # number of columns of the operator
    vals::V         # values of entries
    rows::I         # linear row indices of entries
    offs::K         # columns offsets in arrays of entries and row indices
    rowsiz::Dims{M} # dimensions of rows
    colsiz::Dims{N} # dimensions of columns

    # A private inner constructor is defined to prevent Julia from providing a simple
    # outer constructor, it is not meant to be called directly as it does not check
    # whether arguments are correct.
    global _SparseOperatorCSC
    function _SparseOperatorCSC(m::Int, n::Int,
                                vals::V, rows::I, offs::K,
                                rowsiz::Dims{M},
                                colsiz::Dims{N}) where {T,M,N,
                                                        V<:AbstractVector{T},
                                                        I<:AbstractVector{Int},
                                                        K<:AbstractVector{Int}}
        new{T,M,N,V,I,K}(m, n, vals, rows, offs, rowsiz, colsiz)
    end
end

@callable struct SparseOperatorCOO{T,M,N,
                                   V<:AbstractVector{T},
                                   I<:AbstractVector{Int},
                                   J<:AbstractVector{Int}
                                   } <: CompressedSparseOperator{:COO,T,M,N}
    m::Int          # equivalent number of rows of the operator
    n::Int          # number of columns of the operator
    vals::V         # values of entries
    rows::I         # linear row indices of entries
    cols::J         # linear column indices of entries
    rowsiz::Dims{M} # dimensions of rows
    colsiz::Dims{N} # dimensions of columns

    # A private inner constructor is defined to prevent Julia from providing a simple outer
    # constructor, it is not meant to be called directly as it does not check whether
    # arguments are correct.
    global _SparseOperatorCOO
    function _SparseOperatorCOO(m::Integer, n::Integer,
                                vals::V, rows::I, cols::J,
                                rowsiz::Dims{M},
                                colsiz::Dims{N}) where {T,M,N,
                                                        V<:AbstractVector{T},
                                                        I<:AbstractVector{Int},
                                                        J<:AbstractVector{Int}}
        new{T,M,N,V,I,J}(m, n, vals, rows, cols, rowsiz, colsiz)
    end
end

# Unions of compressed sparse operators that can be considered as being in a given storage
# format. Whatever the format, `T` is the element type, `M` is the number of output
# dimensions, and `N` is the number of input dimensions.

const AnyCSR{T,M,N} = Union{CompressedSparseOperator{:CSR,T,M,N},
                            Adjoint{<:CompressedSparseOperator{:CSC,T,N,M}}}

const AnyCSC{T,M,N} = Union{CompressedSparseOperator{:CSC,T,M,N},
                            Adjoint{<:CompressedSparseOperator{:CSR,T,N,M}}}

const AnyCOO{T,M,N} = Union{CompressedSparseOperator{:COO,T,M,N},
                            Adjoint{<:CompressedSparseOperator{:COO,T,N,M}}}

const CSRorCSC = Union{AnyCSR,AnyCSC,SparseMatrixCSC}

#-----------------------------------------------------------------------------------------
# Accessors and basic methods.

Base.eltype(::Type{<:SparseOperator{T,M,N}}) where {T,M,N} = T
InputShape(::Type{<:SparseOperator{T,M,N}}) where {T,M,N} = HasInputShape{N}()
OutputShape(::Type{<:SparseOperator{T,M,N}}) where {T,M,N} = HasOuputShape{M}()

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
Base.axes(A::SparseOperator) = map(Base.OneTo, size(A))

# Use constructors to perform conversion (the first method is to resolve ambiguities).
Base.convert(::Type{T}, A::T) where {T<:SparseOperator} = A
Base.convert(::Type{T}, A) where {T<:SparseOperator} = T(A)

# Assume that a `copy` of a compressed sparse operator is to keep the same structure for
# the structural non-zeros but possibly change the values. So only duplicate the value
# part. For a `deepcopy` of a compressed sparse operator, all the fields are copied.
for (func, rows, cols, offs) in ((:copy,      :get_rows,  :get_cols,  :get_offs),
                                 (:deepcopy, :copy_rows, :copy_cols, :copy_offs))
    @eval begin
        Base.$func(A::SparseOperatorCSR) =
            _SparseOperatorCSR(nrows(A), ncols(A), copy_vals(A), $cols(A), $offs(A),
                               row_size(A), col_size(A))

        Base.$func(A::SparseOperatorCSC{T,M,N}) where {T,M,N} =
            _SparseOperatorCSC(nrows(A), ncols(A), copy_vals(A), $rows(A), $offs(A),
                               row_size(A), col_size(A))

        Base.$func(A::SparseOperatorCOO{T,M,N}) where {T,M,N} =
            _SparseOperatorCOO(nrows(A), ncols(A), copy_vals(A), $rows(A), $cols(A),
                               row_size(A), col_size(A))
    end
end

# `findnz(A) -> I,J,V` yields the row and column indices and the values of the stored
# values in `A`.
SparseArrays.findnz(A::SparseOperator) = (get_rows(A), get_cols(A), get_vals(A))

# Extend some methods in SparseArrays. The "structural" non-zeros are the entries stored
# by the sparse structure which may or not be equal to zero, un-stored entries are always
# considered as being strictly equal to zero.
SparseArrays.nonzeros(A::SparseOperator) = get_vals(A)
SparseArrays.nnz(A::SparseOperator) = length(nonzeros(A))
SparseArrays.nnz(A::Adjoint{<:SparseOperator}) = length(nonzeros(parent(A)))

"""
    LazyAlgebra.get_vals(A)

yields the array storing the structural non-zeros of the compressed sparse operator `A`.
The returned array is shared with `A`, call [`LazyAlgebra.copy_vals(A)`](@ref
LazyAlgebra.copy_vals) instead if you want to modify the contents of the returned array
with no side effects on `A`.

"""
get_vals(A::SparseOperator) = getfield(A, :vals)
get_vals(A::Adjoint{<:SparseOperator}) = LazyMap{eltype(A)}(conj, get_vals(parent(A)))

"""
    LazyAlgebra.copy_vals([T = eltype(A),] A) -> vals

yields a copy of the values of the structural non-zeros of the sparse operator `A`
converted to type `T`. The result is a vector that is not shared by `A`, the caller may
thus modify its contents with no side effects on `A`.

"""
copy_vals(A::SparseOperator{T}) where {T} = copy_vals(T, A)
function copy_vals(::Type{T}, A::SparseOperator) where {T}
    vals = get_vals(A)
    unsafe_vcopy!(similar(vals, T), vals)
end

"""
    LazyAlgebra.get_rows(A)

yields the row indices of the structural non-zeros of the sparse operator `A`. The
returned array may be shared with `A`, call [`LazyAlgebra.copy_rows(A)`](@ref
LazyAlgebra.copy_rows) instead if you want to modify the contents of the returned array
with no side effects on `A`.

"""
get_rows(A::Union{SparseOperatorCSC,SparseOperatorCOO}) = getfield(A, :rows)
get_rows(A::CompressedSparseOperator{:CSR}) = copy_rows(A) # FIXME: yield an iterator
get_rows(A::Adjoint{<:SparseOperator}) = get_cols(parent(A))

"""
    LazyAlgebra.copy_rows(A) -> rows

yields a copy of the linear row indices of the structural non-zeros of the sparse operator
`A`. The result is a vector that is not shared by `A`, the caller may thus modify its
contents with no side effects on `A`.

"""
function copy_rows(A::SparseOperator)
    rows = get_rows(A)
    unsafe_vcopy!(Vector{Int}(undef, size(rows)), rows)
end
function copy_rows(A::CompressedSparseOperator{:CSR})
    rows = Vector{Int}(undef, length(get_vals(A)))
    @inbounds for i in each_row(A)
        @simd for k in each_nz(A, i)
            rows[k] = i
        end
    end
    return rows
end

"""
    LazyAlgebra.get_cols(A)

yields the column indices of the structural non-zeros of the sparse operator `A`. The
returned array may be shared with `A`, call [`LazyAlgebra.copy_cols(A)`](@ref
LazyAlgebra.copy_cols) instead if you want to modify the contents of the returned array
with no side effects on `A`.

"""
get_cols(A::Union{SparseOperatorCSR,SparseOperatorCOO}) = getfield(A, :cols)
get_cols(A::Union{CompressedSparseOperator{:CSC},SparseMatrixCSC}) =
    copy_cols(A) # FIXME: yield an iterator
get_cols(A::Adjoint{<:SparseOperator}) = get_rows(parent(A))

"""
    LazyAlgebra.copy_cols(A) -> cols

yields a copy of the linear column indices of the structural non-zeros of the sparse
operator `A`. The result is a vector that is not shared by `A`, the caller may thus modify
its contents with no side effects on `A`.

"""
function copy_cols(A::SparseOperator)
    cols = get_cols(A)
    unsafe_vcopy!(Vector{Int}(undef, size(cols)), cols)
end
function copy_cols(A::Union{CompressedSparseOperator{:CSC},SparseMatrixCSC})
    cols = Vector{Int}(undef, length(get_vals(A)))
    @inbounds for j in each_col(A)
        @simd for k in each_nz(A, j)
            cols[k] = j
        end
    end
    return cols
end

"""
    LazyAlgebra.get_offs(A)

yields the table of offsets of the sparse operator `A`. Not all operators extend this
method.

!!! warning
    The interpretation of offsets depend on the type of `A`. For instance, assuming `offs
    = LazyAlgebra.get_offs(A)`, then the index range of the `j`-th column of a
    `SparseMatrixCSC` is `offs[j]:(offs[j+1]-1)` while the index range is
    `(offs[j]+1):offs[j+1]` for a `SparseOperatorCSC`. For this reason, it is recommended
    to call [`each_nz`](@ref) instead or to call `get_offs` with 2 arguments as shown
    below.

"""
get_offs(A::Union{SparseOperatorCSR,SparseOperatorCSC}) = getfield(A, :offs)
get_offs(A::Adjoint{<:CompressedSparseOperator{:CSR}}) = get_offs(parent(A))
get_offs(A::Adjoint{<:CompressedSparseOperator{:CSC}}) = get_offs(parent(A))

"""

For a sparse operator `A` stored in a *Compressed Sparse Coordinate* (COO) format, the
call:

    each_nz(A)

yields an iterator over the indices in the arrays of values and of linear row and column
indices for the `k`-th entry of `A`.

---

For a sparse operator `A` stored in a *Compressed Sparse Column* (CSC) format, the call:

    each_nz(A, j)

yields an iterator over the indices in the arrays of values and linear row indices for the
`j`-th column of `A`.

---

For a sparse operator `A` stored in a *Compressed Sparse Row* (CSR) format, the call:

    each_nz(A, i)

yields an iterator over the indices in the arrays of values and linear column indices for
the `i`-th row of `A`.

"""
@inline each_nz(A::Union{SparseOperatorCOO,Adjoint{<:SparseOperatorCOO}}) =
    Base.OneTo(nnz(A))

@inline first_nz(A::Union{SparseOperatorCOO,Adjoint{<:SparseOperatorCOO}}) = 1

@inline last_nz(A::Union{SparseOperatorCOO,Adjoint{<:SparseOperatorCOO}}) = nnz(A)

@inline function first_nz(A::CSRorCSC, ij::Int)
    @boundscheck check_offset_index(A, ij)
    return unsafe_first_nz(A, ij)
end

@inline function last_nz(A::CSRorCSC, ij::Int)
    @boundscheck check_offset_index(A, ij)
    return unsafe_last_nz(A, ij)
end

@inline function each_nz(A::CSRorCSC, ij::Int)
    @boundscheck check_offset_index(A, ij)
    return unsafe_each_nz(A, ij)
end

# Management of offsets in compressed sparse operator in CSC- or CSR-like formats.

@inline check_offset_index(::Type{Bool}, A::CSRorCSC, ij::Int) =
    1 ≤ ij < length(get_offs(A))

@inline check_offset_index(A::AnyCSR, i::Int) =
    check_offset_index(Bool, A, i) ? nothing : out_of_range_row_index(A, i)

@inline check_offset_index(A::Union{AnyCSC,SparseMatrixCSC}, j::Int) =
    check_offset_index(Bool, A, j) ? nothing : out_of_range_column_index(A, i)

@inline unsafe_first_nz(A::Union{AnyCSR,AnyCSC}, ij::Int) = @inbounds get_offs(A)[ij] + 1

@inline unsafe_last_nz(A::Union{AnyCSR,AnyCSC}, ij::Int) = @inbounds get_offs(A)[ij + 1]

@inline unsafe_each_nz(A::Union{AnyCSR,AnyCSC}, ij::Int) =
    UnitRange(unsafe_first_nz(A, ij), unsafe_last_nz(A, ij))

@noinline out_of_range_row_index(A, i::Integer) =
    throw(ErrorException(string("out of range row index ", i,
                                " for compressed sparse operator with ",
                                nrows(A), " rows")))

@noinline out_of_range_column_index(A, j::Integer) =
    throw(ErrorException(string("out of range column index ", j,
                                " for compressed sparse operator with ",
                                ncols(A), " columns")))

"""
    each_row(A)

yields an iterator over the linear row indices of the structural non-zeros of the sparse
operator `A` stored in a *Compressed Sparse Row* (CSR) format, this includes the adjoint
of a sparse operator in *Compressed Sparse Column* (CSC) format.

"""
each_row(A::CompressedSparseOperator{:CSR}) = Base.OneTo(nrows(A))
each_row(A::Adjoint{<:CompressedSparseOperator{:CSC}}) = each_col(parent(A))

"""
    each_col(A)

yields an iterator over the linear column indices of the structural non-zeros of the
sparse operator `A` stored in a *Compressed Sparse Column* (CSC) format, this includes the
adjoint of a sparse operator in *Compressed Sparse Row* (CSR) format.

"""
each_col(A::CompressedSparseOperator{:CSC}) = Base.OneTo(ncols(A))
each_col(A::Adjoint{<:CompressedSparseOperator{:CSR}}) = each_row(parent(A))

"""
    get_row(A, k) -> i

yields the linear row index of the `k`-th entry of the sparse operator `A` stored in a
*Compressed Sparse Column* (CSC) or *Coordinate* (COO) formats (this includes adjoint of
sparse operators in CSR format).

"""
@propagate_inbounds get_row(A::Union{AnyCOO,AnyCSC}, k::Int) = get_rows(A)[k]

"""
    get_col(A, k) -> j

yields the linear column index of the `k`-th entry of the sparse operator `A` stored in a
*Compressed Sparse Row* (CSR) or *Coordinate* (COO) formats (this includes adjoint of
sparse operators in CSC format).

"""
@propagate_inbounds get_col(A::Union{AnyCOO,AnyCSR}, k::Int) = get_cols(A)[k]

"""
    get_val(A, k) -> v

yields the value of the `k`-th structural non-zero of the sparse operator `A` stored in a
compressed format. Argument `A` may also be the adjoint of a compressed sparse operator.

"""
@inline function get_val(A::CompressedSparseOperator, k::Int)
    vals = get_vals(A)
    @boundscheck checkbounds(vals, k)
    v = @inbounds vals[k]
    return v
end

@inline function get_val(A::Adjoint{<:CompressedSparseOperator}, k::Int)
    vals = get_vals(parent(A))
    @boundscheck checkbounds(vals, k)
    v = @inbounds vals[k]
    return conj(v)
end

"""
    set_val!(A, k, v) -> A

assigns `v` to the value of the `k`-th structural non-zero of the sparse operator `A`
stored in a compressed format. Argument `A` may also be the adjoint of a compressed sparse
operator in which case the call is similar to `set_val!(A', k, conj(v))`.

"""
@inline function set_val!(A::CompressedSparseOperator, k::Int, v)
    vals = get_vals(A)
    @boundscheck checkbounds(vals, k)
    @inbounds vals[k] = v
    return A
end

@inline function set_val!(A::Adjoint{<:CompressedSparseOperator}, k::Int, v)
    vals = get_vals(parent(A))
    @boundscheck checkbounds(vals, k)
    @inbounds vals[k] = conj(v)
    return A
end

# Iterators to deliver (v,i,j).

@inline function Base.iterate(A::AnyCSR, (i, k, kmax)::Tuple{Int,Int,Int} = (0,0,0))
    @inbounds begin
        k += 1
        while k > kmax
            i < nrows(A) || return nothing
            i += 1
            kmax = last_nz(A, i)
        end
        v = get_val(A, k)
        j = get_col(A, k)
        return ((v, i, j), (i, k, kmax))
    end
end

@inline function Base.iterate(A::AnyCSC, (j, k, kmax)::Tuple{Int,Int,Int} = (0,0,0))
    @inbounds begin
        k += 1
        while k > kmax
            j < ncols(A) || return nothing
            j += 1
            kmax = last_nz(A, j)
        end
        v = get_val(A, k)
        i = get_row(A, k)
        return ((v, i, j), (j, k, kmax))
    end
end

@inline function Base.iterate(A::AnyCOO, (k, kmax)::Tuple{Int,Int} = (0, nnz(A)))
    @inbounds begin
        k < kmax || return nothing
        k += 1
        return ((get_val(A, k), get_row(A, k), get_col(A, k)), (k, kmax))
    end
end

#-----------------------------------------------------------------------------------------
# Extend LzyAlgebra sparse operator API for SparseArrays.SparseMatrixCSC.

nrows(A::SparseMatrixCSC) = getfield(A, :m)
ncols(A::SparseMatrixCSC) = getfield(A, :n)
get_vals(A::SparseMatrixCSC) = getfield(A, :nzval)
get_offs(A::SparseMatrixCSC) = getfield(A, :colptr)
get_rows(A::SparseMatrixCSC) = getfield(A, :rowval)
# get_cols is already done elsewhere.
row_size(A::SparseMatrixCSC) = (nrows(A),)
col_size(A::SparseMatrixCSC) = (ncols(A),)
each_col(A::SparseMatrixCSC) = Base.OneTo(ncols(A))

# Provide a specific versions of `check_offset_index`, `unsafe_first_nz`, and
# `unsafe_last_nz` because offsets have a slightly different definition for
# `SparseMatrixCSC` than for our CSC format.
@inline check_offset_index(::Type{Bool}, A::SparseMatrixCSC, j::Int) =
     1 ≤ j < length(get_offs(A))
@inline unsafe_first_nz(A::SparseMatrixCSC, j::Int) = @inbounds get_offs(A)[j]
@inline unsafe_last_nz(A::SparseMatrixCSC, j::Int) = @inbounds get_offs(A)[j + 1] - 1

#-----------------------------------------------------------------------------------------
# Constructors.

"""

Sparse operators in *Compressed Sparse Coordinate* (COO) format store their structural
non-zeros in no particular order, as a vector of values, a vector of linear row indices
and a vector of linear column indices. It is even possible to have repeated entries. This
format is very useful to build a sparse linear operator. It can be converted to a more
efficient format like *Compressed Sparse Column* (CSC) or *Compressed Sparse Row* (CSR)
for fast application of the sparse linear operator or of its adjoint.

A sparse operator in COO storage format can be constructed by providing all necessary
information:

    SparseOperatorCOO(vals, rows, cols, rowsiz, colsiz)

where `vals` is the vector of structural non-zeros, `rows` and `cols` are integer valued
vectors with the linear row and column indices of the structural non-zeros, `rowsiz` and
`colsiz` are the sizes of the row and column dimensions. The value and linear row and
column indices of the `k`-th structural non-zero are respectively given by `vals[k]`,
`rows[k]` and `cols[k]`. For efficiency reasons, sparse operators are currently limited to
*fast* arrays because they can be indexed linearly with no loss of performances. If
`vals`, `rows` and/or `cols` are not fast arrays, they will be automatically converted to
linearly indexed arrays.

A sparse operator in COO storage format can be directly constructed from a 2-dimensional
Julia array `A`:

    SparseOperatorCOO(A, f = (v,i,j) -> !iszero(v))

where optional argument `f` is a predicate function which is called as `f(v,i,j)` with
`v`, `i` and `j` the value, the row and the column linear indices for each entry of `A`
and which yields whether a given entry of `A` is a structural non-zeros. The default
predicate is such that all non-zeros of `A` are considered as being structural non-zeros.

The element type, say `T`, of the values of the structural non-zeros can be imposed by
rewriting the above examples as:

    SparseOperatorCOO{T}(args...)

A sparse operator in COO storage format implementing generalized matrix-vector
multiplication can also be directly constructed from a `L`-dimensional Julia array (with
`L ≥ 2`) `A` by:

    SparseOperatorCOO{T,M}(A[, f])

with `M` the number of leading dimensions of `A` corresponding to the *rows* of the
operator, the trailing `N = L - M` dimensions being assumed to correspond to the *columns*
of the operator. These dimensions are the size of, respectively, the output and the input
arrays when applying the operator. The parameter `N` may be specified (although it can be
automatically determined):

    SparseOperatorCOO{T,M,N}(A[, f])

provided `M + N = ndims(A)` holds.

A last parameter `V` can be specified for the type of the vector to store the values of
the structural non-zeros:

    SparseOperatorCOO{T,M,N,V}(args...)

provided `V` implements standard linear indexing. The default is to take `V = Vector{T}`.
As a special case, you can choose a uniform boolean vector from the `StructuredArrays`
package to store the sparse coefficients:

    SparseOperatorCOO{T,M,N,UniformVector{Bool}}(args...)

to get a compressed sparse operator in COO format whose values are an immutable uniform
vector of true values requiring no storage. This is useful to only store the sparse
structure of the operator, that is the indices in COO format of the sparse coefficients
not their values.

The `SparseOperatorCOO` constructor can also be used to convert a sparse operator in
another storage format into the COO format. In that case, parameter `T` may also be
specified to convert the type of the sparse coefficients.

""" SparseOperatorCOO

"""

Sparse operators in *Compressed Sparse Column* (CSC) format store their structural
non-zeros in a column-wise order, as a vector of values, a vector of corresponding linear
row indices and a vector of offsets indicating, for each column, the range of indices in
the vectors of values and of row indices. This storage format is very suitable for fast
application of the operator, notably its adjoint.

A sparse operator in CSC storage format can be constructed by providing all necessary
information:

    SparseOperatorCSC(vals, rows, offs, rowsiz, colsiz)

where `vals` is the vector of structural non-zeros, `rows` is an integer valued vector
with the linear row indices of the structural non-zeros, `offs` is a column-wise table of
offsets in these arrays, `rowsiz` and `colsiz` are the sizes of the row and column
dimensions. The values of the structural non-zeros of the `j`-th column and their
respective linear row indices are given by `vals[k]` and `rows[k]` with `k ∈
offs[j]+1:offs[j+1]`. The linear column index `j` is in the range `1:n` where `n =
prod(colsiz)` is the equivalent number of columns. For efficiency reasons, sparse
operators are currently limited to *fast* arrays because they can be indexed linearly with
no loss of performances. If `vals`, `rows` and/or `offs` are not fast arrays, they will be
automatically converted to linearly indexed arrays.

A sparse operator in CSC storage format can be directly constructed from a 2-dimensional
Julia array `A`:

    SparseOperatorCSC(A, f = (v,i,j) -> !iszero(v))

where optional argument `f` is a predicate function which is called as `f(v,i,j)` with
`v`, `i` and `j` the value, the row and the column linear indices for each entry of `A`
and which yields whether a given entry of `A` is a structural non-zeros. The default
predicate is such that all non-zeros of `A` are considered as being structural non-zeros.

The element type, say `T`, of the values of the structural non-zeros can be imposed by
rewriting the above examples as:

    SparseOperatorCSC{T}(args...)

A sparse operator in CSC storage format implementing generalized matrix-vector
multiplication can also be directly constructed from a `L`-dimensional Julia array (with
`L ≥ 2`) `A` by:

    SparseOperatorCSC{T,M}(A[, sel])

with `M` the number of leading dimensions of `A` corresponding to the *rows* of the
operator, the trailing `N = L - M` dimensions being assumed to correspond to the *columns*
of the operator. These dimensions are the size of, respectively, the output and the input
arrays when applying the operator. The parameter `N` may be specified (although it can be
automatically determined):

    SparseOperatorCSC{T,M,N}(A[, sel])

provided `M + N = ndims(A)` holds.

A last parameter `V` can be specified for the type of the vector to store the values of
the structural non-zeros:

    SparseOperatorCSC{T,M,N,V}(args...)

provided `V` implements standard linear indexing. The default is to take `V = Vector{T}`.
As a special case, you can choose a uniform boolean vector from the `StructuredArrays`
package to store the sparse coefficients:

    SparseOperatorCSC{T,M,N,UniformVector{Bool}}(args...)

to get a compressed sparse operator in CSC format whose values are an immutable uniform
vector of true values requiring no storage. This is useful to only store the sparse
structure of the operator, that is the indices in CSC format of the sparse coefficients
not their values.

The `SparseOperatorCSC` constructor can also be used to convert a sparse operator in
another storage format into the CSC format. In that case, parameter `T` may also be
specified to convert the type of the sparse coefficients.

""" SparseOperatorCSC

"""

Sparse operators in *Compressed Sparse Row* (CSR) format store their structural non-zeros
in a row-wise order, as a vector of values, a vector of corresponding linear column
indices and a vector of offsets indicating, for each row, the range of indices in the
vectors of values and of column indices. This storage format is very suitable for fast
application of the operator.

A sparse operator in CSR storage format can be constructed by providing all
necessary information:

    SparseOperatorCSR(vals, cols, offs, rowsiz, colsiz)

where `vals` is the vector of values of the structural non-zeros, `cols` is an integer
valued vector with the linear column indices of the structural non-zeros, `offs` is a
column-wise table of offsets in these arrays, `rowsiz` and `colsiz` are the sizes of the
row and column dimensions. The values of the structural non-zeros of the `i`-th row and
their respective linear column indices are given by `vals[k]` and `cols[k]` with `k ∈
offs[i]+1:offs[i+1]`. The linear row index `i` is in the range `1:m` where `m =
prod(rowsiz)` is the equivalent number of rows. For efficiency reasons, sparse operators
are currently limited to *fast* arrays because they can be indexed linearly with no loss
of performances. If `vals`, `cols` and/or `offs` are not fast arrays, they will be
automatically converted to linearly indexed arrays.

A sparse operator in CSR storage format can be directly constructed from a 2-dimensional
Julia array `A`:

    SparseOperatorCSR(A, f = (v,i,j) -> !iszero(v))

where optional argument `f` is a predicate function which is called as `f(v,i,j)` with
`v`, `i` and `j` the value, the row and the column linear indices for each entry of `A`
and which yields whether a given entry of `A` is a structural non-zeros. The default
predicate is such that all non-zeros of `A` are considered as being structural non-zeros.

The element type, say `T`, of the values of the structural non-zeros can be imposed by
rewriting the above examples as:

    SparseOperatorCSR{T}(args...)

A sparse operator in CSR storage format implementing generalized matrix-vector
multiplication can also be directly constructed from a `L`-dimensional Julia array (with
`L ≥ 2`) `A` by:

    SparseOperatorCSR{T,M}(A[, sel])

with `M` the number of leading dimensions of `A` corresponding to the *rows* of the
operator, the trailing `N = L - M` dimensions being assumed to correspond to the *columns*
of the operator. These dimensions are the size of, respectively, the output and the input
arrays when applying the operator. The parameter `N` may be specified (although it can be
automatically determined):

    SparseOperatorCSR{T,M,N}(A[, sel])

provided `M + N = ndims(A)` holds.

A last parameter `V` can be specified for the type of the vector to store the values of
the structural non-zeros:

    SparseOperatorCSR{T,M,N,V}(args...)

provided `V` implements standard linear indexing. The default is to take `V = Vector{T}`.
As a special case, you can choose a uniform boolean vector from the `StructuredArrays`
package to store the sparse coefficients:

    SparseOperatorCSR{T,M,N,UniformVector{Bool}}(args...)

to get a compressed sparse operator in CSR format whose values are an immutable uniform
vector of true values requiring no storage. This is useful to only store the sparse
structure of the operator, that is the indices in CSR format of the sparse coefficients
not their values.

The `SparseOperatorCSR` constructor can also be used to convert a sparse operator in
another storage format into the CSR format. In that case, parameter `T` may also be
specified to convert the type of the sparse coefficients.

""" SparseOperatorCSR

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
        SparseOperator{T,M,N}(A::$F{<:Any,M,N}) where {T,M,N} = $F{T,M,N}(A)
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

# Conversion without changing format (mostly for changing element type).
CompressedSparseOperator{Any}(A::CompressedSparseOperator) = A
CompressedSparseOperator{Any,T}(A::CompressedSparseOperator{F}) where {F,T} =
    CompressedSparseOperator{F,T}(A)
CompressedSparseOperator{Any,T,M}(A::CompressedSparseOperator{F}) where {F,T,M} =
    CompressedSparseOperator{F,T,M}(A)
CompressedSparseOperator{Any,T,M,N}(A::CompressedSparseOperator{F}) where {F,T,M,N} =
    CompressedSparseOperator{F,T,M,N}(A)

@inline isnonzero(v::T, i::Integer, j::Integer) where {T} = (v != zero(T))

# Many constructors have similar code whatever the compressed sparse storage format. We
# therefore use meta-programming to define them.
for (CS, other_args) in ((:SparseOperatorCSR, (:cols, :offs)),
                         (:SparseOperatorCSC, (:rows, :offs)),
                         (:SparseOperatorCOO, (:rows, :cols)))
    # All other arguments are integer-valued vectors.
    other_decl = map(s -> :($s::AbstractVector{<:Integer}), other_args)
    _CS = Symbol("_",CS)
    @eval begin
        # Get rid of the M,N parameters, but keep/set T for conversion of values.
        $CS{T,M,N}(A::SparseOperator{<:Any,M,N}) where {T,M,N} = $CS{T}(A)
        $CS{T,M}(A::SparseOperator{<:Any,M}) where {T,M} = $CS{T}(A)
        $CS(A::SparseOperator{T}) where {T} = $CS{T}(A)

        # Cases which do nothing (it makes sense that a constructor of an immutable type
        # be able to just return its argument if it is already of the correct type).
        $CS{T}(A::$CS{T}) where {T} = A

        # Manage to call constructors of compressed sparse operator given a regular Julia
        # array with correct parameters and selector.
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
            1 ≤ M < L || error("parameters M=$M and L=$L are not such that 1 ≤ M < L")
            $CS{T,M,L-M}(A, args...; kwds...)
        end
        $CS{T,M,N}(A::AbstractArray, args...; kwds...) where {T,M,N} =
            $CS{T,M,N,Vector{T}}(A, args...; kwds...)

        # Constructors that convert array of values. Other fields have already been
        # checked so do not check structure again.
        $CS{T}(A::$CS{S,M,N}) where {S,T,M,N} =
            $_CS(nrows(A), ncols(A), to_values(T, get_vals(A)),
                 $(map(s -> :($(Symbol("get_",s))(A)), other_args)...),
                 row_size(A), col_size(A))

        # Basic outer constructors return a fully checked structure.
        function $CS(vals::AbstractVector, $(other_decl...),
                     rowsiz::Tuple{Vararg{Integer}}, colsiz::Tuple{Vararg{Integer}})
            check_structure($_CS(to_values(vals),
                                 $(map(s -> :(to_indices($s)), other_args)...),
                                 to_size(rowsiz), to_size(colsiz)))
        end

        # Constructors for any compressed format similar to the basic ones but with type
        # parameters that may imply converting arguments.
        function $CS{T,M,N}(vals::AbstractVector, $(other_decl...),
                              rowsiz::Tuple{Vararg{Integer}},
                              colsiz::Tuple{Vararg{Integer}}) where {T,M,N}
            N isa Int || throw(AssertionError("type parameter `N` must be an `Int`"))
            length(colsiz) == N || throw(DimensionMismatch(
                "number of column dimensions is not equal to type parameter `N = $N`"))
            $CS{T,M}(vals, $(other_args...), rowsiz, colsiz)
        end
        function $CS{T,M}(vals::AbstractVector, $(other_decl...),
                            rowsiz::Tuple{Vararg{Integer}},
                            colsiz::Tuple{Vararg{Integer}}) where {T,M}
            M isa Int || throw(AssertionError("type parameter `M` must be an `Int`"))
            length(rowsiz) == M || throw(DimensionMismatch(
                "number of row dimensions is not equal to type parameter `M = $M`"))
            $CS{T}(vals, $(other_args...), rowsiz, colsiz)
        end
        function $CS{T}(vals::AbstractVector, $(other_decl...),
                          rowsiz::Tuple{Vararg{Integer}},
                          colsiz::Tuple{Vararg{Integer}}) where {T}
            M isa Int || throw(AssertionError("type parameter `M` must be an `Int`"))
            length(rowsiz) == M || throw(DimensionMismatch(
                "number of row dimensions must be equal to type parameter `M`"))
            $CS(to_values(T, vals), $(other_args...), rowsiz, colsiz)
        end
   end
end

# Constructors of a sparse operator in various format given a regular Julia array and a
# selector function. Julia arrays are usually in column-major order but this is not always
# the case, to handle various storage orders when extracting selected entries, we convert
# the input array into a equivalent "matrix", that is a 2-dimensional array.

function SparseOperatorCSR{T,M,N,V}(arr::AbstractArray{S,L},
                                    f::Function = isnonzero) where {
                                        S,T,L,M,N,V<:AbstractVector{T}}
    # Get equivalent matrix dimensions.
    nrows, ncols, rowsiz, colsiz = get_equivalent_size(arr, Val(M), Val(N))

    # Convert into equivalent matrix.
    A = as_matrix(arr, nrows, ncols)

    # Count the number of selected entries.
    nvals = count_selection(A, f)

    # Extract the selected entries and their column indices and count the number of
    # selected entries per row. The pseudo-matrix is walked in row-major order.
    cols = Vector{Int}(undef, nvals)
    offs = Vector{Int}(undef, nrows + 1)
    k = 0
    if V <: UniformVector{Bool}
        # Just extract the structure, not the values.
        vals = V(true, nvals)
        @inbounds for i in 1:nrows
            offs[i] = k
            for j in 1:ncols
                if f(A[i,j], i, j)
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
                if f(v, i, j)
                    (k += 1) ≤ nvals || bad_selector()
                    vals[k] = v
                    cols[k] = j
                end
            end
        end
    end
    k == nvals || bad_selector()
    offs[end] = nvals

    # By construction, the sparse structure should be correct so just call the "unsafe"
    # constructor.
    return _SparseOperatorCSR(nrows, ncols, vals, cols, offs, rowsiz, colsiz)
end

function SparseOperatorCSC{T,M,N,V}(arr::AbstractArray{S,L},
                                    f::Function = isnonzero) where {
                                        S,T,L,M,N,V<:AbstractVector{T}}
    # Get equivalent matrix dimensions.
    nrows, ncols, rowsiz, colsiz = get_equivalent_size(arr, Val(M), Val(N))

    # Convert into equivalent matrix.
    A = as_matrix(arr, nrows, ncols)

    # Count the number of selected entries.
    nvals = count_selection(A, f)

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
                if f(A[i,j], i, j)
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
                if f(v, i, j)
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
    return _SparseOperatorCSC(nrows, ncols, vals, rows, offs, rowsiz, colsiz)
end

function SparseOperatorCOO{T,M,N,V}(arr::AbstractArray{S,L},
                                    f::Function = isnonzero) where {
                                        S,T,L,M,N,V<:AbstractVector{T}}
    # Get equivalent matrix dimensions.
    nrows, ncols, rowsiz, colsiz = get_equivalent_size(arr, Val(M), Val(N))

    # Convert into equivalent matrix.
    A = as_matrix(arr, nrows, ncols)

    # Count the number of selected entries.
    nvals = count_selection(A, f)

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
                if f(A[i,j], i, j)
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
                if f(v, i, j)
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
    return _SparseOperatorCOO(nrows, ncols, vals, rows, cols, rowsiz, colsiz)
end

"""
    unpack!(A, S; flatten=false) -> A

unpacks the non-zero coefficients of the sparse operator `S` into the array `A`
and returns `A`.  Keyword `flatten` specifies whether to only consider the
length of `A` instead of its dimensions.  In any cases, `A` must have as many
elements as `length(S)` and standard linear indexing.

Just call `Array(S)` to unpack the coefficients of a sparse operator `S`
without providing the destination array.

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

function prepare_unpack!(dst::AbstractArray,
                         src::SparseOperator,
                         flatten::Bool)
    is_fast_array(dst) || throw_non_standard_indexing("destination array")
    if flatten
        length(dst) == length(src) ||
            throw_incompatible_number_of_elements()
    else
        size(dst) == (row_size(src)..., col_size(src)...,) ||
            throw_incompatible_dimensions()
    end
    fill!(dst, zero(eltype(dst)))
end

function unpack!(B::AbstractArray{T,L},
                 A::SparseOperatorCSR{<:Any,M,N};
                 flatten::Bool = false) where {T,L,M,N}
    prepare_unpack!(B, A, flatten)
    m = nrows(A) # used as the "stride" in B
    @inbounds for i in each_row(A)
        for k in each_nz(A, i)
            j = get_col(A, k)
            v = get_val(A, k)
            B[i + m*(j - 1)] = v
        end
    end
    return B
end

function unpack!(B::AbstractArray{T,L},
                 A::SparseOperatorCSC{<:Any,M,N};
                 flatten::Bool = false) where {T,L,M,N}
    prepare_unpack!(B, A, flatten)
    m = nrows(A) # used as the "stride" in B
    @inbounds for j in each_col(A)
        for k in each_nz(A, j)
            i = get_row(A, k)
            v = get_val(A, k)
            B[i + m*(j - 1)] = v
        end
    end
    return B
end

unpack!(B::AbstractArray, A::SparseOperatorCOO; kwds...) =
    unpack!(B, A, +; kwds...)

unpack!(B::AbstractArray, A::SparseOperatorCOO{Bool}; kwds...) =
    unpack!(B, A, |; kwds...)

function unpack!(B::AbstractArray{T,L},
                 A::SparseOperatorCOO{<:Any,M,N},
                 op::Function; flatten::Bool = false) where {T,L,M,N}
    prepare_unpack!(B, A, flatten)
    m = nrows(A) # used as the "stride" in B
    @inbounds for k in each_nz(A)
        i = get_row(A, k)
        j = get_col(A, k)
        v = get_val(A, k)
        l = i + m*(j - 1)
        B[l] = op(B[l], v)
    end
    return B
end

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
    _SparseOperatorCSR(nrows(A), ncols(A), get_vals(A), get_cols(A), get_offs(A),
                       rowsiz, colsiz)
end

function Base.reshape(A::SparseOperatorCSC,
                      rowsiz::Tuple{Vararg{Int}},
                      colsiz::Tuple{Vararg{Int}})
    check_new_shape(A, rowsiz, colsiz)
    _SparseOperatorCSC(nrows(A), ncols(A), get_vals(A), get_rows(A), get_offs(A),
                       rowsiz, colsiz)
end

function Base.reshape(A::SparseOperatorCOO,
                      rowsiz::Tuple{Vararg{Int}},
                      colsiz::Tuple{Vararg{Int}})
    check_new_shape(A, rowsiz, colsiz)
    _SparseOperatorCOO(nrows(A), ncols(A), get_vals(A), get_rows(A), get_cols(A),
                       rowsiz, colsiz)
end

# Convert from other compressed sparse formats. For compressed sparse row and column (CSR
# and CSC) formats, the compressed sparse coordinate (COO) format is used as an
# intermediate representation and entries are sorted in row/column major order. To avoid
# side-effects, they must be copied first. Unless values are converted, there is no needs
# to copy when converting to a compressed sparse coordinate (COO) format.

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
    coo_to_csr!(vals, rows, cols, rowsiz, colsiz [, op]) -> A

yields the a compressed sparse operator in a CSR format given the components `vals`,
`rows` and `cols` in the COO format and the sizes `rowsiz` and `colsiz` of the row and
column dimensions. Input arrays are modified in-place. Optional argument `op` is a binary
operator to reduce the values of entries having the same row and column indices.

Input arrays must be regular Julia vectors to ensure type stability in case of duplicates.

"""
function coo_to_csr!(vals::Vector{T},
                     rows::Vector{Int},
                     cols::Vector{Int},
                     rowsiz::Dims{M},
                     colsiz::Dims{N},
                     op::Function = (T <: Bool ? (|) : (+))) where {T,M,N}
    # Check row and column sizes.
    nrows = check_size(rowsiz, "row")
    ncols = check_size(colsiz, "column")

    # Check row and column indices.
    check_rows(rows, nrows)
    check_cols(cols, ncols)

    # Sort and reduce entries in row-major order, resize arrays used in the result if
    # needed, and compute offsets.
    nvals = sort_and_reduce!(vals, rows, cols, op)
    if nvals < length(vals)
        vals = vals[1:nvals]::Vector{T}
        cols = cols[1:nvals]::Vector{Int}
    end
    offs = sparse_compressed_offsets(nrows, view(rows, 1:nvals))

    # Since everything will have been checked, we can call the unsafe constructor.
    return _SparseOperatorCSR(nrows, ncols, vals, cols, offs, rowsiz, colsiz)
end

"""
    coo_to_csc!(vals, rows, cols, rowsiz, colsiz [, op]) -> A

yields the a compressed sparse operator in a CSC format given the components `vals`,
`rows` and `cols` in the COO format and the sizes `rowsiz` and `colsiz` of the row and
column dimensions. Input arrays are modified in-place. Optional argument `op` is a binary
operator to reduce the values of entries having the same row and column indices.

Input arrays must be regular Julia vectors to ensure type stability in case of duplicates.

"""
function coo_to_csc!(vals::Vector{T},
                     rows::Vector{Int},
                     cols::Vector{Int},
                     rowsiz::Dims{M},
                     colsiz::Dims{N},
                     op::Function = (T <: Bool ? (|) : (+))) where {T,M,N}
    # Check row and column sizes.
    nrows = check_size(rowsiz, "row")
    ncols = check_size(colsiz, "column")

    # Check row and column indices.
    check_rows(rows, nrows)
    check_cols(cols, ncols)

    # Sort and reduce entries in column-major order, resize arrays used in the result if
    # needed, and compute offsets.
    nvals = sort_and_reduce!(vals, cols, rows, op)
    if nvals < length(vals)
        vals = vals[1:nvals]::Vector{T}
        rows = rows[1:nvals]::Vector{Int}
    end
    offs = sparse_compressed_offsets(ncols, view(cols, 1:nvals))

    # Since everything will have been checked, we can call the unsafe constructor.
    return _SparseOperatorCSC(nrows, ncols, vals, rows, offs, rowsiz, colsiz)
end

"""
    sort_and_reduce!(vals, major, minor, op) -> nvals

sorts entries and reduces duplicates in input arrays `vals`, `major` and `minor`. Entries
consist in the 3-tuples `(vals[k],major[k],minor[k])`. The sorting order of the `k`-th
entry is based the value of `major[k]` and, if equal, on the value of `minor[k]`. After
sorting, duplicate entries, that is those which have the same minor and major indices, are
replaced by a single entry whose value is obtained by reducing the values in `vals` with
the binary operator `op`. All operations are done in-place, the number of unique entries
is returned but inputs arrays are not resized, only the `nvals` first entries are valid.

"""
function sort_and_reduce!(vals::AbstractVector,
                          major::AbstractVector{Int},
                          minor::AbstractVector{Int},
                          op::Function)
    # Check dimensions.
    axes(vals) == axes(major) == axes(minor) || throw(DimensionMismatch(
        "arguments must have the same axes"))
    isempty(vals) && return 0

    # Sort entries in order (this also ensures that all arrays have the same dimensions).
    sort!(ZippedArray(vals, major, minor);
          # In the provided "less-than" method, arguments are 3-tuples `(v,maj,min)` with
          # `v` the structural non-zero value, `maj` the major index, and `min` the minor
          # index. The order only depends on the two latter.
          lt = (x, y) -> (x[2] < y[2]) | ((x[2] == y[2]) & (x[3] < y[3])))

    # Reduce duplicates.
    r = eachindex(IndexLinear(), vals, major, minor)
    j = first(r)
    @inbounds for k in first(r)+1:last(r)
        if (major[k] == major[j]) & (minor[k] == minor[j])
            vals[j] = op(vals[j], vals[k]) # reduce value of duplicate entry
        elseif (j += 1) < k
            vals[j], major[j], minor[j] = vals[k], major[k], minor[k]
        end
    end
    return j - first(r) + 1
end

"""
    sparse_compressed_offsets(n, inds) -> offs

yields a vector of `n+1` offsets for sparse compressed storage and computed from the list
of indices `inds`. Indices in `inds` must be in non-increasing order and in the range
`1:n`.

"""
sparse_compressed_offsets(n::Int, inds::AbstractVector{Int}) =
    sparse_compressed_offsets!(Vector{Int}(undef, n + 1), inds)

function sparse_compressed_offsets!(offs::AbstractVector{Int}, inds::AbstractVector{Int})
    firstindex(offs) == 1 || throw(AssertionError(
        "vector of offsets must have 1-based indices"))
    n = length(offs) - 1
    k1 = firstindex(inds)
    k2 = lastindex(inds)
    i = 0
    @inbounds for k in k1:k2
        j = inds[k]
        if 1 ≤ i == j
            nothing
        elseif i < j ≤ n
            off = k - k1
            while i < j
                i += 1
                offs[i] = off
            end
        else
            throw(AssertionError(1 ≤ j ≤ n ? "indices must be in non-increasing order" :
                "out of bound indices"))
        end
    end
    @inbounds while i ≤ n
        i += 1
        offs[i] = length(inds)
    end
    return offs
end

# This error is due to the non-zeros selector not returning the same results in
# the two selection passes.
bad_selector() = throw_argument_error("inconsistent selector function")

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

yields equivalent matrix dimensions of array `A` assuming the *rows* account for the `M`
leading dimensions while the *columns* account for the other `N` dimensions.

"""
function get_equivalent_size(A::AbstractArray{T,L},
                             ::Val{M}, ::Val{N}) where {T,L,M,N}
    @assert L == M + N
    @assert M ≥ 1
    @assert N ≥ 1
    eachindex(A) == 1:length(A) ||
        throw_argument_error("array must have standard linear indexing")
    siz = size(A)
    rowsiz = siz[1:M]
    colsiz = siz[M+1:end]
    nrows = prod(rowsiz)
    ncols = prod(colsiz)
    return nrows, ncols, rowsiz, colsiz
end

"""
    count_selection(A, f, rowmajor=false) -> nvals

yields the number of selected entries in matrix `A` such that `f(A[i,j],i,j)` is `true`
and with `i` and `j` the row and column indices. if optinal argument `rowmajor` is true,
the array is walked in row-major order; otherwise (the default), the array is walked in
column-major order.

"""
function count_selection(A::AbstractMatrix, f::Function,
                         rowmajor::Bool = false)
    nrows, ncols = size(A)
    nvals = 0
    if rowmajor
        # Walk the coefficients in row-major order.
        @inbounds for i in 1:nrows, j in 1:ncols
            if f(A[i,j], i, j)
                nvals += 1
            end
        end
    else
        # Walk the coefficients in column-major order.
        @inbounds for j in 1:ncols, i in 1:nrows
            if f(A[i,j], i, j)
                nvals += 1
            end
        end
    end
    return nvals
end

"""
    check_structure(A) -> A

checks the structure of the compressed sparse operator `A` throwing an exception if there
are any inconsistencies.

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

checks the array size `siz` and returns the corresponding number of elements. An
`ArgumentError` is thrown if a dimension is invalid, using `id` to identify the argument.

    check_size(A)

checks the validity of the row and column sizes in compressed sparse operator `A` throwing
an exception if there are any inconsistencies.

"""
function check_size(siz::Dims{N}, id::String="array") where {N}
    len = 1
    @inbounds for i in 1:N
        (dim = siz[i]) ≥ 0 || bad_dimension(dim, i, id)
        len *= dim
    end
    return len
end

function check_size(A::SparseOperator)
    check_size(row_size(A), "row") == nrows(A) ||
        throw_dimension_mismatch("incompatible equivalent number of rows and row size")
    check_size(col_size(A), "column") == ncols(A) ||
        throw_dimension_mismatch("incompatible equivalent number of columns and column size")
    nothing
end

@noinline bad_dimension(dim::Integer, i::Integer, id) =
    throw_argument_error("invalid ", i, ordinal_suffix(i), " ", id,
                         " dimension: ", dim)

"""
    check_vals(A)

checks the array of values in compressed sparse operator `A` throwing an exception if
there are any inconsistencies.

"""
function check_vals(A::SparseOperator)
    vals = get_vals(A)
    is_fast_array(vals) || throw_not_fast_array("array of values")
    length(vals) == nnz(A) || throw_argument_error("bad number of values")
    nothing
end

"""
    check_rows(A)

checks the array of linear row indices in the compressed sparse operator `A` stored in a
*Compressed Sparse Column* (CSC) or *Compressed Sparse Coordinate* (COO) format. Throws an
exception in case of inconsistency.

    check_rows(rows, m)

check the array of linear row indices `rows` for being a fast vector of values in the
range `1:m`.

"""
function check_rows(A::Union{<:CompressedSparseOperator{:CSC},
                             <:CompressedSparseOperator{:COO}})
    rows = get_rows(A)
    length(rows) == nnz(A) ||
        throw_argument_error("bad number of row indices")
    check_rows(rows, nrows(A))
    # FIXME: also check sorting for CompressedSparseOperator{:CSC}?
end

function check_rows(rows::AbstractVector{Int}, m::Int)
    is_fast_array(rows) || throw_not_fast_array("array of row indices")
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

checks the array of linear column indices in the compressed sparse operator `A` stored in
a *Compressed Sparse Row* (CSR) or *Compressed Sparse Coordinate* (COO) format. Throws an
exception in case of inconsistency.

    check_cols(cols, n)

check the array of linear column indices `cols` for being a fast vector of values in the
range `1:n`.

"""
function check_cols(A::Union{<:CompressedSparseOperator{:CSR},
                             <:CompressedSparseOperator{:COO}})
    cols = get_cols(A)
    length(cols) == nnz(A) ||
        throw_argument_error("bad number of column indices")
    check_cols(cols, ncols(A))
    # FIXME: also check sorting for CompressedSparseOperator{:CSR}?
end

function check_cols(cols::AbstractVector{Int}, n::Int)
    is_fast_array(cols) || throw_not_fast_array("array of column indices")
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

checks the array of offsets in the compressed sparse operator `A` stored in a *Compressed
Sparse Row* (CSR) or *Compressed Sparse Column* (CSC) format. Throws an exception in case
of inconsistency.

"""
function check_offs(A::T) where {T<:Union{CompressedSparseOperator{:CSR},
                                          CompressedSparseOperator{:CSC}}}
    offs = get_offs(A)
    is_fast_array(offs) || throw_not_fast_array("array of offsets")
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
    _SparseOperatorCSR([m, n,] vals, cols, offs, rowsiz, colsiz)

yields a compressed sparse operator in *Compressed Sparse Row* (CSR) format as an instance
of `SparseOperatorCSR`. This private constructor assumes that arguments are correct and is
mostly used by converters and other constructors.

"""
function _SparseOperatorCSR(vals::AbstractVector,
                    cols::AbstractVector{Int},
                    offs::AbstractVector{Int},
                    rowsiz::Dims{M},
                    colsiz::Dims{N}) where {M,N}
    _SparseOperatorCSR(prod(rowsiz), prod(colsiz), vals, cols, offs, rowsiz, colsiz)
end

"""
    _SparseOperatorCSC([m, n,] vals, rows, offs, rowsiz, colsiz)

yields a compressed sparse operator in *Compressed Sparse Column* (CSC) format as an
instance of `SparseOperatorCSC`. This private constructor assumes that arguments are
correct and is mostly used by converters and other constructors.

"""
function _SparseOperatorCSC(vals::AbstractVector,
                            rows::AbstractVector{Int},
                            offs::AbstractVector{Int},
                            rowsiz::Dims{M},
                            colsiz::Dims{N}) where {M,N}
    _SparseOperatorCSC(prod(rowsiz), prod(colsiz), vals, rows, offs, rowsiz, colsiz)
end

"""
    _SparseOperatorCOO([m, n,] vals, rows, cols, rowsiz, colsiz)

yields a compressed sparse operator in *Compressed Sparse Coordinate* (COO) format as an
instance of `SparseOperatorCOO`. This private constructor assumes that arguments are
correct and is mostly used by converters and other constructors.

"""
function _SparseOperatorCOO(vals::AbstractVector,
                            rows::AbstractVector{Int},
                            cols::AbstractVector{Int},
                            rowsiz::Dims{M},
                            colsiz::Dims{N}) where {M,N}
    _SparseOperatorCOO(prod(rowsiz), prod(colsiz), vals, rows, cols, rowsiz, colsiz)
end

"""
    is_fast_array(A) -> bool

yields whether `A` is a *fast array* that is an array with standard linear indexing.

"""
is_fast_array(A::AbstractArray) = is_fast_indices(eachindex(A))

"""
    is_fast_indices(inds) -> bool

yields whether `inds` is an iterator for *fast indices* that is linear indices starting at
`1`.

"""
is_fast_indices(inds::AbstractUnitRange{Int}) = (first(inds) == 1)
is_fast_indices(inds) = false

"""
    check_argument(A, siz, id="array")

checks whether array `A` has size `siz` and implements standard linear indexing. An
exception is thrown if any of these do not hold.

"""
function check_argument(A::AbstractArray{<:Any,N},
                        siz::Dims{N},
                        id="array") where {N}
    IndexStyle(A) === IndexLinear() || throw_non_linear_indexing(id)
    inds = axes(A)
    @inbounds for i in 1:N
        first(inds[i]) == 1 || throw_non_standard_indexing(id)
        length(inds[i]) == siz[i] || throw_incompatible_dimensions(id)
    end
    nothing
end

"""
    throw_argument_error(args...)

throws an `ArgumentError` exception with a textual message made of `args...`.

"""
throw_argument_error(mesg::AbstractString) = throw(ArgumentError(mesg))
@noinline throw_argument_error(args...) = throw_argument_error(string(args...))

@noinline throw_not_fast_array(id) =
    throw_argument_error(id, " does not implement fast indexing")

@noinline throw_non_linear_indexing(id) =
    throw_argument_error(id, " does not implement linear indexing")

@noinline throw_non_standard_indexing(id) =
    throw_argument_error(id, " has non-standard indexing")

"""
    throw_assertion_error(args...)

throws an `AssertionError` exception with a textual message made of `args...`.

"""
throw_assertion_error(mesg::AbstractString) = throw(AssertionError(mesg))
@noinline throw_assertion_error(args...) = throw_assertion_error(string(args...))

"""
    throw_dimension_mismatch(args...)

throws a `DimensionMismatch` exception with a textual message made of `args...`.

"""
throw_dimension_mismatch(mesg::AbstractString) =
    throw(DimensionMismatch(mesg))

@noinline throw_dimension_mismatch(args...) =
    throw_dimension_mismatch(string(args...))

@noinline throw_incompatible_dimensions(id) =
    throw_dimension_mismatch(id, " has incompatible dimensions")

@noinline throw_incompatible_dimensions() =
    throw_dimension_mismatch("incompatible dimensions")

@noinline throw_incompatible_number_of_dimensions() =
    throw_dimension_mismatch("incompatible number of dimensions")

@noinline throw_incompatible_number_of_elements() =
    throw_dimension_mismatch("incompatible number of elements")

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

#-----------------------------------------------------------------------------------------
# Apply operators.

# Directly extend the `dispatch_vmul!` method for sparse operators in compressed sparse
# row format.
function dispatch_vmul!(α::Number,
                        A::AnyCSR{Ta,M,N},
                        x::AbstractArray{Tx,N},
                        β::Number,
                        y::AbstractArray{Ty,M}) where {Ta,Tx,Ty,M,N}
    # FIXME: check_argument(x, col_size(A))
    # FIXME: check_argument(y, row_size(A))
    if isone(α)
        if iszero(β)
            unsafe_vmul!(axpby_yields_x,     α, A, x, β, y)
        elseif isone(β)
            unsafe_vmul!(axpby_yields_xpy,   α, A, x, β, y)
        else
            unsafe_vmul!(axpby_yields_xpby,  α, A, x, β, y)
        end
    elseif !iszero(α)
        if iszero(β)
            unsafe_vmul!(axpby_yields_ax,    α, A, x, β, y)
        elseif isone(β)
            unsafe_vmul!(axpby_yields_axpy,  α, A, x, β, y)
        else
            unsafe_vmul!(axpby_yields_axpby, α, A, x, β, y)
        end
    else
        dispatch_vscale!(y, β)
    end
    return y
end

# FIXME: Unify API so that the codes of the two following methods are
#        identical (just the type of A change).

function unsafe_vmul!(f::Function,
                      α::Number,
                      A::CompressedSparseOperator{:CSR,Ta,M,N},
                      x::AbstractArray{Tx,N},
                      β::Number,
                      y::AbstractArray{Ty,M}) where {Ta,Tx,Ty,M,N}
    T = sumprod_type(Ta, Tx)
    @inbounds for i in each_row(A)
        s = zero(T)
        for k in each_nz(A, i)
            j = get_col(A, k)
            v = get_val(A, k)
            s += v*x[j]
        end
        y[i] = f(α, s, β, y[i])
    end
    nothing
end

function unsafe_vmul!(f::Function,
                      α::Number,
                      A′::Adjoint{<:CompressedSparseOperator{:CSC,Ta,M,N}},
                      x::AbstractArray{Tx,M},
                      β::Number,
                      y::AbstractArray{Ty,N}) where {Ta,Tx,Ty,M,N}
    A = parent(A′)
    T = sumprod_type(Ta, Tx)
    @inbounds for j in each_col(A)
        s = zero(T)
        for k in each_nz(A, j)
            i = get_row(A, k)
            v = get_val(A, k)
            s += conj(v)*x[i]
        end
        y[j] = f(α, s, β, y[j])
    end
    nothing
end

function unsafe_vmul!(α::Number,
                      A′::Adjoint{<:CompressedSparseOperator{:CSR,Ta,M,N}},
                      x::AbstractArray{Tx,M},
                      β::Number,
                      y::AbstractArray{Ty,N}) where {Ta,Tx,Ty,M,N}
    # Assumptions: (1) all sizes have been checked, (ii) α and β have been converted to a
    # suitable precision, and (iii) α is not zero.
    A = parent(A′)
    # FIXME check_argument(x, row_size(A))
    # FIXME check_argument(y, col_size(A))
    isone(β) || dispatch_vscale!(y, β)
    if isone(α)
        T = real_type(α) # NOTE α has the precision of α*A[i,j]*x[i]
        @inbounds for i in each_row(A)
            q = set_precision(T, x[i])
            if !iszero(q)
                for k in each_nz(A, i)
                    j = get_col(A, k)
                    v = get_val(A, k)
                    y[j] += q*conj(v)
                end
            end
        end
    else
        @inbounds for i in each_row(A)
            q = α*x[i]
            if !iszero(q)
                for k in each_nz(A, i)
                    j = get_col(A, k)
                    v = get_val(A, k)
                    y[j] += q*conj(v)
                end
            end
        end
    end
    return y
end

# Apply a sparse operator, and its adjoint, stored in Compressed Sparse Column (CSC)
# format.

function unsafe_vmul!(α::Number,
                      A::CompressedSparseOperator{:CSC,Ta,M,N},
                      x::AbstractArray{Tx,N},
                      β::Number,
                      y::AbstractArray{Ty,M}) where {Ta,Tx,Ty,M,N}
    # Assumptions: (1) all sizes have been checked, (ii) α and β have been converted to a
    # suitable precision, and (iii) α is not zero.
    # FIXME check_argument(x, col_size(A))
    # FIXME check_argument(y, row_size(A))
    isone(β) || dispatch_vscale!(y, β)
    if isnone(α)
        T = real_type(α) # NOTE α has the precision of α*A[i,j]*x[i]
        @inbounds for j in each_col(A)
            q = set_precision(T, x[j])
            if !iszero(q)
                for k in each_nz(A, j)
                    i = get_row(A, k)
                    v = get_val(A, k)
                    y[i] += q*v
                end
            end
        end
    else
        @inbounds for j in each_col(A)
            q = α*x[j]
            if !iszero(q)
                for k in each_nz(A, j)
                    i = get_row(A, k)
                    v = get_val(A, k)
                    y[i] += q*v
                end
            end
        end
    end
    return y
end

# Apply a sparse operator, and its adjoint, stored in Compressed Sparse
# Coordinate (COO) format.

function unsafe_vmul!(α::Number,
                      A::CompressedSparseOperator{:COO,Ta,M,N},
                      x::AbstractArray{Tx,N},
                      β::Number,
                      y::AbstractArray{Ty,M}) where {Ta,Tx,Ty,M,N}
    # Assumptions: (1) all sizes have been checked, (ii) α and β have been converted to a
    # suitable precision, and (iii) α is not zero.
    # FIXME check_argument(x, col_size(A))
    # FIXME check_argument(y, row_size(A))
    isone(β) || dispatch_vscale!(y, β)
    V, I, J = get_vals(A), get_rows(A), get_cols(A)
    if isone(α)
        @inbounds for k in eachindex(V, I, J)
            v, i, j = V[k], I[k], J[k]
            y[i] += x[j]*v
        end
    elseif α == -one(α)
        @inbounds for k in eachindex(V, I, J)
            v, i, j = V[k], I[k], J[k]
            y[i] -= x[j]*v
        end
    else
        @inbounds for k in eachindex(V, I, J)
            v, i, j = V[k], I[k], J[k]
            y[i] += α*x[j]*v
        end
    end
    return y
end

function unsafe_vmul!(α::Number,
                      A′::Adjoint{<:CompressedSparseOperator{:COO,Ta,M,N}},
                      x::AbstractArray{Tx,M},
                      β::Number,
                      y::AbstractArray{Ty,N}) where {Ta,Tx,Ty,M,N}
    # Assumptions: (1) all sizes have been checked, (ii) α and β have been converted
    # to a suitable precision, and (iii) α is not zero.
    # FIXME: check_argument(x, row_size(A))
    # FIXME: check_argument(y, col_size(A))
    isone(β) || dispatch_vscale!(y, β)
    A = parent(A′) # FIXME use generic API
    V, I, J = get_vals(A), get_rows(A), get_cols(A)
    if isone(α)
        @inbounds for k in eachindex(V, I, J)
            v, i, j = V[k], I[k], J[k]
            y[j] += x[i]*conj(v)
        end
    elseif α == -one(α)
        @inbounds for k in eachindex(V, I, J)
            v, i, j = V[k], I[k], J[k]
            y[j] -= x[i]*conj(v)
        end
    else
        @inbounds for k in eachindex(V, I, J)
            v, i, j = V[k], I[k], J[k]
            y[j] += α*x[i]*conj(v)
        end
    end
    return y
end

end # module SparseOperators

# The following module is to facilitate using compressed sparse operators at a
# lower level than the exported API.
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
    each_nz,
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
    each_nz,
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
