# `sparse.jl` implements various format of compressed sparse linear operators, an API to
# deal with sparse operators, and methods to apply sparse operators and convert between
# different sparse formats. This goes beyond Julia's `SparseArrays` standard package which
# only provides "Compressed Sparse Column" (CSC) format.
#
# See https://en.wikipedia.org/wiki/Sparse_matrix.

#-----------------------------------------------------------------------------------------
# Convert to vector of indices.
to_indices(inds::AbstractVector{<:Integer}) = convert_eltype(Int, inds)

# Convert to vector of values with given element type and make sure it is a fast vector.
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

# Union of types acceptable to define array size.
const ArraySize = Union{Integer,Tuple{Vararg{Integer}}}

#-----------------------------------------------------------------------------------------

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

A simple (but slow for CSR and CSC strage formats) method to loop over the coordinates and
values of the structural non-zero of a sparse operator `A` is to write:

```julia
using LazyAlgebra: row_indices, col_indices
for (i,j,Aij) in zip(row_indices(A), col_indices(A), nonzeros(A))
    ...
end
```

Except for COO storage format, It is however more efficient to access the structural
non-zeros according to their storage order which depends on the compressed format.

- If `A` is in CSC format or is the adjoint of a sparse operator in CSR format:

  ```julia
  using LazyAlgebra: each_nz_index, each_col_index, row_index
  for j in each_col_index(A)       # loop over column index
      for k in each_nz_index(A, j) # loop over structural non-zeros in this column
          i = row_index(A, k)      # get row index of entry
          Aij = A[k]               # get value of entry
          A[k] = ...               # set value of entry
       end
  end
  ```

- If `A` is in CSR format or is the adjoint of a sparse operator in CSC format:

  ```julia
  using LazyAlgebra: each_nz_index, each_row_index, col_index
  for i in each_row_index(A)       # loop over row index
      for k in each_nz_index(A, i) # loop over structural non-zeros in this row
          j = col_index(A, k)      # get column index of entry
          Aij = A[k]               # get value of entry
          A[k] = ...               # set value of entry
       end
  end
  ```

- If `A` is in COO format:

  ```julia
  using LazyAlgebra: each_nz_index, row_index, col_index
  for k in each_nz_index(A) # loop over index of structural non-zeros
       i = row_index(A, k)  # get row index of entry
       j = col_index(A, k)  # get column index of entry
       Aij = A[k]           # get value of entry
       A[k] = ...           # set value of entry
  end
  ```

The low-level methods `each_row_index`, `each_col_index`, `each_nz_index`, `row_index`,
and `col_index` are public but not automatically exported by `LazyAlgebra`.

""" CompressedSparseOperator

# Unions of compressed sparse operators that can be considered as being in a given storage
# format. Whatever the format, `T` is the element type, `M` is the number of output
# dimensions, and `N` is the number of input dimensions.

const AnySparseCSR{T,M,N} = Union{CompressedSparseOperator{:CSR,T,M,N},
                                  Swapped{<:CompressedSparseOperator{:CSC,T,N,M}}}

const AnySparseCSC{T,M,N} = Union{CompressedSparseOperator{:CSC,T,M,N},
                                  Swapped{<:CompressedSparseOperator{:CSR,T,N,M}}}

const AnySparseCOO{T,M,N} = Union{CompressedSparseOperator{:COO,T,M,N},
                                  Swapped{<:CompressedSparseOperator{:COO,T,N,M}}}

#-----------------------------------------------------------------------------------------
# Accessors and basic methods.

Base.eltype(::Type{<:SparseOperator{T,M,N}}) where {T,M,N} = T
InputShape(::Type{<:SparseOperator{T,M,N}}) where {T,M,N} = HasInputShape{N}()
OutputShape(::Type{<:SparseOperator{T,M,N}}) where {T,M,N} = HasOutputShape{M}()

nrows(A::SparseOperator) = getfield(A, :m)
ncols(A::SparseOperator) = getfield(A, :n)
row_size(A::SparseOperator) = getfield(A, :rowsiz) # alias to output_size
col_size(A::SparseOperator) = getfield(A, :colsiz) # alias to input_size
row_axes(A::SparseOperator) = map(Base.OneTo, row_size(A)) # alias to output_axes
col_axes(A::SparseOperator) = map(Base.OneTo, col_size(A)) # alias to input_axes

# Use constructors to perform conversion (the first method is to resolve ambiguities).
Base.convert(::Type{T}, A::T) where {T<:SparseOperator} = A
Base.convert(::Type{T}, A) where {T<:SparseOperator} = T(A)

for f in (:(==), :isequal)
    @eval begin
        function Base.$f(A::SparseOperatorCSR{<:Any,M,N},
                         B::SparseOperatorCSR{<:Any,M,N}) where {M,N}
            A === B || (A.m == B.m && A.n == B.n &&
                A.rowsiz == B.rowsiz && A.colsiz == B.colsiz &&
                A.cols == B.cols && A.offs == B.offs && $f(A.vals, B.vals))
        end
        function Base.$f(A::SparseOperatorCSC{<:Any,M,N},
                         B::SparseOperatorCSC{<:Any,M,N}) where {M,N}
            A === B || (A.m == B.m && A.n == B.n &&
                A.rowsiz == B.rowsiz && A.colsiz == B.colsiz &&
                A.rows == B.rows && A.offs == B.offs && $f(A.vals, B.vals))
        end
        function Base.$f(A::SparseOperatorCOO{<:Any,M,N},
                         B::SparseOperatorCOO{<:Any,M,N}) where {M,N}
            A === B || (A.m == B.m && A.n == B.n &&
                A.rowsiz == B.rowsiz && A.colsiz == B.colsiz &&
                A.rows == B.rows && A.cols == B.cols && $f(A.vals, B.vals))
        end
    end
end

TypeUtils.get_precision(::Type{A}) where {A<:SparseOperator} = get_precision(eltype(A))
TypeUtils.adapt_precision(::Type{T}, A::SparseOperator) where {T<:TypeUtils.Precision} =
    convert_eltype(adapt_precision(T, eltype(A)), A)

for (type, (getfield1, getfield2)) in (:SparseOperatorCSR => (:col_indices, :offsets),
                                       :SparseOperatorCSC => (:row_indices, :offsets),
                                       :SparseOperatorCOO => (:row_indices, :col_indices))
    _type = Symbol("_",type)
    @eval begin
        TypeUtils.convert_eltype(::Type{T}, A::$type{T}) where {T} = A
        TypeUtils.convert_eltype(::Type{T}, A::$type{S}) where {T,S} =
            $_type(nrows(A), ncols(A), convert_eltype(T, nonzeros(A)),
                   $getfield1(A), $getfield2(A), row_size(A), col_size(A))
    end
end

# Assume that a `copy` of a compressed sparse operator is to keep the same structure for
# the structural non-zeros but possibly change the values. So only duplicate the value
# part. For a `deepcopy` of a compressed sparse operator, all the fields are copied.

Base.copy(A::SparseOperatorCSR) = _SparseOperatorCSR(
    nrows(A), ncols(A), copy(nonzeros(A)), col_indices(A), offsets(A),
    row_size(A), col_size(A))

Base.copy(A::SparseOperatorCSC) = _SparseOperatorCSC(
    nrows(A), ncols(A), copy(nonzeros(A)), row_indices(A), offsets(A),
    row_size(A), col_size(A))

Base.copy(A::SparseOperatorCOO) = _SparseOperatorCOO(
    nrows(A), ncols(A), copy(nonzeros(A)), row_indices(A), col_indices(A),
    row_size(A), col_size(A))

Base.deepcopy(A::SparseOperatorCSR) = _SparseOperatorCSR(
    nrows(A), ncols(A), copy(nonzeros(A)), copy(col_indices(A)), copy(offsets(A)),
    row_size(A), col_size(A))

Base.deepcopy(A::SparseOperatorCSC) = _SparseOperatorCSC(
    nrows(A), ncols(A), copy(nonzeros(A)), copy(row_indices(A)), copy(offsets(A)),
    row_size(A), col_size(A))

Base.deepcopy(A::SparseOperatorCOO) = _SparseOperatorCOO(
    nrows(A), ncols(A), copy(nonzeros(A)), copy(row_indices(A)), copy(col_indices(A)),
    row_size(A), col_size(A))

# `findnz(A) -> I,J,V` yields the row and column indices and the values of the stored
# values in `A`.
SparseArrays.findnz(A::SparseOperator) = (row_indices(A), col_indices(A), nonzeros(A))

# Extend some methods in SparseArrays. The "structural" non-zeros are the entries stored
# by the sparse structure which may or not be equal to zero, un-stored entries are always
# considered as being strictly equal to zero.
SparseArrays.nnz(A::SparseOperator) = length(nonzeros(A))
SparseArrays.nnz(A::Swapped{<:SparseOperator}) = length(nonzeros(parent(A)))

"""
    nonzeros(A::LazyAlgebra.SparseOperator)

Return the array storing the structural non-zeros of the compressed sparse operator `A`.

The returned array is shared with `A`, call `copy(nonzeros(A))` or `collect(nonzeros(A))`
instead if you want to modify the contents of the returned array with no side effects on
`A`.

"""
SparseArrays.nonzeros(A::SparseOperator) = getfield(A, :vals)
SparseArrays.nonzeros(A::Transpose{<:SparseOperator}) = nonzeros(parent(A))
SparseArrays.nonzeros(A::Adjoint{<:SparseOperator}) =
    lazymap(eltype(A), conj, nonzeros(parent(A)))

"""
    LazyAlgebra.row_indices(A) -> I

Return the row indices of the structural non-zeros of the sparse operator `A`.

If `A` is a sparse operator in CSC or COO storage format, the result `I` is a vector of
indices shared with `A`; otherwise, `I` is an iterator. In any case, the caller shall not
attempt to modify the contents of `I`. Call `collect(row_indices(A))` to get a vector of
row indices that can be modified with no side effects on `A`.

"""
row_indices(A::Union{SparseOperatorCSC,SparseOperatorCOO}) = getfield(A, :rows)
row_indices(A::CompressedSparseOperator{:CSR}) = SparseIndexIterator(A)
row_indices(A::Swapped{<:SparseOperator}) = col_indices(parent(A))

"""
    LazyAlgebra.col_indices(A) -> J

Return the column indices of the structural non-zeros of the sparse operator `A`.

If `A` is a sparse operator in CSR or COO storage format, the result `J` is a vector of
indices shared with `A`; otherwise, `J` is an iterator. In any case, the caller shall not
attempt to modify the contents of `J`. Call `collect(col_indices(A))` to get a vector of
column indices that can be modified with no side effects on `A`.

"""
col_indices(A::Union{SparseOperatorCSR,SparseOperatorCOO}) = getfield(A, :cols)
col_indices(A::Union{CompressedSparseOperator{:CSC},SparseMatrixCSC}) =
    SparseIndexIterator(A) # FIXME: check whether this works SparseMatrixCSC
col_indices(A::Swapped{<:SparseOperator}) = row_indices(parent(A))

"""
    LazyAlgebra.offsets(A)

Return the table of offsets of the sparse operator `A`. Not all operators extend this
method.

!!! warning
    The interpretation of offsets depend on the type of `A`. For instance, assuming `offs
    = LazyAlgebra.offsets(A)`, then the index range of the `j`-th column of a
    `SparseMatrixCSC` is `offs[j]:(offs[j+1]-1)` while the index range is
    `(offs[j]+1):offs[j+1]` for a `SparseOperatorCSC`. For this reason, it is recommended
    to call [`each_nz_index`](@ref) instead or to call `offsets` with 2 arguments: `A`
    and, depending on the compressed storage format, the row or column index.

"""
offsets(A::Union{SparseOperatorCSR,SparseOperatorCSC}) = getfield(A, :offs)
offsets(A::Swapped{<:CompressedSparseOperator{:CSR}}) = offsets(parent(A))
offsets(A::Swapped{<:CompressedSparseOperator{:CSC}}) = offsets(parent(A))

"""
    LazyAlgebra.each_nz_index(A)

Return an iterator over the indices of the structural non-zeros of the sparse operator `A`
stored in a *Compressed Sparse Coordinate* (COO) format.

---
    LazyAlgebra.each_nz_index(A, j)

Return an iterator over the indices of the structural non-zeros of the `j`-th column of
the sparse operator `A` stored in a *Compressed Sparse Column* (CSC) format.

---
    LazyAlgebra.each_nz_index(A, i)

Return an iterator over the indices of the structural non-zeros of the `i`-th row of the
sparse operator `A` stored in a *Compressed Sparse Row* (CSR) format.

"""
@inline each_nz_index(A::Union{SparseOperatorCOO,Swapped{<:SparseOperatorCOO}}) = 𝟙:nnz(A)

@inline function each_nz_index(A::Union{AnySparseCSR,AnySparseCSC}, ij::Int)
    @boundscheck check_offset_index(A, ij)
    return unsafe_each_nz(A, ij)
end

@inline unsafe_each_nz(A::Union{AnySparseCSR,AnySparseCSC}, ij::Int) =
    UnitRange(unsafe_first_nz_index(A, ij), unsafe_last_nz_index(A, ij))

"""
    LazyAlgebra.first_nz_index(A)

Return the index of the first structural non-zero of the sparse operator `A` stored in a
*Compressed Sparse Coordinate* (COO) format.

---
    LazyAlgebra.first_nz_index(A, j)

Return the index of the first structural non-zero of the `j`-th column of the sparse
operator `A` stored in a *Compressed Sparse Column* (CSC) format.

---
    LazyAlgebra.first_nz_index(A, i)

Return the index of the first structural non-zero of the `i`-th row of the sparse operator
`A` stored in a *Compressed Sparse Row* (CSR) format.

"""
@inline first_nz_index(A::Union{SparseOperatorCOO,Swapped{<:SparseOperatorCOO}}) = 1

@inline function first_nz_index(A::Union{AnySparseCSR,AnySparseCSC}, ij::Int)
    @boundscheck check_offset_index(A, ij)
    return unsafe_first_nz_index(A, ij)
end

"""
    LazyAlgebra.last_nz_index(A)

Return the index of the last structural non-zero of the sparse operator `A` stored in a
*Compressed Sparse Coordinate* (COO) format.

---
    LazyAlgebra.last_nz_index(A, j)

Return the index of the last structural non-zero of the `j`-th column of the sparse
operator `A` stored in a *Compressed Sparse Column* (CSC) format.

---
    LazyAlgebra.last_nz_index(A, i)

Return the index of the last structural non-zero of the `i`-th row of the sparse operator
`A` stored in a *Compressed Sparse Row* (CSR) format.

"""
@inline last_nz_index(A::Union{SparseOperatorCOO,Swapped{<:SparseOperatorCOO}}) = nnz(A)

@inline function last_nz_index(A::Union{AnySparseCSR,AnySparseCSC}, ij::Int)
    @boundscheck check_offset_index(A, ij)
    return unsafe_last_nz_index(A, ij)
end

@inline unsafe_first_nz_index(A::Union{AnySparseCSR,AnySparseCSC}, ij::Int) =
    @inbounds offsets(A)[ij] + 1

@inline unsafe_last_nz_index(A::Union{AnySparseCSR,AnySparseCSC}, ij::Int) =
    @inbounds offsets(A)[ij + 1]

@inline check_offset_index(::Type{Bool}, A::Union{AnySparseCSR,AnySparseCSC}, k::Int) =
    1 ≤ k < length(offsets(A))

@inline check_offset_index(A::AnySparseCSR, i::Int) =
    check_offset_index(Bool, A, i) ? nothing : out_of_range_row_index(A, i)

@inline check_offset_index(A::Union{AnySparseCSC,SparseMatrixCSC}, j::Int) =
    check_offset_index(Bool, A, j) ? nothing : out_of_range_column_index(A, j)

@noinline out_of_range_row_index(A, i::Integer) =
    throw(ErrorException(string("out of range row index ", i,
                                " for compressed sparse operator with ",
                                nrows(A), " rows")))

@noinline out_of_range_column_index(A, j::Integer) =
    throw(ErrorException(string("out of range column index ", j,
                                " for compressed sparse operator with ",
                                ncols(A), " columns")))

"""
    LazyAlgebra.each_row_index(A)

Return an iterator over the linear row indices of the structural non-zeros of the sparse
operator `A` stored in a *Compressed Sparse Row* (CSR) format, this includes the adjoint
of a sparse operator in *Compressed Sparse Column* (CSC) format.

"""
each_row_index(A::CompressedSparseOperator{:CSR}) = 𝟙:nrows(A)
each_row_index(A::Swapped{<:CompressedSparseOperator{:CSC}}) = each_col_index(parent(A))

"""
    LazyAlgebra.each_col_index(A)

Return an iterator over the linear column indices of the structural non-zeros of the
sparse operator `A` stored in a *Compressed Sparse Column* (CSC) format, this includes the
adjoint of a sparse operator in *Compressed Sparse Row* (CSR) format.

"""
each_col_index(A::CompressedSparseOperator{:CSC}) = 𝟙:ncols(A)
each_col_index(A::Swapped{<:CompressedSparseOperator{:CSR}}) = each_row_index(parent(A))

"""
    LazyAlgebra.row_index(A, k) -> i

Return the linear row index of the `k`-th entry of the sparse operator `A` stored in a
*Compressed Sparse Column* (CSC) or *Coordinate* (COO) formats (this includes adjoint of
sparse operators in CSR format).

"""
@propagate_inbounds row_index(A::Union{AnySparseCOO,AnySparseCSC}, k::Int) = row_indices(A)[k]

"""
    LazyAlgebra.col_index(A, k) -> j

Return the linear column index of the `k`-th entry of the sparse operator `A` stored in a
*Compressed Sparse Row* (CSR) or *Coordinate* (COO) formats (this includes adjoint of
sparse operators in CSC format).

"""
@propagate_inbounds col_index(A::Union{AnySparseCOO,AnySparseCSR}, k::Int) = col_indices(A)[k]

# Implement partial API of abstract vectors to access the nonzeros by their linear index `k`.

Base.length(A::CompressedSparseOperator) = nnz(A)

Base.eltype(::Type{<:CompressedSparseOperator{F,T}}) where {F,T} = T

@inline function Base.getindex(A::CompressedSparseOperator, k::Int)
    vals = nonzeros(A)
    @boundscheck checkbounds(vals, k)
    v = @inbounds vals[k]
    return v
end

@inline function Base.getindex(A::Transpose{<:CompressedSparseOperator}, k::Int)
    vals = nonzeros(parent(A))
    @boundscheck checkbounds(vals, k)
    v = @inbounds vals[k]
    return v
end

@inline function Base.getindex(A::Adjoint{<:CompressedSparseOperator}, k::Int)
    vals = nonzeros(parent(A))
    @boundscheck checkbounds(vals, k)
    v = @inbounds vals[k]
    return conj(v)
end

@inline function Base.setindex!(A::CompressedSparseOperator, v, k::Int)
    vals = nonzeros(A)
    @boundscheck checkbounds(vals, k)
    @inbounds vals[k] = v
    return A
end

@inline function Base.setindex!(A::Transpose{<:CompressedSparseOperator}, v, k::Int)
    vals = nonzeros(parent(A))
    @boundscheck checkbounds(vals, k)
    @inbounds vals[k] = v
    return A
end

@inline function Base.setindex!(A::Adjoint{<:CompressedSparseOperator}, v, k::Int)
    vals = nonzeros(parent(A))
    @boundscheck checkbounds(vals, k)
    @inbounds vals[k] = conj(v)
    return A
end

#----------------------------------------------------------------------------- ITERATORS -

# As an iterator, a sparse operator behaves as a vector of the structural non-zero values.
Base.IteratorSize(::Type{<:CompressedSparseOperator}) = Base.HasLength()
Base.IteratorEltype(::Type{<:CompressedSparseOperator}) = Base.HasEltype()

@inline function Base.iterate(A::CompressedSparseOperator, k::Int = firstindex(A))
    vals = nonzeros(A)
    checkbounds(Bool, vals, k) ? (@inbounds(vals[k]), k + 1) : nothing
end

# Iterator over the row/column indices of a sparse operator with CSR or CSC storage.
struct SparseIndexIterator{S<:Union{AnySparseCSR,AnySparseCSC}}
    parent::S
end
Base.parent(iter::SparseIndexIterator) = getfield(iter, :parent)

Base.IteratorSize(::Type{<:SparseIndexIterator}) = Base.HasLength()
Base.IteratorEltype(::Type{<:SparseIndexIterator}) = Base.HasEltype()
Base.length(iter::SparseIndexIterator) = nnz(parent(iter))
Base.eltype(::Type{<:SparseIndexIterator}) = Int

function Base.iterate(iter::SparseIndexIterator{<:AnySparseCSR},
                      (i, k, l)::Tuple{Int,Int,Int} = (
                          1, 0, unsafe_last_nz_index(parent(iter), 1)))
    k += 1
    while k > l
        i += 1
        check_offset_index(Bool, parent(iter), i) || return nothing
        l = unsafe_last_nz_index(parent(iter), i)
    end
    return i, (i, k, l)
end

function Base.iterate(iter::SparseIndexIterator{<:AnySparseCSC},
                      (j, k, l)::Tuple{Int,Int,Int} = (
                          1, 0, unsafe_last_nz_index(parent(iter), 1)))
    k += 1
    while k > l
        j += 1
        check_offset_index(Bool, parent(iter), j) || return nothing
        l = unsafe_last_nz_index(parent(iter), j)
    end
    return j, (j, k, l)
end

# Optimized version of `collect`.
function Base.collect(iter::SparseIndexIterator)
    vect = Vector{eltype(iter)}(undef, length(iter))
    @inbounds for (k, index) in enumerate(iter)
        vect[k] = index
    end
    return vect
end

#-----------------------------------------------------------------------------------------
# Extend LazyAlgebra sparse operator API for SparseArrays.SparseMatrixCSC.

nrows(A::SparseMatrixCSC) = getfield(A, :m)
ncols(A::SparseMatrixCSC) = getfield(A, :n)
offsets(A::SparseMatrixCSC) = getfield(A, :colptr) # like `getcolptr`
row_indices(A::SparseMatrixCSC) = getfield(A, :rowval) # like `rowvals`
# FIXME col_indices is already done elsewhere.
row_size(A::SparseMatrixCSC) = (nrows(A),)
col_size(A::SparseMatrixCSC) = (ncols(A),)
each_col_index(A::SparseMatrixCSC) = 𝟙:ncols(A)

# Provide specific versions of `check_offset_index`, `unsafe_first_nz_index`, and
# `unsafe_last_nz_index` because offsets have a slightly different definition for
# `SparseMatrixCSC` than for our CSC format.
@inline check_offset_index(::Type{Bool}, A::SparseMatrixCSC, j::Int) =
    1 ≤ j < length(offsets(A))
@inline unsafe_first_nz_index(A::SparseMatrixCSC, j::Int) = @inbounds offsets(A)[j]
@inline unsafe_last_nz_index(A::SparseMatrixCSC, j::Int) = @inbounds offsets(A)[j + 1] - 1

@propagate_inbounds each_nz_index(A::SparseMatrixCSC, j::Integer) = nzrange(A, j::Integer)
@propagate_inbounds SparseArrays.nzrange(A::CompressedSparseOperator, ij::Int) =
    each_nz_index(A, ij)

function SparseArrays.rowvals(A::Union{CompressedSparseOperator{:COO},
                                       Swapped{<:CompressedSparseOperator{:COO}},
                                       CompressedSparseOperator{:CSC},
                                       Swapped{<:CompressedSparseOperator{:CSR}}})
    row_indices(A)
end

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

"""
SparseOperatorCSC

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
    f_decl = :(f::Function = isnonzero)
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
        # array with correct parameters and predicate.
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

        # Call generic constructor.
        function $CS{T,M,N,V}(A::AbstractArray{S,L},
                              f::Function = isnonzero) where {S,T,L,M,N,
                                                              V<:AbstractVector{T}}
            return build($CS{T,M,N,V}, A, f)
        end

        # Constructors that convert array of values. Other fields have already been
        # checked so do not check structure again.
        $CS{T}(A::$CS{S,M,N}) where {S,T,M,N} =
            $_CS(nrows(A), ncols(A), to_values(T, nonzeros(A)),
                 $(map(s -> :($(Symbol("get_",s))(A)), other_args)...),
                 row_size(A), col_size(A))

        # Basic outer constructors return a fully checked structure.
        function $CS(vals::AbstractVector, $(other_decl...),
                     rowsiz::Tuple{Vararg{Integer}}, colsiz::Tuple{Vararg{Integer}})
            check_structure($_CS(to_values(vals),
                                 $(map(s -> :(to_indices($s)), other_args)...),
                                 as_array_size(rowsiz), as_array_size(colsiz)))
        end

        # Constructors for any compressed format similar to the basic ones but with type
        # parameters that may imply converting arguments.
        function $CS{T,M,N}(vals::AbstractVector, $(other_decl...),
                              rowsiz::Tuple{Vararg{Integer}},
                              colsiz::Tuple{Vararg{Integer}}) where {T,M,N}
            N isa Int || throw_assertion_error("type parameter `N` must be an `Int`")
            length(colsiz) == N || throw_dimension_mismatch(
                "number of column dimensions is not equal to type parameter `N = $N`")
            $CS{T,M}(vals, $(other_args...), rowsiz, colsiz)
        end
        function $CS{T,M}(vals::AbstractVector, $(other_decl...),
                            rowsiz::Tuple{Vararg{Integer}},
                            colsiz::Tuple{Vararg{Integer}}) where {T,M}
            M isa Int || throw_assertion_error("type parameter `M` must be an `Int`")
            length(rowsiz) == M || throw_dimension_mismatch(
                "number of row dimensions is not equal to type parameter `M = $M`")
            $CS{T}(vals, $(other_args...), rowsiz, colsiz)
        end
        function $CS{T}(vals::AbstractVector, $(other_decl...),
                          rowsiz::Tuple{Vararg{Integer}},
                          colsiz::Tuple{Vararg{Integer}}) where {T}
            M isa Int || throw_assertion_error("type parameter `M` must be an `Int`")
            length(rowsiz) == M || throw_dimension_mismatch(
                "number of row dimensions must be equal to type parameter `M`")
            $CS(to_values(T, vals), $(other_args...), rowsiz, colsiz)
        end
    end
end

# Generic constructor of a sparse operator in various format given a regular Julia array
# and a predicate function. Julia arrays are usually in column-major order but this is not
# always the case, to handle various storage orders when extracting selected entries, we
# convert the input array into a equivalent "matrix", that is a 2-dimensional array.
function build(::Type{W}, arr::AbstractArray{S,L},
               f::Function = isnonzero) where {S,T,L,M,N,V<:AbstractVector{T},
                                               W<:Union{SparseOperatorCOO{T,M,N,V},
                                                        SparseOperatorCSC{T,M,N,V},
                                                        SparseOperatorCSR{T,M,N,V}}}
    # Get equivalent matrix dimensions.
    M isa Int || throw_bad_argument(
        "number of row dimensions `M` must be an `Int`, got an `$(typeof(M))`")
    N isa Int || throw_bad_argument(
        "number of column dimensions `N` must be an `Int`, got an `$(typeof(N))`")
    M ≥ 1 || throw_bad_argument(
        "number of row dimensions must be ≥ 1, got `M = $M`")
    N ≥ 1 || throw_bad_argument(
        "number of column dimensions must be ≥ 1, got `N = $N`")
    M + N == L || throw_bad_argument(
        "sum of numbers of row and column dimensions must be $L, got `M + N = $(M + N)`")
    siz = size(arr)
    rowsiz = siz[1:M]
    colsiz = siz[M+1:end]
    nrows = prod(rowsiz)
    ncols = prod(colsiz)

    # Reshape input array into a matrix if needed.
    A = !(arr isa AbstractMatrix) ? reshape(arr, (nrows, ncols)) :
        size(arr) == (nrows, ncols) ? arr : throw_dimension_mismatch(
            "argument has size $(size(arr)), expecting ($nrows, $ncols)")

    # Count the number of selected entries assuming column-major storage order which is
    # the most common in Julia (this only has a consequence on the speed).
    nvals = 0
    @inbounds for j in 1:ncols, i in 1:nrows
        # Using `ifelse` here asserts that the predicate yields a Boolean and avoid
        # branching.
        nvals += ifelse(f(A[i,j], i, j), 1, 0)
    end

    # Extract the selected entries and, depending on the format, their row and/or column
    # indices and/or offsets.
    if W <: Union{SparseOperatorCOO,SparseOperatorCSC}
        rows = Vector{Int}(undef, nvals)
    end
    if W <: Union{SparseOperatorCOO,SparseOperatorCSR}
        cols = Vector{Int}(undef, nvals)
    end
    if W <: SparseOperatorCSC
        offs = Vector{Int}(undef, ncols + 1)
    end
    if W <: SparseOperatorCSR
        offs = Vector{Int}(undef, nrows + 1)
    end
    if V <: UniformVector{Bool}
        vals = V(true, nvals)
    else
        vals = V(undef, nvals)
    end
    k = 0
    if  W <: SparseOperatorCSR
        # For a row-major compressed storage, the pseudo-matrix is walked in row-major
        # order.
        @inbounds for i in 1:nrows
            offs[i] = k
            for j in 1:ncols
                Aij = A[i,j]
                if f(Aij, i, j)
                    (k += 1) ≤ nvals || throw_bad_predicate()
                    if !(V <: UniformVector{Bool})
                        vals[k] = Aij
                    end
                    cols[k] = j
                end
            end
        end
    else
        # For a column-major compressed storage, the pseudo-matrix is walked in
        # column-major order. This is also suitable for the COO format since most Julia
        # arrays are stored in that order.
        @inbounds for j in 1:ncols
            if W <: SparseOperatorCSC
                offs[j] = k
            end
            for i in 1:nrows
                Aij = A[i,j]
                if f(Aij, i, j)
                    (k += 1) ≤ nvals || throw_bad_predicate()
                    if !(V <: UniformVector{Bool})
                        vals[k] = Aij
                    end
                    rows[k] = i
                    if W <: SparseOperatorCOO
                        cols[k] = j
                    end
                end
            end
        end
    end
    k == nvals || throw_bad_predicate()
    if W <: Union{SparseOperatorCSC,SparseOperatorCSR}
        offs[end] = nvals
    end

    # By construction, the sparse structure should be correct so just call the
    # "unsafe" constructor.
    if W <: SparseOperatorCOO
        return _SparseOperatorCOO(nrows, ncols, vals, rows, cols, rowsiz, colsiz)
    end
    if W <: SparseOperatorCSC
        return _SparseOperatorCSC(nrows, ncols, vals, rows, offs, rowsiz, colsiz)
    end
    if W <: SparseOperatorCSR
        return _SparseOperatorCSR(nrows, ncols, vals, cols, offs, rowsiz, colsiz)
    end
end

"""
    unpack!(A, S; flatten=false) -> A

unpacks the non-zero coefficients of the sparse operator `S` into the array `A` and
returns `A`. Keyword `flatten` specifies whether to only consider the length of `A`
instead of its dimensions. In any cases, `A` must have as many elements as `length(S)` and
standard linear indexing.

Just call `Array(S)` to unpack the coefficients of a sparse operator `S` without providing
the destination array.

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
    @inbounds for i in each_row_index(A)
        for k in each_nz_index(A, i)
            j = col_index(A, k)
            v = A[k]
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
    @inbounds for j in each_col_index(A)
        for k in each_nz_index(A, j)
            i = row_index(A, k)
            v = A[k]
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
    @inbounds for k in each_nz_index(A)
        i = row_index(A, k)
        j = col_index(A, k)
        v = A[k]
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
    reshape(A, as_array_size(rowsiz), as_array_size(colsiz))

function Base.reshape(A::SparseOperatorCSR,
                      rowsiz::Tuple{Vararg{Int}},
                      colsiz::Tuple{Vararg{Int}})
    check_new_shape(A, rowsiz, colsiz)
    _SparseOperatorCSR(nrows(A), ncols(A), nonzeros(A), col_indices(A), offsets(A),
                       rowsiz, colsiz)
end

function Base.reshape(A::SparseOperatorCSC,
                      rowsiz::Tuple{Vararg{Int}},
                      colsiz::Tuple{Vararg{Int}})
    check_new_shape(A, rowsiz, colsiz)
    _SparseOperatorCSC(nrows(A), ncols(A), nonzeros(A), row_indices(A), offsets(A),
                       rowsiz, colsiz)
end

function Base.reshape(A::SparseOperatorCOO,
                      rowsiz::Tuple{Vararg{Int}},
                      colsiz::Tuple{Vararg{Int}})
    check_new_shape(A, rowsiz, colsiz)
    _SparseOperatorCOO(nrows(A), ncols(A), nonzeros(A), row_indices(A), col_indices(A),
                       rowsiz, colsiz)
end

# Convert from other compressed sparse formats. For compressed sparse row and column (CSR
# and CSC) formats, the compressed sparse coordinate (COO) format is used as an
# intermediate representation and entries are sorted in row/column major order. To avoid
# side-effects, they must be copied first. Unless values are converted, there is no needs
# to copy when converting to a compressed sparse coordinate (COO) format.

SparseOperatorCSR{T}(A::SparseOperator) where {T} = # FIXME convert adjoint as well
    coo_to_csr!(copy_with_eltype(T, nonzeros(A)),
                collect(row_indices(A)),
                collect(col_indices(A)),
                row_size(A),
                col_size(A))

SparseOperatorCSC{T}(A::SparseOperator) where {T} = # FIXME convert adjoint as well
    coo_to_csc!(copy_with_eltype(T, nonzeros(A)),
                collect(row_indices(A)),
                collect(col_indices(A)),
                row_size(A),
                col_size(A))

SparseOperatorCOO{T}(A::SparseOperator) where {T} =
    _SparseOperatorCOO(nrows(A), ncols(A),
                       with_eltype(T, nonzeros(A)),
                       as_vector(row_indices(A)),
                       as_vector(col_indices(A)),
                       row_size(A),
                       col_size(A))

with_eltype(::Type{T}, A::AbstractArray{T}) where {T} = A
with_eltype(::Type{T}, A::AbstractArray) where {T} = copy_with_eltype(T, A)

copy_with_eltype(::Type{T}, A::AbstractArray) where {T} = copyto!(similar(A, T), A)

as_vector(vect::AbstractVector) = vect
as_vector(iter::SparseIndexIterator) = collect(iter)

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
    firstindex(offs) == 1 || throw_assertion_error(
        "vector of offsets must have 1-based indices")
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
            throw_assertion_error(1 ≤ j ≤ n ? "indices must be in non-increasing order" :
                "out of bound indices")
        end
    end
    @inbounds while i ≤ n
        i += 1
        offs[i] = length(inds)
    end
    return offs
end

# This error is due to the non-zeros predicate not returning the same results in
# the two selection passes.
throw_bad_predicate() = throw_bad_argument("inconsistent predicate function")

@inline select_non_zeros(v::Bool, i::Int, j::Int) = v
@inline select_non_zeros(v::T, i::Int, j::Int) where {T} = (v != zero(T))

@inline select_non_zeros_in_diagonal(v::T, i::Int, j::Int) where {T} =
    ((i == j)|(v != zero(T)))

@inline select_non_zeros_in_lower_part(v::T, i::Int, j::Int) where {T} =
    ((i ≥ j)&(v != zero(T)))

@inline select_non_zeros_in_upper_part(v::T, i::Int, j::Int) where {T} =
    ((i ≤ j)&(v != zero(T)))

"""
    check_structure(A) -> A

Check the structure of the compressed sparse operator `A` throwing an exception if there
are any inconsistencies and returning `A` otherwise.

"""
function check_structure(A::AbstractSparseOperator{CSR})
    check_size(A)
    check_vals(A)
    check_cols(A)
    check_offs(A)
    return A
end

function check_structure(A::AbstractSparseOperator{CSC})
    check_size(A)
    check_vals(A)
    check_rows(A)
    check_offs(A)
    return A
end

function check_structure(A::AbstractSparseOperator{COO})
    check_size(A)
    check_vals(A)
    check_rows(A)
    check_cols(A)
    return A
end

"""
    check_size(siz, id="array") -> len

Return the corresponding number of elements corresponding to array size `siz` throwing an
exception if any dimension is invalid (using `id` to identify the argument).

"""
function check_size(siz::Dims{N}, id::AbstractString="array") where {N}
    len = 1
    @inbounds for i in 1:N
        (dim = siz[i]) ≥ 0 || throw_bad_dimension(dim, i, id)
        len *= dim
    end
    return len
end

"""
    check_size(A)

Throw and exception if the row and column sizes in compressed sparse operator `A` have any
inconsistencies.

"""
function check_size(A::AbstractSparseOperator)
    check_size(row_size(A), "row") == nrows(A) || throw_dimension_mismatch(
        "incompatible equivalent number of rows and row size")
    check_size(col_size(A), "column") == ncols(A) || throw_dimension_mismatch(
        "incompatible equivalent number of columns and column size")
    return nothing
end

@noinline throw_bad_dimension(dim::Integer, i::Integer, id) =
    throw_bad_argument("invalid ", i, ordinal_suffix(i), " ", id, " dimension: ", dim)

"""
    check_vals(A)

Throw and exception if the values in compressed sparse operator `A` are not stored in a
proper vector.

"""
function check_vals(A::AbstractSparseOperator{<:Union{COO,CSC,CSR}})
    vals = nonzeros(A)
    is_fast_array(vals) || throw_not_fast_array("array of values")
    length(vals) == nnz(A) || throw_bad_argument("bad number of values")
    return nothing
end

"""
    check_rows(A)

Throw and exception if the row indices in the compressed sparse operator `A` stored in a
*Compressed Sparse Column* (CSC) or *Compressed Sparse Coordinate* (COO) format are
inconsistent.

"""
function check_rows(A::AbstractSparseOperator{<:Union{COO,CSC}})
    rows = row_indices(A)
    length(rows) == nnz(A) || throw_bad_argument("bad number of row indices")
    check_rows(rows, nrows(A))
    # FIXME: also check sorting for AbstractSparseOperator{CSC}?
    return nothing
end

"""
    check_rows(rows, m)

Throw and exception if the linear row indices `rows` is not a fast vector of values in the
range `1:m`.

"""
function check_rows(rows::AbstractVector{Int}, m::Int)
    is_fast_array(rows) || throw_not_fast_array("array of row indices")
    anyerror = false
    @inbounds @simd for k in eachindex(rows)
        i = rows[k]
        anyerror |= ((i < 1)|(i > m))
    end
    anyerror && throw_assertion_error("out of range row indices")
    return nothing
end

"""
    check_cols(A)

Throw and exception if the linear column indices in the compressed sparse operator `A`
stored in a *Compressed Sparse Row* (CSR) or *Compressed Sparse Coordinate* (COO) format
are inconsistent.

"""
function check_cols(A::AbstractSparseOperator{<:Union{COO,CSR}})
    cols = col_indices(A)
    length(cols) == nnz(A) || throw_assertion_error("bad number of column indices")
    check_cols(cols, ncols(A))
    # FIXME: also check sorting for AbstractSparseOperator{CSR}?
    return nothing
end

"""
    check_cols(cols, n)

Throw and exception if the linear column indices `cols` is not a fast vector of values in
the range `1:n`.

"""
function check_cols(cols::AbstractVector{Int}, n::Int)
    is_fast_array(cols) || throw_not_fast_array("array of column indices")
    anyerror = false
    @inbounds @simd for k in eachindex(cols)
        j = cols[k]
        anyerror |= ((j < 1)|(j > n))
    end
    anyerror && throw_assertion_error("out of range column indices")
    return nothing
end

"""
    check_offs(A)

Throw and exception if the offsets in the compressed sparse operator `A` stored
in a *Compressed Sparse Row* (CSR) or *Compressed Sparse Column* (CSC) format are
inconsistent.

"""
function check_offs(A::AbstractSparseOperator{F}) where {F<:Union{CSC,CSR}}
    offs = offsets(A)
    is_fast_array(offs) || throw_not_fast_array("array of offsets")
    n = (F <: CSR ? nrows(A) : ncols(A))
    length(offs) == n + 1 || throw_assertion_error("bad number of offsets")
    offs[1] == 0 || throw_assertion_error("bad initial offset")
    len = 0
    anyerrors = false
    @inbounds for i in 1:n
        k1, k2 = offs[i], offs[i+1]
        anyerrors |= (k2 < k1)
        len += ifelse(k2 > k1, k2 - k1, 0)
    end
    anyerrors && throw_assertion_error("offsets must be non-decreasing")
    len == nnz(A) || throw_assertion_error(
        "offsets incompatible with number of structural non-zeros")
    return nothing
end

"""
    _SparseOperatorCSR([m, n,] vals, cols, offs, rowsiz, colsiz)

Build a compressed sparse operator in *Compressed Sparse Row* (CSR) format as an instance
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

Build a compressed sparse operator in *Compressed Sparse Column* (CSC) format as an
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

Build a compressed sparse operator in *Compressed Sparse Coordinate* (COO) format as an
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
is_fast_indices(inds::AbstractUnitRange{Int}) = (first(inds) == 1) # FIXME always true for multi-dimensional arrays
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

#------------------------------------------------------------------ Apply Sparse Operators -

# When calling `unsafe_vmul!`, the following assumptions must hold:
# 1. all axes have been checked;
# 2. α is not zero;
# 3. α and β have been converted to a suitable type.

function unsafe_vmul!(α::Number,
                      A::AnySparseCSR{Ta,M,N},
                      x::AbstractArray{Tx,N},
                      β::Number,
                      y::AbstractArray{Ty,M}) where {Ta,Tx,Ty,M,N}
    Ts = sum_prod_type(eltype(A), eltype(x))
    @inbounds for i in each_row_index(A)
        s = zero(Ts)
        for k in each_nz_index(A, i)
            j = col_index(A, k)
            s += A[k]*x[j]
        end
        y[i] = α*s + β*y[i]
    end
    return y
end

function unsafe_vmul!(α::Number,
                      A::AnySparseCSC{Ta,M,N},
                      x::AbstractArray{Tx,N},
                      β::Number,
                      y::AbstractArray{Ty,M}) where {Ta,Tx,Ty,M,N}
    isone(β) || unsafe_vscale!(y, β)
    @inbounds for j in each_col_index(A)
        αxⱼ = α*x[j]
        if !iszero(αxⱼ)
            for k in each_nz_index(A, j)
                i = row_index(A, k)
                y[i] += A[k]*αxⱼ
            end
        end
    end
    return y
end

function unsafe_vmul!(α::Number,
                      A::AnySparseCOO{Ta,M,N},
                      x::AbstractArray{Tx,N},
                      β::Number,
                      y::AbstractArray{Ty,M}) where {Ta,Tx,Ty,M,N}
    isone(β) || unsafe_vscale!(y, β)
    @inbounds for k in each_nz_index(A)
        i = row_index(A, k)
        j = col_index(A, k)
        y[i] += α*A[k]*x[j]
    end
    return y
end

#-------------------------------------------------------------------------------------------
