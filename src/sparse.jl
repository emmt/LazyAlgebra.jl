# `sparse.jl` implements various format of compressed sparse linear operators, an API to
# deal with sparse operators, and methods to apply sparse operators and convert between
# different sparse formats. This goes beyond Julia's `SparseArrays` standard package which
# only provides "Compressed Sparse Column" (CSC) format.
#
# See https://en.wikipedia.org/wiki/Sparse_matrix.

#-------------------------------------------------------------------------- Sparse formats -

"""
    SparseFormat(x)
    SparseFormat(typeof(x))

Return a singleton representing the sparse storage format of `x`. The formats implemented by
`LazyAlgebra` are:

- [`CompressedSparseCoordinate()`](@ref CompressedSparseCoordinate) for *Compressed Sparse
  Coordinate* (COO) storage format;

- [`CompressedSparseColumn()`](@ref CompressedSparseColumn) for *Compressed Sparse Column*
  (CSR) storage format;

- [`CompressedSparseRow()`](@ref CompressedSparseRow) for *Compressed Sparse Row* (CSR)
  storage format.

Other formats may be implemented by foreign packages.

"""
SparseFormat(x::SparseFormat) = x
SparseFormat(x::Any) = SparseFormat(typeof(x))
SparseFormat(::Type{T}) where {T<:SparseFormat} = T()
SparseFormat(::Type{T}) where {T<:Any} = throw_bad_argument(
    "type `$T` has no known sparse storage format")

SparseFormat(::Type{<:Unswapped{<:SparseOperator{F}}}) where {F} = F()
SparseFormat(::Type{<:Swapped{  <:SparseOperator{F}}}) where {F} = transpose(F())

Base.transpose(trait::CompressedSparseRow) = CompressedSparseColumn()
Base.transpose(trait::CompressedSparseColumn) = CompressedSparseRow()
Base.transpose(trait::CompressedSparseCoordinate) = CompressedSparseCoordinate()

"""
    COO()
    CompressedSparseCoordinate()

Singleton representing *Compressed Sparse Coordinate* (COO) storage format.

Sparse operators with this storage format store their structural non-zeros in no particular
order, as a vector of values, a vector of linear row indices, and a vector of linear column
indices. It is even possible to have repeated entries. This format is very useful to build a
sparse linear operator. It can be converted to a more efficient format like *Compressed
Sparse Column* (CSC) or *Compressed Sparse Row* (CSR) for fast application of the sparse
linear operator or of its adjoint (see [`CompressedSparseColumn`](@ref) and
[`CompressedSparseRow`](@ref)).

Accessing the structural non-zeros and their respective row and column linear indices of a
sparse operator `A` in COO format is typically done by:

```julia
using LazyAlgebra: each_nz_index, row_index, col_index
for k in each_nz_index(A) # loop over index of structural non-zeros
     i = row_index(A, k)  # get row index of `k`-th structural non-zero
     j = col_index(A, k)  # get column index of `k`-th structural non-zero
     Aᵢⱼ = A[k]           # get value of `k`-th structural non-zero
     A[k] = ...           # set value of `k`-th structural non-zero
end
```

This API is also applicable to the conjugate, adjoint, and transpose of sparse operators in
COO format.

See also [`SparseFormat`](@ref), [`CompressedSparseColumn`](@ref), and
[`CompressedSparseRow`](@ref).

""" CompressedSparseCoordinate

"""
    CSC()
    CompressedSparseColumn()

Singleton representing *Compressed Sparse Column* (CSC) storage format.

Sparse operators with this storage format store their structural non-zeros in a column-major
order, as a vector of values, a vector of corresponding linear row indices, and a vector of
offsets indicating, for each column, the range of indices in the vectors of values and of
row indices. This storage format is very suitable for fast application of the operator,
notably its adjoint or its transpose.

Accessing the structural non-zeros and their respective row and column linear indices of a
sparse operator `A` in CSC format is typically done by:

```julia
using LazyAlgebra: each_nz_index, each_col_index, row_index
for j in each_col_index(A)       # loop over column index
    for k in each_nz_index(A, j) # loop over structural non-zeros in this column
        i = row_index(A, k)      # get row index of `k`-th structural non-zero
        Aᵢⱼ = A[k]               # get value of `k`-th structural non-zero
        A[k] = ...               # set value of `k`-th structural non-zero
     end
end
```

This API is also applicable to the conjugate of sparse operators in CSC format and to the
adjoint or transpose of sparse operators in CSR format.

See also [`SparseFormat`](@ref), [`CompressedSparseCoordinate`](@ref), and
[`CompressedSparseRow`](@ref).

""" CompressedSparseColumn

"""
    CSR()
    CompressedSparseRow()

Singleton representing *Compressed Sparse Row* (CSR) storage format.

Sparse operators with this format store their structural non-zeros in a row-major order, as
a vector of values, a vector of corresponding linear column indices, and a vector of offsets
indicating, for each row, the range of indices in the vectors of values and of column
indices. This storage format is very suitable for fast application of the operator.

Accessing the structural non-zeros and their respective row and column linear indices of a
sparse operator `A` in CSR format is typically done by:

```julia
using LazyAlgebra: each_nz_index, each_row_index, col_index
for i in each_row_index(A)       # loop over row index
    for k in each_nz_index(A, i) # loop over structural non-zeros in this row
        j = col_index(A, k)      # get column index of entry
        Aᵢⱼ = A[k]               # get value of entry
        A[k] = ...               # set value of entry
     end
end
```

This API is also applicable to the conjugate of sparse operators in CSR format and to the
adjoint or transpose of sparse operators in CSC format.

See also [`SparseFormat`](@ref), [`CompressedSparseCoordinate`](@ref), and
[`CompressedSparseColumn`](@ref).

""" CompressedSparseRow

#--------------------------------------------------------------- Abstract sparse operators -

"""
    SparseOperator{F}(args...; kwds...)
    SparseOperator{F,T}(args...; kwds...)
    SparseOperator{F,T,M}(args...; kwds...)
    SparseOperator{F,T,M,N}(args...; kwds...)

Build a sparse operator from arguments `args...` and keywords `kwds...`. Mandatory parameter
`F` is the storage format of the structural non-zeros. Optional parameters are the type `T`
of the structural non-zeros, the number `M` of dimensions of the *rows* of the operator, and
the number `N` of dimensions of the *columns* of the operator. If not specified, optional
parameters are inferred from the arguments.

The following formats `F` are providied by `LAzyAlgebra`:

- `COO` or [`CompressedSparseCoordinate`](@ref) for *Compressed Sparse Coordinate* storage
  format. This format is not the most efficient, it is mostly used as an intermediate for
  building a sparse operator in one of the other formats. See [`SparseOperatorCOO`](@ref)
  for a description of supported arguments and keywords.

- `CSC` or [`CompressedSparseColumn`](@ref) for *Compressed Sparse Column* storage format.
  This format is very efficient for applying the adjoint of the sparse operator. See
  [`SparseOperatorCSC`](@ref) for a description of supported arguments and keywords.

- `CSR` or [`CompressedSparseRow`](@ref) for *Compressed Sparse Row* storage format. This
  format is very efficient for applying the sparse operator. See [`SparseOperatorCSR`](@ref)
  for a description of supported arguments and keywords.

A simple (but slow for CSR and CSC storage formats) method to loop over the coordinates and
values of the structural non-zero of a sparse operator `A` is to write:

```julia
using LazyAlgebra: row_indices, col_indices
for (i, j, Aᵢⱼ) in zip(row_indices(A), col_indices(A), nonzeros(A))
    ...
end
```

or equivalently:

```julia
using SparseArrays
for (i, j, Aᵢⱼ) in zip(findnz(A)...)
    ...
end
```

Except for COO storage format, it is however more efficient to access the structural
non-zeros according to their storage order which depends on the compressed format. See
[`CompressedSparseCoordinate`](@ref), [`CompressedSparseColumn`](@ref), and
[`CompressedSparseRow`](@ref) for the implemented API.

"""
SparseOperator(A::SparseOperator) = A
SparseOperator{F}(A::SparseOperator{F}) where {F} = A
SparseOperator{F,T}(A::SparseOperator{F,T}) where {F,T} = A
SparseOperator{F,T,M}(A::SparseOperator{F,T,M}) where {F,T,M} = A
SparseOperator{F,T,M,N}(A::SparseOperator{F,T,M,N}) where {F,T,M,N} = A

# The above methods is to avoid doing anything if possible. Otherwise, call concrete
# constructors according to the specified format.
for (sym, fmt) in (:COO => :CompressedSparseCoordinate,
                   :CSC => :CompressedSparseColumn,
                   :CSR => :CompressedSparseRow,)
    constructor = Symbol("SparseOperator", sym)
    @eval begin
        function SparseOperator{$fmt}(args...; kwds...)
            return $constructor(args...; kwds...)
        end
        function SparseOperator{$fmt,T}(args...; kwds...) where {T}
            return $constructor{T}(args...; kwds...)
        end
        function SparseOperator{$fmt,T,M}(args...; kwds...) where {T,M}
            return $constructor{T,M}(args...; kwds...)
        end
        function SparseOperator{$fmt,T,M,N}(args...; kwds...) where {T,M,N}
            return $constructor{T,M,N}(args...; kwds...)
        end
    end
end

# Use constructors to perform conversion.
Base.convert(::Type{T}, A::T) where {T<:SparseOperator} = A
Base.convert(::Type{T}, A) where {T<:SparseOperator} = T(A)::T

#------------------------------------------------------------- Accessors and basic methods -

Base.eltype(::Type{<:SparseOperator{F,T,M,N}}) where {F,T,M,N} = T

InputShape( ::Type{<:SparseOperator{F,T,M,N}}) where {F,T,M,N} = HasInputShape{N}()
input_shape(A::BareSparseOperator) = getfield(A, :colsiz)
input_length(A::BareSparseOperator) = getfield(A, :n)

OutputShape(::Type{<:SparseOperator{F,T,M,N}}) where {F,T,M,N} = HasOutputShape{M}()
output_shape(A::BareSparseOperator) = getfield(A, :rowsiz)
output_length(A::BareSparseOperator) = getfield(A, :m)

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

TypeUtils.get_precision(::Type{A}) where {A<:SparseOperatorLike} = get_precision(eltype(A))
TypeUtils.adapt_precision(::Type{T}, A::SparseOperatorLike) where {T<:TypeUtils.Precision} =
    convert_eltype(adapt_precision(T, eltype(A)), A)

for (type, (get_1st_field, get_2nd_field)) in (:SparseOperatorCSR => (:col_indices, :offsets),
                                               :SparseOperatorCSC => (:row_indices, :offsets),
                                               :SparseOperatorCOO => (:row_indices, :col_indices))
    unsafe_constructor = Symbol("_",type)
    @eval begin
        TypeUtils.convert_eltype(::Type{T}, A::$type{T}) where {T} = A
        TypeUtils.convert_eltype(::Type{T}, A::$type{S}) where {T,S} =
            $unsafe_constructor(output_length(A), input_length(A),
                                convert_eltype(T, nonzeros(A)),
                                $get_1st_field(A), $get_2nd_field(A),
                                output_size(A), input_size(A))
    end
end

# Assume that a `copy` of a compressed sparse operator is to keep the same structure for the
# structural non-zeros but possibly change the values. So only duplicate the value part. For
# a `deepcopy` of a compressed sparse operator, all the fields are copied.

Base.copy(A::SparseOperatorCSR) = _SparseOperatorCSR(
    output_length(A), input_length(A), copy(nonzeros(A)), col_indices(A), offsets(A),
    output_size(A), input_size(A))

Base.copy(A::SparseOperatorCSC) = _SparseOperatorCSC(
    output_length(A), input_length(A), copy(nonzeros(A)), row_indices(A), offsets(A),
    output_size(A), input_size(A))

Base.copy(A::SparseOperatorCOO) = _SparseOperatorCOO(
    output_length(A), input_length(A), copy(nonzeros(A)), row_indices(A), col_indices(A),
    output_size(A), input_size(A))

Base.deepcopy(A::SparseOperatorCSR) = _SparseOperatorCSR(
    output_length(A), input_length(A), copy(nonzeros(A)), copy(col_indices(A)), copy(offsets(A)),
    output_size(A), input_size(A))

Base.deepcopy(A::SparseOperatorCSC) = _SparseOperatorCSC(
    output_length(A), input_length(A), copy(nonzeros(A)), copy(row_indices(A)), copy(offsets(A)),
    output_size(A), input_size(A))

Base.deepcopy(A::SparseOperatorCOO) = _SparseOperatorCOO(
    output_length(A), input_length(A), copy(nonzeros(A)), copy(row_indices(A)), copy(col_indices(A)),
    output_size(A), input_size(A))

# Extend some methods in SparseArrays. The "structural" non-zeros are the entries stored by
# the sparse structure which may or not be equal to zero, un-stored entries are always
# considered as being strictly equal to zero.
SparseArrays.nnz(A::SparseOperator) = length(nonzeros(A))
function SparseArrays.nnz(A::Union{Adjoint{<:SparseOperator},
                                   Transpose{<:SparseOperator},
                                   Conjugate{<:SparseOperator}})
    return nnz(parent(A))
end
#
# `findnz(A) -> I,J,V` yields the row and column indices and the values of the stored values
# in `A`.
SparseArrays.findnz(A::SparseOperator) = (row_indices(A), col_indices(A), nonzeros(A))

"""
    nonzeros(A::SparseOperatorLike)

Return the array storing the structural non-zeros of the compressed sparse operator `A`.

The returned array is shared with `A`, call `copy(nonzeros(A))` or `collect(nonzeros(A))`
instead if you want to modify the contents of the returned array with no side effects on
`A`.

"""
SparseArrays.nonzeros(A::BareSparseOperator) = getfield(A, :vals)
SparseArrays.nonzeros(A::Transpose{<:SparseOperator}) = nonzeros(parent(A))
function SparseArrays.nonzeros(A::Union{Adjoint{<:SparseOperator},
                                        Conjugate{<:SparseOperator}})
    return lazymap(eltype(A), conj, nonzeros(parent(A)), conj)
end

"""
    LazyAlgebra.row_indices(A) -> I

Return the row indices of the structural non-zeros of the sparse operator `A`.

If `A` is a sparse operator in CSC or COO storage format, the result `I` is a vector of
indices shared with `A`; otherwise, `I` is an iterator. In any case, the caller shall not
attempt to modify the contents of `I`. Call `collect(row_indices(A))` to get a vector of row
indices that can be modified with no side effects on `A`.

"""
row_indices(A::Union{SparseOperatorCSC,SparseOperatorCOO}) = getfield(A, :rows)
row_indices(A::SparseOperator{CSR}) = SparseIndexIterator(A)
row_indices(A::Swapped{<:SparseOperator}) = col_indices(parent(A))
row_indices(A::Conjugate{<:SparseOperator}) = row_indices(parent(A))

"""
    LazyAlgebra.col_indices(A) -> J

Return the column indices of the structural non-zeros of the sparse operator `A`.

If `A` is a sparse operator in CSR or COO storage format, the result `J` is a vector of
indices shared with `A`; otherwise, `J` is an iterator. In any case, the caller shall not
attempt to modify the contents of `J`. Call `collect(col_indices(A))` to get a vector of
column indices that can be modified with no side effects on `A`.

"""
col_indices(A::Union{SparseOperatorCSR,SparseOperatorCOO}) = getfield(A, :cols)
col_indices(A::Union{SparseOperator{CSC},SparseMatrixCSC}) =
    SparseIndexIterator(A) # FIXME: check whether this works for SparseMatrixCSC
col_indices(A::Swapped{<:SparseOperator}) = row_indices(parent(A))
col_indices(A::Conjugate{<:SparseOperator}) = col_indices(parent(A))

"""
    LazyAlgebra.offsets(A)

Return the table of offsets of the sparse operator `A`. Not all operators extend this
method.

!!! warning
    The interpretation of offsets depend on the type of `A`. For instance, assuming `offs =
    LazyAlgebra.offsets(A)`, then the index range of the `j`-th column of a
    `SparseMatrixCSC` is `offs[j]:(offs[j+1]-1)` while the index range is
    `(offs[j]+1):offs[j+1]` for a `SparseOperatorCSC`. For this reason, it is recommended to
    call [`each_nz_index`](@ref) instead or to call `offsets` with 2 arguments: `A` and,
    depending on the compressed storage format, the row or column index.

"""
offsets(A::Union{SparseOperatorCSR,SparseOperatorCSC}) = getfield(A, :offs)
function offsets(A::Union{Adjoint{T},Transpose{T},Conjugate{T}}) where {T<:SparseOperator}
    return offsets(parent(A))
end

"""
    LazyAlgebra.each_nz_index(A)

Return an iterator over the indices of the structural non-zeros of the sparse operator `A`
stored in a *Compressed Sparse Coordinate* (COO) format.

---
    LazyAlgebra.each_nz_index(A, j)

Return an iterator over the indices of the structural non-zeros of the `j`-th column of the
sparse operator `A` stored in a *Compressed Sparse Column* (CSC) format.

---
    LazyAlgebra.each_nz_index(A, i)

Return an iterator over the indices of the structural non-zeros of the `i`-th row of the
sparse operator `A` stored in a *Compressed Sparse Row* (CSR) format.

"""
@inline each_nz_index(A::AnySparseCOO) = 𝟙:nnz(A)

@inline function each_nz_index(A::Union{AnySparseCSR,AnySparseCSC}, ij::Int)
    @boundscheck check_offset_index(A, ij)
    return UnitRange(unsafe_first_nz_index(A, ij),
                     unsafe_last_nz_index(A, ij))
end

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
@inline first_nz_index(A::AnySparseCOO) = 1

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

Return the index of the last structural non-zero of the `j`-th column of the sparse operator
`A` stored in a *Compressed Sparse Column* (CSC) format.

---
    LazyAlgebra.last_nz_index(A, i)

Return the index of the last structural non-zero of the `i`-th row of the sparse operator
`A` stored in a *Compressed Sparse Row* (CSR) format.

"""
@inline last_nz_index(A::AnySparseCOO) = nnz(A)

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

@noinline out_of_range_row_index(A, i::Integer) = throw(ErrorException(string(
    "out of range row index ", i, " for compressed sparse operator with ",
    output_length(A), " rows")))

@noinline out_of_range_column_index(A, j::Integer) = throw(ErrorException(string(
    "out of range column index ", j, " for compressed sparse operator with ",
    input_length(A), " columns")))

"""
    LazyAlgebra.each_row_index(A)

Return an iterator over the linear row indices of the structural non-zeros of the sparse
operator `A` stored in a *Compressed Sparse Row* (CSR) format.

"""
each_row_index(A::SparseOperator{CSR}) = 𝟙:output_length(A)
each_row_index(A::Conjugate{<:SparseOperator{CSR}}) = each_row_index(parent(A))
each_row_index(A::Swapped{<:SparseOperator{CSC}}) = each_col_index(parent(A))

"""
    LazyAlgebra.each_col_index(A)

Return an iterator over the linear column indices of the structural non-zeros of the sparse
operator `A` stored in a *Compressed Sparse Column* (CSC) format.

"""
each_col_index(A::SparseOperator{CSC}) = 𝟙:input_length(A)
each_col_index(A::Conjugate{<:SparseOperator{CSC}}) = each_col_index(parent(A))
each_col_index(A::Swapped{<:SparseOperator{CSR}}) = each_row_index(parent(A))

"""
    LazyAlgebra.row_index(A, k) -> i

Return the linear row index of the `k`-th entry of the sparse operator `A` stored in a
*Compressed Sparse Column* (CSC) or *Compressed Sparse Coordinate* (COO) formats.

"""
@propagate_inbounds row_index(A::Union{AnySparseCOO,AnySparseCSC}, k::Int) =
    row_indices(A)[k]

"""
    LazyAlgebra.col_index(A, k) -> j

Return the linear column index of the `k`-th entry of the sparse operator `A` stored in a
*Compressed Sparse Row* (CSR) or *Compressed Sparse Coordinate* (COO) formats.

"""
@propagate_inbounds col_index(A::Union{AnySparseCOO,AnySparseCSR}, k::Int) =
    col_indices(A)[k]

#--------------------------------------------------------------------- Abstract vector API -

# Implement API for sparse operators to behave as abstract vectors of their nonzeros.

Base.length(A::SparseOperatorLike) = nnz(A)

# TODO nonzeros(A) -> nonzeros(parent(A))
Base.eachindex(style::IndexStyle, A::SparseOperatorLike) = eachindex(style, nonzeros(A))

Base.checkbounds(A::SparseOperatorLike, k::Int) = checkbounds(nonzeros(A), k)
Base.checkbounds(::Type{Bool}, A::SparseOperatorLike, k::Int) =
    checkbounds(Bool, nonzeros(A), k)

Base.IndexStyle(A::SparseOperatorLike) = IndexStyle(typeof(A))

function Base.IndexStyle(::Type{<:Union{Adjoint{A},
                                        Transpose{A},
                                        Conjugate{A}}}) where {A<:SparseOperator}
    return IndexStyle(typeof(A))
end

Base.IndexStyle(::Type{<:SparseOperatorCSR{T,M,N,V}}) where {T,M,N,V} = IndexStyle(V)
Base.IndexStyle(::Type{<:SparseOperatorCSC{T,M,N,V}}) where {T,M,N,V} = IndexStyle(V)
Base.IndexStyle(::Type{<:SparseOperatorCOO{T,M,N,V}}) where {T,M,N,V} = IndexStyle(V)

for func in (:eachindex, :firstindex, :lastindex, :keys)
    @eval Base.$func(A::SparseOperatorLike) = $func(nonzeros(A))
end

for (T,(p,f)) in (:SparseOperator                => (identity, identity),
                  :(Adjoint{<:SparseOperator})   => (parent,   conj),
                  :(Transpose{<:SparseOperator}) => (parent,   identity),
                  :(Conjugate{<:SparseOperator}) => (parent,   conj))

    @eval begin
        @inline function Base.getindex(A::$T, k::Int)
            vals = nonzeros($p(A))
            @boundscheck checkbounds(vals, k)
            v = @inbounds vals[k]
            return $f(v)
        end
        @inline function Base.setindex!(A::$T, v, k::Int)
            vals = nonzeros($p(A))
            @boundscheck checkbounds(vals, k)
            @inbounds vals[k] = $f(v)
            return A
        end
    end
end

# As an iterator, a sparse operator behaves as a vector of the structural non-zero values.
Base.IteratorSize(::Type{<:SparseOperatorLike}) = Base.HasLength()
Base.IteratorEltype(::Type{<:SparseOperatorLike}) = Base.HasEltype()
@inline function Base.iterate(A::SparseOperatorLike, k::Int = firstindex(A))
    checkbounds(Bool, A, k) ? (@inbounds(A[k]), k + 1) : nothing
end

#-------------------------------------------------------------- Row/column index iterators -

# Iterator over the row/column indices of a sparse operator with CSR or CSC storage.
struct SparseIndexIterator{S<:Union{AnySparseCSR,AnySparseCSC}}
    parent::S
end
Base.parent(iter::SparseIndexIterator) = getfield(iter, :parent)

Base.IteratorSize(::Type{<:SparseIndexIterator}) = Base.HasLength()
Base.length(iter::SparseIndexIterator) = nnz(parent(iter))

Base.IteratorEltype(::Type{<:SparseIndexIterator}) = Base.HasEltype()
Base.eltype(::Type{<:SparseIndexIterator}) = Int

function Base.iterate(iter::SparseIndexIterator{<:Union{AnySparseCSC,AnySparseCSR}},
                      (ij, k, l)::Tuple{Int,Int,Int} = (
                          1, 0, unsafe_last_nz_index(parent(iter), 1)))
    k += 1
    while k > l
        ij += 1
        check_offset_index(Bool, parent(iter), ij) || return nothing
        l = unsafe_last_nz_index(parent(iter), ij)
    end
    return i, (i, k, l)
end

# Optimized version of `collect`.
function Base.collect(iter::SparseIndexIterator)
    vect = Vector{eltype(iter)}(undef, length(iter))
    @inbounds for (k, index) in enumerate(iter)
        vect[k] = index
    end
    return vect
end

#---------------------------------------------------- API for SparseArrays.SparseMatrixCSC -

output_length(A::SparseMatrixCSC) = getfield(A, :m)
input_length(A::SparseMatrixCSC) = getfield(A, :n)
offsets(A::SparseMatrixCSC) = getfield(A, :colptr) # like `getcolptr`
row_indices(A::SparseMatrixCSC) = getfield(A, :rowval) # like `rowvals`
# FIXME col_indices is already done elsewhere.
output_size(A::SparseMatrixCSC) = (output_length(A),)
input_size(A::SparseMatrixCSC) = (input_length(A),)
each_col_index(A::SparseMatrixCSC) = 𝟙:input_length(A)

# Provide specific versions of `check_offset_index`, `unsafe_first_nz_index`, and
# `unsafe_last_nz_index` because offsets have a slightly different definition for
# `SparseMatrixCSC` than for our CSC format.
@inline check_offset_index(::Type{Bool}, A::SparseMatrixCSC, j::Int) =
    1 ≤ j < length(offsets(A))
@inline unsafe_first_nz_index(A::SparseMatrixCSC, j::Int) = @inbounds offsets(A)[j]
@inline unsafe_last_nz_index(A::SparseMatrixCSC, j::Int) = @inbounds offsets(A)[j + 1] - 1

@propagate_inbounds each_nz_index(A::SparseMatrixCSC, j::Integer) = nzrange(A, j::Integer)
@propagate_inbounds SparseArrays.nzrange(A::SparseOperator, ij::Int) =
    each_nz_index(A, ij)

function SparseArrays.rowvals(A::Union{SparseOperator{COO},
                                       Swapped{<:SparseOperator{COO}},
                                       SparseOperator{CSC},
                                       Swapped{<:SparseOperator{CSR}}})
    row_indices(A)
end

#---------------------------------------------------------------------------- Constructors -

"""
    SparseOperatorCOO(vals, rows, cols, rowsiz, colsiz)

Build a sparse operator in *Compressed Sparse Coordinate* (COO) storage format given all
necessary data: `vals` is the vector of structural non-zeros, `rows` and `cols` are integer
valued vectors with the linear row and column indices of the structural non-zeros, `rowsiz`
and `colsiz` are the sizes of the row and column dimensions.

The value and linear row and column indices of the `k`-th structural non-zero are
respectively given by `vals[k]`, `rows[k]` and `cols[k]`. For efficiency reasons, sparse
operators are currently limited to *fast* arrays because they can be indexed linearly with
no loss of performances. If `vals`, `rows` and/or `cols` are not fast arrays, they will be
automatically converted to linearly indexed arrays.

A sparse operator in COO storage format can be constructed from a Julia array or LazyAlgebra
[`PseudoMatrix`](@ref) `A` in different ways:

    SparseOperatorCOO(         [f,] A)
    SparseOperatorCOO{T}(      [f,] A)
    SparseOperatorCOO{T,M}(    [f,] A)
    SparseOperatorCOO{T,M,N}(  [f,] A)
    SparseOperatorCOO{T,M,N,V}([f,] A)

where arguments and parameters are:

* `f` is an optional predicate function which is called as `f(v,i,j)` with `v`, `i` and `j`
  the value, the row and the column linear indices for each entry of `A` and which yields
  whether a given entry of `A` is a structural non-zeros. If omitted, the default predicate
  is similar to:

      f(v,i,j) = !iszero(v)

  to assume that all non-zeros of `A` are structural non-zeros.

* Parameter `T` is the element type of the structural non-zeros; if not specified, `T =
  eltype(A)` is assumed.

* Parameters `M` and `N` are the number of leading and trailing dimensions of `A`
  corresponding to the *rows* and the *columns* of the operator and `M = N = ndims(A)` must
  hold. These dimensions are the size of, respectively, the output and the input arrays when
  applying the operator. If `A` is a matrix, `M` and `N` must be both `1` and may be
  omitted. If `A` is a pseudo-matrix, these parameters are inferred from `A` and must not be
  specified.

* Optional parameter `V` is to specify the type of the vector backing the storage of the
  values of the structural non-zeros. `V` must implement standard linear indexing. The
  default is to take `V = Vector{T}`. As a special case, you can choose a uniform boolean
  vector from the `StructuredArrays` package to store the sparse coefficients:

      SparseOperatorCOO{T,M,N,UniformVector{Bool}}(args...)

  to get a compressed sparse operator in COO format whose values are an immutable uniform
  vector of true values requiring no storage. This is useful to only store the sparse
  structure of the operator, that is the indices in COO format of the sparse coefficients
  not their values.

The `SparseOperatorCOO` constructor can also be used to convert a sparse operator in another
storage format into the COO format. In that case, parameter `T` may also be specified to
convert the type of the sparse coefficients.

""" SparseOperatorCOO

"""
    SparseOperatorCSC(vals, rows, offs, rowsiz, colsiz)

Build a sparse operator in *Compressed Sparse Column* (CSC) storage format given all
necessary data: `vals` is the vector of structural non-zeros, `rows` is an integer valued
vector with the linear row indices of the structural non-zeros, `offs` is a column-major
table of offsets in these arrays, `rowsiz` and `colsiz` are the sizes of the row and column
dimensions.

The values of the structural non-zeros of the `j`-th column and their respective linear row
indices are given by `vals[k]` and `rows[k]` with `k ∈ offs[j]+1:offs[j+1]`. The linear
column index `j` is in the range `1:n` where `n = prod(colsiz)` is the equivalent number of
columns. For efficiency reasons, sparse operators are currently limited to *fast* arrays
because they can be indexed linearly with no loss of performances. If `vals`, `rows` and/or
`offs` are not fast arrays, they will be automatically converted to linearly indexed arrays.

A sparse operator in CSC storage format can be constructed from a Julia array or LazyAlgebra
[`PseudoMatrix`](@ref) `A` in different ways:

    SparseOperatorCSC(         [f,] A)
    SparseOperatorCSC{T}(      [f,] A)
    SparseOperatorCSC{T,M}(    [f,] A)
    SparseOperatorCSC{T,M,N}(  [f,] A)
    SparseOperatorCSC{T,M,N,V}([f,] A)

See [`SparseOperatorCOO`](@ref) for a description of above arguments and parameters.

The `SparseOperatorCSC` constructor can also be used to convert a sparse operator in another
storage format into the CSC format. In that case, parameter `T` may also be specified to
convert the type of the sparse coefficients.

"""
SparseOperatorCSC

"""
    SparseOperatorCSR(vals, cols, offs, rowsiz, colsiz)

Build a sparse operator in *Compressed Sparse Row* (CSR) storage format given all necessary
data: `vals` is the vector of values of the structural non-zeros, `cols` is an integer
valued vector with the linear column indices of the structural non-zeros, `offs` is a
column-major table of offsets in these arrays, `rowsiz` and `colsiz` are the sizes of the
row and column dimensions.

The values of the structural non-zeros of the `i`-th row and their respective linear column
indices are given by `vals[k]` and `cols[k]` with `k ∈ offs[i]+1:offs[i+1]`. The linear row
index `i` is in the range `1:m` where `m = prod(rowsiz)` is the equivalent number of rows.
For efficiency reasons, sparse operators are currently limited to *fast* arrays because they
can be indexed linearly with no loss of performances. If `vals`, `cols` and/or `offs` are
not fast arrays, they will be automatically converted to linearly indexed arrays.

A sparse operator in CSR storage format can be constructed from a Julia array or LazyAlgebra
[`PseudoMatrix`](@ref) `A` in different ways:

    SparseOperatorCSR(         [f,] A)
    SparseOperatorCSR{T}(      [f,] A)
    SparseOperatorCSR{T,M}(    [f,] A)
    SparseOperatorCSR{T,M,N}(  [f,] A)
    SparseOperatorCSR{T,M,N,V}([f,] A)

See [`SparseOperatorCOO`](@ref) for a description of above arguments and parameters.

The `SparseOperatorCSR` constructor can also be used to convert a sparse operator in another
storage format into the CSR format. In that case, parameter `T` may also be specified to
convert the type of the sparse coefficients.

""" SparseOperatorCSR

@inline isnonzero(v::Any, i::Integer, j::Integer) = !iszero(v)

# Many constructors have similar code whatever the compressed sparse storage format. We
# therefore use meta-programming to define them.
for (constructor, other_args) in (:SparseOperatorCSR => (:cols, :offs),
                                  :SparseOperatorCSC => (:rows, :offs),
                                  :SparseOperatorCOO => (:rows, :cols),)
    # All other arguments are integer-valued vectors.
    other_decl = map(s -> :($s::AbstractVector{<:Integer}), other_args)
    f_decl = :(f::Function = isnonzero)
    unsafe_constructor = Symbol("_",constructor)
    @eval begin
        # Get rid of the M and N parameters, but keep/set T for conversion of values.
        $constructor{T,M,N}(A::SparseOperatorLike{<:Any,<:Any,<:Any,M,N}) where {T,M,N} =
            $constructor{T}(A)
        $constructor{T,M}(A::SparseOperatorLike{<:Any,<:Any,<:Any,M}) where {T,M} =
            $constructor{T}(A)
        $constructor(A::SparseOperatorLike{<:Any,<:Any,T}) where {T} =
            $constructor{T}(A)

        # Do nothing cases (it makes sense that a constructor of an immutable type be able
        # to just return its argument if it is already of the correct type).
        $constructor{T}(A::$constructor{T}) where {T} = A

        # Provide default predicate.
        function $constructor(A::Union{AbstractMatrix,PseudoMatrix})
            return $constructor(isnonzero, A)
        end
        function $constructor{T}(A::Union{AbstractMatrix,PseudoMatrix}) where {T}
            return $constructor{T}(isnonzero, A)
        end
        function $constructor{T,M}(A::AbstractArray) where {T,M}
            return $constructor{T,M}(isnonzero, A)
        end
        function $constructor{T,M,N}(A::AbstractArray) where {T,M,N}
            return $constructor{T,M,N}(isnonzero, A)
        end
        function $constructor{T,M,N,V}(A::AbstractArray) where {T,M,N,V}
            return $constructor{T,M,N,V}(isnonzero, A)
        end

        # Provide type `T` of the structural non-zeros.
        function $constructor(f, A::Union{AbstractMatrix{T},PseudoMatrix{T}}) where {T}
            return $constructor{T}(f, A)
        end

        # Provide the number `M` of row dimensions.
        function $constructor{T}(f, A::AbstractMatrix) where {T}
            return $constructor{T,1}(f, A)
        end
        function $constructor{T}(f, A::PseudoMatrix{<:Any,M}) where {T,M}
            A isa FlexibleMatrix && throw_bad_argument(
                "flexible matrices cannot be converted to sparse operators")
            return $constructor{T,M}(f, parent(A))
        end

        # Provide the number `N` of column dimensions.
        function $constructor{T,M}(f, A::AbstractArray{<:Any,L}) where {T,M,L}
            1 ≤ M < L || throw_bad_argument("1 ≤ M < ndims(A) = $L` must hold, got `M=$M`")
            return $constructor{T,M,L-M}(f, A)
        end

        # Provide the type `V` of the array to store the structural non-zeros.
        function $constructor{T,M,N}(f, A::AbstractArray) where {T,M,N}
            return $constructor{T,M,N,Vector{T}}(f, A)
        end

        # Call generic constructor.
        function $constructor{T,M,N,V}(f, A::AbstractArray) where {T,M,N,
                                                                   V<:AbstractVector{T}}
            return build($constructor{T,M,N,V}, f, A)
        end

        # Constructors that convert array of values. Other fields have already been checked
        # so do not check structure again.
        function  $constructor{T}(A::$constructor{S,M,N}) where {S,T,M,N}
            return $unsafe_constructor(output_length(A), input_length(A),
                                       to_values(T, nonzeros(A)),
                                       $(map(s -> :($(Symbol("get_",s))(A)), other_args)...),
                                       output_size(A), input_size(A))
        end

        # Basic outer constructors return a fully checked structure.
        function $constructor(vals::AbstractVector, $(other_decl...),
                              rowsiz::Tuple{Vararg{Integer}},
                              colsiz::Tuple{Vararg{Integer}})
            return check_structure(
                $unsafe_constructor(to_values(vals),
                                    $(map(s -> :(to_indices($s)), other_args)...),
                                    as_array_size(rowsiz), as_array_size(colsiz)))
        end

        # Constructors for any compressed format similar to the basic ones but with type
        # parameters that may imply converting arguments.
        function $constructor{T,M,N}(vals::AbstractVector, $(other_decl...),
                                     rowsiz::Tuple{Vararg{Integer}},
                                     colsiz::Tuple{Vararg{Integer}}) where {T,M,N}
            N isa Int || throw_assertion_error("type parameter `N` must be an `Int`")
            length(colsiz) == N || throw_dimension_mismatch(
                "number of column dimensions is not equal to type parameter `N = $N`")
            $constructor{T,M}(vals, $(other_args...), rowsiz, colsiz)
        end
        function $constructor{T,M}(vals::AbstractVector, $(other_decl...),
                                   rowsiz::Tuple{Vararg{Integer}},
                                   colsiz::Tuple{Vararg{Integer}}) where {T,M}
            M isa Int || throw_assertion_error("type parameter `M` must be an `Int`")
            length(rowsiz) == M || throw_dimension_mismatch(
                "number of row dimensions is not equal to type parameter `M = $M`")
            $constructor{T}(vals, $(other_args...), rowsiz, colsiz)
        end
        function $constructor{T}(vals::AbstractVector, $(other_decl...),
                                 rowsiz::Tuple{Vararg{Integer}},
                                 colsiz::Tuple{Vararg{Integer}}) where {T}
            M isa Int || throw_assertion_error("type parameter `M` must be an `Int`")
            length(rowsiz) == M || throw_dimension_mismatch(
                "number of row dimensions must be equal to type parameter `M`")
            $constructor(to_values(T, vals), $(other_args...), rowsiz, colsiz)
        end
    end
end

# Generic constructor of a sparse operator in various format given a regular Julia array and
# a predicate function. Julia arrays are usually in column-major order but this is not
# always the case, to handle various storage orders when extracting selected entries, we
# convert the input array into a equivalent "matrix", that is a 2-dimensional array.
function build(::Type{W}, f,
               arr::AbstractArray{S,L}) where {S,T,L,M,N,V<:AbstractVector{T},
                                               W<:Union{SparseOperatorCOO{T,M,N,V},
                                                        SparseOperatorCSC{T,M,N,V},
                                                        SparseOperatorCSR{T,M,N,V}}}
    # Get equivalent matrix dimensions.
    M isa Int || throw_bad_argument(
        "number of row dimensions `M` must be an `Int`, got an instance of `$(typeof(M))`")
    N isa Int || throw_bad_argument(
        "number of column dimensions `N` must be an `Int`, got an instance of `$(typeof(N))`")
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

    # Count the number of selected entries assuming column-major storage order which is the
    # most common in Julia (this only has a consequence on the speed).
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
                Aᵢⱼ = A[i,j]
                if f(Aᵢⱼ, i, j)
                    (k += 1) ≤ nvals || throw_bad_predicate()
                    if !(V <: UniformVector{Bool})
                        vals[k] = Aᵢⱼ
                    end
                    cols[k] = j
                end
            end
        end
    else
        # For a column-major compressed storage, the pseudo-matrix is walked in column-major
        # order. This is also suitable for the COO format since most Julia arrays are stored
        # in that order.
        @inbounds for j in 1:ncols
            if W <: SparseOperatorCSC
                offs[j] = k
            end
            for i in 1:nrows
                Aᵢⱼ = A[i,j]
                if f(Aᵢⱼ, i, j)
                    (k += 1) ≤ nvals || throw_bad_predicate()
                    if !(V <: UniformVector{Bool})
                        vals[k] = Aᵢⱼ
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

    # By construction, the sparse structure should be correct so just call the "unsafe"
    # constructor.
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

Unpack the non-zero coefficients of the sparse operator `S` into the array `A` and returns
`A`. Keyword `flatten` specifies whether to only consider the length of `A` instead of its
dimensions. In any cases, `A` must have as many elements as `length(S)` and standard linear
indexing.

Just call `Array(S)` to unpack the coefficients of a sparse operator `S` without providing
the destination array.

""" unpack!

# Convert to standard Julia arrays which are stored in column-major order, hence the stride
# is the equivalent number of rows. For COO format, as duplicates are allowed, values must
# be combined by an operator.

Base.Array(A::SparseOperatorLike) = Array{eltype(A)}(A)
Base.Array{T}(A::SparseOperatorLike) where {T} =
    Array{T, output_ndims(A) + input_ndims(A)}(A)
function Base.Array{T,N}(A::SparseOperatorLike) where {T,N}
    N == output_ndims(A) + input_ndims(A) || throw_incompatible_number_of_dimensions()
    return unpack!(Array{T}(undef, (output_size(A)..., input_size(A)...,)), A)
end

function prepare_unpack!(dst::AbstractArray,
                         src::SparseOperator,
                         flatten::Bool)
    is_fast_array(dst) || throw_non_standard_indexing("destination array")
    if flatten
        length(dst) == length(src) ||
            throw_incompatible_number_of_elements()
    else
        size(dst) == (output_size(src)..., input_size(src)...,) ||
            throw_incompatible_dimensions()
    end
    fill!(dst, zero(eltype(dst)))
end

# FIXME this is unsafe as destination may have another kind of indexing, use a view?
function unpack!(B::AbstractArray{T,L},
                 A::SparseOperatorCSR{<:Any,M,N};
                 flatten::Bool = false) where {T,L,M,N}
    prepare_unpack!(B, A, flatten)
    m = output_length(A) # used as the "stride" in B
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
    m = output_length(A) # used as the "stride" in B
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
    m = output_length(A) # used as the "stride" in B
    @inbounds for k in each_nz_index(A)
        i = row_index(A, k)
        j = col_index(A, k)
        v = A[k]
        l = i + m*(j - 1)
        B[l] = op(B[l], v)
    end
    return B
end

Base.reshape(A::SparseOperator, rowsiz::ArraySize, colsiz::ArraySize) =
    reshape(A, as_array_size(rowsiz), as_array_size(colsiz))

Base.reshape(A::SparseOperatorCSR, rowsiz::Dims, colsiz::Dims) =
    _SparseOperatorCSR(check_new_shape(A, rowsiz, colsiz)...,
                       nonzeros(A), col_indices(A), offsets(A), rowsiz, colsiz)

Base.reshape(A::SparseOperatorCSC, rowsiz::Dims, colsiz::Dims) =
    _SparseOperatorCSC(check_new_shape(A, rowsiz, colsiz)...,
                       nonzeros(A), row_indices(A), offsets(A), rowsiz, colsiz)

Base.reshape(A::SparseOperatorCOO, rowsiz::Dims, colsiz::Dims) =
    _SparseOperatorCOO(check_new_shape(A, rowsiz, colsiz)...,
                       nonzeros(A), row_indices(A), col_indices(A), rowsiz, colsiz)

# FIXME use axes not size
function check_new_shape(A::SparseOperator, rowsiz::Dims, colsiz::Dims)
    m = output_length(A)
    n = input_length(A)
    check_size(m, rowsiz, "row")
    check_size(m, colsiz, "column")
    return (m, n)
end

# Convert from other compressed sparse formats. For compressed sparse row and column (CSR
# and CSC) formats, the compressed sparse coordinate (COO) format is used as an intermediate
# representation and entries are sorted in row/column major order. To avoid side-effects,
# they must be copied first. Unless values are converted, there is no needs to copy when
# converting to a compressed sparse coordinate (COO) format.

SparseOperatorCSR{T}(A::SparseOperatorLike) where {T} =
    coo_to_csr!(copy_with_eltype(T, nonzeros(A)),
                collect(row_indices(A)),
                collect(col_indices(A)),
                output_size(A),
                input_size(A))

SparseOperatorCSC{T}(A::SparseOperatorLike) where {T} =
    coo_to_csc!(copy_with_eltype(T, nonzeros(A)),
                collect(row_indices(A)),
                collect(col_indices(A)),
                output_size(A),
                input_size(A))

SparseOperatorCOO{T}(A::SparseOperatorLike) where {T} =
    _SparseOperatorCOO(output_length(A), input_length(A),
                       with_eltype(T, nonzeros(A)),
                       as_vector(row_indices(A)),
                       as_vector(col_indices(A)),
                       output_size(A),
                       input_size(A))

with_eltype(::Type{T}, A::AbstractArray{T}) where {T} = A
with_eltype(::Type{T}, A::AbstractArray) where {T} = copy_with_eltype(T, A)

copy_with_eltype(::Type{T}, A::AbstractArray) where {T} = copyto!(similar(A, T), A)

as_vector(vect::AbstractVector) = vect
as_vector(iter::SparseIndexIterator) = collect(iter)

"""
    coo_to_csr!(vals, rows, cols, rowsiz, colsiz [, op]) -> A

Return a compressed sparse operator in a CSR format given the components `vals`, `rows` and
`cols` in the COO format and the sizes `rowsiz` and `colsiz` of the row and column
dimensions. Input arrays are modified in-place. Optional argument `op` is a binary operator
to reduce the values of entries having the same row and column indices.

Input arrays must be regular Julia vectors to ensure type stability in case of duplicates.

"""
function coo_to_csr!(vals::Vector{T},
                     rows::Vector{Int},
                     cols::Vector{Int},
                     rowsiz::Dims{M},
                     colsiz::Dims{N},
                     op::Function = (T <: Bool ? (|) : (+))) where {T,M,N}
    # Check row and column sizes.
    nrows = check_size(rowsiz)
    ncols = check_size(colsiz)

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
    offs = sparse_compressed_offsets(nrows, view(rows, 𝟙:nvals))

    # Since everything will have been checked, we can call the unsafe constructor.
    return _SparseOperatorCSR(nrows, ncols, vals, cols, offs, rowsiz, colsiz)
end

"""
    coo_to_csc!(vals, rows, cols, rowsiz, colsiz [, op]) -> A

Return a compressed sparse operator in a CSC format given the components `vals`, `rows` and
`cols` in the COO format and the sizes `rowsiz` and `colsiz` of the row and column
dimensions. Input arrays are modified in-place. Optional argument `op` is a binary operator
to reduce the values of entries having the same row and column indices.

Input arrays must be regular Julia vectors to ensure type stability in case of duplicates.

"""
function coo_to_csc!(vals::Vector{T},
                     rows::Vector{Int},
                     cols::Vector{Int},
                     rowsiz::Dims{M},
                     colsiz::Dims{N},
                     op::Function = (T <: Bool ? (|) : (+))) where {T,M,N}
    # Check row and column sizes.
    nrows = check_size(rowsiz)
    ncols = check_size(colsiz)

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
    offs = sparse_compressed_offsets(ncols, view(cols, 𝟙:nvals))

    # Since everything will have been checked, we can call the unsafe constructor.
    return _SparseOperatorCSC(nrows, ncols, vals, rows, offs, rowsiz, colsiz)
end

"""
    sort_and_reduce!(vals, major, minor, op) -> nvals

Sort entries and reduce duplicates in input arrays `vals`, `major` and `minor`.

Entries consist in the 3-tuples `(vals[k],major[k],minor[k])`. The sorting order of the
`k`-th entry is based the value of `major[k]` and, if equal, on the value of `minor[k]`.
After sorting, duplicate entries, that is those which have the same minor and major indices,
are replaced by a single entry whose value is obtained by reducing the values in `vals` with
the binary operator `op`. All operations are done in-place, the number of unique entries is
returned but inputs arrays are not resized, only the `nvals` first entries are valid.

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
            vals[ j] = vals[ k]
            major[j] = major[k]
            minor[j] = minor[k]
        end
    end
    return j - first(r) + 1
end

"""
    sparse_compressed_offsets(n, inds) -> offs

Return a vector of `n+1` offsets for sparse compressed storage and computed from the list of
indices `inds`. Indices in `inds` must be in non-increasing order and in the range `1:n`.

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
            throw_assertion_error(1 ≤ j ≤ n ?
                "indices must be in non-increasing order" :
                "out of bound indices")
        end
    end
    @inbounds while i ≤ n
        i += 1
        offs[i] = length(inds)
    end
    return offs
end

# This error is due to the non-zeros predicate not returning the same results in the two
# selection passes.
throw_bad_predicate() = throw_bad_argument("inconsistent predicate function")

@inline select_non_zeros(v::Any, i::Int, j::Int) = !iszero(v)

@inline select_non_zeros_in_diagonal(v::Any, i::Int, j::Int) =
    ((i == j)&(!iszero(v)))

@inline select_non_zeros_in_lower_part(v::Any, i::Int, j::Int) =
    ((i ≥ j)&(!iszero(v)))

@inline select_non_zeros_in_upper_part(v::Any, i::Int, j::Int) =
    ((i ≤ j)&(!iszero(v)))

"""
    check_structure(A) -> A

Check the structure of the compressed sparse operator `A` throwing an exception if there are
any inconsistencies and returning `A` otherwise.

"""
function check_structure(A::SparseOperator{CSR})
    check_size(A)
    check_vals(A)
    check_cols(A)
    check_offs(A)
    return A
end

function check_structure(A::SparseOperator{CSC})
    check_size(A)
    check_vals(A)
    check_rows(A)
    check_offs(A)
    return A
end

function check_structure(A::SparseOperator{COO})
    check_size(A)
    check_vals(A)
    check_rows(A)
    check_cols(A)
    return A
end

"""
    check_size(siz::Dims) -> len

Return the corresponding number of elements corresponding to array size `siz` throwing an
exception if any dimension is invalid.

"""
function check_size(siz::Dims{N}) where {N}
    len = 1
    @inbounds for d in 1:N
        (dim = siz[d]) ≥ 0 || throw_bad_argument(
            "invalid ", d, ordinal_suffix(d), " array dimension: ", dim)
        len *= dim
    end
    return len
end

"""
    check_size(A)

Throw an exception if the row and column sizes in compressed sparse operator `A` have any
inconsistencies.

"""
function check_size(A::SparseOperator)
    check_size(output_length(A), output_size(A), "output")
    check_size(input_length(A), input_size(A), "input")
    return nothing
end

check_size(len::Int, siz::Dims, name::AbstractString) =
    (n = check_size(siz)) == len ? nothing : throw_dimension_mismatch(
        "products of ", what, " dimensions must be equal to ", len, "got ", n)

"""
    check_vals(A)

Throw an exception if the values in compressed sparse operator `A` are not stored in a
proper vector.

"""
function check_vals(A::SparseOperator{<:Union{COO,CSC,CSR}})
    vals = nonzeros(A)
    is_fast_array(vals) || throw_not_fast_array("array of values")
    length(vals) == nnz(A) || throw_bad_argument("bad number of values")
    return nothing
end

"""
    check_rows(A)

Throw an exception if the row indices in the compressed sparse operator `A` stored in a
*Compressed Sparse Column* (CSC) or *Compressed Sparse Coordinate* (COO) format are
inconsistent.

"""
function check_rows(A::SparseOperator{<:Union{COO,CSC}})
    rows = row_indices(A)
    length(rows) == nnz(A) || throw_bad_argument("bad number of row indices")
    check_rows(rows, output_length(A))
    # FIXME: also check sorting for SparseOperator{CSC}?
    return nothing
end

"""
    check_rows(rows, m)

Throw an exception if the linear row indices `rows` is not a fast vector of values in the
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

Throw an exception if the linear column indices in the compressed sparse operator `A` stored
in a *Compressed Sparse Row* (CSR) or *Compressed Sparse Coordinate* (COO) format are
inconsistent.

"""
function check_cols(A::SparseOperator{<:Union{COO,CSR}})
    cols = col_indices(A)
    length(cols) == nnz(A) || throw_assertion_error("bad number of column indices")
    check_cols(cols, input_length(A))
    # FIXME: also check sorting for SparseOperator{CSR}?
    return nothing
end

"""
    check_cols(cols, n)

Throw an exception if the linear column indices `cols` is not a fast vector of values in the
range `1:n`.

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

Throw an exception if the offsets in the compressed sparse operator `A` stored in a
*Compressed Sparse Row* (CSR) or *Compressed Sparse Column* (CSC) format are inconsistent.

"""
function check_offs(A::SparseOperator{F}) where {F<:Union{CSC,CSR}}
    offs = offsets(A)
    is_fast_array(offs) || throw_not_fast_array("array of offsets")
    n = (F <: CSR ? output_length(A) : input_length(A))
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

Build a compressed sparse operator in *Compressed Sparse Row* (CSR) format as an instance of
`SparseOperatorCSR`. This private constructor assumes that arguments are correct and is
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

Build a compressed sparse operator in *Compressed Sparse Column* (CSC) format as an instance
of `SparseOperatorCSC`. This private constructor assumes that arguments are correct and is
mostly used by converters and other constructors.

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
instance of `SparseOperatorCOO`. This private constructor assumes that arguments are correct
and is mostly used by converters and other constructors.

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

Return whether `A` is a *fast array* that is an array with standard linear indexing.

"""
is_fast_array(A::AbstractArray) = is_fast_indices(eachindex(A))

"""
    is_fast_indices(inds) -> bool

Return whether `inds` is an iterator for *fast indices* that is linear indices starting at
`1`.

"""
is_fast_indices(inds::AbstractUnitRange{Int}) = (first(inds) == 1) # FIXME always true for multi-dimensional arrays
is_fast_indices(inds) = false

"""
    check_argument(A, siz, id="array")

Check whether array `A` has size `siz` and implements standard linear indexing. An exception
is thrown if any of these do not hold.

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

#------------------------------------------------------------------ Apply sparse operators -

# When calling `unsafe_vmul!`, the following assumptions must hold:
# 1. all axes have been checked;
# 2. α is not zero;
# 3. α and β have been converted to a suitable type.

function unsafe_vmul!(α::Number,
                      A::AnySparseCSR{Ta,M,N},
                      x::AbstractArray{Tx,N},
                      β::Number,
                      y::AbstractArray{Ty,M}) where {Ta,Tx,Ty,M,N}
    Ts = sum_prod_type(Ta, Tx)
    @inbounds for i in each_row_index(A)
        s = zero(Ts)
        for k in each_nz_index(A, i)
            j = col_index(A, k)
            s += convert(Ts, A[k]*x[j])
        end
        y[i] = α*s + β*y[i]
    end
    return nothing
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
    return nothing
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
    return nothing
end

#------------------------------------------------------------------------------- Utilities -

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

#-------------------------------------------------------------------------------------------
