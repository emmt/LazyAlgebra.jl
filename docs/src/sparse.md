# Sparse operators

A sparse operator ([`SparseOperator`](@ref)) in `LazyAlgebra` is the
generalization of a sparse matrix.  Like a [`GeneralMatrix`](@ref), rows and
columns may be multi-dimensional.  However unlike a [`GeneralMatrix`](@ref), a
sparse operator only stores its *structural non-zero* entries and thus requires
fewer memory and is usually faster to apply.

There are many different possibilities for storing a sparse operator, hence
`SparseOperator{T,M,N}` is an abstract type inherited by the concrete types
implementing compressed sparse storage in various formats.  Parameter `T` is
the type of the elements while parameters `M` and `N` are the number of
dimensions of the *rows* and of the *columns* respectively.  Objects of this
kind are a generalization of sparse matrices in the sense that they implement
linear mappings which can be applied to `N`-dimensional arguments to produce
`M`-dimensional results (as explained below).  The construction of a sparse
operator depends on its storage format.  Several concrete implementations are
provided: [*Compressed Sparse Row* (CSR)](#Compressed-sparse-row-format),
[*Compressed Sparse Column* (CSC)](#Compressed-sparse-column-format) and
[*Compressed Sparse Coordinate* (COO)](#Compressed-sparse-coordinate-format).


## Basic methods

The following methods are generally applicable to any sparse operator `A`:

- `eltype(A)` yields `T`, the type of the elements of `A`;

- `row_size(A)` yields an `M`-tuple of `Int`: the size of the rows of `A`, this
  is equivalent to `output_size(A)`;

- `col_size(A)` yields an `N`tuple of `Int`: the size of the columns of `A`,
  this is equivalent to `input_size(A)`;

- `nrows(A)` yields `prod(row_size(A))`, the equivalent number of rows of `A`;

- `ncols(A)` yields `prod(col_size(A))`, the equivalent number of columns of
  `A`.

- `ndims(A)` yields `M + N` the number of dimensions of the regular array
   corresponding to the sparse operator `A`;

- `size(A)` yields `(row_size(A)..., col_size(A)...)` the size of the regular
   array corresponding to the sparse operator `A`;

- `length(A)` yields `prod(size(A))` the number of elements of the regular
   array corresponding to the sparse operator `A`;

- `nnz(A)` yields the number of *structural non-zeros* in `A`;

- `nonzeros(A)` yields the vector of *structural non-zeros* in `A`.

The *structural non-zeros* are the entries stored by the sparse structure, they
may or not be equal to zero, un-stored entries are always considered as being
equal to zero.

As can be seen above, `eltype`, `ndims`, `size` and `length` yield the same
results as if applied to the multi-dimensional array corresponding to the
sparse operator.


## Generalized matrix multplication by a sparse operator

A sparse operator `A` can be directly used as any linear mapping in
`LazyAlgebra`:

```julia
A*x
```

yields the generalized matrix multiplication of `x` by `A`.  The size of `x`
must be that of the *columns* of `A`, that is `col_size(A)`.  The result is an
array whose size is that of the *rows* of `A`, that is `size(A*x) =
row_size(A)`.  Applying the adjoint of `A` is also implemented by the usual
syntax:

```julia
A'*y
```

to produce an array of size `col_size(A)` provided `y` is of suitable size,
i.e. `size(y) = row_size(A)`.

Sparse operators can be used as iterators, the value returned by the iterator
is a 3-tuple `(v,i,j)` with the value, the linear row index and the linear
column index of the entry.  For instance:

```julia
for (v,i,j) in A
    println("A[$i,$j] = $v")
end
```

This can be used to illustrate how `w = A*x` and `z = A'*y` could be computed
for the sparse operator `A`:

```julia
# Compute w = A*x:
@assert !Base.has_offset_axes(x)
@assert size(x) == col_size(A)
T = promote_type(eltype(A),eltype(x))
w = zeros(T, row_size(A))
@inbounds for (v,i,j) in A
    w[i] += v*x[j]
end
```

and

```julia
# Compute z = A'*y:
@assert !Base.has_offset_axes(y)
@assert size(y) == row_size(A)
T = promote_type(eltype(A),eltype(y))
z = zeros(T, col_size(A))
@inbounds for (v,i,j) in A
    z[j] += conj(v)*y[i]
end
```

Actual implementations of sparse operators in `LazyAlgebra` are equivalent to
the above examples but should be more efficient because they exploit the
specific storage format of a compressed sparse operator (see
[`CompressedSparseOperator`](@ref), [`SparseOperatorCSR`](@ref),
[`SparseOperatorCSC`](@ref) and [`SparseOperatorCOO`](@ref)).

!!! note
    For now, row and column indices are restricted to be linear indices and
    arguments to the linear mappings implemented by compressed sparse
    operators must be arrays with conventional indexing (1-based linear
    indices) and preferably implementing linear indices (not Cartesian ones).


## Simple construction of compressed sparse operators

Compressed sparse operators only store their structural non-zero elements.  The
abstract super-type of these sparse operators is
`CompressedSparseOperator{F,T,M,N}` which is a direct sub-type of
`SparseOperator{T,M,N}` with an additional parameter `F` to specify the storage
format.  The parameter `F` is specificed as a symbol and can be:

- `:COO` for *Compressed Sparse Coordinate* storage format.  This format is not
  the most efficient, it is mostly used as an intermediate for building a
  sparse operator in one of the following format.

- `:CSC` for *Compressed Sparse Column* storage format.  This format is very
  efficient for applying the adjoint of the sparse operator.

- `:CSR` for *Compressed Sparse Row* storage format.  This format is very
  efficient for directly applying the sparse operator.

To construct a compressed sparse operator in a given format `F` from the values
in a 2-dimensional array `A` call:

```julia
CompressedSparseOperator{F}(A, sel = (v,i,j) -> (v != zero(v)))
```

where optional argument `sel` is a selector function which is called as
`sel(v,i,j)` with `v`, `i` and `j` the value, the row and the column linear
indices for each entries of `A` and which is assumed to yield `true` for the
entries of `A` to be selected in the sparse structure and `false` for the
entries of `A` to discard.  The default selector is such that all non-zeros of
`A` are selected.  As an example, to select the non-zeros of the lower
triangular part of `A`, the constructor can be called as:

```julia
CompressedSparseOperator{F}(A, (v,i,j) -> ((i ≥ j)&(v != zero(v))))
```

Note the (slight) optimization of the expression with a biwise AND `&` instead
of a short-circuiting logical AND `&&` to avoid branching.

By default the values of the structural non-zeros of the sparse operator have
the same type as the elements of `A`, you can enforce a specific element type
`T` with:

```julia
CompressedSparseOperator{F,T}(A[, sel])
```

To generalize the matrix-vector product, a sparse operator can emulate an array
with more than 2 dimensions.  In that case, you must specify the number `M` of
leading dimensions that are considered as the *rows*:

```julia
CompressedSparseOperator{F,T,M}(A[, sel])
```

The number `N` of trailing dimensions that are considered as the *columns* can
also be specified (although they can be automatically guessed):

```julia
CompressedSparseOperator{F,T,M,N}(A[, sel])
```

with the restriction that `M ≥ 1`, `N ≥ 1` and `M + N = ndims(A)`.  Note that
parameter `T` can be `Any` to keep the same element type as `A`.

Finally, the type `V` of the vector used to store the coefficients of the sparse
operator may also be specified:

```julia
CompressedSparseOperator{F,T,M,N,V}(A[, sel])
```

with the restriction that `V` must have standard linear indexing.  The default
is to take `V = Vector{T}`.  As a special case, you can choose a uniform
boolean vector from the
[`StructuredArrays`](https://github.com/emmt/StructuredArrays.jl) package to
store the sparse coefficients:

```julia
CompressedSparseOperator{F,T,M,N,UniformVector{Bool}}(A[, sel])
```

yields a compressed sparse operator whose values are an immutable uniform
vector of `true` values requiring no storage.  This is useful if you want to
only store the sparse structure of the selected values, that is their indices
in the compressed format `F` not their values.

As explained in the last sections, compressed sparse operators can also be
consructed by providing the values of the structural non-zeros and their
respective row and column indices.  As a general rule, to construct (or convert
to) a sparse operator with compressed storage format `F`, you can call:

```julia
CompressedSparseOperator{F}(args...; kwds...)
CompressedSparseOperator{F,T}(args...; kwds...)
CompressedSparseOperator{F,T,M}(args...; kwds...)
CompressedSparseOperator{F,T,M,N}(args...; kwds...)
CompressedSparseOperator{F,T,M,N,V}(args...; kwds...)
```

where given parameters `T`, `M`, `N` and `V`, arguments `args...`
and optional keywords `kwds...` will be passed to the concrete constructor
[`SparseOperatorCOO`](@ref), [`SparseOperatorCSC`](@ref) or
[`SparseOperatorCSR`](@ref) corresponding to the format `F`.  For instance,

```julia
CompressedSparseOperator{:CSR}(A) -> SparseOperatorCSR(A)
```


## Accessing the structural non-zeros

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
purpose of the statement `using LazyAlgebra.SparseMethods`.  These methods may
be extended to implement variants of compressed sparse operators.


## Sparse operators in COO format

Sparse operators in *Compressed Sparse Coordinate* (COO) format store the
significant entries in no particular order, as a vector of values, a vector of
linear row indices and a vector of linear column indices.  It is even possible
to have repeated entries.  This format is very useful to build a sparse
operator.  It can be converted to a more efficient format like *Compressed
Sparse Column* or *Compressed Sparse Row* for fast application of the sparse
linear mapping or of its adjoint.

A sparse operator in with COO storage can be directly constructed by:

```julia
CompressedSparseOperator{:COO}(vals, rows, cols, rowsiz, colsiz)
```

which is the same as:

```julia
SparseOperatorCOO(vals, rows, cols, rowsiz, colsiz)
```

or, if you want to force the element type of the result, one of the following:

```julia
CompressedSparseOperator{:COO,T}(vals, rows, cols, rowsiz, colsiz)
SparseOperatorCOO{T}(vals, rows, cols, rowsiz, colsiz)
```

Here, `vals` is the vector of values of the sparse entries, `rows` and `cols`
are integer valued vectors with the linear row and column indices of the sparse
entries, `rowsiz` and `colsiz` are the sizes of the row and column dimensions.
The entries values and respective linear row and column indices of the `k`-th
sparse entry are given by `vals[k]`, `rows[k]` and `cols[k]`.

For efficiency reasons, sparse operators are currently limited to *fast* arrays
because they can be indexed linearly with no loss of performances.  If `vals`,
`rows` and/or `cols` are not fast arrays, they will be automatically converted
to linearly indexed arrays.


## Sparse operators in CSC format

Sparse operators in *Compressed Sparse Column* (CSC) format store the
significant entries in a column-wise order, as a vector of values, a vector of
corresponding linear row indices and a vector of offsets indicating, for each
column, the range of indices in the vectors of values and of row indices.

A sparse operator in with CSC storage can be directly constructed by:

```julia
CompressedSparseOperator{:CSC}(vals, rows, offs, rowsiz, colsiz)
```

which is the same as:

```julia
SparseOperatorCSC(vals, rows, offs, rowsiz, colsiz)
```

or, if you want to force the element type of the result, one of the following:

```julia
CompressedSparseOperator{:CSC,T}(vals, rows, offs, rowsiz, colsiz)
SparseOperatorCSC{T}(vals, rows, offs, rowsiz, colsiz)
```

Here, `vals` is the vector of values of the sparse entries, `rows` is an
integer valued vector of the linear row indices of the sparse entries, `offs`
is a column-wise table of offsets in these arrays, `rowsiz` and `colsiz` are
the sizes of the row and column dimensions.  The entries values and respective
linear row indices of the `j`-th column are given by `vals[k]` and `rows[k]`
with `k ∈ offs[j]+1:offs[j+1]`.  The linear column index `j` is in the range
`1:n` where `n = prod(colsiz)` is the equivalent number of columns.

For efficiency reasons, sparse operators are currently limited to *fast* arrays
because they can be indexed linearly with no loss of performances.  If `vals`,
`rows` and/or `offs` are not fast arrays, they will be automatically converted
to linearly indexed arrays.


## Sparse operators in CSR format

Sparse operators in *Compressed Sparse Row* (CSR) format store the significant
entries in a row-wise order, as a vector of values, a vector of corresponding
linear column indices and a vector of offsets indicating, for each row, the
range of indices in the vectors of values and of column indices.

A sparse operator in with CSR storage can be directly constructed by:

```julia
CompressedSparseOperator{:CSR}(vals, cols, offs, rowsiz, colsiz)
```

which is the same as:

```julia
SparseOperatorCSR(vals, cols, offs, rowsiz, colsiz)
```

or, if you want to force the element type of the result, one of the following:

```julia
CompressedSparseOperator{:CSR,T}(vals, cols, offs, rowsiz, colsiz)
SparseOperatorCSR{T}(vals, cols, offs, rowsiz, colsiz)
```

Here, `vals` is the vector of values of the sparse entries, `cols` is an
integer valued vector of the linear column indices of the sparse entries,
`offs` is a column-wise table of offsets in these arrays, `rowsiz` and `colsiz`
are the sizes of the row and column dimensions.  The entries values and
respective linear column indices of the `i`-th row are given by `vals[k]` and
`cols[k]` with `k ∈ offs[j]+1:offs[j+1]`.  The linear row index `i` is in the
range `1:m` where `m = prod(rowsiz)` is the equivalent number of rows.

For efficiency reasons, sparse operators are currently limited to *fast* arrays
because they can be indexed linearly with no loss of performances.  If `vals`,
`cols` and/or `offs` are not fast arrays, they will be automatically converted
to linearly indexed arrays.


## Conversion

Calling a sparse operator constructor can also be used to convert between
different formats or to change the type of the stored values.  For example, to
convert a sparse operator `A` into a Compressed Spase Row (CSR) format, the
following calls are equivalent:

```julia
SparseOperatorCSR(A)
CompressedSparseOperator{:CSR}(A)
convert(SparseOperatorCSR, A)
convert(CompressedSparseOperator{:CSR}, A)
```

If `A` is in Compressed Sparse Coordinate (COO) format, entries are sorted and
duplicates merged.  This also occurs when converting from COO format to
Compressed Sparse Column (CSC) format.  Such conversions are very useful as
building a sparse operator in COO format is easier while CSC and CSR formats
are more efficients.

It is sufficient to specify the element type `T` to convert the storage format
and the type of the stored values.  For example, any of the following will
convert `A` to CSC format with element type `T`:

```julia
SparseOperatorCSC{T}(A)
CompressedSparseOperator{:CSC,T}(A)
convert(SparseOperatorCSC{T}, A)
convert(CompressedSparseOperator{:CSC,T}, A)
```

If you just want to convert the type of the values stored by the sparse
operator `A` to type `T` while keeping its storage format, any of the following
will do the job:

```julia
SparseOperator{T}(A)
CompressedSparseOperator{Any,T}(A)
convert(SparseOperator{T}, A)
convert(CompressedSparseOperator{Any,T}, A)
```

As can be seen, specifying `Any` for the format parameter in
`CompressedSparseOperator` is a mean to keep the same storage format.



## Other methods

A sparse operator `S` can be reshaped:

```julia
reshape(S, rowdims, coldims)
```

where `rowdims` and `coldims` are the new list of dimensions for the rows and
the columns, their product must be equal to the product of the former lists of
dimensions (which means that you cannot change the number of elements of the
input and output of a sparse operator).  The reshaped sparse operator and `S`
share the arrays of non-zero coefficients and corresponding row and column
indices, hence reshaping is a fast operation.

The non-zero coefficients of a sparse operator `S` can be unpacked into
a provided array `A`:

```julia
unpack!(A, S; flatten=false) -> A
```

Keyword `flatten` specifies whether to only consider the length of `A` instead
of its dimensions.  In any cases, `A` must have as many elements as `length(S)`
and standard linear indexing.  Just call `Array(S)` to unpack the coefficients
of the sparse operator `S` without providing the destination array or
`Array{T}(S)` if you want to a specific element type `T`.



[LinearInterpolators]: https://github.com/emmt/LinearInterpolators.jl
[SparseArrays]: https://docs.julialang.org/en/latest/stdlib/SparseArrays/#Sparse-Arrays-1
[sparse]: https://docs.julialang.org/en/latest/stdlib/SparseArrays/#SparseArrays.sparse
[SparseMatrixCSC]: https://docs.julialang.org/en/latest/stdlib/SparseArrays/#SparseArrays.SparseMatrixCSC
