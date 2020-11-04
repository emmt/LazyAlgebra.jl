# Sparse operators

A sparse operator ([`SparseOperator`](@ref)) in LazyAlgebra is the
generalization of a sparse matrix.  Like a [`GeneralMatrix`](@ref), rows and
columns may be multi-dimensional.  However unlike a [`GeneralMatrix`](@ref), a
sparse operator only stores its *structural non-zero* entries and thus require
fewer memory and is usually faster.


## Generalities

There are many different possibilities for storing a sparse operator, hence
`SparseOperator{T,M,N}` is an abstract type inherited by the concrete types
implementing compressed sparse storage in various formats.  Parameter `T` is
the type of the elements while parameters `M` and `N` are the number of
dimensions of the *rows* and of the *columns* respectively.  Objects of this
kind are a generalization of sparse matrices in the sense that they implement
linear mappings which can be applied to `N`-dimensonal arguments to produce
`M`-dimensional results (as explained below).  The construction of a sparse
operator depends on its storage format.  Several concrete implementations are
provided: [*Compressed Sparse Row* (CSR)](#Compressed-sparse-row-format),
[*Compressed Sparse Column* (CSC)](#Compressed-sparse-column-format) and
[*Compressed Sparse Coordinate* (COO)](#Compressed-sparse-coordinate-format).

Sparse operators can be used as iterators, the value returned by the iterator
is a 3-tuple `(v,i,j)` with the value, the linear row index and the linear
column index of the entry.  For instance:

```julia
for (v,i,j) in A
    println("A[$i,$j] = $v")
end
```

The following methods are generally applicable to any sparse linear operator
`A`:

- `eltype(A)` yields `T`, the type of the elements of `A`;

- `row_size(A)` yields an `M`-tuple of `Int`: the size of the rows of `A`, this
  is equivalent to `output_size(A)`;

- `col_size(A)` yields an `N`tuple of `Int`: the size of the columns of `A`,
  this is equivalent to `input_size(A)`;

- `nrows(A)` yields `prod(row_size(A))`, the equivalent number of rows of `A`;

- `ncols(A)` yields `prod(col_size(A))`, the equivalent number of columns of
  `A`.

- `ndims(A)` yields `M + N` the number of dimensions of the regular array
   corresponding to the sparse linear operator `A`;

- `size(A)` yields `(row_size(A)..., col_size(A)...)` the size of the regular
   array corresponding to the sparse linear operator `A`;

- `length(A)` yields `prod(size(A))` the number of elements of the regular
   array corresponding to the sparse linear operator `A`;

- `nnz(A)` yields the number of *structural non-zeros* in `A`;

- `nonzeros(A)` yields the vector of *structural non-zeros* in `A`.

The *structural non-zeros* are the entries stored by the sparse structure, they
may or not be equal to zero, un-stored entries are always considered as being
equal to zero.

As can be seen above, `eltype`, `ndims`, `size` and `length` yield the same
results as if applied to the multi-dimensional array corresponding to the
sparse operator.

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
i.e. `size(y) = row_size(A)`.  The following pieces of code show how `w = A*x`
and `z = A'*y` could be computed for the sparse operator `A` using the methods
exported by `LazyAlgebra`:

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
specific storage format of a compressed sparse linear operator (see
[`AbstractSparseOperatorCSR`](@ref), [`AbstractSparseOperatorCSC`](@ref) and
[`AbstractSparseOperatorCOO`](@ref)).

!!! note
    For now, row and column indices are restricted to be linear indices and
    arguments to the linear mappings implemented by compressed sparse linear
    operators must be arrays with conventional indexing (1-based linear
    indices).





















> ## Construction
>
> A sparse operator can be built as follows:
>
> ```julia
> SparseOperator(I, J, C, rowdims, coldims)
> ```
>
> where `I` and `J` are row and column indices of the non-zero coefficients whose
> values are specified by `C` and with `rowdims` and `coldims` the dimensions of
> the rows and of the columns.  Appart from the fact that the rows and columns
> may be multi-dimensional, this is very similar to the [`sparse`][sparse] method
> in [`SparseArrays`][SparseArrays] standard Julia module.
>
> Another possibility is to build a sparse operator from an array or from
> a sparse matrix:
>
> ```julia
> S = SparseOperator(A)
> ```
>
> where `A` is an array (of an y sub-type of `AbstractArray`) or a sparse matrix
> (of type [`SparseMatrixCSC`][SparseMatrixCSC]).  If `A` is an array with more
> than 2 dimensions, the number `n` of dimensions correspondinrg to the *rows* of
> the operator can be specified:
>
> ```julia
> S = SparseOperator(A, n)
> ```
>
> If not specified, `n=1` is assumed.
>
> A sparse operator can be converted to a regular array, to a regular matrix or
> to a sparse matrix.  Assuming `S` is a `SparseOperator`, convertions to other
> representations are done by:
>
> ```julia
> A = Array(S)     # convert S to an array
> M = Matrix(S)    # convert S to a matrix
>
> using SparseArrays
> sp = sparse(S)   # convert S to a sparse matrix
> ```
>
> !!! note
>     If the sparse operator `S` has multi-dimensional columns/rows, these
>     dimensions are preserved when `S` is converted to an array but are
>     silently flattened when `S` is converted to a matrix or to a sparse
>     matrix.
>
> Package [LinearInterpolators][LinearInterpolators] provides a
> `SparseInterpolator` which is a LazyAlgebra `LinearMapping` and which can also
> be converted to a `SparseOperator` (sse the documentation of this package).
>
>
> ## Usage
>
> A sparse operator can be used as any other LazyAlgebra linear mapping, *e.g.*,
> `S*x` yields the result of applying the sparse operator `S` to `x` (unless `x`
> is a scalar, see below).
>
> A sparse operator can be reshaped:
>
> ```julia
> reshape(S, rowdims, coldims)
> ```
>
> where `rowdims` and `coldims` are the new list of dimensions for the rows and
> the columns, their product must be equal to the product of the former lists of
> dimensions.  The reshaped sparse operator and `S` share the arrays of non-zero
> coefficients and corresponding row and column indices.
>
> Left or right composition of a by a sparse operator by
> [`NonuniformScalingOperator`](@ref) (with comptable dimensions) yields another
> sparse operator whith same row and column indices but scaled coefficients.  A
> similar simplification is performed when a sparse operator is left or right
> multiplied by a scalar.
>
> The non-zero coefficients of a sparse operator `S` can be unpacked into
> a provided array `A`:
>
> ```julia
> unpack!(A, S) -> A
> ```
>
> where `A` must have the same element type as the coefficients of `S` and the
> same number of elements as the the products of the row and of the column
> dimensions of `S`.  Unpacking is perfomed by adding the non-zero coefficients
> of `S` to the correponding elements of `A` (or using the `|` operator for
> boolean elements).  Hence unpacking into an array of zeros with appropriate
> dimensions yields the same result as `Array(S)`.





## Compressed sparse row format

Sparse operators in *Compressed Sparse Row* (CSR) format store the significant
entries in a row-wise order, as a vector of values, a vector of corresponding
linear column indices and a vector of offsets indicating, for each row, the
range of indices in the vectors of values and of column indices.

### Construction of sparse operators in CSR format

`AbstractSparseOperatorCSR{T,M,N}` is an abstract sub-type of
`SparseOperator{T,M,N}` and is inherited by the concrete types
implementing CSR storage of sparse operators.
[`SparseOperatorCSR`](@ref) is such a concrete implementation and let you
build a CSR operator as:

```julia
SparseOperatorCSR{T,M,N}(A, sel = (v,i,j) -> (v != zero(v)))
```

which yields a sparse operator in CSR format whose structure and values are
taken from the selected entries in array `A`.  Parameter `T` is the type of the
values stored by the sparse linear operator,  parameters `M` and `N` are the
number of leading and trailing dimensions of `A` to group to form the
equivalent *rows* and *columns* of the sparse linear operator.  See Section
[Generalities](#Generalities) for a definition of equivalent rows and
columns and for explanations about aplying the linear mapping (or its adjoint)
implemented by a sparse operator.

Optional argument `sel` is a selector function which is called as `sel(v,i,j)`
with `v`, `i` and `j` the value, the row and the column linear indices for each
entries of `A` and which is assumed to yield `true` for the entries of `A` to
be selected in the sparse structure and `false` for the entries of `A` to
discard.  The default selector is such that all non-zeros of `A` are selected.
As an example, to select the non-zeros of the lower triangular part of `A`, the
constructor can be called as:

```julia
SparseOperatorCSR{T,M,N}(A, (v,i,j) -> ((i ≥ j)&(v != zero(v))))
```

The equality `M + N = ndims(A)` must hold, so it is sufficient to only
specifify `M`:

```julia
SparseOperatorCSR{T,M}(A, sel)
```

If `A` is a simple matrix, that is a two-dimensional array, parameters `M` and
`N` must be equal to 1 and may be omitted.  In that case, the type `T` may also
be omitted and is `eltype(A)` by default.  The simplest way to create a sparse
operator in CSR format from a 2-dimensional array `A` is therefore:


```julia
SparseOperatorCSR(A)
```

The components of the CSR storage can also be directly provided:

```julia
SparseOperatorCSR(vals, cols, offs, rowsiz, colsiz)
```

or

```julia
SparseOperatorCSR{T}(vals, cols, offs, rowsiz, colsiz)
```

to force the element type of the result.  Here, `vals` is the vector of values
of the sparse entries, `cols` is an integer valued vector of the linear column
indices of the sparse entries, `offs` is a column-wise table of offsets in
these arrays, `rowsiz` and `colsiz` are the sizes of the row and column
dimensions.  The entries values and respective linear column indices of the
`i`-th row are given by `vals[k]` and `cols[k]` with `k ∈ offs[j]+1:offs[j+1]`.
The linear row index `i` is in the range `1:m` where `m = prod(rowsiz)` is the
equivalent number of rows.

For efficiency reasons, sparse operators are currently limited to *fast* arrays
because they can be indexed linearly with no loss of performances.  If `vals`,
`cols` and/or `offs` are not fast arrays, they will be automatically converted
to linearly indexed arrays.

A sparse linear operator in CSR format can also be built given a sparse linear
operator `A` in another storage format:

```julia
SparseOperatorCSR(A)
```

or

```julia
SparseOperatorCSR{T}(A)
```

to force the element type of the result.  If `A` is in Compressed Sparse
Coordinate (COO) format, entries are sorted and duplicates merged.  See
[`SparseOperatorCSR`](@ref) and [`SparseOperatorCOO`](@ref) for other storage
formats.


### Accessing the entries of a sparse operator in CSR format

The following piece of code shows how to navigate into a CSR sparse operator
`A` and retrieve all values `v` and their respective row `i` and column `j`
indices:

```julia
using LazyAlgebra.SparseMethods
for i in each_row(A)        # loop over row index
    for k in each_off(A, i) # loop over structural non-zeros in this row
        j = get_col(A, k)   # get column number of entry
        v = get_val(A, k)   # get value of entry
     end
end
```

The above code delivers the entries in their storage order.  Obviously a CSR
operator is in row-major storage order.  It is also possible to use the sparse
operator as an iterator (see [Generalities](#Generalities) above) to write
simpler loops to the cost of efficiency .  The low-level methods `each_row`,
`each_off`, `get_col` and `get_val` are not automatically exported by
`LazyAlgebra`, this is the purpose of the statement `using
LazyAlgebra.SparseMethods`.


## Compressed sparse column format

`AbstractSparseOperatorCSC{T,M,N}` is an abstract sub-type of
`SparseOperator{T,M,N}` and is inherited by the concrete types implementing
*Compressed Sparse Column* (CSC) storage of sparse linear operators.

CSC operators store the significant entries in a column-wise order, as a vector
of values, a vector of corresponding linear row indices and a vector of offsets
indicating, for each column, the range of indices in the vectors of values and
of row indices.

The following piece of code shows how to navigate into a CSC operator `A` and
retrieve all values `v` and their respective row `i` and column `j` indices:

```julia
using LazyAlgebra.SparseMethods
for j in each_col(A)
    for k in each_off(A, j)
        i = get_row(A, k)
        v = get_val(A, k)
     end
end
```

The above code delivers the entries in their storage order.  Obviously a CSC
operator is in column-major storage order.  It is also possible to use the
sparse operator as an iterator (see [`SparseOperator`](@ref)) to write simpler
loops.  The low-level methods `each_col`, `each_off`, `get_row` and `get_val`
are not automatically exported by `SparseOperators`, this is the purpose of the
statement `using LazyAlgebra.SparseMethods`.

See [`SparseOperatorCSC`](@ref) for a concrete implementation of this
compressed sparse format.

### Accessing the entries of a sparse operator in CSR format



## Compressed sparse coordinate format

`AbstractSparseOperatorCOO{T,M,N}` is an abstract sub-type of
`SparseOperator{T,M,N}` and is inherited by the concrete types
implementing *Compressed Sparse Coordinate* (COO) storage of sparse linear
operators.

Sparse operators in *Compressed Sparse Coordinate* (COO) format store the
significant entries in no particular order, as a vector of values, a vector of
linear row indices and a vector of linear column indices.  It is even possible
to have repeated entries.  This format is very useful to build a sparse linear
operator.  It can be converted to a more efficient format like *Compressed
Sparse Column* or *Compressed Sparse Row* for fast application of the sparse
linear mapping or of its adjoint.

The following piece of code shows how to navigate into a COO sparse operator
`A` and retrieve all values `v` and their respective row `i` and column `j`
indices:

```julia
using LazyAlgebra.SparseMethods
for k in each_off(A)   # loop over structural non-zeros
     v = get_val(A, k) # get value of entry
     i = get_row(A, k) # get row number of entry
     j = get_col(A, k) # get column number of entry
end
```

Note that no specific order is assumed for the entries, duplicates are allowed.
Duplicates are summed together when converting to a more efficient format.  It
is also possible to use the sparse linear operator as an iterator (see
[`SparseOperator`](@ref)) to write simpler loops.  The low-level methods
`each_off`, `get_val`, `get_row` and `get_col` are not automatically exported
by `LazyAlgebra.SparseOperators`, this is the purpose of the statement `using
LazyAlgebra.SparseMethods`.

[`SparseOperatorCOO`](@ref) for a concrete implementation of this
compressed sparse format.

## Extracting the sparse structure only

If the purpose of a sparse linear operator is to store a compressed sparse
structure in CSR or CSC format, a uniform vector (from the
[`StructuredArrays`](https://github.com/emmt/StructuredArrays.jl) package) of
`true` values can be used to represent the selected values.  There are special
constructor calls for that:

```julia
SparseOperatorCSR{Bool,M,N,UniformVector{Bool}}(args...)
SparseOperatorCSS{Bool,M,N,UniformVector{Bool}}(args...)
SparseOperatorCOO{Bool,M,N,UniformVector{Bool}}(args...)
```

which yield the sparse structure (respectively in a compressed sparse column,
row or coordinate format) made out of arguments `args...`.  The result is a
compressed sparse operator whose values are an immutable uniform vector
of `true` values requiring no storage.


[LinearInterpolators]: https://github.com/emmt/LinearInterpolators.jl
[SparseArrays]: https://docs.julialang.org/en/latest/stdlib/SparseArrays/#Sparse-Arrays-1
[sparse]: https://docs.julialang.org/en/latest/stdlib/SparseArrays/#SparseArrays.sparse
[SparseMatrixCSC]: https://docs.julialang.org/en/latest/stdlib/SparseArrays/#SparseArrays.SparseMatrixCSC
