# Sparse operators

A sparse operator ([`SparseOperator`](@ref)) in LazyAlgebra is the
generalization of a sparse matrix.  Like a [`GeneralMatrix`](@ref), rows and
columns may be multi-dimensional.


## Construction

A sparse operator can be built as follows:

```julia
SparseOperator(I, J, C, rowdims, coldims)
```

where `I` and `J` are row and column indices of the non-zero coefficients whose
values are specified by `C` and with `rowdims` and `coldims` the dimensions of
the rows and of the columns.  Appart from the fact that the rows and columns
may be multi-dimensional, this is very similar to the [`sparse`][sparse] method
in [`SparseArrays`][SparseArrays] standard Julia module.

Another possibility is to build a sparse operator from an array or from
a sparse matrix:

```julia
S = SparseOperator(A)
```

where `A` is an array (of an y sub-type of `AbstractArray`) or a sparse matrix
(of type [`SparseMatrixCSC`][SparseMatrixCSC]).  If `A` is an array with more
than 2 dimensions, the number `n` of dimensions corresponding to the *rows* of
the operator can be specified:

```julia
S = SparseOperator(A, n)
```

If not specified, `n=1` is assumed.

A sparse operator can be converted to a regular array, to a regular matrix or
to a sparse matrix.  Assuming `S` is a `SparseOperator`, convertions to other
representations are done by:

```julia
A = Array(S)     # convert S to an array
M = Matrix(S)    # convert S to a matrix

using SparseArrays
sp = sparse(S)   # convert S to a sparse matrix
```

!!! note
    If the sparse operator `S` has multi-dimensional columns/rows, these
    dimensions are preserved when `S` is converted to an array but are
    silently flattened when `S` is converted to a matrix or to a sparse
    matrix.

Package [LinearInterpolators][LinearInterpolators] provides a
`SparseInterpolator` which is a LazyAlgebra `LinearMapping` and which can also
be converted to a `SparseOperator` (sse the documentation of this package).


## Usage

A sparse operator can be used as any other LazyAlgebra linear mapping, *e.g.*,
`S*x` yields the result of applying the sparse operator `S` to `x` (unless `x`
is a scalar, see below).

A sparse operator can be reshaped:

```julia
reshape(S, rowdims, coldims)
```

where `rowdims` and `coldims` are the new list of dimensions for the rows and
the columns, their product must be equal to the product of the former lists of
dimensions.  The reshaped sparse operator and `S` share the arrays of non-zero
coefficients and corresponding row and column indices.

Left or right composition of a by a sparse operator by
[`NonuniformScalingOperator`](@ref) (with comptable dimensions) yields another
sparse operator whith same row and column indices but scaled coefficients.  A
similar simplification is performed when a sparse operator is left or right
multiplied by a scalar.

The non-zero coefficients of a sparse operator `S` can be unpacked into
a provided array `A`:

```julia
unpack!(A, S) -> A
```

where `A` must have the same element type as the coefficients of `S` and the
same number of elements as the the products of the row and of the column
dimensions of `S`.  Unpacking is perfomed by adding the non-zero coefficients
of `S` to the correponding elements of `A` (or using the `|` operator for
boolean elements).  Hence unpacking into an array of zeros with appropriate
dimensions yields the same result as `Array(S)`.

[LinearInterpolators]: https://github.com/emmt/LinearInterpolators.jl
[SparseArrays]: https://docs.julialang.org/en/latest/stdlib/SparseArrays/#Sparse-Arrays-1
[sparse]: https://docs.julialang.org/en/latest/stdlib/SparseArrays/#SparseArrays.sparse
[SparseMatrixCSC]: https://docs.julialang.org/en/latest/stdlib/SparseArrays/#SparseArrays.SparseMatrixCSC