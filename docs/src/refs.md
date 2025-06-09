# Reference

The following provides detailed documentation about types and methods provided by the
`LazyAlgebra` package. This information is also available from the REPL by typing `?`
followed by the name of a method or a type.

## Methods for linear operators

```@docs
nrows
ncols
row_size
col_size
```

## Sparse operators

### Types and compressed storage formats

```@docs
SparseOperator
CompressedSparseOperator
SparseOperatorCOO
SparseOperatorCSC
SparseOperatorCSR
```

### Methods

```@docs
nonzeros
nnz
LazyAlgebra.SparseOperators.unpack!
```

### Low-level interface

These methods are provided by `using LazyAlgebra.SparseMethods`.

```@docs
LazyAlgebra.each_row_index
LazyAlgebra.each_col_index
LazyAlgebra.each_nz_index
LazyAlgebra.row_index
LazyAlgebra.row_indices
LazyAlgebra.col_index
LazyAlgebra.col_indices
LazyAlgebra.offsets
```
