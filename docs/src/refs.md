# Reference

The following provides detailled documentation about types and methods provided
by the `LazyAlgebra` package.  This information is also available from
the REPL by typing `?` followed by the name of a method or a type.

## Methods for linear mappings

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
LazyAlgebra.SparseOperators.unpack!
```

### Low-level interface

These methods are provided by `using LazyAlgebra.SparseMethods`.

```@docs
LazyAlgebra.SparseMethods.each_row
LazyAlgebra.SparseMethods.each_col
LazyAlgebra.SparseMethods.each_off
LazyAlgebra.SparseMethods.get_row
LazyAlgebra.SparseMethods.get_rows
LazyAlgebra.SparseMethods.get_col
LazyAlgebra.SparseMethods.get_cols
LazyAlgebra.SparseMethods.get_val
LazyAlgebra.SparseMethods.get_vals
LazyAlgebra.SparseMethods.set_val!
LazyAlgebra.SparseMethods.get_offs
LazyAlgebra.SparseMethods.copy_rows
LazyAlgebra.SparseMethods.copy_cols
LazyAlgebra.SparseMethods.copy_vals
```
