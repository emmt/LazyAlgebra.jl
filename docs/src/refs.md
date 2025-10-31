# Reference

The following provides detailed documentation about types and methods provided by the
`LazyAlgebra` package. This information is also available from the REPL by typing `?`
followed by the name of a method or a type.

## Methods for linear operators

```@docs
LazyAlgebra.nrows
LazyAlgebra.ncols
LazyAlgebra.row_size
LazyAlgebra.col_size
LazyAlgebra.input_eltype
LazyAlgebra.input_axes
LazyAlgebra.input_size
LazyAlgebra.output_eltype
LazyAlgebra.output_axes
LazyAlgebra.output_size
```

## Constructions

```@docs
LazyAlgebra.Adjoint
LazyAlgebra.Inverse
LazyAlgebra.InverseAdjoint
LazyAlgebra.Sum
LazyAlgebra.Prod
LazyAlgebra.Scaled
```


### Traits

```@docs
LazyAlgebra.InputShape
LazyAlgebra.InputShapeUnknown
LazyAlgebra.HasInputShape
LazyAlgebra.InputEltype
LazyAlgebra.InputEltypeUnknown
LazyAlgebra.HasInputEltype
LazyAlgebra.OutputShape
LazyAlgebra.OutputShapeUnknown
LazyAlgebra.HasOutputShape
LazyAlgebra.OutputEltype
LazyAlgebra.OutputEltypeUnknown
LazyAlgebra.HasOutputEltype
```

## Sparse operators

### Sparse storage formats

```@docs
SparseFormat
CompressedSparseCoordinate
CompressedSparseColumn
CompressedSparseRow
```

### Constructors of sparse operators

```@docs
SparseOperator
SparseOperatorLike
SparseOperatorCOO
SparseOperatorCSC
SparseOperatorCSR
```

### Methods for sparse operators

```@docs
nonzeros
nnz
LazyAlgebra.unpack!
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
