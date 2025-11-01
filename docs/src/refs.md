# Reference

The following provides detailed documentation about types and methods provided by the
`LazyAlgebra` package. This information is also available from the REPL by typing `?`
followed by the name of a method or a type.

## Traits and methods for linear operators

### Input shape of linear operators

The trait indicating whether the input shape of an operator is known is implemented by:

```@docs
LazyAlgebra.InputShape
LazyAlgebra.InputShapeUnknown
LazyAlgebra.HasInputShape
```

For linear operators with known input shape, the following methods are available:

```@docs
LazyAlgebra.input_shape
LazyAlgebra.input_ndims
LazyAlgebra.input_axes
LazyAlgebra.input_size
LazyAlgebra.input_length
```

### Output shape of linear operators

The output shape of `A*x` is inferred by:

```@docs
LazyAlgebra.output_axes(::Operator,::AbstractArray)
```

The trait indicating whether the output shape of an operator is known is implemented by:

```@docs
LazyAlgebra.OutputShape
LazyAlgebra.OutputShapeUnknown
LazyAlgebra.HasOutputShape
```

For linear operators with known output shape, the following methods are available:

```@docs
LazyAlgebra.output_shape
LazyAlgebra.output_ndims
LazyAlgebra.output_axes(::Operator)
LazyAlgebra.output_size
LazyAlgebra.output_length
```

### Input element type

The trait indicating whether the input element type for an operator is known and related
methods are implemented by:

```@docs
LazyAlgebra.InputEltype
LazyAlgebra.InputEltypeUnknown
LazyAlgebra.HasInputEltype
LazyAlgebra.input_eltype
```

### Output element type

The trait indicating whether the output element type for an operator is known and related
methods are implemented by:

```@docs
LazyAlgebra.OutputEltype
LazyAlgebra.OutputEltypeUnknown
LazyAlgebra.HasOutputEltype
LazyAlgebra.output_eltype
```

### Constructions

```@docs
LazyAlgebra.Adjoint
LazyAlgebra.Transpose
LazyAlgebra.Conjugate
LazyAlgebra.Inverse
LazyAlgebra.InverseAdjoint
LazyAlgebra.Sum
LazyAlgebra.Prod
LazyAlgebra.Scaled
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
