* Provide `unpack!` method to unpack the non-zero coefficients of a sparse
  operator and extend `reshape` to be applicable to a sparse operator.

* Make constructor of a sparse operator (`SparseOperator`) reminiscent of the
  `sparse` method.  Row and column dimensions can be a single scalar.

* Provide utility method `makedims` which yields a dimension list out of its
  arguments.

* A sparse operator (`SparseOperator`) can be converted to a regular array or
  to a sparse matrix (`SparseMatrixCSC`) and reciprocally.

* Trait constructors now return trait instances (instead of type).  This is
  more *natural* in Julia and avoid having different method names.

* Skip bound checking when applying a `SparseOperator` (unless the operator
  structure has been corrupted, checking the dimensions of the arguments is
  sufficient to insure that inidices are correct).

* Methods `is_flat_array`, `has_oneto_axes`, `densearray`, `densevector` and
  `densematrix` have been deprecated in favor of `isflatarray`,
  `has_standard_indexing`, `flatarray`, `flatvector` and `flatmatrix`.

* Provide `lgemv` and `lgemv!` for *Lazily Generalized Matrix-Vector
  mutiplication* and `lgemm` and `lgemm!` for *Lazily Generalized Matrix-Matrix
  mutiplication*.  The names of these methods are reminiscent of `xGEMV` and
  `xGEMM` BLAS subroutines in LAPACK (with `x` the prefix corresponding to the
  type of the arguments).

* Deprecated `fastrange` is replaced by `allindices` which is extended to
  scalar dimension and index intervals.

* Complete rewrite of the rules for simplying complex constructions involving
  compositions and linear combination of mappings.

* Add rule for left-division by a scalar.

* `UniformScalingOperator` has been suppressed (was deprecated).

* `show` has been extend for mapping constructions.

* `contents`, too vague, is about to be suppressed and replaced by `operands` or
  `operand`.  Getter `multiplier` is provided to query the multiplier of a
  scaled mapping.  Methods `getindex`, `first` and `last` are extended.  In
  principle, direct reference to a field of any base mapping structures is no
  longer needed.

* The multiplier of a scaled mapping can now be any number although applying
  linear combination of mappings is still limited to real-valued multipliers.

* Add `fftfreq`, `rfftdims`, `goodfftdim` and `goodfftdims` in `LazyAlgebra.FFT`
  and re-export `fftshift` and `ifftshift` when `using LazyAlgebra.FFT`.

* Add `is_same_mapping` to allow for automatic simplications when building-up
  sums and compositions.

* Optimal, an more general, management of temporaries is now done via the
  `scratch` argument of the `vcreate` and `apply!` methods.  `InPlaceType`
  trait and `is_applicable_in_place` method have been removed.

* `promote_scalar` has been modified and renamed as `convert_multipler`.

* Provide `SimpleFiniteDifferences` operator.

* Provide `SparseOperator`.

* `LinearAlgebra.UniformScaling` can be combined with mappings in LazyAlgebra.

* `UniformScalingOperator` has been deprecated in favor of a `Scaled` version
  of the identity.

* Compatible with Julia 0.6, 0.7 and 1.0.

* Provide (partial) support for complex-valued arrays.

* Traits replace abstract types such as `Endomorphism`, `SelfAdjointOperator`,
  etc.  Some operator may be endomorphisms or not.  For instance the
  complex-to-complex `FFTOperator` is an endomorphism while the real-to-complex
  FFT is not.  Another example: `NonuniformScalingOperator` is self-adjoint if
  its coefficients are reals, not if they are complexes. This also overcomes
  the fact that multiple heritage is not possible in Julia.

* The `apply!` method has been rewritten to allow for optimized combination to
  do `y = α*Op(A)⋅x + β*y` (as in LAPACK and optimized if scalars have values
  0, ±1):

  ```julia
  apply!(α::Real, Op::Type{<:Operations}, A::LinearMapping, x, β::Real, y)
  apply!(β::Real, y, α::Real, Op::Type{<:Operations}, A::LinearMapping, x)
  ```
