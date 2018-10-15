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
