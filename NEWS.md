# Visible changes in LazyAlgebra

Future:

* Exported methods and types have been limited to the ones for the end-user.
  Use `using LazyAlgebra.LowLevel` to get the others.

* The notation `A'` is only allowed for linear mappings and always denote the
  adjoint of `A`.  The Jacobian of a non-linear mapping is a linear mapping
  (not clear how to apply it though) whose adjoint can be taken.  Call `∇(A,x)`
  or `jacobian(A,x)` to build the representation of the Jacobian of a mapping
  `A` at the point `x`. The Jacobian of a linear-mapping `A` is `A` whatever `x`.

Done:

* `Gram` type is no longer an alias, it is now part of the *decorated types*
  and some constructions are automaticaly recognized as valid Gram operators.
  Making this work for more complex constructions (like sums and compositions)
  would require to change the simplification rules (notably for the adjoint of
  such constructions).

* New `Jacobian` decoration type to denote the Jacobian of a non-linear mapping
  and associated `jacobian` (aliased to `∇`) method.  For now, `adjoint` and
  `jacobian` behave the same.  In part this is because I could not figure out
  how to directly extend the ' postfix operator and could only extend the
  `adjoint` method.

* Methods `has_oneto_axes`, `densearray`, `densevector` and `densematrix` have
  been replaced by `has_standard_indexing` and `to_flat_array` from `ArrayTools`.

* The exported constant `I = Identity()` has been renamed as `Id` to avoid
  conflicts with standard `LinearAlgebra` package.  `Id` is systematically
  exported while `I` was only exported if not already defined in the `Base`
  module.  The constant `LinearAlgebra.I` and, more generally, any instance of
  `LinearAlgebra.UniformScaling` is recognized by LazyAlgebra in the sense that
  they behave as the identity when combined with any LazyAlgebra mapping.

* `operand` and `operands` are deprecated in favor of `unveil` and `terms`
  which are less confusing.  The `terms` method behaves exactly like the former
  `operands` method.  Compared to `operand`, the `unveil` method has a better
  defined behavior: for a *decorated* mapping (that is an instance of
  `Adjoint`, `Inverse` or `InverseAdjoint`), it yields the embedded mapping;
  for other LazyAlgebra mappings (including scaled ones), it returns its
  argument; for an instance of `LinearAlgebra.UniformScaling`, it returns the
  equivalent LazyAlgebra mapping (that is `λ⋅Id`).  To get the mapping embedded
  in a scaled mapping, call the `unscaled` method.

* `unscaled` is introduced as the counterpart of `multiplier` so that
  `multiplier(A)*unscaled(A) === A` always holds.  Previously it was wrongly
  suggested to use `operand` (now `unveil`) for that but, then the strict
  equality was only true for `A` being a scaled mapping.  These methods also
  work for instances of `LinearAlgebra.UniformScaling`.

* `NonuniformScalingOperator` deprecated in favor of `NonuniformScaling`.

* In most cases, complex-valued arrays and multipliers are supported.

* Argument `scratch` is no longer optional in low-level `vcreate`.

* Not so well defined `HalfHessian` and `Hessian` have been removed
  (`HalfHessian` is somewhat equivalent to `Gram`).

* New `gram(A)` method which yields `A'*A` and alias `Gram{typeof(A)}` to
  represent the type of this construction.

* The `CroppingOperators` sub-module has been renamed `Cropping`.

* Add cropping and zero-padding operators.

* Left multiplication by a scalar and left/right multiplication by a
  non-uniform scaling (a.k.a. diagonal operator) is optimized for sparse
  and non-uniform scaling operators.

* Provide `unpack!` method to unpack the non-zero coefficients of a sparse
  operator and extend `reshape` to be applicable to a sparse operator.

* Make constructor of a sparse operator (`SparseOperator`) reminiscent of the
  `sparse` method.  Row and column dimensions can be a single scalar.

* Provide utility method `dimensions` which yields a dimension list out of its
  arguments and associated union type `Dimensions`.

* A sparse operator (`SparseOperator`) can be converted to a regular array or
  to a sparse matrix (`SparseMatrixCSC`) and reciprocally.

* Trait constructors now return trait instances (instead of type).  This is
  more *natural* in Julia and avoid having different method names.

* Skip bound checking when applying a `SparseOperator` (unless the operator
  structure has been corrupted, checking the dimensions of the arguments is
  sufficient to insure that inidices are correct).

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

* `contents`, too vague, has been suppressed and replaced by `operands` or
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

* `promote_scalar` has been modified and renamed as `promote_multipler`.

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
  FFT is not.  Another example: `NonuniformScaling` is self-adjoint if
  its coefficients are reals, not if they are complexes. This also overcomes
  the fact that multiple heritage is not possible in Julia.

* The `apply!` method has been rewritten to allow for optimized combination to
  do `y = α*Op(A)⋅x + β*y` (as in LAPACK and optimized if scalars have values
  0, ±1):

  ```julia
  apply!(α::Real, Op::Type{<:Operations}, A::LinearMapping, x, β::Real, y)
  apply!(β::Real, y, α::Real, Op::Type{<:Operations}, A::LinearMapping, x)
  ```
