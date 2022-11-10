# Visible changes in LazyAlgebra

## Wish list for future developments

* Optimizations and simplifications of expressions involving mappings belong to
  two categories:
  * Automatic simplifications performed by`LazyAlgebra`. These can change the
    values of multipliers but must not change the coefficients of the mappings.
    Sadly, it is not possible to have every possible automatic simplifications
    be type-stable.
  * The user may explicitly call `simplify(A)` to apply further simplifications
    that may change the coefficients of the mappings in `A`. For example,
    assuming `a` is an array, `inv(Diag(a))` automatically yields
    `Inverse(Diag(a))` while `simplify(inv(Diag(a)))` yields `Diag(1 ./ a)`.
    These simplifications may not be type-stable. For example the multiplier of
    a scaled mapping becoming equal to one can be eliminated.

* Calling BLAS should be avoided in some cases, either because BLAS is slower
  than optimized Julia code, or because BLAS may use more than one thread in
  inappropriate places (e.g., Julia multi-threaded code).

* As far as possible, make the code more agnostic of the element type of the
  arguments.  This would be useful to deal with arrays whose elements have
  non-standard numerical types as physical quantities in the `Unitful` package.

## Branch 0.3

### Breaking changes

* Abstract type `Mapping{L}` has a built-in parameter indicating whether the
  mapping is linbear of not. This breaks compatibility but simplifies a lot
  many parts of the code and makes the linear trait decidable at compile time.

* **Traits** must be type-stable. That is deciding the value of a specific
  mapping trait must be done on the sole basis of the type of the mapping. The
  linear trait is useful and trivial to propagate ub constructions (this is
  part of the decision to add it as a parameter to the abstract mapping type
  `Mapping`). Other traits, such as the morphism type, may not be so useful or
  can be tricky to determine, they may be eliminated in a near future.

### Other changes

* Methods `multipler`, `unscaled`, `terms`, `nterms`, and `unveil` can be
  applied to a mapping type to yield the corresponding type.

## Branch 0.2

### Version 0.2.5

* Make `set_val!` for sparse operators returns the same result as `setindex!`.

### Version 0.2.3

* Simplify and generalize `vfill!` and `vzero!` to be able to work with
  `Unitful` elements.

* Automatically specialize `multiplier_type` for `Unitful.AbstractQuantity`.

### Version 0.2.2

* Improve `promote_multiplier` and make it easy to extend.  The work done by
  `promote_multiplier` is break in sevral functions: `multiplier_type(x)`
  yields the *element type* corresponding to `x` (which can be a number, an
  array of numbers, or a number type), `multiplier_floatingpoint_type(args...)`
  combines the types given by `multiplier_type` for all `args...` to yield a
  concrete floating-point type.  The method `multiplier_type` is intended to be
  extended by other packages.

### Version 0.2.1

* Replace `@assert` by `@certify`.  Compared to `@assert`, the assertion made
  by `@certify` may never be disabled whatever the optimization level.

* Provide default `vcreate` method for Gram operators.

### Version 0.2.0

* Sub-module `LazyAlgebra.Foundations` (previously
  `LazyAlgebra.LazyAlgebraLowLevel`) exports types and methods needed to extend
  or implement `LazyAlgebra` mappings.

* The finite difference operator was too limited (finite differences were
  forcibly computed along all dimensions and only 1st order derivatves were
  implemented) and slow (because the leading dimension was used to store the
  finite differences along each dimension).  The new family of operators can
  compute 1st or 2nd derivatives along all or given dimensions.  The last
  dimension of the result is used to store finite differences along each chosen
  dimensions; the operators are much faster (at least 3 times faster for
  200×200 arrays for instance).  Applying the Gram composition `D'*D` of a
  finite difference operator `D` is optimized and is about 2 times faster than
  applying `D` and then `D'`. Type `SimpleFiniteDifferences` is no longer
  available, use `Diff` instead (`Diff` was available as a shortcut in previous
  releases).

## Branch 0.1

### Version 0.1.0

* New rules: `α/A -> α*inv(A)`.

* Exported methods and types have been limited to the ones for the end-user.
  Use `using LazyAlgebra.LazyAlgebraLowLevel` to use low-level symbols.

* Large sub-package for sparse operators which are linear mappings with few
  non-zero coefficients (see doc. for `SparseOperator` and
  `CompressedSparseOperator`).  All common compressed sparse storage formats
  (COO, CSC and CSR) are supported and easy conversion between them is
  provided.  Generalized matrix-vector multiplication is implemented and is as
  fast or significantly faster than with `SparseArrays.SparseMatrixCSC`.

* Method `∇(A,x)` yields the Jacobian of the mapping `A` at the variables `x`.
  If `A` is a linear-mapping, then `∇(A,x)` yields `A` whatever `x`.  The new
  type `Jacobian` type is used to denote the Jacobian of a non-linear mapping.
  The notation `A'`, which is strictly equivalent to `adjoint(A)`, is only
  allowed for linear mappings and always denote the adjoint (conjugate
  transpose) of `A`.

* Method `gram(A)` yields `A'*A` for the linear mapping `A`.  An associated
  *decorated type* `Gram` is used to denote this specific expression and some
  constructions are automaticaly recognized as valid Gram operators.  Making
  this work for more complex constructions (like sums and compositions) would
  require to change the simplification rules (notably for the adjoint of such
  constructions).

* Methods `has_oneto_axes`, `densearray`, `densevector` and `densematrix` have
  been replaced by `has_standard_indexing` and `to_flat_array` from
  `ArrayTools`.

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
