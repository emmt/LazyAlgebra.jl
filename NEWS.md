# User visible changes in `LazyAlgebra`

This page describes the most important changes in `LazyAlgebra`. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to
[Semantic Versioning](https://semver.org).

## Wish list for future developments

* Simplifications that are automatically done by`LazyAlgebra` may change multipliers but
  must not change the coefficients of the mappings. Call `simplify(A)` to apply further
  simplifications that may change the coefficients of the mappings in `A`. For instance,
  assuming `a` is an array, `inv(Diag(a))` automatically yields `Inverse(Diag(a))` while
  `simplify(inv(Diag(a)))` yields `Diag(1 ./ a)`.

* Calling BLAS should be avoided in some cases, either because BLAS is slower than
  optimized Julia code, or because BLAS may use more than one thread in inappropriate
  places (e.g., Julia multi-threaded code).

* As far as possible, make the code more agnostic of the element type of the arguments.
  This would be useful to deal with arrays whose elements have non-standard numerical
  types as physical quantities in the `Unitful` package.

- Functions `vproduct` and `vproduct!` to compute the Hadamar (elementwise) product of two
  *vectors* have been removed. Instead of `vproduct(x, y)`, simply call `Diag(x)*y`, `x .*
  y`, or `@. x*y` to compute this product efficiently. However note that the 2 latter
  statements do not impose that the axes of `x` and `y` be the same. Similarly, instead of
  `vproduct!(dst, x, y)`, simply call `vmul!(dst, Diag(x), y)`, or `@. dst = x*y`.

## Unreleased

This new major version of `LazyAlgebra` introduces a lot of improvements, simplifications,
and changes.

- Non-linear mappings have not been found to be really useful and are no longer supported.
  As a result all operators, their adjoint, their inverse, their sums, their compositions,
  or a mixture of all this are linear operators. This simplifies a lot of things. Abstract
  type `Operator` replaces `LinearMapping`.

- `LazyAlgebra` consider 3 different kinds of objects:

  - **Linear operators** are instances of `Operator` which can be arbitrarily associated
    in sums and compositions. The coefficients of these operators may not be explicitly
    stored. Adjoint, inverse, sums and compositions of operators are lazily remembered.

  - **Vectors** are instances of `AbstractArray` which can be multiplied (in a similar
    sense as the matrix-vector multiplication) by operators or linearly combined to
    produce other *vectors*.

  - **Multipliers** are scalar factors represented by instances of `Number` and which can
    multiply (or scale) operators and vectors. In operations that involve the scaling of a
    vector by a scalar factor, the storage type (not the units if any) of the factor is
    converted to be the same as the floating-point precision of the vector. Hence no
    unwanted conversion occurs due to the precision of a multiplier.

- Using neutral numbers (from the [`Neutrals.jl`](https://github.com/emmt/LazyAlgebra.jl)
  package) for the multipliers considerably simplifies the code and reduces its size and
  the number of alternatives to consider. For example, [`src/diff.jl`](src/diff.jl) is now
  around 800 lines, compared to 1300 previously. This reduction is without sacrificing
  performances and with a gain in generality as the methods accept dimensionful values.

- Number may have units and complex numbers should be fully supported with their usual
  meaning in linear algebra.


### Removed

- Non-linear mappings are no longer supported. As a result, the `Jacobian` type, the
  `jacobian`, `∇`, `primitive`, `variables`, and `is_linear` functions, and the
  `LinearType` trait and its sub-types `Linear`, and `NonLinear` have been suppressed.

- Type `NonuniformScaling` replaced by its alias `Diag`.

- The `unveil` function has been removed. Call `parent(A)` for adjoint, inverse, or Gram
  operators and `parent(parent(A))` on inverse-adjoint operators.

- The `coefficients` function has been removed. Call `parent(A)` for diagonal operator or
  pseudo-matrix `A`.

- `GeneralMatrix` has been replaced by `FlexibleMatrix` which is a special case of
  the `PseudoMatrix` operator.


### Changed

- Abstract type `LinearMapping` renamed `Operator`.

- Scalar multipliers, array elements, and operator coefficients may have units.

- Methods `vmul`, `vmul!`, and `LazyAlgebra.unsafe_vmul!` replace `LazyAlgebra.apply` and
  `LazyAlgebra.apply!`. `vmul`, and `vmul!` were already existing wand were exported by
  `using LazyAlgebra`. Instead of extending `vmul!` directly for each operator type, it is
  sufficient to specialize `LazyAlgebra.unsafe_vmul!(α, A, x, β, y)` which is called after
  checking that `x` and `y` have suitable dimensions or axes and after converting the
  floating-point type of the multipliers `α` and `β` to the ones of respectively `A*x` and
  `y`. The `scratch` argument that was in `apply!` is no longer used. This disallows an
  optimization that was possible but little used and difficult to implement without bugs.

- Method `vcreate` renamed as `LazyAlgebra.create_output` which is public but not exported
  and which has a slightly different semantic: `y = create_output(α,A,x)` is called to
  create an array `y` suitable to store `α*A*x` with `α` a scalar factor, `A` a linear
  operator, and `x` an input array. This change was needed because (i) the `scratch`
  argument is no longer supported and (ii) multipliers may have units which has an
  incidence on the element type of the result of `α*A*x` even though the floating-point
  type of `α` is given by that of `A*x`. The method `create_output(α,A,x)` shall throw a
  `DimensionMismatch` exception if the dimensions or axes of `x` are not compatible with
  `A` so that `@inbounds` can be assumed by `LazyAlgebra.unsafe_vmul!` for computing
  `α*A*x + β*y`.

- The inner product computed by `vdot` and the norms computed by `vnorm1`, `vnorm2`, and
  `vnorminf` treat complexes as usually done in linear algebra. The only difference is that
  multi-dimensional arguments are considered as *vectors*.

- Extending vectorized methods to other *vector* types shall only require to specialize
  the *unsafe* version of the methods (the ones with the `unsafe_` prefix).

- `vzero!` renamed `vzeros!`.

- Constructor of finite difference operator has a different syntax (for type-stability).
  It is called as `Diff{L,D}()` with `L` the order of differentiation and `D` the
  dimension(s) along which to perform the differentiation. If unspecified, `L=1` and
  `D=Colon` are assumed. The latter indicates to differentiate along all dimensions.

- Sparse operator API has been improved and simplified.
  - Non-exported public method `LazyAlgebra.each_off` has been renamed
    `LazyAlgebra.each_nz`.
  - Non-exported public method `LazyAlgebra.get_offs` can only take a single argument.
  - Non-exported public methods `LazyAlgebra.each_nz`, `LazyAlgebra.first_nz`, and
    `LazyAlgebra.last_nz` are provided to query the range, first, and last indices of the
    structural non-zeros.
  - To extend the package for new sparse compressed operators, non-exported public methods
    `LazyAlgebra.check_offset_index(Bool,A,ij)`, `LazyAlgebra.unsafe_first_nz(A,ij)`, and
    `LazyAlgebra.unsafe_last_nz(A,ij)` may be specialized in the type of `A` and with `ij`
    the row or column index depending on whether `A` is in row- or in column-wise format.
  - `LazyAlgebra.get_vals(A')` yields a lazily conjugated array.


### Added

- Most methods can be specialized for specific operator and/or array types by extending
  the `LazyAlgebra.unsafe_$f` method that is called by method `$f` after having checked
  that arguments have compatible axes and converted scalar multipliers (if any) to
  suitable floating-point type without changing their units. This makes easier to extend
  `LazyAlgebra`.

- Non-exported but public methods `LazyAlgebra.multiplier_type` and
  `LazyAlgebra.convert_multiplier` may be used to infer the type of a scalar multiplier
  and to convert it to a given floating-point type. These methods replace
  `multiplier_floatingpoint_type` and `promote_multiplier`.

- `GeneralMatrix{T,M}` is a generalization of a matrix built over a multi-dimensional
  array of coefficients of type `T` and whose `M` leading dimensions are considered as the
  *rows* of the pseudo-matrix.

- `FlexibleMatrix` is a pseudo-matrix whose number of row dimensions depends on its
  input argument.

- Exported functions `get_precision(A)` and `with_precision(T, A)` to retrieve the
  numerical precision of `A` and to change the numerical precision of `A` to be the
  floating-point type `T`.

- New methods `vones!`, `vnans`, and `vnans!` to fill an array with ones or NaNs.

- Non-exported public method `LazyAlgebra.test_API` to test the implementation of an
  operator.


## Version 0.2.7 (2024-03-08)

### Fixed

- Extend compatibility with `ZippedArrays`.

## Version 0.2.6 (2023-07-16)

### Fixed

- Extend compatibility with `ArrayTools`.

## Version 0.2.5 (2022-11-08)

- Make `set_val!` for sparse operators returns the same result as `setindex!`.

## Version 0.2.4 (2023-07-16)

### Fixed

- Extend compatibility with `MayOptimize`.

## Version 0.2.3 (2023-07-16)

### Fixed

- Simplify and generalize `vfill!` and `vzero!` to be able to work with `Unitful`
  elements.

- Automatically specialize `multiplier_type` for `Unitful.AbstractQuantity`.

## Version 0.2.2

## Added

- Improve `promote_multiplier` and make it easy to extend. The work done by
  `promote_multiplier` is broken in several functions: `multiplier_type(x)` yields the
  *element type* corresponding to `x` (which can be a number, an array of numbers, or a
  number type), `multiplier_floatingpoint_type(args...)` combines the types given by
  `multiplier_type` for all `args...` to yield a concrete floating-point type. The method
  `multiplier_type` is intended to be extended by other packages.

## Version 0.2.1

- Replace `@assert` by `@certify`. Compared to `@assert`, the assertion made by
  `@certify` may never be disabled whatever the optimization level.

- Provide default `vcreate` method for Gram operators.

## Version 0.2.0

### Changed

- Sub-module `LazyAlgebra.Foundations` (previously `LazyAlgebra.LazyAlgebraLowLevel`)
  exports types and methods needed to extend or implement `LazyAlgebra` mappings.

- The finite difference operator was too limited (finite differences were forcibly
  computed along all dimensions and only 1st order derivatives were implemented) and slow
  (because the leading dimension was used to store the finite differences along each
  dimension). The new family of operators can compute 1st or 2nd derivatives along all or
  given dimensions. The last dimension of the result is used to store finite differences
  along each chosen dimensions; the operators are much faster (at least 3 times faster for
  200×200 arrays for instance). Applying the Gram composition `D'*D` of a finite
  difference operator `D` is optimized and is about 2 times faster than applying `D` and
  then `D'`. Type `SimpleFiniteDifferences` is no longer available, use `Diff` instead
  (`Diff` was available as a shortcut in previous releases).

## Version 0.1.0

### Added

- Large sub-package for sparse operators which are linear mappings with few non-zero
  coefficients (see doc. for `SparseOperator` and `CompressedSparseOperator`). All common
  compressed sparse storage formats (COO, CSC and CSR) are supported and easy conversion
  between them is provided. Generalized matrix-vector multiplication is implemented and is
  as fast or significantly faster than with `SparseArrays.SparseMatrixCSC`.

- Method `∇(A,x)` yields the Jacobian of the mapping `A` at the variables `x`. If `A` is a
  linear-mapping, then `∇(A,x)` yields `A` whatever `x`. The new type `Jacobian` type is
  used to denote the Jacobian of a non-linear mapping. The notation `A'`, which is
  strictly equivalent to `adjoint(A)`, is only allowed for linear mappings and always
  denote the adjoint (conjugate transpose) of `A`.

- Method `gram(A)` yields `A'*A` for the linear mapping `A`. An associated *decorated
  type* `Gram` is used to denote this specific expression and some constructions are
  automatically recognized as valid Gram operators. Making this work for more complex
  constructions (like sums and compositions) would require to change the simplification
  rules (notably for the adjoint of such constructions).

- New `gram(A)` method which yields `A'*A` and alias `Gram{typeof(A)}` to represent the
  type of this construction.

- Add cropping and zero-padding operators.

- Provide `unpack!` method to unpack the non-zero coefficients of a sparse operator and
  extend `reshape` to be applicable to a sparse operator.

- Provide utility method `dimensions` which yields a dimension list out of its arguments
  and associated union type `Dimensions`.

- Provide `lgemv` and `lgemv!` for *Lazily Generalized Matrix-Vector multiplication* and
  `lgemm` and `lgemm!` for *Lazily Generalized Matrix-Matrix multiplication*. The names of
  these methods are reminiscent of `xGEMV` and `xGEMM` BLAS subroutines in LAPACK (with
  `x` the prefix corresponding to the type of the arguments).

- Add `fftfreq`, `rfftdims`, `goodfftdim` and `goodfftdims` in `LazyAlgebra.FFT` and
  re-export `fftshift` and `ifftshift` when `using LazyAlgebra.FFT`.

- Add `is_same_mapping` to allow for automatic simplifications when building-up sums and
  compositions.

- Provide `SimpleFiniteDifferences` operator.

- Provide `SparseOperator`.

### Fixed

- New rules: `α/A -> α*inv(A)`.

- Add rule for left-division by a scalar.

- In most cases, complex-valued arrays and multipliers are supported.

- Left multiplication by a scalar and left/right multiplication by a non-uniform scaling
  (a.k.a. diagonal operator) is optimized for sparse and non-uniform scaling operators.

- Provide (partial) support for complex-valued arrays.

- Compatibility with Julia 0.6, 0.7 and 1.0.

### Changed

- Exported methods and types have been limited to the ones for the end-user. Use `using
  LazyAlgebra.LazyAlgebraLowLevel` to use low-level symbols.

- Methods `has_oneto_axes`, `densearray`, `densevector` and `densematrix` have been
  replaced by `has_standard_indexing` and `to_flat_array` from `ArrayTools`.

- The exported constant `I = Identity()` has been renamed as `Id` to avoid conflicts with
  standard `LinearAlgebra` package. `Id` is systematically exported while `I` was only
  exported if not already defined in the `Base` module. The constant `LinearAlgebra.I`
  and, more generally, any instance of `LinearAlgebra.UniformScaling` is recognized by
  `LazyAlgebra` in the sense that they behave as the identity when combined with any
  `LazyAlgebra` mapping.

- `operand` and `operands` are deprecated in favor of `unveil` and `terms` which are less
  confusing. The `terms` method behaves exactly like the former `operands` method.
  Compared to `operand`, the `unveil` method has a better defined behavior: for a
  *decorated* mapping (that is an instance of `Adjoint`, `Inverse` or `InverseAdjoint`),
  it yields the embedded mapping; for other `LazyAlgebra` mappings (including scaled
  ones), it returns its argument; for an instance of `LinearAlgebra.UniformScaling`, it
  returns the equivalent `LazyAlgebra` mapping (that is `λ⋅Id`). To get the mapping
  embedded in a scaled mapping, call the `unscaled` method.

- `unscaled` is introduced as the counterpart of `multiplier` so that
  `multiplier(A)*unscaled(A) === A` always holds. Previously it was wrongly suggested to
  use `operand` (now `unveil`) for that but, then the strict equality was only true for
  `A` being a scaled mapping. These methods also work for instances of
  `LinearAlgebra.UniformScaling`.

- `NonuniformScalingOperator` deprecated in favor of `NonuniformScaling`.

- Argument `scratch` is no longer optional in low-level `vcreate`.

- The `CroppingOperators` sub-module has been renamed `Cropping`.

- Make constructor of a sparse operator (`SparseOperator`) reminiscent of the
  `sparse` method. Row and column dimensions can be a single scalar.

- A sparse operator (`SparseOperator`) can be converted to a regular array or
  to a sparse matrix (`SparseMatrixCSC`) and reciprocally.

- Trait constructors now return trait instances (instead of type). This is more
  *natural* in Julia and avoid having different method names.

- Skip bound checking when applying a `SparseOperator` (unless the operator
  structure has been corrupted, checking the dimensions of the arguments is
  sufficient to insure that indices are correct).

- The `apply!` method has been rewritten to allow for optimized combination to do `y =
  α*Op(A)⋅x + β*y` (as in LAPACK and optimized if scalars have values 0, ±1):

  ```julia
  apply!(α::Real, Op::Type{<:Operations}, A::LinearMapping, x, β::Real, y)
  apply!(β::Real, y, α::Real, Op::Type{<:Operations}, A::LinearMapping, x)
  ```

- Traits replace abstract types such as `Endomorphism`, `SelfAdjointOperator`, etc. Some
  operators may be endomorphisms or not. For instance the complex-to-complex `FFTOperator`
  is an endomorphism while the real-to-complex FFT is not. Another example:
  `NonuniformScaling` is self-adjoint if its coefficients are reals, not if they are
  complexes. This also overcomes the fact that multiple heritage is not possible in Julia.

- `contents`, too vague, has been suppressed and replaced by `operands` or
  `operand`. Accessor `multiplier` is provided to query the multiplier of a
  scaled mapping. Methods `getindex`, `first` and `last` are extended. In
  principle, direct reference to a field of any base mapping structures is no
  longer needed.

- Complete rewrite of the rules for simplifying complex constructions involving
  compositions and linear combination of mappings.

- `show` has been extend for mapping constructions.

- `promote_scalar` has been modified and renamed as `promote_multipler`.

- `LinearAlgebra.UniformScaling` can be combined with mappings in `LazyAlgebra`.

- The multiplier of a scaled mapping can now be any number although applying linear
  combination of mappings is still limited to real-valued multipliers.

- Optimal, an more general, management of temporaries is now done via the `scratch`
  argument of the `vcreate` and `apply!` methods. `InPlaceType` trait and
  `is_applicable_in_place` method have been removed.

### Removed

- Not so well defined `HalfHessian` and `Hessian` have been removed (`HalfHessian` is
  somewhat equivalent to `Gram`).

- Deprecated `fastrange` is replaced by `allindices` which is extended to scalar dimension
  and index intervals.

- `UniformScalingOperator` has been deprecated in favor of a `Scaled` version of the
  identity.

- `UniformScalingOperator` has been suppressed (was deprecated).
