* `convert_multiplier` should have 2 different behaviors: to allow for using
  BLAS routines, all multipliers must be converted to complexes if array
  arguments are complex-valued.

* Remove `REQUIRE` file (`Project.toml` is sufficient) and re-enable Julia 0.7
  in TravisCI.

* Fix `H'` when `H = HalfHessian(SimpleFiniteDifferences())`.

* Optimize composition of cropping and zero-padding operators.  The adjoint of
  a cropping or zero-padding operator is the pseudo-inverse of the operator,
  hence extend the `pinv` method.  If input and ouput dimnesions are the same
  (and offsets are all zeros), a cropping/zero-padding operator is the
  identity.

* `vscale`, `vscale!` and others should be able to take a complex multiplier.

* Implement *preconditioned* conjugate gradient.

* Simplify left/right multiplication of a sparse/diagonal operator by a diagonal
  operator. Same thing for sparse interpolator.  Take care of scaling by
  a multiplier (otherwise this makes little sense).

* Automatically simplify composition of diagonal operators.

* Do not make `A*x` equivalent to `A(x)` for non-linear mappings.

* Provide means to convert a sparse operator to a regular array or to a sparse
  matrix and reciprocally.  Use BLAS/LAPACK routines for sparse operators?

* Write an implementation of the L-BFGS operator and of the SR1 operator and
  perhaps of other low-rank operators.

* Use more extensively BLAS subroutines.  Fix usage of BLAX `dot` and `axpy`
  routines for dense arrays (use flat arrays).

* Cleanup: `is_identical` is not really needed? `===` does the job?

* SelfAdjoint should not be a trait?  Perhaps better to extend `adjoint(A::T) =
  A` when `T` is self-adjoint.

* Provide simplification rules for sums and compositions of diagonal operators
  (which are also easy to invert).

* Add rules so that the composition of two scaled linear operators, say
  `(αA)⋅(βB)`, automatically yields `((αβ)A)⋅B` when `A` is a linear mapping.
  This cannot easily propagate to have, *e.g.* `(αA)⋅(βB)⋅(γC)` automatically
  yields `((αβγ)A)⋅B⋅C` when `A` and `B` are linear mappings.  Perhaps this
  could be solved with some `simplify(...)` method to be applied to constructed
  mappings.  In fact, the solution is to have `(αA)⋅(βB)` automatically
  represented (by the magic of the constructors chain) as a *scaled composition*
  that is `(αβ)(A⋅B)` (that is pull the scale factor outside the expression).
  Indeed, `(αA)⋅(βB)⋅(γC)` will then automatically becomes `(αβ)(A⋅B)⋅(γC)` and
  then `(αβγ)(A⋅B⋅C)` with no additional efforts.

  - `α*A` => `A` if `α = 1`

  - `α*A` => `O` if `α = 0` with `O` the null mapping which is represented as
    `0` times a mapping, here `A`.  This is needed to know the result of
    applying the null mapping.  In other words, there is no *universal* neutral
    element for the addition of mappings; whereas the identity `I` is the
    *universal* neutral element for the composition of mappings.

  - `A*(β*B)` => `β*(A*B)` if `A` is a a linear mapping.

  - `(α*A)*(β*B)` => `(α*β)*(A*B)` if `A` is a linear mapping.

  - As a consequence of the above rules, `(α*A)*(β*B)*(γ*C)` =>
    `(α*βγ*)*(A*B*C)` if `A` and `B` are linear mappings, and so on.

  - `α\A` => `(1/α)*A`

  - `A/(β*B)` => `β\(A/B)` if `A` is a linear mapping.

  - `(α*A)/(β*B)` => `(α/β)*(A/B)` if `A` is a linear mapping.

  - `A\(β*B)` => `β*(A\B)` if `A` is a linear mapping.

  - `(α*A)\(β*B)` => `(β/α)*(A\B)` if `A` is a linear mapping.

  - `(α*I)*A` => `α*A` where `I` is the identity.

  - `A/A`, `A\A`, or `inv(A)*A` => `I` for `A` *invertible* (this trait means
    that `A` can be safely assumed invertible, possibilities: `Invertible`,
    `NonInvertible` to trigger an error on attempt to invert,
    `PossiblyInvertible` for mappings that may be invertible but not always and
    for which it is too costly to check.  For intance, checking for a uniform
    scaling `(α*I)` is trivial as it suffices to check whether `α` is
    non-zero).

* Concrete implementation of mappings on arrays is not consistent for
  complex valued arrays.

* Decide that, unless forbidden, inv is always possible (may be clash when
  trying to apply).  Or decide the opposite.

* Optimize `FiniteDifferences` for other multipliers.

* `Hessian` and `HalfHessian` should not both exist (one is 1/2 or 2 times the
  other).

* Make a demo like:

  ```julia
  using LazyAlgebra
  psf = read_image("psf.dat")
  dat = read_image("data.dat")
  wgt = read_image("weights.dat")
  µ = 1e-3 # choose regularization level
  .... # deal with sizes, zero-padding, or cropping etc.
  F = FFTOperator(dat)    # make a FFT operator to work with arrays similar to dat
  # Build instrumental model H (convolution by the PSF)
  H = F\Diag(F*ifftshift(psf))*F
  W = Diag(wgt)           # W is the precision matrix for independent noise
  D = FiniteDifferences() # D will be used for the regularization
  A = H'*W*H + µ*D'*D     # left hand-side matrix of the normal equations
  b = H'*W*y              # right hand-side vector of the normal equations
  img = conjgrad(A, b)    # solve the normal equations using linear conjugate gradients
  save_image(img, "result.dat")
  ```

  Notes: (1) `D'*D` is automatically simplified into a `HalfHessian`
  construction whose application to a *vector*, say `x`, is faster than
  `D'*(D*x))`.  (2) The evaluation of `H'*W*H` automatically uses the least
  temporary workspace(s).

* Replace `Coder` by using available meta-programming tools.
