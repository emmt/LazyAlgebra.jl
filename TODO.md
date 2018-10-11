* Implement *preconditioned* conjugate gradient.

* Use more extensively BLAS subroutines.

* Add tests for rules.

* Rename traits types and us their constructors to return trait instances.
  This is more *natural* in Julia and avoid having different method names.

* Make *scalars* any `Number`.

* Provide simplification rules for sums and compositions of diagonal operators
  (which are also easy to invert).

* Write rules when an operator is supplied as an instance of
  `LinearAlgebra.UniformScaling`.

* Extend basic methods for `Base.LinAlg.UniformScaling` and rename
  `UniformScalingOperator` and `NonuniformScalingOperator` as `UniformScaling`
  and `NonuniformScaling`?

* Add rules so that: `A*B` yields `α*B` when `A` is a uniform scaling of
  parameter `α`.  See remark above for rewriting uniform scaling as a scaled
  identity.

* Add rules so that the composition of two scaled linear operators, say
  `(αA)⋅(βB)`, automatically yields `((αβ)A)⋅B` when `A` is a linear mapping.
  This cannot easily propagate to have, *e.g.* `(αA)⋅(βB)⋅(γC)` automatically
  yields `((αβγ)A)⋅B⋅C` when `A` and `B` are linear mappings.  Perhaps this
  could be solved with some `simplify(...)` method to be applied to constructed
  mappings.  In fact, the solution is to have `(αA)⋅(βB)` automatically
  represented (by the magic of the constructors chain) as a *scaled compsition*
  that is `(αβ)(A⋅B)` (that is pull the scale factor outside the expression).
  Indeed, `(αA)⋅(βB)⋅(γC)` will then automatically becomes `(αβ)(A⋅B)⋅(γC)` and
  then `(αβγ)(A⋅B⋅C)` with no additional efforts.

  - `α*A` => `A` if `α = 1`
  - `α*A` => `O` if `α = 0` with `O` the null mapping, but how to represent it?
  - `A*(β*B)` => `β*(A*B)` if `A` is a linear mapping
  - `(α*A)*(β*B)` => `(α*β)*(A*B)` if `A` is a linear mapping
  - as a consequence of the above `(α*A)*(β*B)*(γ*C)` => `(α*βγ*)*(A*B*C)`
    if `A` and `B` are linears, and etc.

  - `α\A` => `(1/α)*A`
  - `A/(β*B)` => `β\(A/B)` if `A` is a linear mapping
  - `(α*A)/(β*B)` => `(α/β)*(A/B)` if `A` is a linear mapping

  - `A\(β*B)` => `β*(A\B)` if `A` is a linear mapping
  - `(α*A)\(β*B)` => `(β/α)*(A\B)` if `A` is a linear mapping

  - `(α*I)*A` => `α*A` where `I` is the identity

  - `A/A`, `A\A`, or `inv(A)*A` => `I` for `A` *invertible* (this trait means
    that `A` can be safely assumed invertible, possibilities: `Invertible`,
    `NonInvertible` to trigger an error on attempt to invert,
    `PossiblyInvertible` for mappings that may be invertible but not always and
    for which it is too costly to check.  For intance, checking for a uniform
    scaling `(α*I)` is trivial as it suffices to check whether `α` is
    non-zero).

* Concrete implementation of mappings on arrays is not consistent for
  complex valued arrays.

* Write an implementation of the L-BFGS operator and of the SR1 operator and
  perhaps of other low-rank operators.

* Implement `isequal` and do simplifications like `A + 2A => 3A`

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
  H = F\Diag(F*psf)*F     # H is the instrumental model (convolution by the PSF)
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
