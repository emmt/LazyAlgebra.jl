
* Beware that not all diagonal operators are self-adjoint and for some operator
  this may depends on the settings (*e.g.* a diagonal operator is self adjoint
  if its coefficients are reals not if they are complexes but it is symmetric).
  Suggestion: use abstract type `SelfAdjointOperator` as a helper and add
  method `isselfadjoint()` method.

* Possible implementation of `DiagonalOperator`:

```julia
abstract type DiagonalOperator{T<:LinearOperator} <: T end
DiagonalOperator() = Identity()
DiagonalOperator(α::Real) =
    (α == one(alpha) ? Identity() : UniformScalingOperator(α))
DiagonalOperator(u) = NonuniformScalingOperator(u)
```

* Concrete implementation of operators on arrays is not consistent for
  complex valued arrays.

* Use more extensively BLAS subroutines.

* Rewrite `apply!` to allow for optimized combination to do `y = α*Op(A)⋅x +
  β*y` (as in LAPACK and optimized if scalars have values 0, ±1):

```julia
apply!(α::Real, Op::Type{<:Operations}, A::LinearOperator, x, β::Real, y)
apply!(β::Real, y, α::Real, Op::Type{<:Operations}, A::LinearOperator, x)
```

* Write an implementation of the L-BFGS operator and of the SR1 operator and
  perhaps of other low-rank operators.
