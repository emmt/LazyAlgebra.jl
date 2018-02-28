
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
