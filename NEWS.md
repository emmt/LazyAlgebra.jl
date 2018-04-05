
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

