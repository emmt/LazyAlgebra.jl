* Use more extensively BLAS subroutines.

* Add tests for rules.

* Replace `UniformScaling` by a `Scaled` version of the identity.

* Provide simplification rules for sums of diagonal operators (which are also
  easy to invert).

* Extend basic methods for `Base.LinAlg.UniformScaling` and rename
  `UniformScalingOperator` and `NonuniformScalingOperator` as `UniformScaling`
  and `NonuniformScaling`?

* Add rules so that: `(αA)⋅(βB)` automatically yields `((αβ)A)⋅B` when `A` is a
  linear mapping.  This cannot easily propagate to have, *e.g.*
  `(αA)⋅(βB)⋅(γC)` automatically yields `((αβγ)A)⋅B⋅C` when `A` and `B` are
  linear mappings.  Perhaps this could be solved with some `simplify(...)`
  method to be applied to constructed mappings.

* Add rules so that: `A*B` yields `αB` when `A` is a uniform mapping of
  parameter `α`.

* Concrete implementation of mappings on arrays is not consistent for
  complex valued arrays.

* Write an implementation of the L-BFGS operator and of the SR1 operator and
  perhaps of other low-rank operators.
