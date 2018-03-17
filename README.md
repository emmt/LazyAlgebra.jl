# Lazy algebra framework

[![License](http://img.shields.io/badge/license-MIT-brightgreen.svg?style=flat)](LICENSE.md)

`LazyAlgebra` is a Julia package to generalize the notion of matrices and
vectors used in [linear algebra](https://en.wikipedia.org/wiki/Linear_algebra).

Many numerical methods (*e.g.* in numerical optimization or digital signal
processing) involve essentially linear operations on the considered variables.
`LazyAlgebra` provides a framework to implement these kind of numerical methods
independently of the specific type of the variables.

`LazyAlgebra` also provides a flexible and extensible framework for creating
complex mappings and linear operators to operate on the variables.

A few concepts are central to `LazyAlgebra`:
* *vectors* represent the variables of interest and can be anything provided a
  few methods are implemented for their specific type;
* *mappings* are any functions between such vectors;
* *linear operators* are linear mappings.

`LazyAlgebra` features:
* flexible and extensible framework for creating complex mappings and linear
  operators;
* *lazy* evaluation of the mappings;
* *lazy* assumptions when combining mappings;
* efficient memory allocation by avoiding temporaries;

(https://en.wikipedia.org/wiki/Vector_space)

Similar Julia packages:
* [LinearMaps](https://github.com/Jutho/LinearMaps.jl)
* [LinearOperators](https://github.com/JuliaSmoothOptimizers/LinearOperators.jl)


The rationale to have special methods for basic vector operations, *e.g* `vdot`
instead of `dot`, is to distinguish the assumptions made in Julia where `dot`
yields the inner product of two objects whose type is derived from the
`AbstractArray` type with the only requirement that they have the same number
of elements while `vdot` yields the sum of the product of the corresponding
elements between two objects of the same kind (they can be two arrays but they
must have the same dimensions and complex valued elements are considere as pair
of reals).


## Mappings and linear operators

### General mappings

A `Mapping` can be any function between two variables spaces.  Assuming upper
case Latin letters denote *mappings*, lower case Latin letters denote
*variables*, and Greek letters denote *scalars*, then:

* `A*x` or `A⋅x` yields the result of applying the mapping `A` to `x`;

* `A\x` yields the result of applying the inverse of `A` to `x`;

Simple constructions are allowed for any kind of mappings and can be used to
create new instances of mappings which behave correctly.  For instance:

* `B = α*A` (where `α` is a real) is a mapping which behaves as `A` times `α`;
  that is `B⋅x` yields the same result as `α*(A⋅x)`.

* `C = A + B + ...` is a mapping which behaves as the sum of the mappings `A`,
  `B`, ...; that is `C⋅x` yields the same result as `A⋅x + B⋅x + ...`.

* `C = A*B` or `C = A⋅B` is a mapping which behaves as the composition of the
  mappings `A` and `B`; that is `C⋅x` yields the same result as `A⋅(B.x)`.  As
  for the sum of mappings, there may be an arbitrary number of mappings in a
  composition; for example, if `D = A*B*C` then `D⋅x` yields the same result as
  `A⋅(B⋅(C⋅x))`.

* `C = A\B` is a mapping such that `C⋅x` yields the same result as `A\(B⋅x)`.

* `C = A/B` is a mapping such that `C⋅x` yields the same result as `A⋅(B\x)`.

These constructions can be combined to build up more complex mappings.  For
example:

* `D = A*(B + C)` is a mapping such that `C⋅x` yields the same result as
  `A⋅(B⋅x + C⋅x)`.


### Linear operators

A `LinearOperator` can be any linear mapping between two spaces.  This abstract
subtype of `Mapping` is introduced to extend the notion of *matrices* and
*vectors*.  Assuming the type of `A` inherits from `LinearOperator`, then:

* `A'⋅x` and `A'*x` yields the result of applying the adjoint of the operator
  `A` to `x`;

* `A'\x` yields the result of applying the adjoint of the inverse of operator
  `A` to `x`.

* `B = A'` is a mapping such that `B⋅x` yields the same result as `A'⋅x`.

`LazyAlgebra` provides a number of mappings and linear operators.  Creating
new primitive mapping types (not by combining existing mappings as explained
above) is explained [here](doc/mappings.md).
