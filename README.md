# Lazy algebra framework

[![License](http://img.shields.io/badge/license-MIT-brightgreen.svg?style=flat)](LICENSE.md)

`LazyAlgebra` is a Julia package to generalize the notion of matrices and
vectors used in [linear algebra](https://en.wikipedia.org/wiki/Linear_algebra).

Many numerical methods (*e.g.* in numerical optimization or digital signal
processing) involve essentially linear operations on the considered variables.
`LazyAlgebra` provides a framework to implement these kind of numerical methods
independently of the specific type of the variables.  This is exploited in
[OptimPackNextGen](https://github.com/emmt/OptimPackNextGen.jl) package, an
attempt to provide all optimization algorithms of
[OptimPack](https://github.com/emmt/OptimPack) in pure Julia.

`LazyAlgebra` also provides a flexible and extensible framework for creating
complex mappings and linear mappings to operate on the variables.

A few concepts are central to `LazyAlgebra`:
* *vectors* represent the variables of interest and can be anything provided a
  few methods are implemented for their specific type;
* *mappings* are any functions between such vectors;
* *linear mappings* behave linearly with respect to their arguments.

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


## Mappings

`LazyAlgebra` features:
* flexible and extensible framework for creating complex mappings;
* *lazy* evaluation of the mappings;
* *lazy* assumptions when combining mappings;
* efficient memory allocation by avoiding temporaries.


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

* `D = A*(B + 3C)` is a mapping such that `D⋅x` yields the same result as
  `A⋅(B⋅x + 3*C⋅x)`.


### Linear mappings

A `LinearMapping` can be any linear mapping between two spaces.  This abstract
subtype of `Mapping` is introduced to extend the notion of *matrices* and
*vectors*.  Assuming the type of `A` inherits from `LinearMapping`, then:

* `A'⋅x` and `A'*x` yields the result of applying the adjoint of the mapping
  `A` to `x`;

* `A'\x` yields the result of applying the adjoint of the inverse of mapping
  `A` to `x`.

* `B = A'` is a mapping such that `B⋅x` yields the same result as `A'⋅x`.


### Automatic simplifications

An important feature of `LazyAlgebra` framework for mappings is that a *number
of simplifications are automatically made at contruction time*.  For instance,
assuming `A` is a mapping:

```julia
B = A'
C = B'
```

yields `C` which is just a reference to `A`.  Likely

```julia
D = inv(A)
E = inv(D)
```

yields `E` which is another reference to `A`.  Note that following the
principles of laziness, `inv(inv(A))` just yields `A` assuming by default that
it is invertible.  It is however, possible to prevent this by extended the
`Base.inv` method so as to throw an exception when applied to the specific type
of `A`:

```julia
Base.inv(::SomeNonInvertibleMapping) = error("non-invertible mapping")
```

where `SomeNonInvertibleMapping <: Mapping` is the type of `A`.

Other example of simplifications:

```julia
B = 3A
C = 7B'
```

where mappings `B` and `C` are such that `B*x ≡ 3*(A*x)` and `C*x ≡ 21*(A*x)`
for any *vector* `x`.  That is `C*x` is evaluated as `21*(A*x)` not as
`7*(3*(A*x))` thanks to simplifications occurring at the contruction of the
mapping `C`.

Using the `≡` to denote in the right-hand side the actual construction made by
`LazyAlgebra` for the expression in the left-hand side and assuming `A`, `B`
and `C` are mappings, the following simplications will occur:

```julia
(A + B + 3C)' ≡ (A' + B' + 3C')
(A*B*(3C))'   ≡ (3C'*A'*B')
inv(A*B*(3C)) ≡ (((1/3)*inv(C))*inv(A)*inv(B))
```

Note the necessary parentheses around `3C` in the last examples above to
overcome the associative rule applied by Julia.  Otherwise, `A*B*3C` is
interpreted as `((A*B)*3)*C`; that is, compose `A` and `B`, apply `A*B` to `3`
and right multiply the result by `C`.


### Creating new mappings

`LazyAlgebra` provides a number of simple mappings.  Creating new primitive
mapping types (not by combining existing mappings as explained above) which
benefit from the `LazyAlgebra` framework is as simple as declaring a new
mapping subtype of `Mapping` (or one of its abstract subtypes) and extending
two methods `vcreate` and `apply!` specialized for the new mapping type.  More
mode details, see [here](doc/mappings.md).
