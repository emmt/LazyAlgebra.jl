# Lazy algebra framework

[![License](http://img.shields.io/badge/license-MIT-brightgreen.svg?style=flat)](LICENSE.md)
[![Build Status](https://travis-ci.org/emmt/LazyAlgebra.jl.svg?branch=master)](https://travis-ci.org/emmt/LazyAlgebra.jl)
[![Build Status](https://ci.appveyor.com/api/projects/status/github/emmt/LazyAlgebra.jl?branch=master)](https://ci.appveyor.com/project/emmt/LazyAlgebra-jl/branch/master)
[![Coverage Status](https://coveralls.io/repos/emmt/LazyAlgebra.jl/badge.svg?branch=master&service=github)](https://coveralls.io/github/emmt/LazyAlgebra.jl?branch=master)
[![codecov.io](http://codecov.io/github/emmt/LazyAlgebra.jl/coverage.svg?branch=master)](http://codecov.io/github/emmt/LazyAlgebra.jl?branch=master)

**LazyAlgebra** is a [Julia](http://julialang.org/) package to generalize the
notion of matrices and vectors used in
[linear algebra](https://en.wikipedia.org/wiki/Linear_algebra).

Many numerical methods (*e.g.* in numerical optimization or digital signal
processing) involve essentially
[linear operations](https://en.wikipedia.org/wiki/Vector_space) on the
considered variables.  `LazyAlgebra` provides a framework to implement these
kind of numerical methods independently of the specific type of the variables.
This is exploited in
[OptimPackNextGen](https://github.com/emmt/OptimPackNextGen.jl) package, an
attempt to provide most optimization algorithms of
[OptimPack](https://github.com/emmt/OptimPack) in pure Julia.

`LazyAlgebra` also provides a flexible and extensible framework for creating
complex mappings and linear mappings to operate on the variables.

A few concepts are central to `LazyAlgebra`:
* *vectors* represent the variables of interest and can be anything providing a
  few methods are implemented for their specific type;
* *mappings* are any functions between such vectors;
* *linear mappings* (a.k.a. linear operators) behave linearly with respect to
  their arguments.

Similar Julia packages:
* [LinearMaps](https://github.com/Jutho/LinearMaps.jl)
* [LinearOperators](https://github.com/JuliaSmoothOptimizers/LinearOperators.jl)

There are several reasons to have special methods for basic vector operations
rather than relying on Julia linear algebra methods.  First, the notion of
*vector* is different, in Julia a mono-dimensional array is a vector while,
here any object with embedded values can be assumed to be a vector providing a
subset of methods are specialized for this type of object.  For instance,
LazyAlgebra provides such methods specialized for real-valued and
complex-valued (with real components) arrays of any dimensionality.  Second,
the meaning of the methods may have to be different.  For instance, only
real-valued functions can be minimized (or maximized) and for this task,
complex-valued variables can just be considered as real-valued variables (each
complex value being equivalent to a pair of reals).


## Mappings

`LazyAlgebra` features:
* flexible and extensible framework for creating complex mappings;
* *lazy* evaluation of the mappings;
* *lazy* assumptions when combining mappings;
* efficient memory allocation by avoiding temporaries.


### General mappings

A `Mapping` can be any function between two variables spaces.  Using
Householder-like notation (that is upper case Latin letters denote *mappings*,
lower case Latin letters denote *variables*, and Greek letters denote
*scalars*), then:

* `A(x)`, `A*x` or `A⋅x` yields the result of applying the mapping `A` to `x`;

* `A\x` yields the result of applying the inverse of `A` to `x`;

Simple constructions are allowed for any kind of mappings and can be used to
create new instances of mappings which behave correctly.  For instance:

* `B = α*A` (where `α` is a number) is a mapping which behaves as `A` times
  `α`; that is `B(x)` yields the same result as `α*(A(x))`.

* `C = A + B + ...` is a mapping which behaves as the sum of the mappings `A`,
  `B`, ...; that is `C(x)` yields the same result as `A(x) + B(x) + ...`.

* `C = A*B`, `C = A∘B` or `C = A⋅B` is a mapping which behaves as the
  composition of the mappings `A` and `B`; that is `C⋅x` yields the same result
  as `A(B(x))`.  As for the sum of mappings, there may be an arbitrary number
  of mappings in a composition; for example, if `D = A*B*C` then `D(x)` yields
  the same result as `A(B(C(x)))`.

* `C = A\B` is a mapping such that `C(x)` yields the same result as
  `inv(A)(B(x))`.

* `C = A/B` is a mapping such that `C(x)` yields the same result as
  `A(inv(B)(x))`.

These constructions can be combined to build up more complex mappings.  For
example:

* `D = A*(B + 3C)` is a mapping such that `D⋅x` yields the same result as
  `A(B(x) + 3*C(x))`.


Beware that, due to the priority of operators in Julia, `A*B(x)` is the same
as `A(B(x))` not `(A*B)(x)`.


### Linear mappings

A `LinearMapping` can be any linear mapping between two spaces.  This abstract
subtype of `Mapping` is introduced to extend the notion of *matrices* and
*vectors*.  Assuming the type of `A` inherits from `LinearMapping`, then:

* for linear mappings `A` and `B`, `A⋅B` is the same as `A∘B` or `A*B` which
  yields the composition of `A` and `B` whose effect is to apply `B` and then
  `A`;

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

yields `C` which is just a reference to `A`. In other words, that is
`adjoint(adjoint(A)) -> A` holds.  Likely

```julia
D = inv(A)
E = inv(D)
```

yields `E` which is another reference to `A`.  In other words,
`inv(inv(A)) -> A` holds assuming by default that `A` is invertible.  This
follows the principles of laziness.  It is however, possible to prevent this by
extended the `Base.inv` method so as to throw an exception when applied to the
specific type of `A`:

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
and `C` are linear mappings, the following simplications will occur:

```julia
(A + C + B + 3C)' ≡ A' + B' + 4C'
(A*B*3C)'         ≡ 3C'*B'*A'
inv(A*B*3C)       ≡ 3\inv(C)*inv(B)*inv(A)
```

However, if `M` is a non-linear mapping, then:

```julia
inv(A*B*3M) ≡ inv(M)*(3\inv(B))*inv(A)
```

which can be compared to `inv(A*B*3C)` when all operands are liner mappings.

Note that, due to the associative rules applied by Julia, parentheses are
needed around constructions like `3*C` if it has to be interpreted as `3C` in
all contexes.  Otherwise, `A*B*(3*C)` is equivalent to `A*B*3C` while `A*B*3*C`
is interpreted as `((A*B)*3)*C`; that is, compose `A` and `B`, apply `A*B` to
`3` and right multiply the result by `C`.


### Creating new mappings

`LazyAlgebra` provides a number of simple mappings.  Creating new primitive
mapping types (not by combining existing mappings as explained above) which
benefit from the `LazyAlgebra` framework is as simple as declaring a new
mapping subtype of `Mapping` (or one of its abstract subtypes) and extending
two methods `vcreate` and `apply!` specialized for the new mapping type.  For
mode details, see [here](doc/mappings.md).
