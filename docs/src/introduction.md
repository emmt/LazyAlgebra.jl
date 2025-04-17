# Lazy algebra framework

`LazyAlgebra` is a [Julia](http://julialang.org/) package to generalize the notion of
matrices and vectors used in [linear
algebra](https://en.wikipedia.org/wiki/Linear_algebra).

Many numerical methods (*e.g.* in numerical optimization or in digital signal processing)
involve essentially [linear operations](https://en.wikipedia.org/wiki/Vector_space) on the
considered variables. `LazyAlgebra` provides a framework to implement these kind of
numerical methods independently of the specific type of the variables.

`LazyAlgebra` also provides a flexible and extensible framework for creating complex
linear operators to operate on the variables.

A few concepts are central to `LazyAlgebra`:
* *Multipliers* are scalar factors (of type `Number`) that may scale the following terms.
* *Vectors* represent the variables of interest and can be any abstract array providing a
  few methods are implemented for their specific type.
* Linear *operators* of type [`LazyAlgebra.Operator`](@ref) are linear mappings that take
  a *vector* as input and produce a *vector* as output. An *operator* in LazyAlgebra
  generalizes the notion of *matrix* in Julia.

There are several reasons to have special methods for basic vector operations rather than
relying on Julia linear algebra methods. First, the notion of *vector* is different, in
Julia a mono-dimensional array is a vector while, here any object with embedded values can
be assumed to be a vector providing a subset of methods are specialized for this type of
objects.


## Operators

`LazyAlgebra` features:
* flexible and extensible framework for creating complex operators;
* *lazy* evaluation of the operators;
* *lazy* assumptions when combining operators;
* efficient memory allocation by avoiding temporaries.


### General operators

An `Operator` extend the notion of *matrix* and can be any linear function between two
variables spaces. Using Householder-like notation (that is upper-case Latin letters denote
*operators*, lower-case Latin letters denote *variables*, and Greek letters denote
*scalars*), then:

* `A*x` yields the result of applying the operator `A` to `x`;

* `A\x` and `inv(A)*x` yield the result of applying the inverse of `A` to `x`;

* `A'*x` and `adjoint(A)*x` yield the result of applying the adjoint of `A` to `x`;

* `A'\x`, `inv(A')*x`, and `inv(A)'*x` yield the result of applying the inverse of the
  adjoint of `A` (or the adjoint of the inverse of `A`, this is the same thing) to `x`;

Simple constructions are allowed and can be used to create new instances of operators
which behave correctly:

* `B = α*A` (where `α` is a number) is an operator which behaves as `A` times `α`; that is
  `B*x -> α*(A*x)`.

* `C = A + B + ...` is an operator which behaves as the sum of the operators `A`, `B`,
  ...; that is `C*x -> A*x + B*x + ...` or `(A + B + ...)*x`.

* `C = A*B` or `C = A∘B` is an operator which behaves as the composition of the operators
  `A` and `B`; that is `C*x -> A*(B*x)`. As for the sum of operators, there may be an
  arbitrary number of operators in a composition; for example, if `D = A*B*C` then `D*x ->
  A*(B*(C*x))`.

* `B = A'` or `B = adjoint(A)` is an operator such that `B*x -> A'*x`.

* `B = inv(A)` is an operator such that `B*x -> inv(A)*x`.

* `C = A\B` is an operator such that `C*x -> inv(A)*(B*x)`.

* `C = A/B` is an operator such that `C*x -> A*(inv(B)*x)` or `A*(B\x)`.

These constructions can be combined to build up more complex operators. For example:

* `D = A*(B + 3C)` is an operator such that `D*x -> A*(B*x + 3*(C*x))`.

!!! note
    An important feature of `LazyAlgebra` is that any complex construction of operator is
    itself an operator but whose coefficients are not immediately computed: a constructed
    operator keeps its structure reflecting how it has been built (apart from a few
    automatic simplifications explained next) and *knows* how to behave when applied to an
    input vector. This *lazy* behavior explains the name of the package.

!!! note
    As a facility, most operators may be called as a function: `A(x)` and `A*x` are the
    same thing. However note that , due to the priority of operators in Julia, `A*B(x)` is
    the same as `A(B(x))` not `(A*B)(x)` which is the same as `A*B*x`.


## Automatic simplifications

An important feature of `LazyAlgebra` framework when combining operators is that a *number
of simplifications are automatically made at construction time*. These automatic
simplifications are type-stable and their result is therefore inferable (this was not the
case in old versions of the package).

A few simplification rules occur while building combinations of operators:

* `Id` is the identity operator exported by `LazyAlgebra`, as you can guess, `Id*A`,
  `Id\A`, and `A/Id` yield `A` while `Id/A` and `A\Id` yield `inv(A)`,

For instance,
assuming `A` is an operator:

```julia
B = A'
C = B'
```

yields `C` which is just a reference to `A`. In other words, `adjoint(adjoint(A)) -> A`
holds. Likely

```julia
D = inv(A)
E = inv(D)
```

yields `E` which is another reference to `A`. In other words, `inv(inv(A)) -> A` holds
assuming by default that `A` is invertible. This follows the principles of laziness. It is
however, possible to prevent this by extending the `Base.inv` method so as to throw an
exception when applied to the specific type of `A`:

```julia
Base.inv(::SomeNonInvertibleOperator) = error("non-invertible operator")
```

where `SomeNonInvertibleOperator <: Operator` is the type of `A`.

Other example of simplifications:

```julia
B = 3A
C = 7B'
```

where operators `B` and `C` are such that `B*x -> 3*(A*x)` and `C*x -> 21*(A*x)` for any
*vector* `x`. That is `C*x` is evaluated as `21*(A*x)` not as `7*(3*(A*x))` thanks to
simplifications occurring while the operator `C` is constructed.

Using the `->` to denote in the right-hand side the actual construction made by
`LazyAlgebra` for the expression in the left-hand side and assuming `A`, `B` and `C` are
linear operators, the following simplifications will occur:

```julia
(A + C + B + 3C)' -> A' + B' + 4C'
(A*B*3C)'         -> 3C'*B'*A'
inv(A*B*3C)       -> 3\inv(C)*inv(B)*inv(A)
```

However, if `M` is a non-linear operator, then:

```julia
inv(A*B*3M) -> inv(M)*(3\inv(B))*inv(A)
```

which can be compared to `inv(A*B*3C)` when all operands are linear operators.

!!! note
    Due to the associative rules applied by Julia, parentheses are needed
    around constructions like `3*C` if it has to be interpreted as `3C` in
    all contexts.  Otherwise, `A*B*(3*C)` is equivalent to `A*B*3C` while
    `A*B*3*C` is interpreted as `((A*B)*3)*C`; that is, compose `A` and `B`,
    apply `A*B` to `3` and right multiply the result by `C`.


## Creating new operators

`LazyAlgebra` provides a number of simple operators. Creating new primitive operator types
(not by combining existing operators as explained above) which benefit from the
`LazyAlgebra` framework is as simple as declaring a new operator sub-type of `Operator`
(or one of its abstract sub-types) and specializing a couple of methods for the new
operator type. This is explained in details [here](operators.md).
