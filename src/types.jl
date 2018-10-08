#
# types.jl -
#
# Type definitions and (some) constructors for linear algebra.
#
#-------------------------------------------------------------------------------
#
# This file is part of the LazyAlgebra (https://github.com/emmt/LazyAlgebra.jl)
# package released under the MIT "Expat" license.
#
# Copyright (c) 2017-2018 Éric Thiébaut.
#

struct SingularSystem <: Exception
    msg::String
end
showerror(io::IO, err::SingularSystem) =
    print(io, "singular linear system ($(err.msg))")

struct NonPositiveDefinite <: Exception
    msg::String
end
showerror(io::IO, err::NonPositiveDefinite) =
    print(io, "non-positive definite operator ($(err.msg))")

struct UnimplementedOperation <: Exception
    msg::String
end
showerror(io::IO, err::UnimplementedOperation) =
    print(io, err.msg)

struct UnimplementedMethod <: Exception
    msg::String
end
showerror(io::IO, err::UnimplementedMethod) =
    print(io, err.msg)

"""

Type `Reals` is the set of the floating point types.  It is the numerical
approximation of reals in the mathematical sense.

"""
const Reals = AbstractFloat


"""

Type `Complexes` is the set of the complexes whose real and imaginary parts are
floating point.  It is the numerical approximation of complexes in the
mathematical sense.

"""
const Complexes = Complex{<:Reals}

"""

A `Scalar` is used to represent multipliers or scaling factors when combining
mappings.  For now, scalars are double precision floating-point.

"""
const Scalar = Float64

"""

A `Mapping` is any function between two variables spaces.  Assuming upper case
Latin letters denote mappings, lower case Latin letters denote variables, and
Greek letters denote scalars, then:

* `A*x` or `A⋅x` yields the result of applying the mapping `A` to `x`;

* `A\\x` yields the result of applying the inverse of `A` to `x`;

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

* `C = A\\B` is a mapping such that `C⋅x` yields the same result as `A\\(B⋅x)`.

* `C = A/B` is a mapping such that `C⋅x` yields the same result as `A⋅(B\\x)`.

These constructions can be combined to build up more complex mappings.  For
example:

* `D = A*(B + C)` is a mapping such that `C⋅x` yields the same result as
  `A⋅(B⋅x + C⋅x)`.

A `LinearMapping` is any linear mapping between two spaces.  This abstract
subtype of `Mapping` is introduced to extend the notion of *matrices* and
*vectors*.  Assuming the type of `A` inherits from `LinearMapping`, then:

* `A'⋅x` and `A'*x` yields the result of applying the adjoint of the mapping
  `A` to `x`;

* `A'\\x` yields the result of applying the adjoint of the inverse of mapping
  `A` to `x`.

* `B = A'` is a mapping such that `B⋅x` yields the same result as `A'⋅x`.

The following methods should be implemented for a mapping `A` of specific type
`M <: Mapping`:

```julia
vcreate(::Type{P}, A::M, x) -> y
apply!(α::Scalar, ::Type{P}, A::M, x, β::Scalar, y) -> y
```

for any supported operation `P ∈ Operations` (`Direct`, `Adjoint`, `Inverse`
and/or `InverseAdjoint`).  See the documentation of these methods for
explanations.  Optionally, methods `P(A)` may be extended, *e.g.* to throw
exceptions if operation `P` is forbidden (or not implemented).  By default, all
these operations are assumed possible (except `Adjoint` and `InverseAdjoint`
for a nonlinear mapping).

See also: [`apply`](@ref), [`apply!`](@ref), [`vcreate`](@ref),
          [`LinearType`](@ref), [`is_applicable_in_place`](@ref),
          [`Scalar`](@ref), [`Direct`](@ref), [`Adjoint`](@ref),
          [`Inverse`](@ref), [`InverseAdjoint`](@ref).

"""
abstract type Mapping <: Function end

abstract type LinearMapping <: Mapping end
@doc @doc(Mapping) LinearMapping

"""

The abstract type `Trait` is inherited by types indicating specific traits.

See also: [`LinearType`](@ref), [`SelfAdjointType`](@ref),
          [`DiagonalType`](@ref), [`MorphismType`](@ref).

"""
abstract type Trait end

"""

The *linear* trait indicates whether a mapping is certainly linear.  Abstract
type `LinearType` has two concrete singleton sub-types: `Linear` for linear
mappings and `NonLinear` for other mappings.  The call:

```julia
LinearType(A)
```

yields the *linear* type of mapping `A`, that is one of `Linear` for linear
maps or `NonLinear` for other mappings.

See also: [`Trait`](@ref), [`is_linear`](@ref).

"""
abstract type LinearType <: Trait end

for T in (:NonLinear, :Linear)
    @eval begin
        struct $T <: LinearType end
        @doc @doc(LinearType) $T
    end
end

"""

The *self-adjoint* trait indicates whether a mapping is certainly a
self-adjoint linear map.  Abstract type `SelfAdjointType` has two concrete
singleton sub-types: `SelfAdjoint` for self-adjoint linear maps and
`NonSelfAdjoint` for other mappings.  The call:

```julia
SelfAdjointType(A)
```

yields the *self-adjoint* type of mapping `A`, that is one of `SelfAdjoint` for
self-adjoint linear maps or `NonSelfAdjoint` for other mappings.

See also: [`Trait`](@ref), [`is_selfadjoint`](@ref).

"""
abstract type SelfAdjointType <: Trait end

for T in (:NonSelfAdjoint, :SelfAdjoint)
    @eval begin
        struct $T <: SelfAdjointType end
        @doc @doc(SelfAdjointType) $T
    end
end

"""

The *morphism* trait indicates whether a mapping is certainly an endomorphism
(its input and output spaces are the same).  Abstract type `MorphismType` has
two concrete singleton sub-types: `Endomorphism` for endomorphisms and
`Morphism` for other mappings.  The call:

```julia
MorphismType(A)
```

yields the *morphism* type of mapping `A`, that is one of `Endomorphism` for
mappings whose input and output spaces are the same or `Morphism` for other
mappings.

See also: [`Trait`](@ref), [`is_endomorphism`](@ref).

"""
abstract type MorphismType <: Trait end

for T in (:Morphism, :Endomorphism)
    @eval begin
        struct $T <: MorphismType end
        @doc @doc(MorphismType) $T
    end
end

"""

The *diagonal* trait indicates whether a mapping is certainly a diagonal linear
mapping.  Abstract type `DiagonalType` has two concrete singleton sub-types:
`DiagonalMapping` for diagonal linear mappings and `NonDiagonalMapping` for
other mappings.  The call:

```julia
DiagonalType(A)
```

yields the *diagonal* type of mapping `A`, that is one of `DiagonalMapping` for
diagonal linear maps or `NonDiagonalMapping` for other mappings.

See also: [`Trait`](@ref), [`is_diagonal`](@ref).

"""
abstract type DiagonalType <: Trait end

for T in (:DiagonalMapping, :NonDiagonalMapping)
    @eval begin
        struct $T <: DiagonalType end
        @doc @doc(DiagonalType) $T
    end
end

"""

The *in-place* trait indicates whether a mapping is applicable in-place.
Abstract type `InPlaceType` has two concrete singleton sub-types: `InPlace` for
mappings which are applicable with the same input and output arguments or
`OutOfPlace` for mappings which must be applied with different input and output
arguments.  The call:

```julia
InPlaceType([P=Direct,] A)
```

yields whether the mapping `A` is applicable in-place for operation `P`.  The
retuned value is one of `InPlace` or `OutOfPlace`.

See also: [`Trait`](@ref), [`is_applicable_in_place`](@ref).

"""
abstract type InPlaceType <: Trait end

for T in (:InPlace, :OutOfPlace)
    @eval begin
        struct $T <: InPlaceType end
        @doc @doc(InPlaceType) $T
    end
end

"""

Type `Direct` is a singleton type to indicate that a linear mapping should
be directly applied.  This type is part of the union `Operations`.

See also: [`LinearMapping`](@ref), [`apply`](@ref), [`Operations`](@ref).

"""
struct Direct; end

"""

Types `Adjoint`, `Inverse` and `InverseAdjoint` are used to *decorate* a
mapping to indicate the conjugate transpose and/or inverse of the mapping.  The
`adjoint` method is extended, so that in the code, it is sufficient (and
recommended) to write `A'` instead of `Adjoint(A)`.  Furthermore, `A'` or
`adjoint(A)` may be able to perform some simplications resulting in improved
efficiency.  `AdjointInverse` is just an alias for `InverseAdjoint`.  Note that
the adjoint only makes sense for linear mappings.

See also: [`LinearMapping`](@ref), [`apply`](@ref), [`Operations`](@ref).

"""
struct Adjoint{T<:Mapping} <: LinearMapping
    op::T
    # The inner constructors make sure that the argument is a linear mapping.
    Adjoint{T}(A::T) where {T<:LinearMapping} = new{T}(A)
    function Adjoint{T}(A::T) where {T<:Mapping}
        is_linear(A) ||
            error("taking the adjoint of non-linear mappings is not allowed")
        return new{T}(A)
    end
end

# Outer constructor must be provided.
Adjoint(A::T) where {T<:Mapping} = Adjoint{T}(A)

struct Inverse{T<:Mapping} <: Mapping
    op::T
end

struct InverseAdjoint{T<:LinearMapping} <: LinearMapping
    op::T

    # The inner constructors ensure that the argument is a linear mapping.
    InverseAdjoint{T}(A::T) where {T<:LinearMapping} = new{T}(A)
    function InverseAdjoint{T}(A::T) where {T<:Mapping}
        is_linear(A) ||
            error("taking the inverse adjoint of non-linear mappings is not allowed")
        return new{T}(A)
    end
end

# Outer constructor must be provided.
InverseAdjoint(A::T) where {T<:Mapping} = Adjoint{T}(A)

const AdjointInverse{T} = InverseAdjoint{T}

for T in (:Inverse, :InverseAdjoint, :AdjointInverse)
    @eval @doc @doc(Adjoint) $T
end

"""

`Operations` is the union of the possible ways to apply a linear mapping:
`Direct`, `Adjoint`, `Inverse` and `InverseAdjoint` (or its alias
`AdjointInverse`).

See also: [`LinearMapping`](@ref), [`apply`](@ref), [`Direct`](@ref),
          [`Adjoint`](@ref).

"""
const Operations = Union{Direct,Adjoint,Inverse,InverseAdjoint}

"""

A `Scaled` mapping is used to represent a mappings times a scalar.  End-users
should not use a `Scaled` constructor directly but rather use the `*` operator
(with a scalar left operand) as it may be able to make some simplifications
resulting in improved efficiency.

"""
struct Scaled{T<:Mapping} <: Mapping
    sc::Scalar
    op::T
end

"""

A `Sum` is used to represent an arbitrary sum of mappings.  End-users should
not use a `Sum` constructor directly but rather use the `+` operator as it may
be able to make some simplifications resulting in improved efficiency.

"""
struct Sum{N,T<:NTuple{N,Mapping}} <: Mapping
    ops::T

    # The inner constructor ensures that the number of arguments is at least 2.
    function Sum{N,T}(ops::T) where {T<:NTuple{N,Mapping}} where {N}
        length(ops) ≥ 2 || error("a sum of mappings has at least 2 components")
        return new{N,T}(ops)
    end
end
Sum(ops::T) where {T<:NTuple{N,Mapping}} where {N} = Sum{N,T}(ops)
Sum(ops::Mapping...) = Sum(ops)

"""

A `Composition` is used to represent an arbitrary composition of mappings.
Constructor `Composition(A,B)` may be extended in code implementing specific
mappings of linear operators to provide *automatic* simplifications.  The
end-user should not use `Composition` constructors directly but use the `.` or
`*` operators instead as they may be able to make some simplifications
resulting in improved efficiency.

"""
struct Composition{N,T<:NTuple{N,Mapping}} <: Mapping
    ops::T

    # The inner constructor ensures that the number of arguments is at least 2.
    function Composition{N,T}(ops::T) where {T<:NTuple{N,Mapping}} where {N}
        length(ops) ≥ 2 ||
            error("a composition of mappings has at least 2 components")
        return new{N,T}(ops)
    end
end
Composition(ops::T) where {T<:NTuple{N,Mapping}} where {N} =
    Composition{N,T}(ops)
Composition(ops::Mapping...) = Composition(ops)

"""

`Hessian(A)` is a container to be interpreted as the linear mapping
representing the second derivatives of some objective function at some point
both represented by `A` (which can be anything).  Given `H = Hessian(A)`, the
contents `A` is retrieved by `contents(H)`.

For a simple quadratic objective function like:

```
f(x) = ‖D⋅x‖²
```

the Hessian is:

```
H = 2 D'⋅D
```

As the Hessian is symmetric, a single method `apply!` has to be implemented to
apply the direct and adjoint of the mapping, the signature of the method is:

```julia
apply!(α::Scalar, ::Type{Direct}, H::Hessian{typeof(A)}, x::T,
       β::Scalar, y::T)
```

where `y` is overwritten by `α*H*x + β*y` with `H*x` the result of applying `H`
(or its adjoint) to the argument `x`.  Here `T` is the relevant type of the
variables.

To allocate a new object to store the result of applying the mapping to `x`,
the default method is `vcreate(x)`.  If this is not suitable, it is sufficient
to implement the specific method:

```julia
vcreate(::Type{Direct}, H::Hessian{typeof(A)}, x::T)
```

See also: [`apply!`](@ref), [`vcreate`](@ref), [`LinearMapping`][@ref),
          [`Trait`][@ref), [`HalfHessian`][@ref).

"""
struct Hessian{T} <: Mapping
    obj::T
end

"""

`HalfHessian(A)` is a container to be interpreted as the linear mapping
representing the second derivatives (times 1/2) of some objective function at
some point both represented by `A` (which can be anything).  Given `H =
HalfHessian(A)`, the contents `A` is retrieved by `contents(H)`.

For a simple quadratic objective function like:

```
f(x) = ‖D⋅x‖²
```

the half-Hessian is:

```
H = D'⋅D
```

As the half-Hessian is symmetric, a single method `apply!` has to be
implemented to apply the direct and adjoint of the mapping, the signature of
the method is:

```julia
apply!(α::Scalar, ::Type{Direct}, H::HalfHessian{typeof(A)}, x::T,
       β::Scalar, y::T)
```

where `y` is overwritten by `α*H*x + β*y` with `H*x` the result of applying `H`
(or its adjoint) to the argument `x`.  Here `T` is the relevant type of the
variables.

To allocate a new object to store the result of applying the mapping to `x`,
the default method is `vcreate(x)`.  If this is not suitable, it is sufficient
to implement the specific method:

```julia
vcreate(::Type{Direct}, H::HalfHessian{typeof(A)}, x::T)
```

See also: [`apply!`](@ref), [`vcreate`](@ref), [`LinearMapping`][@ref),
          [`Trait`][@ref), [`Hessian`][@ref).

"""
struct HalfHessian{T} <: Mapping
    obj::T
end
