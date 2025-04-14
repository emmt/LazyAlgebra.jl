#
# types.jl -
#
# Type definitions and (some) constructors for linear algebra.
#
#-------------------------------------------------------------------------------
#
# This file is part of LazyAlgebra (https://github.com/emmt/LazyAlgebra.jl)
# released under the MIT "Expat" license.
#
# Copyright (c) 2017-2025 Éric Thiébaut.
#

struct SingularSystem <: Exception
    msg::String
end
showerror(io::IO, err::SingularSystem) =
    print(io, "singular linear system (", err.msg, ")")

struct NonPositiveDefinite <: Exception
    msg::String
end
showerror(io::IO, err::NonPositiveDefinite) =
    print(io, "non-positive definite operator (", err.msg, ")")

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
    Reals

is the set of the floating point types. It is the numerical approximation of
reals in the mathematical sense.

This definition closely follows the semantic used in the BLAS module that
`BlasReal` are all the real types supported by the BLAS library.

"""
const Reals = AbstractFloat

"""
    Complexes

is the set of the complexes whose real and imaginary parts are floating point.
It is the numerical approximation of complexes in the mathematical sense.

This definition closely follows the semantic used in the BLAS module that
`BlasComplex` are all the complex types supported by the BLAS library.

"""
const Complexes = Complex{<:Reals}

"""
    Floats

is the union of all floating-point types (reals and complexes).

This definition closely follows the semantic used in the BLAS module that
`BlasFloat` are all floating-point types supported by the BLAS library.

"""
const Floats = Union{Reals,Complexes}

"""

An `Operator` is any linear function between two variables spaces. Assuming upper case
Latin letters denote mappings, lower case Latin letters denote variables, and Greek
letters denote scalars, then:

* `A*x` or `A⋅x` yields the result of applying the mapping `A` to `x`;

* `A\\x` yields the result of applying the inverse of `A` to `x`;

Simple constructions are allowed for any kind of mappings and can be used to create new
instances of mappings which behave correctly. For instance:

* `B = α*A` (where `α` is a real) is a mapping which behaves as `A` times `α`; that is
  `B⋅x` yields the same result as `α*(A⋅x)`.

* `C = A + B + ...` is a mapping which behaves as the sum of the mappings `A`, `B`, ...;
  that is `C⋅x` yields the same result as `A⋅x + B⋅x + ...`.

* `C = A*B` or `C = A⋅B` is a mapping which behaves as the composition of the mappings `A`
  and `B`; that is `C⋅x` yields the same result as `A⋅(B.x)`. As for the sum of mappings,
  there may be an arbitrary number of mappings in a composition; for example, if `D =
  A*B*C` then `D⋅x` yields the same result as `A⋅(B⋅(C⋅x))`.

* `C = A\\B` is a mapping such that `C⋅x` yields the same result as `A\\(B⋅x)`.

* `C = A/B` is a mapping such that `C⋅x` yields the same result as `A⋅(B\\x)`.

These constructions can be combined to build up more complex mappings. For
example:

* `D = A*(B + C)` is a mapping such that `C⋅x` yields the same result as
  `A⋅(B⋅x + C⋅x)`.

An `Operator` is any linear mapping between two spaces. This abstract sub-type of
`Operator` is introduced to extend the notion of *matrices* and *vectors*. Assuming the
type of `A` inherits from `Operator`, then:

* `A'⋅x` and `A'*x` yields the result of applying the adjoint of the mapping `A` to `x`;

* `A'\\x` yields the result of applying the adjoint of the inverse of mapping `A` to `x`.

* `B = A'` is a mapping such that `B⋅x` yields the same result as `A'⋅x`.

The following methods should be implemented for a mapping `A` of specific type `M <:
Operator`:

```julia
vcreate(::Type{P}, A::M, x, scratch::Bool) -> y
apply!(α::Number, ::Type{P}, A::M, x, , scratch::Bool, β::Number, y) -> y
```

for any supported operation `P ∈ Operations` (`Direct`, `Adjoint`, `Inverse` and/or
`InverseAdjoint`). See the documentation of these methods for explanations. Optionally,
methods `P(A)` may be extended, *e.g.* to throw exceptions if operation `P` is forbidden
(or not implemented). By default, all these operations are assumed possible (except
`Adjoint` and `InverseAdjoint` for a nonlinear mapping).

See also: [`apply`](@ref), [`apply!`](@ref), [`vcreate`](@ref), [`Scalar`](@ref),
          [`Direct`](@ref), [`Adjoint`](@ref), [`Inverse`](@ref),
          [`InverseAdjoint`](@ref).

"""
abstract type Operator end

"""
    Identity()

yields the identity operator. The identity is a singleton and is also available as:

    Id

The `LinearAlgebra` module of the standard library exports a constant `I` which also
corresponds to the identity (but in the sense of a matrix). When `I` is combined with any
`LazyAlgebra` operator, it is recognized as an alias of `Id`. So that, for instance,
`I/A`, `A\\I`, `Id/A` and `A\\Id` all yield `inv(A)` for any `LazyAlgebra` mapping `A`.

"""
struct Identity <: Operator end

"""
    Trait

is the abstract type inherited by types indicating specific traits.

See also: [`SelfAdjointType`](@ref),
          [`DiagonalType`](@ref), [`MorphismType`](@ref).

"""
abstract type Trait end

# Trait indicating whether a mapping is certainly a self-adjoint linear map.
abstract type SelfAdjointType <: Trait end
struct NonSelfAdjoint <: SelfAdjointType end
struct SelfAdjoint <: SelfAdjointType end

# Trait indicating whether a mapping is certainly an endomorphism.
abstract type MorphismType <: Trait end
struct Morphism <: MorphismType end
struct Endomorphism <: MorphismType end

# Trait indicating whether a mapping is certainly a diagonal linear mapping.
abstract type DiagonalType <: Trait end
struct NonDiagonalOperator <: DiagonalType end
struct DiagonalOperator <: DiagonalType end

"""

Type `Direct` is a singleton type to indicate that a linear mapping should
be directly applied.  This type is part of the union `Operations`.

See also: [`Operator`](@ref), [`apply`](@ref), [`Operations`](@ref).

"""
struct Direct; end

"""
    Adjoint(A) -> obj

yields an object instance `obj` representing `A'`, the adjoint of the linear
mapping `A`.

Directly calling this constructor is discouraged, use an expression like `A'`
instead and benefit from automatic simplification rules.

Call [`unveil(obj)`](@ref) to reveal the linear mapping `A` embedded in `obj`.

See also [`DecoratedOperator`](@ref).

"""
struct Adjoint{T<:Operator} <: Operator
    op::T

    # The outer constructors prevent most illegal calls to `Adjoint(A)` we
    # just have to check that the argument is a simple linear mapping.
    function Adjoint{T}(A::T) where {T<:Operator}
        return new{T}(A)
    end
end

"""
    Inverse(A) -> obj

yields an object instance `obj` representing the inverse of the mapping `A`.

Directly calling this constructor is discouraged, call `inv(A)` or use an
expression like `Id/A` instead and benefit from automatic simplification rules.

Call [`unveil(obj)`](@ref) to reveal the mapping `A` embedded in `obj`.

See also [`DecoratedOperator`](@ref).

"""
struct Inverse{T<:Operator} <: Operator
    op::T

    # The outer constructors prevent all illegal calls to `Inverse(A)` so there
    # is nothing more to check.
    Inverse{T}(A::T) where {T<:Operator} = new{T}(A)
end

"""
    InverseAdjoint(A) -> obj

yields an object instance `obj` representing the inverse of the adjoint of the
linear mapping `A`.

Directly calling this constructor is discouraged, use expressions like
`inv(A')`, `inv(A')` or `Id/A'` instead and benefit from automatic
simplification rules.

Call [`unveil(obj)`](@ref) to reveal the mapping `A` embedded in `obj`.

`AdjointInverse` is an alias for `InverseAdjoint`.

See also [`DecoratedOperator`](@ref).

"""
struct InverseAdjoint{T<:Operator} <: Operator
    op::T

    # The outer constructors prevent most illegal calls to `InverseAdjoint(A)`
    # we just have to check that the argument is a simple linear mapping.
    function InverseAdjoint{T}(A::T) where {T<:Operator}
        return new{T}(A)
    end
end

const AdjointInverse{T} = InverseAdjoint{T}
@doc @doc(InverseAdjoint) AdjointInverse

"""
    Gram(A) -> obj

yields an object instance `obj` representing the composition `A'*A` for the
linear mapping `A`.

Directly calling this constructor is discouraged, call [`gram(A)`](@ref) or use
expression `A'*A` instead and benefit from automatic simplification rules.

Call [`unveil(obj)`](@ref) to reveal the linear mapping `A` embedded in `obj`.

See also [`gram`](@ref), [`unveil`](@ref) and [`DecoratedOperator`](@ref).

"""
struct Gram{T<:Operator} <: Operator
    op::T

    # The outer constructors prevent most illegal calls to `Gram(A)` we
    # just have to check that the argument is a simple linear mapping.
    function Gram{T}(A::T) where {T<:Operator}
        return new{T}(A)
    end
end

"""
    DecoratedOperator

is the union of the *decorated* mapping types: [`Adjoint`](@ref),
[`Inverse`](@ref), [`InverseAdjoint`](@ref), and [`Gram`](@ref).

The method [`unveil(A)`](@ref) can be called to reveal the mapping embedded in
a decorated mapping `A`.

"""
const DecoratedOperator = Union{Adjoint,Inverse,InverseAdjoint,Gram}

"""
    Operations

is the union of the possible variants to apply a mapping: [`Direct`](@ref),
[`Adjoint`](@ref), [`Inverse`](@ref) and [`InverseAdjoint`](@ref) (or its alias
[`AdjointInverse`](@ref)).

See also: [`apply`](@ref) and [`apply!`](@ref).

"""
const Operations = Union{Direct,Adjoint,Inverse,InverseAdjoint}

"""
    Scaled(λ, M) -> obj

yields an object instance `obj` representing `λ*M`, the mapping `M` multiplied
by a scalar `λ`.

Directly calling this constructor is discouraged, use expressions like `λ*M`
instead and benefit from automatic simplification rules.

Call [`multiplier(obj)`](@ref) and [`unscaled(obj)`](@ref) with a scaled
mapping `obj = λ*M` to retrieve `λ` and `M` respectively.

"""
struct Scaled{T<:Operator,S<:Number} <: Operator
    λ::S
    M::T
    Scaled{T,S}(λ::S, M::Operator) where {S<:Number,T<:Operator} =
        new{T,S}(λ, M)
end

"""
    Sum(A, B, ...) -> obj

yields an object instance `obj` representing the sum `A + B + ...` of the
mappings `A`, `B`, ...

Directly calling this constructor is discouraged, use expressions like `A + B +
...` instead and benefit from automatic simplification rules.

Call [`terms(obj)`](@ref) retrieve the tuple `(A,B,...)` of the terms of the
sum stored in `obj`.

"""
struct Sum{N,T<:NTuple{N,Operator}} <: Operator
    ops::T

    # The inner constructor ensures that the number of arguments is at least 2.
    function Sum{N,T}(ops::T) where {N,T<:NTuple{N,Operator}}
        N ≥ 2 ||
            throw(ArgumentError("a sum of mappings has at least 2 components"))
        new{N,T}(ops)
    end
end

"""
    Composition(A, B, ...) -> obj

yields an object instance `obj` representing the composition `A*B*...` of the
mappings `A`, `B`, ...

Directly calling this constructor is discouraged, use expressions like
`A*B*...` `A∘B∘...` or `A⋅B⋅...` instead and benefit from automatic
simplification rules.

Call [`terms(obj)`](@ref) retrieve the tuple `(A,B,...)` of the terms of the
composition stored in `obj`.

"""
struct Composition{N,T<:NTuple{N,Operator}} <: Operator
    ops::T

    # The inner constructor ensures that the number of arguments is at least 2.
    function Composition{N,T}(ops::T) where {N,T<:NTuple{N,Operator}}
        N ≥ 2 ||
            throw(ArgumentError("a composition of mappings has at least 2 components"))
        new{N,T}(ops)
    end
end
