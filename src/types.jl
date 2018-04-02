#
# types.jl -
#
# Type definitions and (some) constructors for linear algebra.
#
#-------------------------------------------------------------------------------
#
# This file is part of the LazyAlgebra package released under the MIT "Expat"
# license.
#
# Copyright (c) 2017-2018 Éric Thiébaut.
#

struct SingularSystem <: Exception
    msg::String
end
Base.showerror(io::IO, err::SingularSystem) =
    print(io, "singular linear system ($(err.msg))")

struct NonPositiveDefinite <: Exception
    msg::String
end
Base.showerror(io::IO, err::NonPositiveDefinite) =
    print(io, "non-positive definite operator ($(err.msg))")

"""

A `Scalar` is used to represent multipliers or scaling factors when combining
mappings.  For now, scalars are reals.

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
          [`lineartype`](@ref), [`is_applicable_in_place`](@ref),
          [`Scalar`](@ref), [`Direct`](@ref), [`Adjoint`](@ref),
          [`Inverse`](@ref), [`InverseAdjoint`](@ref).

"""
abstract type Mapping end

abstract type LinearMapping <: Mapping end
@doc @doc(Mapping) LinearMapping

"""

The linear trait indicates whether a mapping is linear or not.  Abstract type
`LinearType` has two concrete singleton subtypes: `Linear` for linear mappings
and `Nonlinear` for other mappings.

See also: [`lineartype`](@ref).

"""
abstract type LinearType end

for T in (:Nonlinear, :Linear)
    @eval begin
        struct $T <: LinearType end
        @doc @doc(LinearType) $T
    end
end

"""

Abstract type `SelfAdjointOperator` is to be inherited by linear mappings
whose adjoint is equal to themself.  Such mappings must only implement
methods for the `Direct` and `Inverse` operations (if applicable).

See also: [`LinearMapping`](@ref).

"""
abstract type SelfAdjointOperator <: LinearMapping end

"""

Type `Direct` is a singleton type to indicate that a linear mapping should
be directly applied.  This type is part of the union `Operations`.

See also: [`LinearMapping`](@ref), [`apply`](@ref), [`Operations`](@ref).

"""
struct Direct; end

"""

Types `Adjoint`, `Inverse` and `InverseAdjoint` are used to *decorate* a
mapping to indicate the conjugate transpose and/or inverse of the mapping.  The
`ctranspose` method is extended, so that in the code, it is sufficient (and
recommended) to write `A'` instead of `Adjoint(A)`.  Furthermore, `A'` or
`ctranspose(A)` may be able to perform some simplications resulting in improved
efficiency.  `AdjointInverse` is just an alias for `InverseAdjoint`.  Note that
the adjoint only makes sense for linear mappings.

See also: [`LinearMapping`](@ref), [`apply`](@ref), [`Operations`](@ref).

"""
struct Adjoint{T<:Mapping} <: LinearMapping
    op::T
    # The inner constructors make sure that the argument is a linear mapping.
    Adjoint{T}(A::T) where {T<:LinearMapping} = new{T}(A)
    function Adjoint{T}(A::T) where {T<:Mapping}
        if lineartype(A) != Linear
            error("taking the adjoint of non-linear mappings is not allowed")
        end
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
        if lineartype(A) != Linear
            error("taking the inverse adjoint of non-linear mappings is not allowed")
        end
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

A `Scaled` mapping is used to represent a mappings times a scalar, it should
not be used directly.  Use the `*` operator (with a scalar left operand)
instead as it may be able to make some simplifications resulting in improved
efficiency.

"""
struct Scaled{T<:Mapping} <: Mapping
    sc::Scalar
    op::T
end

"""

A `Sum` is used to represent an arbitrary sum of mappings, it should not be
used directly.  Use the `+` operator instead as it may be able to make some
simplifications resulting in improved efficiency.

"""
struct Sum{T<:Tuple{Vararg{Mapping}}} <: Mapping
    ops::T

    # The inner constructor ensures that the number of arguments is at least 2.
    function Sum{T}(ops::T) where {T<:Tuple{Vararg{Mapping}}}
        if length(ops) < 2
            error("a sum of mappings has at least 2 components")
        end
        return new{T}(ops)
    end
end
Sum(ops::T) where {T<:Tuple{Vararg{Mapping}}} = Sum{T}(ops)
Sum(ops::Mapping...) = Sum(ops)

"""

A `Composition` is used to represent an arbitrary composition of mappings, it
should be used directly.  Use the `.` or `*` operators instead as they may be
able to make some simplifications resulting in improved efficiency.

"""
struct Composition{T<:Tuple{Vararg{Mapping}}} <: Mapping
    ops::T

    # The inner constructor ensures that the number of arguments is at least 2.
    function Composition{T}(ops::T) where {T<:Tuple{Vararg{Mapping}}}
        if length(ops) < 2
            error("a composition of mappings has at least 2 components")
        end
        return new{T}(ops)
    end
end
Composition(ops::T) where {T<:Tuple{Vararg{Mapping}}} = Composition{T}(ops)
Composition(ops::Mapping...) = Composition(ops)
