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
# Copyright (c) 2017-2020 Éric Thiébaut.
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

This definition closely follows the semantic used in the BLAS module that
`BlasReal` are all the real types supported by the BLAS library.

"""
const Reals = AbstractFloat

"""

Type `Complexes` is the set of the complexes whose real and imaginary parts are
floating point.  It is the numerical approximation of complexes in the
mathematical sense.

This definition closely follows the semantic used in the BLAS module that
`BlasComplex` are all the complex types supported by the BLAS library.

"""
const Complexes = Complex{<:Reals}

"""

Type `Floats` is the union of all floating-point types (reals and complexes).

This definition closely follows the semantic used in the BLAS module that
`BlasFloat` are all floating-point types supported by the BLAS library.

"""
const Floats = Union{Reals,Complexes}

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
vcreate(::Type{P}, A::M, x, scratch::Bool) -> y
apply!(α::Number, ::Type{P}, A::M, x, , scratch::Bool, β::Number, y) -> y
```

for any supported operation `P ∈ Operations` (`Direct`, `Adjoint`, `Inverse`
and/or `InverseAdjoint`).  See the documentation of these methods for
explanations.  Optionally, methods `P(A)` may be extended, *e.g.* to throw
exceptions if operation `P` is forbidden (or not implemented).  By default, all
these operations are assumed possible (except `Adjoint` and `InverseAdjoint`
for a nonlinear mapping).

See also: [`apply`](@ref), [`apply!`](@ref), [`vcreate`](@ref),
          [`LinearType`](@ref), [`Scalar`](@ref), [`Direct`](@ref),
          [`Adjoint`](@ref), [`Inverse`](@ref), [`InverseAdjoint`](@ref).

"""
abstract type Mapping <: Function end

abstract type LinearMapping <: Mapping end
@doc @doc(Mapping) LinearMapping

"""
```julia
Identity()
```

yields the identity linear mapping.  The purpose of this mapping is to be as
efficient as possible, hence the result of applying this mapping may be the
same as the input argument.

The identity is a singleton and is also available as:

```julia
LazyAlgebra.I
```

which is not exported by default (to avoid collisions with the `LinearAlgebra`
module).  You may do it explicitly:

```julia
import LazyAlgebra: I
```

This should not be necessary if you are `using LinearAlgebra` as the identity
`I` defined in this standard library as `UniformScaling(1)` can be
automatically converted into `LazyAlgebra` own version of the identity when
combined with other LazyAlgebra mappings.  If you are using any of these
versions of the identity `I`, then:

```julia
I/A
A\\I
```

both yield `inv(A)` for any LazyAlgebra mappings `A`.

"""
struct Identity <: LinearMapping; end

"""

The abstract type `Trait` is inherited by types indicating specific traits.

See also: [`LinearType`](@ref), [`SelfAdjointType`](@ref),
          [`DiagonalType`](@ref), [`MorphismType`](@ref).

"""
abstract type Trait end

# Trait indicating whether a mapping is certainly linear.
abstract type LinearType <: Trait end
struct NonLinear <: LinearType end
struct Linear <: LinearType end

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
struct NonDiagonalMapping <: DiagonalType end
struct DiagonalMapping <: DiagonalType end

# What kind of argument is allowed or not when calling a constructor is highly
# dependent on which simplifications are possible and should have been applied.
# A trait must be used to determine this because not all decoration types are
# known when inner constructors are defined.

abstract type CanBuildAdjointTrait end
struct CanBuildAdjoint <: CanBuildAdjointTrait end
struct CannotBuildAdjoint <: CanBuildAdjointTrait end

abstract type CanBuildInverseTrait end
struct CanBuildInverse <: CanBuildInverseTrait end
struct CannotBuildInverse <: CanBuildInverseTrait end

abstract type CanBuildInverseAdjointTrait end
struct CanBuildInverseAdjoint <: CanBuildInverseAdjointTrait end
struct CannotBuildInverseAdjoint <: CanBuildInverseAdjointTrait end

abstract type CanBuildScaledTrait end
struct CanBuildScaled <: CanBuildScaledTrait end
struct CannotBuildScaled <: CanBuildScaledTrait end

"""

Type `Direct` is a singleton type to indicate that a linear mapping should
be directly applied.  This type is part of the union `Operations`.

See also: [`LinearMapping`](@ref), [`apply`](@ref), [`Operations`](@ref).

"""
struct Direct; end

"""

Types `Adjoint`, `Inverse` and `InverseAdjoint` are used to *decorate* a
mapping to indicate the conjugate transpose and/or inverse of the mapping.  The
`adjoint` method is extended, so that in the code, it is sufficient to write
`A'` or `adjoint(A)`.  Simarly, use `inv(A)` or `I/A` (with `I` the identity)
to get the inverse of `A`.  Furthermore, `A'`, `adjoint(A)`, `inv(A)` or `I/A`
may be able to perform some simplications resulting in improved efficiency.

FIXME: To benefit from such simplications, directly calling `Adjoint(A)`, `Inverse(A)`
FIXME: or `InverseAdjoint(A)` is forbidden (it will throw an error).  Only the inner
FIXME: constructors can be used if really needed.

`AdjointInverse` is just an alias for `InverseAdjoint`.  Note that the adjoint
only makes sense for linear mappings.

See also: [`LinearMapping`](@ref), [`apply`](@ref), [`Operations`](@ref).

"""
struct Adjoint{T<:Mapping} <: LinearMapping
    op::T

    # The inner constructor prevents illegal calls to `Adjoint(A)` and make
    # sure that the argument is a simple linear mapping.
    function Adjoint{T}(A::T) where {T<:Mapping}
        CanBuildAdjointTrait(T) === CanBuildAdjoint() ||
            illegal_call_to(Adjoint)
        is_linear(A) ||
            bad_argument("taking the adjoint of non-linear mappings is not allowed")
        return new{T}(A)
    end

    # Identity is treated specially.
    Adjoint{T}(A::T) where {T<:Identity} = A
end

struct Inverse{T<:Mapping} <: Mapping
    op::T

    # The inner constructor prevents illegal calls like `Inverse(A)`.
    function Inverse{T}(A::T) where {T<:Mapping}
        CanBuildInverseTrait(T) === CanBuildInverse() ||
            illegal_call_to(Inverse)
        return new{T}(A)
    end

    # Identity is treated specially.
    Inverse{T}(A::T) where {T<:Identity} = A
end

struct InverseAdjoint{T<:Mapping} <: LinearMapping
    op::T

    # The inner constructor prevents illegal calls to `InverseAdjoint(A)` and
    # make sure that the argument is a simple linear mapping or a sum of linear mappings.
    function InverseAdjoint{T}(A::T) where {T<:Mapping}
        CanBuildInverseAdjointTrait(T) === CanBuildInverseAdjoint() ||
            illegal_call_to(InverseAdjoint)
        is_linear(A) ||
            bad_argument("taking the inverse adjoint of non-linear mappings is not allowed")
        return new{T}(A)
    end

    # Identity is treated specially.
    InverseAdjoint{T}(A::T) where {T<:Identity} = A
end

const AdjointInverse{T} = InverseAdjoint{T}

for T in (:Inverse, :InverseAdjoint, :AdjointInverse)
    @eval @doc @doc(Adjoint) $T
end

"""

`Operations` is the union of the possible ways to apply a mapping: `Direct`,
`Adjoint`, `Inverse` and `InverseAdjoint` (or its alias `AdjointInverse`).

See also: [`LinearMapping`](@ref), [`apply`](@ref), [`Direct`](@ref),
          [`Adjoint`](@ref).

"""
const Operations = Union{Direct,Adjoint,Inverse,InverseAdjoint}

"""

A `Scaled` mapping is used to represent a mapping times a scalar.  End-users
should not use athe `Scaled` constructor directly but rather use expressions
like `λ*A` with `λ` a scalar number and `A` a mapping, as LazyAlgebra may be
able to make some simplifications resulting in improved efficiency.

"""
struct Scaled{T<:Mapping,S<:Number} <: Mapping
    λ::S
    M::T
    Scaled{T,S}(λ::S, M::Mapping) where {S<:Number,T<:Mapping} =
        new{T,S}(λ, M)
end

"""

A `Sum` is used to represent an arbitrary sum of mappings.  End-users should
not use the `Sum` constructor directly but rather use the `+` operator as
LazyAlgebra may be able to make some simplifications resulting in improved
efficiency.

"""
struct Sum{N,T<:NTuple{N,Mapping}} <: Mapping
    ops::T

    # The inner constructor ensures that the number of arguments is at least 2.
    function Sum{N,T}(ops::T) where {N,T<:NTuple{N,Mapping}}
        N ≥ 2 || error("a sum of mappings has at least 2 components")
        new{N,T}(ops)
    end
end

"""

A `Composition` is used to represent an arbitrary composition of mappings.
Constructor `Composition(A,B)` may be extended in code implementing specific
mappings of linear operators to provide *automatic* simplifications.  The
end-user should not use the `Composition` constructor directly but use the
operators `*`, `∘` or `⋅` instead as LazyAlgebra may be able to make some
simplifications resulting in improved efficiency.

"""
struct Composition{N,T<:NTuple{N,Mapping}} <: Mapping
    ops::T

    # The inner constructor ensures that the number of arguments is at least 2.
    function Composition{N,T}(ops::T) where {N,T<:NTuple{N,Mapping}}
        N ≥ 2 || error("a composition of mappings has at least 2 components")
        new{N,T}(ops)
    end
end

"""

`Gram{typeof(A)}` is an alias to represent the type of the construction
`gram(A) = A'*A` for the linear mapping `A`.

See also [`gram`](@ref).

"""
const Gram{T<:LinearMapping} = Composition{2,Tuple{Adjoint{T},T}}
