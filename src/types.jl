#
# types.jl -
#
# Definition of types and constants for linear algebra.
#
#-----------------------------------------------------------------------------------------
#
# This file is part of LazyAlgebra (https://github.com/emmt/LazyAlgebra.jl) released under
# the MIT "Expat" license.
#
# Copyright (c) 2017-2025, Éric Thiébaut.
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
vmul!(α::Number, ::Type{P}, A::M, x, , scratch::Bool, β::Number, y) -> y
```

for any supported operation `P ∈ Operations` (`Direct`, `Adjoint`, `Inverse` and/or
`InverseAdjoint`). See the documentation of these methods for explanations. Optionally,
methods `P(A)` may be extended, *e.g.* to throw exceptions if operation `P` is forbidden
(or not implemented). By default, all these operations are assumed possible (except
`Adjoint` and `InverseAdjoint` for a nonlinear mapping).

See also [`vmul`](@ref), [`vmul!`](@ref), [`vcreate`](@ref),
[`LazyAlgebra.Adjoint`](@ref), [``LazyAlgebra.Inverse`](@ref),
[``LazyAlgebra.InverseAdjoint`](@ref).

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
    Id

is the identity operator in `LazyAlgebra`; it is a *singleton*: the only instance of
[`LazyAlgebra.Identity`](@ref).

The `LinearAlgebra` module of the standard library exports a constant `I` which also
corresponds to the identity (but in the sense of a matrix). When `I` is combined with any
`LazyAlgebra` operator, it is recognized as an alias of `Id`. So that, for instance,
`I/A`, `A\\I`, `I/A` and `A\\I` all yield `inv(A)` for any `LazyAlgebra` operator `A`.

"""
const Id = Identity()

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
    B = A'
    B = adjoint(A)
    B = LazyAlgebra.Adjoint(A)

yield a linear operator `B` representing the *adjoint* (conjugate transpose) of the linear
operator `A`.

Taking the adjoint of `B` yields back `A`, that is `B' === A` holds. `B[]` and
`Base.parent(B)` also yield the linear operator `A` embedded in `B = A'`.

Also see [`LazyAlgebra.Inverse`](@ref).

"""
struct Adjoint{T<:Operator} <: Operator
    parent::T
end

"""
    B = inv(A)
    B = A\\Id
    B = Id/A
    B = LazyAlgebra.Inverse(A)

yield a linear operator `B` representing the *inverse* of the linear operator `A`
regardless whether this inverse exists or not.

Taking the inverse of `B` yields back `A`, that is `inv(B)' === A` holds. `B[]` and
`Base.parent(B)` also yield the linear operator `A` embedded in `B = inv(A)`.

Also see [`LazyAlgebra.Adjoint`](@ref).

"""
struct Inverse{T<:Operator} <: Operator
    parent::T
end

"""
    LazyAlgebra.InverseAdjoint{A}

is an alias for the type of an operator that is the inverse adjoint (or adjoint inverse)
of an operator of type `A`.

See also [`LazyAlgebra.Adjoint`](@ref) and [`LazyAlgebra.Inverse`](@ref).

"""
const InverseAdjoint{A} = Union{Inverse{Adjoint{A}},Adjoint{Inverse{A}}}

"""
    B = Gram(A)
    B = simplify(A'*A)

yield a linear operator `B` representing the composition `A'*A` for the linear mapping
`A`.

Method [`LazyAlgebra.unveil(B)`](@ref) may be used to reveal the bare linear operator `A`
embedded in `B`.

"""
struct Gram{T<:Operator} <: Operator
    parent::T
end

"""
    LazyAlgebra.DecoratedOperator{A}

is the union of the *decorated* operator types: [`LazyAlgebra.Adjoint`](@ref),
[`LazyAlgebra.Inverse`](@ref), and [`LazyAlgebra.Gram`](@ref). Parameter `A` is the type
of the decorated operator.

The `Base.parent` method can be called to reveal the operator embedded in a decorated
operator.

"""
const DecoratedOperator{A<:Operator} = Union{Adjoint{A},Inverse{A},Gram{A}}

"""
    Operations

is the union of the possible variants to apply a mapping: [`Direct`](@ref),
[`Adjoint`](@ref), [`Inverse`](@ref) and [`InverseAdjoint`](@ref) (or its alias
[`AdjointInverse`](@ref)).

See also [`vmul`](@ref) and [`vmul!`](@ref).

"""
const Operations = Union{Adjoint,Inverse} # FIXME:

# Type of operands in a product.
const Operand = Union{Number,Operator}

"""
    C = A*B # if at least one of A or B is an Operator
    C = LazyAlgebra.Prod(A::Union{Number,Operator}, B::Union{Number,Operator})

yield the result of multiplying operand `A` by operand `B`. If both operands are scalars
the result is a scalar; otherwise an instance of `Prod` is returned. If any operand is an
operator, `A*B` yields the same thing as `Prod(A, B)`.

If `C` is an instance of `LazyAlgebra.Prod`, then `C[1]` and `C[2]` respectively yield the
left and right operands of `C`. However, due to simplifications that may occur, these are
not necessarily `A` and `B`.

When composing instances of `Prod` whose operands are operators, right associativity is
applied so as to keep the operands in suitable order when applying the composite operator:

```julia
A*B*C   -> Prod(A, Prod(B, C))
(A*B)*C -> Prod(A, Prod(B, C))
A*(B*C) -> Prod(A, Prod(B, C))
```

"""
struct Prod{L<:Operand,R<:Operator} <: Operator
    # In a `Prod` object, only the left operand can be a scalar, the right operand must be
    # an operator. This is to force factorization of scalar multipliers to the left of
    # products.
    operands::Tuple{L,R}
    Prod(left::L, right::R) where {L<:Operand,R<:Operator} = new{L,R}((left, right))
end

# Alias representing `λ*A`, the linear operator `A` multiplied by a scalar `λ`.
# Call [`LazyAlgebra.multiplier(B)`](@ref) and [`unscaled(B)`](@ref) with a scaled
# operator `B = λ*A` to retrieve `λ` and `A` respectively.
const Scaled{L<:Number,R} = Prod{L,R}

const Composition = Prod # FIXME:

"""
    C = A + B
    C = LazyAlgebra.Sum(A::Operator, B::Operator)

yields a linear operator `C` representing the sum of the linear operators `A` and `B`.

If `C` is an instance of `LazyAlgebra.Sum`, then `C[1]` and `C[2]` respectively yield the
left and right operands of `C`. However, due to simplifications that may occur, these are
not necessarily `A` and `B`.

"""
struct Sum{L<:Operator,R<:Operator} <: Operator
    operands::Tuple{L,R}
    Sum(left::L, right::R) where {L<:Operator,R<:Operator} = new{L,R}((left, right))
end
