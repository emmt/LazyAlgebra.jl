#
# types.jl -
#
# Definition of types and constants for linear algebra.
#
#-----------------------------------------------------------------------------------------

"""
    Operator

is the abstract type representing any linear function between two variables spaces in
`LazyAlgebra`. Using upper-case Latin letters to denote operators, lower-case Latin
letters to denote variables, and Greek letters to denote scalars, then:

* `A*x` or `A(x)` yields the result of applying the operator `A` to `x`;

* `A'*x` yields the result of applying the adjoint of `A` to `x`;

* `inv(A)*x` and `A\\x` yield the result of applying the inverse of `A` to `x`;

* `A'\\x`, `inv(A')*x`, and `inv(A)'*x` yield the result of applying the inverse-adjoint
  of the operator `A` to `x`.

Constructions are allowed for any kind of operators and can be used to create new
operators which behave as above. Compared to matrices in Julia, operators are not limited
to be 2-dimensional. Moreover, combining operators does not (generally) involve explicitly
computing the coefficients of the resulting operator but, instead, yields a composite
operator storing the structure and the components of the combination. This combination can
be manipulated (e.g. simplified), re-used in another combination, or applied to an
argument. For example:

* `B = A'` and `B = adjoint(A)` build an operator `B` such that `B*x` yields the same
  result as `A'*x`.

* `B = inv(A)`, `B = Id/A`, and `B = A\\Id` (where [`Id`](@ref) represents the *universal
  identity*) , build an operator `B` such that `B*x` yields the same result as `inv(A)*x`.

* `B = α*A` (where `α` is a real) builds an operator `B` which behaves as `A` times `α`;
  that is `B*x` yields the same result as `α*(A*x)`.

* `C = A + B + ...` builds an operator `C` which behaves as the sum of the operators `A`,
  `B`, ...; that is `C*x` yields the same result as `A*x + B*x + ...`.

* `C = A*B` builds an operator `C` which behaves as the composition of the operators `A`
  and `B`; that is `C*x` yields the same result as `A*(B*x)`. As for the sum of operators,
  there may be an arbitrary number of operators in a composition; for example, if `D =
  A*B*C` then `D*x` yields the same result as `A*(B*(C*x))`.

* `C = A\\B` yields an operator `C` such that `C*x` yields the same result as `A\\(B*x)`.

* `C = A/B` yields an operator `C` such that `C*x` yields the same result as `A*(B\\x)`.

These constructions can be combined to build up more complex operators. For example:

* `D = A*(B + C)` yields an operator `D` such that `D*x` yields the same result as `A*(B*x
  + C*x)`.

See also [`vmul`](@ref) and [`vmul!`](@ref).

"""
abstract type Operator end

if VERSION ≥ v"1.3.0"
    # Only since Julia 1.3, methods can be added to an abstract type.
    @callable Operator
end

"""
    B = A'
    B = adjoint(A)
    B = LazyAlgebra.Adjoint(A)

yield a linear operator `B` representing the *adjoint* (conjugate transpose) of the linear
operator `A`.

Taking the adjoint of `B` yields back `A`, that is `B' === A` holds. Calling
`Base.parent(B)` or `B[]` also reveals the linear operator `A` embedded in `B = A'`.

Also see [`LazyAlgebra.Inverse`](@ref).

"""
struct Adjoint{T<:Operator} <: Operator
    parent::T
end

@callable Adjoint

"""
    B = inv(A)
    B = A\\Id
    B = Id/A
    B = LazyAlgebra.Inverse(A)

yield a linear operator `B` representing the *inverse* of the linear operator `A`
regardless whether this inverse exists or not.

Taking the inverse of `B` yields back `A`, that is `inv(B)' === A` holds. Calling
`Base.parent(B)` or `B[]` also reveals the linear operator `A` embedded in `B = inv(A)`.

Also see [`LazyAlgebra.Adjoint`](@ref).

"""
struct Inverse{T<:Operator} <: Operator
    parent::T
end

@callable Inverse

"""
    LazyAlgebra.InverseAdjoint{A}

is an alias for the type of an operator that is the inverse adjoint (or adjoint inverse)
of an operator of type `A`.

See also [`LazyAlgebra.Adjoint`](@ref) and [`LazyAlgebra.Inverse`](@ref).

"""
const InverseAdjoint{A} = Union{Inverse{Adjoint{A}},Adjoint{Inverse{A}}}

# Any of A, A', inv(A), inv(A'), or inv(A)'.
const AnyVariant{A} = Union{A,Adjoint{A},Inverse{A},InverseAdjoint{A}}

"""
    B = Gram(A)
    B = simplify(A'*A)

yield a linear operator `B` representing the composition `A'*A` for the linear operator
`A`.

Calling `Base.parent(B)` or `B[]` reveals the bare linear operator `A` embedded in `B`.

"""
struct Gram{T<:Operator} <: Operator
    parent::T
end

@callable Gram

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

@callable Sum

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

@callable Prod

# Alias representing `λ*A`, the linear operator `A` multiplied by a scalar `λ`.
# Call [`LazyAlgebra.multiplier(B)`](@ref) and [`unscaled(B)`](@ref) with a scaled
# operator `B = λ*A` to retrieve `λ` and `A` respectively.
const Scaled{L<:Number,R} = Prod{L,R}

abstract type InputShape end
struct InputShapeUnknown <: InputShape end
struct HasInputShape{N}  <: InputShape end

abstract type OutputShape end
struct OutputShapeUnknown <: OutputShape end
struct HasOutputShape{N}  <: OutputShape end

abstract type InputEltype end
struct InputEltypeUnknown <: InputEltype end
struct HasInputEltype     <: InputEltype end

abstract type OutputEltype end
struct OutputEltypeUnknown <: OutputEltype end
struct HasOutputEltype     <: OutputEltype end

struct Identity{I} <: Operator
    shape::I
    # A private constructor is needed to "filter" the input shape.
    global _Identity
    _Identity(shape::I) where {I<:Union{Colon,Dims,ArrayAxes}} = new{I}(shape)
end

@callable Identity

"""
    Id

is the *universal identity* operator in `LazyAlgebra`; it is a *singleton* of type
[`Identity{Colon}`](@ref LazyAlgebra.Identity).

The `LinearAlgebra` module of the standard library exports a constant `I` which also
corresponds to the identity (but in the sense of a matrix). When `I` is combined with any
`LazyAlgebra` operator, it is recognized as an alias of `Id`. So that, for instance,
`I/A`, `A\\I`, `I/A` and `A\\I` all yield `inv(A)` for any `LazyAlgebra` operator `A`.

"""
const Id = _Identity(:)
const UniversalIdentity = Identity{Colon}
const ShapedIdentity{N} = Identity{<:Union{Dims{N},ArrayAxes{N}}}

struct Diag{D<:AbstractArray} <: Operator
    diag::D
end

@callable Diag

struct LazyMap{T,N,L,F,A<:AbstractArray{<:Any,N}} <: AbstractArray{T,N}
    func::F
    arr::A
    LazyMap{T}(func::F, arr::A) where {T,N,F<:Function,A<:AbstractArray{<:Any,N}} =
        new{T,N,IndexStyle(A)==IndexLinear(),F,A}(func, arr)
end

abstract type AbstractRankOneOperator{U<:AbstractArray,V<:AbstractArray} <: Operator end

struct RankOneOperator{U,V} <: AbstractRankOneOperator{U,V}
    u::U
    v::V
end

@callable RankOneOperator

struct SymmetricRankOneOperator{U} <: AbstractRankOneOperator{U,U}
    u::U
end

@callable SymmetricRankOneOperator
