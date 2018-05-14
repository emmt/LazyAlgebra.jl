#
# rules.jl -
#
# Implement rules for basic operations involving mappings.
#
#-------------------------------------------------------------------------------
#
# This file is part of the LazyAlgebra package released under the MIT "Expat"
# license.
#
# Copyright (c) 2017-2018 Éric Thiébaut.
#

import Base: *, ⋅, ∘, +, -, \, /, ctranspose, inv, A_mul_B!

"""
```julia
@callable T
```

makes concrete type `T` callable as a regular mapping that is `A(x)` yields
`apply(A,x)` for any `A` of type `T`.

"""
macro callable(T)
    quote
	(A::$T)(x) = apply(A, x)
    end
end

for T in (Adjoint, Inverse, InverseAdjoint, Scaled, Sum, Composition, Hessian)
    @eval (A::$T)(x) = apply(A, x)
end

const UnsupportedInverseOfSumOfMappings = "automatic dispatching of the inverse of a sum of mappings is not supported"

# As a general rule, do not use the constructors of tagged types directly but
# use `A'` or `ctranspose(A)` instead of `Adjoint(A)`, `inv(A)` instead of
# `Inverse(A)`, etc.  This is somewhat enforced by the following constructors
# which systematically throw an error.
Inverse(::Union{Adjoint,Inverse,InverseAdjoint,Scaled,Sum,Composition}) =
    error("use `inv(A)` instead of `Inverse(A)`")
Adjoint(::Union{Adjoint,Inverse,InverseAdjoint,Scaled,Sum,Composition}) =
    error("use `A'` or `ctranspose(A)` instead of `Adjoint(A)`")
InverseAdjoint(::Union{Adjoint,Inverse,InverseAdjoint,Scaled,Sum,Composition}) =
    error("use `inv(A')`, `inv(A)'`, `inv(ctranspose(A))` or `ctranspose(inv(A))` instead of `InverseAdjoint(A)` or `AdjointInverse(A)")

# Extend the `length` method to yield the number of components of a sum or
# composition of mappings.
Base.length(A::Union{Sum,Composition}) = length(contents(A))

"""
```julia
lineartype(A)
```

yields the *linear* type of mapping `A`, that is one of `Linear` for linear
maps or `NonLinear` for other mappings.

See also: [`Trait`](@ref).

"""
lineartype(::LinearMapping) = Linear
lineartype(::Scaled{<:LinearMapping}) = Linear
lineartype(A::Union{Scaled,Inverse}) = lineartype(A.op)
lineartype(::Mapping) = NonLinear # anything else is non-linear
function lineartype(A::Union{Sum,Composition})
    @inbounds for i in 1:length(A)
        if lineartype(A.ops[i]) != Linear
            return NonLinear
        end
    end
    return Linear
end

"""
```julia
selfadjointtype(A)
```

yields the *self-adjoint* type of mapping `A`, that is one of `SelfAdjoint` for
self-adjoint linear maps or `NonSelfAdjoint` for other mappings.

See also: [`Trait`](@ref).

"""
selfadjointtype(::Mapping) = NonSelfAdjoint
selfadjointtype(A::Union{Scaled,Inverse}) = selfadjointtype(contents(A))
function selfadjointtype(A::Union{Sum,Composition})
    @inbounds for i in 1:length(A)
        if selfadjointtype(A.ops[i]) != SelfAdjoint
            return NonSelfAdjoint
        end
    end
    return SelfAdjoint
end

"""
```julia
morphismtype(A)
```

yields the *morphism* type of mapping `A`, that is one of `Endomorphism` for
mappings whose input and output spaces are the same or `Morphism` for other
mappings.

See also: [`Trait`](@ref).

"""
morphismtype(::Mapping) = Morphism
morphismtype(A::Union{Scaled,Inverse}) = morphismtype(contents(A))
function morphismtype(A::Union{Sum,Composition})
    @inbounds for i in 1:length(A)
        if morphismtype(A.ops[i]) != Endomorphism
            return Morphism
        end
    end
    return Endomorphism
end

"""
```julia
diagonaltype(A)
```

yields the *diagonal* type of mapping `A`, that is one of `DiagonalMapping` for
diagonal linear maps or `NonDiagonalMapping` for other mappings.

See also: [`Trait`](@ref).

"""
diagonaltype(::Mapping) = NonDiagonalMapping
diagonaltype(A::Union{Scaled,Inverse}) = diagonaltype(contents(A))
function diagonaltype(A::Union{Sum,Composition})
    @inbounds for i in 1:length(A)
        if diagonaltype(A.ops[i]) != DiagonalMapping
            return NonDiagonalMapping
        end
    end
    return DiagonalMapping
end

"""
```julia
inplacetype([P=Direct,] A)
```

yields whether the mapping `A` is applicable in-place for operation `P`.  The
retuned value is one of `InPlace` or `OutOfPlace`.

See also: [`Trait`](@ref).

"""
inplacetype(::Mapping) = OutOfPlace
inplacetype(A::Union{Scaled,Inverse}) = inplacetype(contents(A))
function inplacetype(A::Union{Sum,Composition})
    @inbounds for i in 1:length(A)
        if inplacetype(A.ops[i]) != InPlace
            return OutOfPlace
        end
    end
    return InPlace
end

is_linear(A::Mapping) = (lineartype(A) == Linear)
is_selfadjoint(A::Mapping) = (selfadjointtype(A) == SelfAdjoint)
is_endomorphism(A::Mapping) = (morphismtype(A) == Endomorphism)
is_diagonal(A::Mapping) = (diagonaltype(A) == DiagonalMapping)
is_applicable_in_place(A::Mapping) = is_applicable_in_place(Direct, A)
is_applicable_in_place(::Type{P}, A::Mapping) where {P<:Operations} =
    (inplacetype(P, A) == InPlace)

# Unary minus and unary plus.
-(A::Mapping) = -1*A
+(A::Mapping) = A

# Sum of mappings.
+(A::Sum, B::Mapping) = Sum(A.ops..., B)
+(A::Mapping, B::Sum) = Sum(A, B.ops...)
+(A::Sum, B::Sum) = Sum(A.ops..., B.ops...)
+(A::Mapping, B::Mapping) = Sum(A, B)

# Dot operator involving a mapping acts a s the multiply operator.
⋅(α::Real, B::Mapping) = α*B
⋅(A::Mapping, b::T) where {T} = A*b

# Left scalar muliplication of a mapping.
*(alpha::Real, A::Scaled) = (alpha*A.sc)*A.op
*(alpha::Real, A::T) where {T<:Mapping} =
    (alpha == one(alpha) ? A : Scaled{T}(alpha, A))

# Composition of mappings and right multiplication of a mapping by a vector.
∘(A::Mapping, B::Mapping) = A*B
*(A::Composition, B::Mapping) = Composition(A.ops..., B)
*(A::Composition, B::Composition) = Composition(A.ops..., B.ops...)
*(A::Mapping, B::Composition) = Composition(A, B.ops...)
*(A::Mapping, B::Mapping) = Composition(A, B)
*(A::Mapping, x::T) where {T} = apply(A, x)

\(A::Mapping, x::T) where {T} = apply(Inverse, A, x)
\(A::Mapping, B::Mapping) = inv(A)*B
/(A::Mapping, B::Mapping) = A*inv(B)

ctranspose(A::Mapping) = _adjoint(selfadjointtype(A), A)
ctranspose(A::Adjoint) = A.op
ctranspose(A::InverseAdjoint) = inv(A.op)
ctranspose(A::Scaled) = conj(A.sc)*ctranspose(A.op)
ctranspose(A::Sum) = Sum(ntuple(i -> ctranspose(A.ops[i]), length(A)))
function ctranspose(A::Composition)
    n = length(A)
    Composition(ntuple(i -> ctranpose(A.ops[n + 1 - i]), n))
end
_adjoint(::Type{SelfAdjoint}, A::Mapping) = A
_adjoint(::Type{NonSelfAdjoint}, A::Mapping) = Adjoint(A)
_adjoint(::Type{SelfAdjoint}, A::Inverse) = A
_adjoint(::Type{NonSelfAdjoint}, A::Inverse) = InverseAdjoint(A)

inv(A::Mapping) = Inverse(A)
inv(A::Adjoint) = InverseAdjoint(A.op)
inv(A::Inverse) = A.op
inv(A::InverseAdjoint) = ctranpose(A.op)
inv(A::Scaled) = (one(Scalar)/A.sc)*inv(A.op)
inv(A::Sum) = error(UnsupportedInverseOfSumOfMappings)
function inv(A::Composition)
    n = length(A)
    Composition(ntuple(i -> inv(A.ops[n + 1 - i]), n))
end

"""
```julia
apply([P,] A, x) -> y
```

yields the result `y` of applying mapping `A` to the argument `x`.
Optional parameter `P` can be used to specify how `A` is to be applied:

* `Direct` (the default) to apply `A` and yield `y = A⋅x`;
* `Adjoint` to apply the adjoint of `A` and yield `y = A'⋅x`;
* `Inverse` to apply the inverse of `A` and yield `y = A\\x`;
* `InverseAdjoint` or `AdjointInverse` to apply the inverse of `A'` and
  yield `y = A'\\x`.

Note that not all operations may be implemented by the different types of
mappings and `Adjoint` and `InverseAdjoint` may only be applicable for linear
mappings.

Julia methods are provided so that `apply(A', x)` automatically calls
`apply(Adjoint, A, x)` so the shorter syntax may be used without performances
impact.

See also: [`Mapping`](@ref), [`apply!`](@ref), [`vcreate`](@ref).

"""
apply(A::Mapping, x) = apply(Direct, A, x)

apply(::Type{P}, A::Mapping, x) where {P<:Operations} =
    apply!(one(Scalar), P, A, x, zero(Scalar), vcreate(P, A, x))

"""
```julia
apply!([α = 1,] [P = Direct,] A::Mapping, x, [β = 0,] y) -> y
```

overwrites `y` with `α*P(A)⋅x + β*y` where `P ∈ Operations` can be `Direct`,
`Adjoint`, `Inverse` and/or `InverseAdjoint` to indicate which variant of the
mapping `A` to apply.  The convention is that the prior contents of `y` is not
used at all if `β = 0` so `y` does not need to be properly initialized in that
case.

The order of arguments can be changed and the same result as above is obtained
with:

```julia
apply!([β = 0,] y, [α = 1,] [P = Direct,] A::Mapping, x) -> y
```

The result `y` may have been allocated by:

```julia
y = vcreate([Op,] A, x)
```

Mapping sub-types only need to implement `apply!` with the specific signature:

```julia
apply!(α::Real, ::Type{P}, A::M, x, β::Real, y) -> y
```

for any supported operation `P` and where `M` is the type of the mapping.  Of
course, the types of arguments `x` and `y` may be specified as well.

See also: [`Mapping`](@ref), [`apply`](@ref), [`vcreate`](@ref).

""" apply!

# Provide fallbacks so that `Direct` is the default operation and only the
# method with signature:
#
#     apply!(α::Real, ::Type{P}, A::MappingType, x,
#            β::Real, y) where {P<:Operations}
#
# has to be implemented by mapping subtypes so we provide the necessary
# mechanism to dispatch derived methods.
apply!(A::Mapping, x, y) =
    apply!(one(Scalar), Direct, A, x, zero(Scalar), y)
apply!(α::Real, A::Mapping, x, y) =
    apply!(α, Direct, A, x, zero(Scalar), y)
apply!(A::Mapping, x, β::Real, y) =
    apply!(one(Scalar), Direct, A, x, β, y)
apply!(::Type{P}, A::Mapping, x, y) where {P<:Operations} =
    apply!(one(Scalar), P, A, x, zero(Scalar), y)
apply!(α::Real, ::Type{P}, A::Mapping, x, y) where {P<:Operations} =
    apply!(α, P, A, x, zero(Scalar), y)
apply!(::Type{P}, A::Mapping, x, β::Real, y) where {P<:Operations} =
    apply!(one(Scalar), P, A, x, β, y)

# Change order of arguments.
apply!(y, A::Mapping, x) = apply!(A, x, y)
apply!(y, ::Type{P}, A::Mapping, x) where {P<:Operations} =
    apply!(P, A, x, y)
apply!(y, α::Real, A::Mapping, x) = apply!(α, A, x, y)
apply!(y, α::Real, ::Type{P}, A::Mapping, x) where {P<:Operations} =
    apply!(α, P, A, x, y)
apply!(β::Real, y, A::Mapping, x) = apply!(A, x, β, y)
apply!(β::Real, y, ::Type{P}, A::Mapping, x) where {P<:Operations} =
    apply!(P, A, x, β, y)
apply!(β::Real, y, α::Real, A::Mapping, x) = apply!(α, A, x, β, y)
apply!(β::Real, y, α::Real, ::Type{P}, A::Mapping, x) where {P<:Operations} =
    apply!(α, P, A, x, β, y)

# Extend `A_mul_B!` so that are no needs to extend `Ac_mul_B` `Ac_mul_Bc`,
# etc. to have `A'*x`, `A*B*C*x`, etc. yield the expected result.
A_mul_B!(y::Ty, A::Mapping, x::Tx) where {Tx,Ty} =
    apply!(one(Scalar), Direct, A, x, zero(Scalar), y)

# Implemention of the `apply!(α,P,A,x,β,y)` and `vcreate(P,A,x)` methods for a
# scaled mapping.
for (P, expr) in ((:Direct, :(α*A.sc)),
                  (:Adjoint, :(α*conj(A.sc))),
                  (:Inverse, :(α/A.sc)),
                  (:InverseAdjoint, :(α/conj(A.sc))))
    @eval begin

        vcreate(::Type{$P}, A::Scaled, x) =
            vcreate($P, A.op, x)

        apply!(α::Real, ::Type{$P}, A::Scaled, x, β::Real, y) =
            apply!($expr, $P, A.op, x, β, y)

    end
end

# Implemention of the `apply!(α,P,A,x,β,y)` and `vcreate(P,A,x)` methods for
# the various decorations of a mapping so as to automativcally unveil the
# embedded mapping.
for (T1, T2, T3) in ((:Direct,         :Adjoint,        :Adjoint),
                     (:Adjoint,        :Adjoint,        :Direct),
                     (:Inverse,        :Adjoint,        :InverseAdjoint),
                     (:InverseAdjoint, :Adjoint,        :Inverse),
                     (:Direct,         :Inverse,        :Inverse),
                     (:Adjoint,        :Inverse,        :InverseAdjoint),
                     (:Inverse,        :Inverse,        :Direct),
                     (:InverseAdjoint, :Inverse,        :Adjoint),
                     (:Direct,         :InverseAdjoint, :InverseAdjoint),
                     (:Adjoint,        :InverseAdjoint, :Inverse),
                     (:Inverse,        :InverseAdjoint, :Adjoint),
                     (:InverseAdjoint, :InverseAdjoint, :Direct))
    @eval begin

        apply!(α::Real, ::Type{$T1}, A::$T2, x, β::Real, y) =
            apply!(α, $T3, A.op, x, β, y)

        vcreate(::Type{$T1}, A::$T2, x) =
            vcreate($T3, A.op, x)

    end
end

# Implementation of the `vcreate(P,A,x)` and `apply!(α,P,A,x,β,y)` and methods
# for a sum of mappings.  Note that `Sum` instances are warranted to have at
# least 2 components.

vcreate(::Type{P}, A::Sum, x) where {P<:Union{Direct,Adjoint}} =
    vcreate(P, A.ops[1], x)

function apply!(α::Real, ::Type{P}, A::Sum, x,
                β::Real, y) where {P<:Union{Direct,Adjoint}}
    b = convert(Scalar, β)
    for i in 1:length(A)
        apply!(α, P, A.ops[i], x, b, y)
        b = one(Scalar)
    end
    return y
end

vcreate(::Type{P}, A::Sum, x) where {P<:Union{Inverse,InverseAdjoint}} =
    error(UnsupportedInverseOfSumOfMappings)

apply(::Type{P}, A::Sum, x) where {P<:Union{Inverse,InverseAdjoint}} =
    error(UnsupportedInverseOfSumOfMappings)

function apply!(α::Real, ::Type{P}, A::Sum, x,
                β::Real, y) where {P<:Union{Inverse,InverseAdjoint}}
    error(UnsupportedInverseOfSumOfMappings)
end

# Implementation of the `apply!(α,P,A,x,β,y)` method for a composition of
# mappings.  There is no possible `vcreate(P,A,x)` method for a composition so
# we directly extend the `apply(P,A,x)` method.  Note that `Composition`
# instances are warranted to have at least 2 components.

apply(::Type{P}, A::Composition, x) where {P<:Union{Direct,InverseAdjoint}} =
    _apply(P, A, x, 1, length(A))

function apply!(α::Real, ::Type{P}, A::Composition, x,
                β::Real, y) where {P<:Union{Direct,InverseAdjoint}}
    # Apply mappings in order.
    return apply!(α, P, A.ops[1], _apply(P, A, x, 2, length(A)), β, y)
end

apply(::Type{P}, A::Composition, x) where {P<:Union{Adjoint,Inverse}} =
    (n = length(A); _apply(P, A, x, n, n))

function apply!(α::Real, ::Type{P}, A::Composition, x,
                β::Real, y) where {P<:Union{Adjoint,Inverse}}
    # Apply mappings in reverse order.
    n = length(A)
    return apply!(α, P, A.ops[n], _apply(P, A, x, n - 1, n), β, y)
end

function _apply(::Type{P}, A::Composition, x,
                i::Int, n::Int) where {P<:Union{Direct,InverseAdjoint}}
    if i < n
        return apply(P, A.ops[i], _apply(P, A, x, i + 1, n))
    else
        return apply(P, A.ops[i], x)
    end
end

function _apply(::Type{P}, A::Composition, x,
                i::Int, n::Int) where {P<:Union{Adjoint,Inverse}}
    if i > 1
        return apply(P, A.ops[i], _apply(P, A, x, i - 1, n))
    else
        return apply(P, A.ops[i], x)
    end
end

"""
```julia
vcreate([P,] A, x) -> y
```

yields a new instance `y` suitable for storing the result of applying mapping
`A` to the argument `x`.  Optional parameter `P ∈ Operations` can be `Direct`
(the default), `Adjoint`, `Inverse` and/or `InverseAdjoint` can be used to
specify how `A` is to be applied as explained in the documentation of the
[`apply`](@ref) method.

The method `vcreate(::Type{P}, A, x)` should be implemented by linear mappings
for any supported operations `P` and argument type for `x`.

See also: [`Mapping`](@ref), [`apply`](@ref).

"""
vcreate(A::Mapping, x) = vcreate(Direct, A, x)
