#
# rules.jl -
#
# Implement rules for basic operations involving operators.
#
#-------------------------------------------------------------------------------
#
# This file is part of the LazyAlgebra package released under the MIT "Expat"
# license.
#
# Copyright (c) 2017-2018 Éric Thiébaut.
#

import Base: *, ⋅, +, -, \, /, ctranspose, inv, A_mul_B!

Base.length(A::Sum) = length(A.ops)
Base.length(A::Composition) = length(A.ops)

"""
```julia
lineartype(A)
```

yields the linear type of a operator (or mapping) `A`, that is one of `Linear`
or `NonLinear` which are singleton types.

"""
lineartype(::LinearOperator) = Linear
lineartype(::Scaled{<:LinearOperator}) = Linear
lineartype(A::Union{Scaled,Inverse}) = lineartype(A.op)
lineartype(::Mapping) = Nonlinear # anything else is non-linear
function lineartype(A::Union{Sum,Composition})
    @inbounds for i in 1:length(A)
        if lineartype(A.ops[i]) != Linear
            return Nonlinear
        end
    end
    return Linear
end

is_linear(x) = (lineartype(x) == Linear)
is_nonlinear(x) = ! is_linear(x)

# Unary minus and unary plus.
-(A::Mapping) = -1*A
+(A::Mapping) = A

# Sum of mappings.
+(A::Sum, B::Mapping) = Sum((A.ops..., B))
+(A::Mapping, B::Sum) = Sum((A, B.ops...))
+(A::Sum, B::Sum) = Sum((A.ops..., B.ops...))
+(A::Mapping, B::Mapping) = Sum((A, B))

# Left scalar muliplication of a mapping.
*(alpha::Real, A::Scaled) = (alpha*A.sc)*A.op
*(alpha::Real, A::T) where {T<:Mapping} =
    (alpha == one(alpha) ? A : Scaled{T}(alpha, A))

# Composition of mappings and right muliplication of a mapping by a vector.
for op in (:*, :⋅)
    @eval begin
        $op(A::Composition, B::Mapping) = Composition((A.ops..., B))
        $op(A::Composition, B::Composition) = Composition((A.ops..., B.ops...))
        $op(A::Mapping, B::Composition) = Composition((A, B.ops...))
        $op(A::Mapping, B::Mapping) = Composition((A, B))
        $op(A::Mapping, x::AbstractArray) = apply(A, x)
    end
end

\(A::Mapping, x::AbstractArray) = apply(Inverse, A, x)
\(A::Mapping, B::Mapping) = inv(A)*B
/(A::Mapping, B::Mapping) = A*inv(B)

ctranspose(A::Mapping) = Adjoint(A)
ctranspose(A::SelfAdjointOperator) = A
ctranspose(A::Adjoint) = A.op
ctranspose(A::Inverse) = InverseAdjoint(A.op)
ctranspose(A::InverseAdjoint) = inv(A.op)
ctranspose(A::Scaled) = conj(A.sc)*ctranspose(A.op)
ctranspose(A::Sum) = Sum(ntuple(i -> ctranspose(A.ops[i]), length(A)))
function ctranspose(A::Composition)
    n = length(A)
    Composition(ntuple(i -> ctranpose(A.ops[n + 1 - i]), n))
end

inv(A::Mapping) = Inverse(A)
inv(A::Adjoint) = InverseAdjoint(A.op)
inv(A::Inverse) = A.op
inv(A::InverseAdjoint) = ctranpose(A.op)
inv(A::Scaled) = (one(Scalar)/A.sc)*inv(A.op)
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
* `Inverse` to apply the inverse of `A` and yield `y = A\x`;
* `InverseAdjoint` or `AdjointInverse` to apply the inverse of `A'` and
  yield `y = A'\x`.

Note that not all operations may be implemented by the different types of
mappings and `Adjoint` and `InverseAdjoint` may only be applicable for linear
operators.

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

Mapping subtypes only need to implement `apply!` with the specific signature:

```julia
apply!(α::Scalar, ::Type{P}, A::M, x, β::Scalar, y) -> y
```

for any supported operation `P` and where `M` is the type of the mapping.  Of
course, the types of arguments `x` and `y` may be specified as well.

See also: [`Mapping`](@ref), [`apply`](@ref), [`vcreate`](@ref).

"""
function apply! end

# `Direct` is the default operation.
apply!(A::Mapping, x, y) =
    apply!(one(Scalar), Direct, A, x, zero(Scalar), y)
apply!(α::Real, A::Mapping, x, y) =
    apply!(convert(Scalar, α), Direct, A, x, zero(Scalar), y)
apply!(A::Mapping, x, β::Real, y) =
    apply!(one(Scalar), Direct, A, x, convert(Scalar, β), y)
apply!(α::Real, A::Mapping, x, β::Real, y) =
    apply!(convert(Scalar, α), Direct, A, x, convert(Scalar, β), y)

# Only the method with signature:
#
#     apply!(α::Scalar, ::Type{P}, A::MappingType, x,
#            β::Scalar, y) where {P<:Operations}
#
# has to be implemented by mapping subtypes so we provide the necessary
# mechanism to dispatch derived methods.
apply!(::Type{P}, A::Mapping, x, y) where {P<:Operations} =
    apply!(one(Scalar), P, A, x, zero(Scalar), y)
apply!(α::Real, ::Type{P}, A::Mapping, x, y) where {P<:Operations} =
    apply!(convert(Scalar, α), P, A, x, zero(Scalar), y)
apply!(::Type{P}, A::Mapping, x, β::Real, y) where {P<:Operations} =
    apply!(one(Scalar), P, A, x, convert(Scalar, β), y)
apply!(α::Real, ::Type{P}, A::Mapping, x, β::Real, y) where {P<:Operations} =
    apply!(convert(Scalar, α), P, A, x, convert(Scalar, β), y)

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

# Implemention of the `apply!(α,P,A,x,β,y)` method for a scaled mapping.
for (T, expr) in ((:Direct, :(α*A.sc)),
                  (:Adjoint, :(α*conj(A.sc))),
                  (:Inverse, :(α/A.sc)),
                  (:InverseAdjoint, :(α/conj(A.sc))))
    @eval apply!(α::Scalar, ::Type{$T}, A::Scaled, x, β::Scalar, y) =
        apply!($expr, $T, A.op, x, β, y)
end

# Implemention of the `apply!(α,P,A,x,β,y)` method for a the adjoint of a linear
# operator.
for (T1, T2) in ((:Direct, :Adjoint),
                 (:Adjoint, :Direct),
                 (:Inverse, :InverseAdjoint),
                 (:InverseAdjoint, :Inverse))
    @eval apply!(α::Scalar, ::Type{$T1}, A::Adjoint, x, β::Scalar, y) =
        apply!(α, $T2, A.op, x, β, y)
end

# Implemention of the `apply!(α,P,A,x,β,y)` method for the inverse of a
# mapping.
for (T1, T2) in ((:Direct, :Inverse),
                 (:Adjoint, :InverseAdjoint),
                 (:Inverse, :Direct),
                 (:InverseAdjoint, :Adjoint))
    @eval apply!(α::Scalar, ::Type{$T1}, A::Inverse, x, β::Scalar, y) =
        apply!(α, $T2, A.op, x, β, y)
end

# Implementation of the `apply!(α,P,A,x,β,y)` method for a sum of mappings.
function apply!(α::Scalar, ::Type{P}, A::Sum, x,
                β::Scalar, y) where {P<:Union{Direct,Adjoint}}
    n = length(A)
    if n == 0
        vscale!(β, y)
    else
        apply!(α, P, A.ops[1], x, β, y)
        for i in 2:n
            apply!(α, P, A.ops[i], x, one(Scalar), y)
        end
    end
    return y
end

function apply!(α::Scalar, ::Type{P}, A::Sum, x,
                β::Scalar, y) where {P<:Union{Inverse,InverseAdjoint}}
    error("automatic dispatching of the inverse of a sum of mappings is not supported")
end

# Implementation of the `apply!(α,P,A,x,β,y)` method for a composition of
# mappings.
function apply!(α::Scalar, ::Type{P}, A::Composition, x,
                β::Scalar, y) where {P<:Union{Direct,InverseAdjoint}}
    # Apply operators in order.
    n = length(A)
    if n > 1
        apply!(α, P, A.ops[1], _apply(P, A, x, 2, n), β, y)
    elseif n == 1
        apply!(α, P, A.ops[1], x, β, y)
    else
        vscale!(β, y)
    end
    return y
end

function apply!(α::Scalar, ::Type{P}, A::Composition, x,
                β::Scalar, y) where {P<:Union{Adjoint,Inverse}}
    # Apply operators in reverse order.
    n = length(A)
    if n > 1
        apply!(α, P, A.ops[n], _apply(P, A, x, n - 1, n), β, y)
    elseif n == 1
        apply!(α, P, A.ops[1], x, β, y)
    else
        vscale!(β, y)
    end
    return y
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

The method `vcreate(::Type{P}, A, x)` should be implemented by linear operators
for any supported operations `P` and argument type for `x`.

See also: [`Mapping`](@ref), [`apply`](@ref).

"""
vcreate(A::LinearOperator, x) = vcreate(Direct, A, x)
