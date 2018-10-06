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

"""
```julia
@callable T
```

makes concrete type `T` callable as a regular mapping that is `A(x)` yields
`apply(A,x)` for any `A` of type `T`.

"""
macro callable(T)
    quote
	(A::$(esc(T)))(x) = apply(A, x)
    end
end

for T in (Adjoint, Inverse, InverseAdjoint, Scaled, Sum, Composition, Hessian)
    @eval (A::$T)(x) = apply(A, x)
end

const UnsupportedInverseOfSumOfMappings = "automatic dispatching of the inverse of a sum of mappings is not supported"

# As a general rule, do not use the constructors of tagged types directly but
# use `A'` or `adjoint(A)` instead of `Adjoint(A)`, `inv(A)` instead of
# `Inverse(A)`, etc.  This is somewhat enforced by the following constructors
# which systematically throw an error.
Inverse(::Union{Adjoint,Inverse,InverseAdjoint,Scaled,Sum,Composition}) =
    error("use `inv(A)` instead of `Inverse(A)`")
Adjoint(::Union{Adjoint,Inverse,InverseAdjoint,Scaled,Sum,Composition}) =
    error("use `A'` or `adjoint(A)` instead of `Adjoint(A)`")
InverseAdjoint(::Union{Adjoint,Inverse,InverseAdjoint,Scaled,Sum,Composition}) =
    error("use `inv(A')`, `inv(A)'`, `inv(adjoint(A))` or `adjoint(inv(A))` instead of `InverseAdjoint(A)` or `AdjointInverse(A)")

# Extend the `length` method to yield the number of components of a sum or
# composition of mappings.
length(A::Union{Sum,Composition}) = length(contents(A))

# Non-specific constructors for the *linear* trait.
LinearType(::LinearMapping) = Linear
LinearType(::Scaled{<:LinearMapping}) = Linear
LinearType(A::Union{Scaled,Inverse}) = LinearType(A.op)
LinearType(::Mapping) = NonLinear # anything else is non-linear
LinearType(A::Union{Sum,Composition}) =
    all(is_linear, A.ops) ? Linear : NonLinear

# Non-specific constructors for the *self-adjoint* trait.
SelfAdjointType(::Mapping) = NonSelfAdjoint
SelfAdjointType(A::Union{Scaled,Inverse}) = SelfAdjointType(A.op)
SelfAdjointType(A::Sum) =
    all(is_selfadjoint, A.ops) ? SelfAdjoint : NonSelfAdjoint

# Non-specific constructors for the *morphism* trait.
MorphismType(::Mapping) = Morphism
MorphismType(A::Union{Scaled,Inverse}) = MorphismType(A.op)
MorphismType(A::Union{Sum,Composition}) =
    all(is_endomorphism, ops) ? Endomorphism : Morphism

# Non-specific constructors for the *diagonal* trait.
DiagonalType(::Mapping) = NonDiagonalMapping
DiagonalType(A::Union{Scaled,Inverse}) = DiagonalType(A.op)
DiagonalType(A::Union{Sum,Composition}) =
    all(is_diagonal, A.ops) ? DiagonalMapping : NonDiagonalMapping

# Non-specific constructors for the *in-place* trait.
InPlaceType(A::Mapping) = InPlaceType(Direct, A)
InPlaceType(::Type{<:Operations}, ::Mapping) = OutOfPlace
InPlaceType(::Type{P}, A::Union{Scaled,Inverse}) where {P<:Operations} =
    InPlaceType(P, A.op)
InPlaceType(::Type{P}, A::Union{Sum,Composition}) where {P<:Operations} =
    all(op -> is_applicable_in_place(P, op), A.ops) ? InPlace : OutOfPlace

"""
```julia
is_linear(A)
```

yields whether `A` is certainly a linear mapping.

!!! note
    This method is intended to perform certain automatic simplifications or
    optimizations.  It is guaranted to return `true` when its argument is
    certainly a linear mapping but it may return `false` even though its
    argument behaves linearly because it is not always possible to figure out
    that a complex mapping assemblage has this property.

See also: [`LinearType`](@ref).

"""
is_linear(A::Mapping) = _is_linear(LinearType(A))
_is_linear(::Type{Linear}) = true
_is_linear(::Type{NonLinear}) = false

"""
```julia
is_selfadjoint(A)
```

yields whether mapping `A` is certainly a self-adjoint linear mapping.

!!! note
    This method is intended to perform certain automatic simplifications or
    optimizations.  It is guaranted to return `true` when its argument is
    certainly a self-adjoint linear mapping but it may return `false` even
    though its argument behaves like a self-adjoint linear map because it is
    not always possible to figure out that a complex mapping assemblage has
    this property.

See also: [`SelfAdjointType`](@ref).

"""
is_selfadjoint(A::Mapping) = _is_selfadjoint(SelfAdjointType(A))
_is_selfadjoint(::Type{SelfAdjoint}) = true
_is_selfadjoint(::Type{NonSelfAdjoint}) = false

"""
```julia
    is_endomorphism(A)
```

yields whether mapping `A` is certainly an endomorphism.

!!! note
    This method is intended to perform certain automatic simplifications or
    optimizations.  It is guaranted to return `true` when its argument is
    certainly an endomorphism but it may return `false` even though its
    argument behaves like an endomorphism because it is not always possible to
    figure out that a complex mapping assemblage has this property.

See also: [`MorphismType`](@ref).

"""
is_endomorphism(A::Mapping) = _is_endomorphism(MorphismType(A))
_is_endomorphism(::Type{Endomorphism}) = true
_is_endomorphism(::Type{Morphism}) = false

"""
```julia
    is_diagonal(A)
```

yields whether mapping `A` is certainly a diagonal linear map.

!!! note
    This method is intended to perform certain automatic simplifications or
    optimizations.  It is guaranted to return `true` when its argument is
    certainly a diagonal linear map but it may return `false` even though its
    argument behaves like a diagonal linear map because it is not always
    possible to figure out that a complex mapping assemblage has this property.

See also: [`DiagonalType`](@ref).

"""
is_diagonal(A::Mapping) = _is_diagonal(DiagonalType(A))
_is_diagonal(::Type{DiagonalMapping}) = true
_is_diagonal(::Type{NonDiagonalMapping}) = false

"""
```julia
    is_applicable_in_place([P=Direct,] A)
```

yields whether operation `P` (`Direct` by default) of mapping `A` can be
applied in-place.

See also: [`InPlaceType`](@ref).

"""
is_applicable_in_place(A::Mapping) = is_applicable_in_place(Direct, A)
is_applicable_in_place(::Type{P}, A::Mapping) where {P<:Operations} =
    _is_applicable_in_place(InPlaceType(P, A))
_is_applicable_in_place(::Type{InPlace}) = true
_is_applicable_in_place(::Type{OutOfPlace}) = false

# Unary minus and unary plus.
-(A::Mapping) = -1*A
+(A::Mapping) = A

# Sum of mappings.
+(A::Sum, B::Mapping) = Sum(A.ops..., B)
+(A::Mapping, B::Sum) = Sum(A, B.ops...)
+(A::Sum, B::Sum) = Sum(A.ops..., B.ops...)
+(A::Mapping, B::Mapping) = Sum(A, B)

# Dot operator involving a mapping acts as the multiply operator.
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

adjoint(A::Mapping) = _adjoint(SelfAdjointType(A), A)
adjoint(A::Adjoint) = A.op
adjoint(A::InverseAdjoint) = inv(A.op)
adjoint(A::Scaled) = conj(A.sc)*adjoint(A.op)
adjoint(A::Sum) = Sum(map(adjoint, A.ops))
adjoint(A::Composition) = Composition(reversemap(adjoint, A.ops))
_adjoint(::Type{SelfAdjoint}, A::Mapping) = A
_adjoint(::Type{SelfAdjoint}, A::Inverse) = A
_adjoint(::Type{NonSelfAdjoint}, A::Mapping) = Adjoint(A)
_adjoint(::Type{NonSelfAdjoint}, A::Inverse) = InverseAdjoint(A)

inv(A::Mapping) = Inverse(A)
inv(A::Adjoint) = InverseAdjoint(A.op)
inv(A::Inverse) = A.op
inv(A::InverseAdjoint) = adjoint(A.op)
inv(A::Scaled) = (one(Scalar)/A.sc)*inv(A.op)
inv(A::Sum) = error(UnsupportedInverseOfSumOfMappings)
inv(A::Composition) = Composition(reversemap(inv, A.ops))

"""
```julia
reversemap(f, args)
```

applies the function `f` to arguments `args` in reverse order and return the
result.  For now, the arguments `args` must be in the form of a simple tuple
and the result is the tuple: `(f(args[end]),f(args[end-1]),...,f(args[1])`.

"""
reversemap(f::Function, args::NTuple{N,Any}) where {N} =
    ntuple(i -> f(args[(N + 1) - i]), Val(N))

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

# Extend `mul!` so that `A'*x`, `A*B*C*x`, etc. yield the expected result.
mul!(y::Ty, A::Mapping, x::Tx) where {Tx,Ty} =
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
`A` to the argument `x`.  Optional parameter `P ∈ Operations` is one of
`Direct` (the default), `Adjoint`, `Inverse` and/or `InverseAdjoint` and can be
used to specify how `A` is to be applied as explained in the documentation of
the [`apply`](@ref) method.

The method `vcreate(::Type{P}, A, x)` should be implemented by linear mappings
for any supported operations `P` and argument type for `x`.

See also: [`Mapping`](@ref), [`apply`](@ref).

"""
vcreate(A::Mapping, x) = vcreate(Direct, A, x)
