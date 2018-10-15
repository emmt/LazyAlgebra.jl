#
# rules.jl -
#
# Implement rules for basic operations involving mappings.
#
#-------------------------------------------------------------------------------
#
# This file is part of LazyAlgebra (https://github.com/emmt/LazyAlgebra.jl)
# released under the MIT "Expat" license.
#
# Copyright (c) 2017-2018 Éric Thiébaut.
#

const UnsupportedInverseOfSumOfMappings =
    "automatic dispatching of the inverse of a sum of mappings is not supported"

# As a general rule, do not use the constructors of tagged types directly but
# use `A'` or `adjoint(A)` instead of `Adjoint(A)`, `inv(A)` instead of
# `Inverse(A)`, etc.  This is enforced by the following constructors which
# systematically throw an error.
Inverse(A) = error("use `inv(A)` instead of `Inverse(A)`")
Adjoint(A) = error("use `A'` or `adjoint(A)` instead of `Adjoint(A)`")
InverseAdjoint(A) = error("use `inv(A')`, `inv(A)'`, `inv(adjoint(A))` or `adjoint(inv(A))` instead of `InverseAdjoint(A)` or `AdjointInverse(A)")

# Non-specific constructors for the *linear* trait.
LinearType(::LinearMapping) = Linear
LinearType(::Scaled{<:LinearMapping}) = Linear
LinearType(A::Union{Scaled,Inverse}) = LinearType(operand(A))
LinearType(::Mapping) = NonLinear # anything else is non-linear
LinearType(A::Union{Sum,Composition}) =
    all(is_linear, operands(A)) ? Linear : NonLinear

# Non-specific constructors for the *self-adjoint* trait.
SelfAdjointType(::Mapping) = NonSelfAdjoint
SelfAdjointType(A::Union{Scaled,Inverse}) = SelfAdjointType(operand(A))
SelfAdjointType(A::Sum) =
    all(is_selfadjoint, operands(A)) ? SelfAdjoint : NonSelfAdjoint

# Non-specific constructors for the *morphism* trait.
MorphismType(::Mapping) = Morphism
MorphismType(A::Union{Scaled,Inverse}) = MorphismType(operand(A))
MorphismType(A::Union{Sum,Composition}) =
    all(is_endomorphism, operands(A)) ? Endomorphism : Morphism

# Non-specific constructors for the *diagonal* trait.
DiagonalType(::Mapping) = NonDiagonalMapping
DiagonalType(A::Union{Scaled,Inverse}) = DiagonalType(operand(A))
DiagonalType(A::Union{Sum,Composition}) =
    all(is_diagonal, operands(A)) ? DiagonalMapping : NonDiagonalMapping

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

# Unary minus and unary plus.
-(A::Mapping) = -1*A
+(A::Mapping) = A

# Sum of mappings.
+(A::Sum, B::Mapping) = Sum(operands(A)..., B)
+(A::Mapping, B::Sum) = Sum(A, operands(B)...)
+(A::Sum, B::Sum) = Sum(operands(A)..., operands(B)...)
+(A::Mapping, B::Mapping) = Sum(A, B)

# Subtraction.
-(A::Mapping, B::Mapping) = A + (-1)*B

# Dot operator (\cdot) involving a mapping acts as the multiply or compose
# operator.
⋅(A::Mapping, B::Mapping) = A*B
⋅(A::Mapping, B::T) where {T} = A*B
⋅(A::T, B::Mapping) where {T} = A*B

# Left scalar muliplication of a mapping.
*(α::Number, A::Scaled) = (α*multiplier(A))*operand(A)
*(α::S, A::T) where {S<:Number,T<:Mapping} =
    (α == one(α) ? A : Scaled{T,S}(α, A))

# Composition of mappings and right multiplication of a mapping by a vector.
∘(A::Mapping, B::Mapping) = A*B
*(A::Composition, B::Mapping) = Composition(operands(A)..., B)
*(A::Composition, B::Composition) = Composition(operands(A)..., operands(B)...)
*(A::Mapping, B::Composition) = Composition(A, operands(B)...)
*(A::Mapping, B::Mapping) = Composition(A, B)
*(A::Mapping, x::T) where {T} = apply(A, x)

\(A::Mapping, x::T) where {T} = apply(Inverse, A, x)
\(A::Mapping, B::Mapping) = inv(A)*B
/(A::Mapping, B::Mapping) = A*inv(B)

adjoint(A::Mapping) = _adjoint(SelfAdjointType(A), A)
adjoint(A::Inverse) = inv(adjoint(operand(A)))
adjoint(A::Adjoint) = operand(A)
adjoint(A::InverseAdjoint) = inv(operand(A))
adjoint(A::Scaled) = conj(multiplier(A))*adjoint(operand(A))
adjoint(A::Sum) = Sum(map(adjoint, operands(A)))
adjoint(A::Composition) = Composition(reversemap(adjoint, operands(A)))
_adjoint(::Type{SelfAdjoint}, A::Mapping) = A
_adjoint(::Type{NonSelfAdjoint}, A::T) where {T<:Mapping} = Adjoint{T}(A)

inv(A::T) where {T<:Mapping} = Inverse{T}(A)
inv(A::Adjoint{T}) where {T} = InverseAdjoint{T}(operand(A))
inv(A::Inverse) = operand(A)
inv(A::InverseAdjoint) = adjoint(operand(A))
inv(A::Scaled) = inv(operand(A))*(inv(multiplier(A))*I)
inv(A::Sum) = error(UnsupportedInverseOfSumOfMappings)
inv(A::Composition) = Composition(reversemap(inv, operands(A)))

#------------------------------------------------------------------------------
# APPLY AND VCREATE

"""
```julia
apply([P,] A, x, scratch=false) -> y
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

Optional argument `scratch` indicates whether input argument `x` can be
overwritten by the operation.  This may be exploited to avoid allocating
temporary workspace(s).  The caller should set `scratch = true` if `x` is not
needed after calling `apply`.  If `scratch = true`, then it is possible that
`y` be the same object as `x`; otherwise, `y` is a new object unless applying
the operation yields the same contents as `y` for the result `x` (this is
always true for the identity for instance).  Thus, in general, it should not be
assumed that the result of applying a mapping is different from the input.

Julia methods are provided so that `apply(A', x)` automatically calls
`apply(Adjoint, A, x)` so the shorter syntax may be used without performances
impact.

See also: [`Mapping`](@ref), [`apply!`](@ref), [`vcreate`](@ref).

"""
apply(A::Mapping, x, scratch::Bool=false) = apply(Direct, A, x, scratch)

apply(P::Type{<:Operations}, A::Mapping, x, scratch::Bool=false) =
    apply!(1, P, A, x, scratch, 0, vcreate(P, A, x, scratch))

"""
```julia
apply!([α = 1,] [P = Direct,] A::Mapping, x, [scratch=false,] [β = 0,] y) -> y
```

overwrites `y` with `α*P(A)⋅x + β*y` where `P ∈ Operations` can be `Direct`,
`Adjoint`, `Inverse` and/or `InverseAdjoint` to indicate which variant of the
mapping `A` to apply.  The convention is that the prior contents of `y` is not
used at all if `β = 0` so `y` does not need to be properly initialized in that
case and can be directly used to store the result.  The `scratch` optional
argument indicates whether the input `x` is no longer needed by the caller and
can thus be used as a scratch array.  Having `scratch = true` or `β = 0` may be
exploited by the specific implementation of the `apply!` method for the mapping
type to avoid allocating temporary workspace(s).

The order of arguments can be changed and the same result as above is obtained
with:

```julia
apply!([β = 0,] y, [α = 1,] [P = Direct,] A::Mapping, x, scratch=false) -> y
```

The result `y` may have been allocated by:

```julia
y = vcreate([Op,] A, x, scratch=false)
```

Mapping sub-types only need to extend `vcreate` and `apply!` with the specific
signatures:

```julia
vcreate(::Type{P}, A::M, x, scratch::Bool=false) -> y
apply!(α::Real, ::Type{P}, A::M, x, scratch::Bool, β::Real, y) -> y
```

for any supported operation `P` and where `M` is the type of the mapping.  Of
course, the types of arguments `x` and `y` may be specified as well.

Optionally, the method with signature:

```julia
apply(::Type{P}, A::M, x, scratch::Bool=false) -> y
```

may also be extended to improve the default implementation which is:

```julia
apply(P::Type{<:Operations}, A::Mapping, x, scratch::Bool=false) =
    apply!(1, P, A, x, scratch, 0, vcreate(P, A, x, scratch))
```

See also: [`Mapping`](@ref), [`apply`](@ref), [`vcreate`](@ref).

""" apply!

# Provide fallbacks so that `Direct` is the default operation and only the
# method with signature:
#
#     apply!(α::Real, ::Type{P}, A::MappingType, x, scratch::Bool,
#            β::Real, y) where {P<:Operations}
#
# has to be implemented by mapping subtypes so we provide the necessary
# mechanism to dispatch derived methods.
apply!(A::Mapping, x, y) =
    apply!(1, Direct, A, x, false, 0, y)
apply!(α::Real, A::Mapping, x, y) =
    apply!(α, Direct, A, x, false, 0, y)
apply!(A::Mapping, x, β::Real, y) =
    apply!(1, Direct, A, x, false, β, y)
apply!(α::Real, A::Mapping, x, β::Real, y) =
    apply!(α, Direct, A, x, false, β, y)

apply!(P::Type{<:Operations}, A::Mapping, x, y) =
    apply!(1, P, A, x, false, 0, y)
apply!(α::Real, P::Type{<:Operations}, A::Mapping, x, y) =
    apply!(α, P, A, x, false, 0, y)
apply!(P::Type{<:Operations}, A::Mapping, x, β::Real, y) =
    apply!(1, P, A, x, false, β, y)
apply!(α::Real, P::Type{<:Operations}, A::Mapping, x, β::Real, y) =
    apply!(α, P, A, x, false, β, y)

apply!(A::Mapping, x, scratch::Bool, y) =
    apply!(1, Direct, A, x, scratch, 0, y)
apply!(α::Real, A::Mapping, x, scratch::Bool, y) =
    apply!(α, Direct, A, x, scratch, 0, y)
apply!(A::Mapping, x, scratch::Bool, β::Real, y) =
    apply!(1, Direct, A, x, scratch, β, y)

apply!(P::Type{<:Operations}, A::Mapping, x, scratch::Bool, y) =
    apply!(1, P, A, x, scratch, 0, y)
apply!(α::Real, P::Type{<:Operations}, A::Mapping, x, scratch::Bool, y) =
    apply!(α, P, A, x, scratch, 0, y)
apply!(P::Type{<:Operations}, A::Mapping, x, scratch::Bool, β::Real, y) =
    apply!(1, P, A, x, scratch, β, y)

# Change order of arguments.
apply!(y, A::Mapping, x, scratch::Bool=false) =
    apply!(1, Direct, A, x, scratch, 0, y)
apply!(y, P::Type{<:Operations}, A::Mapping, x, scratch::Bool=false) =
    apply!(1, P, A, x, scratch, 0, y)
apply!(y, α::Real, A::Mapping, x, scratch::Bool=false) =
    apply!(α, Direct, A, x, scratch, 0, y)
apply!(y, α::Real, P::Type{<:Operations}, A::Mapping, x, scratch::Bool=false) =
    apply!(α, P, A, x, scratch, 0, y)
apply!(β::Real, y, A::Mapping, x, scratch::Bool=false) =
    apply!(1, Direct, A, x, scratch, β, y)
apply!(β::Real, y, P::Type{<:Operations}, A::Mapping, x, scratch::Bool=false) =
    apply!(1, P, A, x, scratch, β, y)
apply!(β::Real, y, α::Real, A::Mapping, x, scratch::Bool=false) =
    apply!(α, Direct, A, x, scratch, β, y)
apply!(β::Real, y, α::Real, P::Type{<:Operations}, A::Mapping, x, scratch::Bool=false) =
    apply!(α, P, A, x, scratch, β, y)

# Extend `mul!` so that `A'*x`, `A*B*C*x`, etc. yield the expected result.
mul!(y::Ty, A::Mapping, x::Tx) where {Tx,Ty} =
    apply!(1, Direct, A, x, false, 0, y)

# Implemention of the `apply!(α,P,A,x,scratch,β,y)` and
# `vcreate(P,A,x,scratch)` methods for a scaled mapping.
for (P, expr) in ((:Direct, :(α*multiplier(A))),
                  (:Adjoint, :(α*conj(multiplier(A)))),
                  (:Inverse, :(α/multiplier(A))),
                  (:InverseAdjoint, :(α/conj(multiplier(A)))))
    @eval begin

        vcreate(::Type{$P}, A::Scaled, x, scratch::Bool=false) =
            vcreate($P, operand(A), x)

        apply!(α::Real, ::Type{$P}, A::Scaled, x, scratch::Bool, β::Real, y) =
            apply!($expr, $P, operand(A), x, scratch, β, y)

    end
end

# Implemention of the `apply!(α,P,A,x,scratch,β,y)` and
# `vcreate(P,A,x,scratch=false)` methods for the various decorations of a
# mapping so as to automativcally unveil the embedded mapping.
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

        vcreate(::Type{$T1}, A::$T2, x, scratch::Bool=false) =
            vcreate($T3, operand(A), x, scratch)

        apply!(α::Real, ::Type{$T1}, A::$T2, x, β::Real, y) =
            apply!(α, $T3, operand(A), x, false, β, y)

        apply!(α::Real, ::Type{$T1}, A::$T2, x, scratch::Bool, β::Real, y) =
            apply!(α, $T3, operand(A), x, scratch, β, y)

    end
end

# Implementation of the `vcreate(P,A,x,scratch=false)` and
# `apply!(α,P,A,x,scratch,β,y)` and methods for a sum of mappings.  Note that
# `Sum` instances are warranted to have at least 2 components.

function vcreate(::Type{P}, A::Sum, x,
                 scratch::Bool=false) where {P<:Union{Direct,Adjoint}}
    # The sum only makes sense if all mappings yields the same kind of result.
    # Hence we just call the vcreate method for the first mapping of the sum.
    vcreate(P, A[1], x, scratch)
end

function apply!(α::Real, ::Type{P}, A::Sum{N}, x, scratch::Bool,
                β::Real, y) where {N,P<:Union{Direct,Adjoint}}
    # Apply first mapping with β and then other with β=1.  Scratch flag is
    # always false until last mapping because we must preserve x as there is
    # more than one term.
    @assert N ≥ 2 "bug in Sum constructor"
    apply!(α, P, A[1], x, false, β, y)
    for i in 2:N
        apply!(α, P, A[i], x, (scratch && i == N), 1, y)
    end
    return y
end

vcreate(::Type{<:Union{Inverse,InverseAdjoint}}, A::Sum, x, scratch::Bool=false) =
    error(UnsupportedInverseOfSumOfMappings)

apply(::Type{<:Union{Inverse,InverseAdjoint}}, A::Sum, x, scratch::Bool=false) =
    error(UnsupportedInverseOfSumOfMappings)

function apply!(α::Real, ::Type{<:Union{Inverse,InverseAdjoint}}, A::Sum, x,
                scratch::Bool, β::Real, y)
    error(UnsupportedInverseOfSumOfMappings)
end

# Implementation of the `apply!(α,P,A,x,scratch,β,y)` method for a composition
# of mappings.  There is no possible `vcreate(P,A,x,scratch)` method for a
# composition so we directly extend the `apply(P,A,x,scratch)` method.  Note
# that `Composition` instances are warranted to have at least 2 components.
#
# The unrolled code (taking care of allowing as few temporaries as possible and
# for the Direct or InverseAdjoint operation) writes:
#
#     w1 = apply(P, A[N], x, scratch)
#     scratch = (scratch || ! is_same_mutable_object(w1, x))
#     w2 = apply!(1, P, A[N-1], w1, scratch)
#     scratch = (scratch || ! is_same_mutable_object(w2, w1))
#     w3 = apply!(1, P, A[N-2], w2, scratch)
#     scratch = (scratch || ! is_same_mutable_object(w3, w2))
#     ...
#     return apply!(α, P, A[1], wNm1, scratch, β, y)
#
# To break the type barrier, this is done by a recursion.  The recursion is
# just done in the other direction for the Adjoint or Inverse operation.

function apply!(α::Real, P::Type{<:Union{Direct,InverseAdjoint}},
                A::Composition{N}, x, scratch::Bool, β::Real, y) where {N}
    # Apply mappings in order.
    @assert N ≥ 2 "bug in Composition constructor"
    w = _apply!(P, A, Val(2), x, scratch)
    scratch = (scratch || ! is_same_mutable_object(w, x))
    return apply!(α, P, A[1], w, scratch, β, y)
end

function apply(P::Type{<:Union{Direct,InverseAdjoint}},
               A::Composition{N}, x, scratch::Bool=false) where {N}
    # Apply mappings in order.
    @assert N ≥ 2 "bug in Composition constructor"
    w = _apply!(P, A, Val(2), x, scratch)
    scratch = (scratch || ! is_same_mutable_object(w, x))
    return apply(P, A[1], w, scratch)
end

function apply!(α::Real, P::Type{<:Union{Adjoint,Inverse}},
                A::Composition{N}, x, scratch::Bool, β::Real, y) where {N}
    # Apply mappings in reverse order.
    @assert N ≥ 2 "bug in Composition constructor"
    w = _apply!(P, A, Val(N-1), x, scratch)
    scratch = (scratch || ! is_same_mutable_object(w, x))
    return apply!(α, P, A[N], w, scratch, β, y)
end

function apply(P::Type{<:Union{Adjoint,Inverse}},
               A::Composition{N}, x, scratch::Bool=false) where {N}
    # Apply mappings in reverse order.
    @assert N ≥ 2 "bug in Composition constructor"
    w = _apply!(P, A, Val(N-1), x, scratch)
    scratch = (scratch || ! is_same_mutable_object(w, x))
    return apply(P, A[N], w, scratch)
end

# Apply intermediate mappings of a composition for Direct or InverseAdjoint
# operation.
function _apply!(P::Type{<:Union{Direct,InverseAdjoint}}, A::Composition{N},
                 ::Val{i}, x, scratch::Bool) where {N,i}
    @assert 1 < i < N
    w = _apply!(P, A, Val(i+1), x, scratch)
    scratch = (scratch || ! is_same_mutable_object(w, x))
    return apply(P, A[i], w, scratch)
end

# Apply last mapping of a composition for Direct or InverseAdjoint operation.
function _apply!(P::Type{<:Union{Direct,InverseAdjoint}}, A::Composition{N},
                 ::Val{N}, x, scratch::Bool) where {N}
    return apply(P, A[N], x, scratch)
end

# Apply intermediate mappings of a composition for Adjoint or InverseDirect
# operation.
function _apply!(P::Type{<:Union{Adjoint,Inverse}}, A::Composition{N},
                 ::Val{i}, x, scratch::Bool) where {N,i}
    @assert 1 < i < N
    w = _apply!(P, A, Val(i-1), x, scratch)
    scratch = (scratch || ! is_same_mutable_object(w, x))
    return apply(P, A[i], w, scratch)
end

# Apply first mapping of a composition for Adjoint or Inverse operation.
function _apply!(P::Type{<:Union{Adjoint,Inverse}}, A::Composition{N},
                 ::Val{1}, x, scratch::Bool) where {N}
    return apply(P, A[1], x, scratch)
end

"""
```julia
vcreate([P,] A, x, scratch=false) -> y
```

yields a new instance `y` suitable for storing the result of applying mapping
`A` to the argument `x`.  Optional parameter `P ∈ Operations` is one of
`Direct` (the default), `Adjoint`, `Inverse` and/or `InverseAdjoint` and can be
used to specify how `A` is to be applied as explained in the documentation of
the [`apply`](@ref) method.

Optional argument `scratch` indicates whether input argument `x` can be
overwritten by the operation and thus used to store the result.  This may be
exploited by some mappings (which are able to operate *in-place*) to avoid
allocating a new object for the result `y`.

The caller should set `scratch = true` if `x` is not needed after calling
`apply`.  If `scratch = true`, then it is possible that `y` be the same object
as `x`; otherwise, `y` is a new object unless applying the operation yields the
same contents as `y` for the result `x` (this is always true for the identity
for instance).  Thus, in general, it should not be assumed that the returned
`y` is different from the input `x`.

The method `vcreate(::Type{P}, A, x)` should be implemented by linear mappings
for any supported operations `P` and argument type for `x`.

See also: [`Mapping`](@ref), [`apply`](@ref).

"""
vcreate(A::Mapping, x, scratch::Bool=false) = vcreate(Direct, A, x, scratch)
