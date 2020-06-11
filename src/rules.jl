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
# Copyright (c) 2017-2020 Éric Thiébaut.
#

"""

```julia
LinearType(A)
```

yields the *linear* trait of mapping `A` indicating whether `A` is certainly
linear.  The returned value is one of the singletons `Linear()` for linear maps
or `NonLinear()` for other mappings.

See also: [`Trait`](@ref), [`is_linear`](@ref).

"""
LinearType(::Mapping) = NonLinear() # any mapping assumed non-linear by default
LinearType(::LinearMapping) = Linear()
LinearType(::Inverse{<:LinearMapping}) = Linear()
LinearType(::Scaled{<:LinearMapping}) = Linear()
LinearType(A::Inverse) = LinearType(unveil(A))
LinearType(A::Scaled) = LinearType(unscaled(A))
LinearType(A::Union{Sum,Composition}) =
    (allof(x -> LinearType(x) === Linear(), terms(A)...) ?
     Linear() : NonLinear())
LinearType(A::Scaled{T,S}) where {T,S} =
    # If the multiplier λ of a scaled mapping A = (λ⋅M) is zero, then A behaves
    # linearly even though M is not a linear mapping.  FIXME: But acknowledging
    # this as a linear mapping may give rise to troubles later.
    (multiplier(A) == zero(S) ? Linear() : LinearType(unscaled(A)))

@doc @doc(LinearType) Linear
@doc @doc(LinearType) NonLinear

"""

```julia
SelfAdjointType(A)
```

yields the *self-adjoint* trait of mapping `A` indicating whether `A` is
certainly a self-adjoint linear map.  The returned value is one of the
singletons `SelfAdjoint()` for self-adjoint linear maps and `NonSelfAdjoint()`
for other mappings.

See also: [`Trait`](@ref), [`is_selfadjoint`](@ref).

"""
SelfAdjointType(::Mapping) = NonSelfAdjoint()
SelfAdjointType(A::DecoratedMapping) = SelfAdjointType(unveil(A))
SelfAdjointType(A::Scaled) = SelfAdjointType(unscaled(A))
SelfAdjointType(A::Sum) =
    (allof(x -> SelfAdjointType(x) === SelfAdjoint(), terms(A)...) ?
     SelfAdjoint() : NonSelfAdjoint())
SelfAdjointType(A::Gram) = SelfAdjoint()

@doc @doc(SelfAdjointType) SelfAdjoint
@doc @doc(SelfAdjointType) NonSelfAdjoint

"""

```julia
MorphismType(A)
```

yields the *morphism* trait of mapping `A` indicating whether `A` is certainly
an endomorphism (its input and output spaces are the same).  The returned value
is one of the singletons `Endomorphism()` for mappings whose input and output
spaces are the same or `Morphism()` for other mappings.

See also: [`Trait`](@ref), [`is_endomorphism`](@ref).

"""
MorphismType(::Mapping) = Morphism()
MorphismType(A::DecoratedMapping) = MorphismType(unveil(A))
MorphismType(A::Gram) = Endomorphism()
MorphismType(A::Scaled) = MorphismType(unscaled(A))
MorphismType(A::Union{Sum,Composition}) =
    (allof(x -> MorphismType(x) === Endomorphism(), terms(A)...) ?
     Endomorphism() : Morphism())

@doc @doc(MorphismType) Morphism
@doc @doc(MorphismType) Endomorphism

"""

```julia
DiagonalType(A)
```

yields the *diagonal* trait of mapping `A` indicating whether `A` is certainly
a diagonal linear mapping.  The returned value is one of the singletons
`DiagonalMapping()` for diagonal linear maps or `NonDiagonalMapping()` for other
mappings.

See also: [`Trait`](@ref), [`is_diagonal`](@ref).

"""
DiagonalType(::Mapping) = NonDiagonalMapping()
DiagonalType(A::DecoratedMapping) = DiagonalType(unveil(A))
DiagonalType(A::Scaled) = DiagonalType(unscaled(A))
DiagonalType(A::Union{Sum,Composition}) =
    (allof(x -> DiagonalType(x) === DiagonalMapping(), terms(A)...) ?
     DiagonalMapping() : NonDiagonalMapping())

@doc @doc(DiagonalType) NonDiagonalMapping
@doc @doc(DiagonalType) DiagonalMapping

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
is_linear(A::LinearMapping) = true
is_linear(A::Mapping) = _is_linear(LinearType(A))
_is_linear(::Linear) = true
_is_linear(::NonLinear) = false

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
_is_selfadjoint(::SelfAdjoint) = true
_is_selfadjoint(::NonSelfAdjoint) = false

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
_is_endomorphism(::Endomorphism) = true
_is_endomorphism(::Morphism) = false

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
_is_diagonal(::DiagonalMapping) = true
_is_diagonal(::NonDiagonalMapping) = false

#------------------------------------------------------------------------------
# General simplification rules:
#
#  - Factorize multipliers to the left.
#
#  - Adjoint of a sum (or a composition) of terms is rewritten as the sum
#    (respectively composition) of the adjoint of the terms.
#
#  - Adjoint of a scaled mapping is rewritten as a scaled adjoint of the
#    mapping.  Similarly, inverse of a scaled mapping is rewritten as a scaled
#    inverse of the mapping, if the mapping is linear, or as the inverse of the
#    mapping times a scaled identity otherwise.
#
#  - Adjoint of the inverse is rewritten as inverse of the adjoint.
#
#  - Inner constructors are fully qualified but check arguments.  Un-qualified
#    outer constructors just call the inner constructors with the suitable
#    parameters.
#
#  - To simplify a sum, the terms corresponding to identical mappings (possibly
#    scaled) are first grouped to produce a single mapping (possibly scaled)
#    per group, the resulting terms are sorted (so that all equivalent
#    expressions yield the same result) and the "zeros" eliminated (if all
#    terms are "zero", the sum simplifies to the first one).  For now, the
#    sorting is not absolutely perfect as it is based on `objectid()` hashing
#    method.  The odds of having the same identifier for two different things
#    are however extremely low.
#
#  - To simplify a composition, a fusion algorithm is applied and "ones" are
#    eliminated.  It is assumed that composition is non-commutative so the
#    ordering of terms is left unchanged.  Thanks to this, simplification rules
#    for simple compositions (made of two non-composition mappings) can be
#    automatically performed by proper dispatching rules.  Calling the fusion
#    algorithm is only needed for more complex compositions.
#
# The simplication algorithm is not perfect (LazyAlgebra is not intended to be
# for symbolic computations) but do a reasonnable job.  In particular complex
# mappings built using the same sequences should be simplified in the same way
# and thus be correctly identified as being identical.

#------------------------------------------------------------------------------
# NEUTRAL ELEMENTS

# The neutral element ("zero") for the addition is zero times a mapping of the
# proper type.
zero(A::Mapping) = 0*A

iszero(A::Scaled) = iszero(multiplier(A))
iszero(::Mapping) = false

# The neutral element ("one") for the composition is the identity.
const Id = Identity()
one(::Mapping) = Id

isone(::Identity) = true
isone(::Mapping) = false

#------------------------------------------------------------------------------
# UNQUALIFIED OUTER CONSTRUCTORS

# Unqualified outer constructors are provided which call the corresponding
# inner constructors with all suitable parameters and rely on the inner
# constructors to check whether the call was allowed or not.  A constraint that
# must hold is that T(A), with T an unqualified type constructor, always yields
# an instance of T.
Direct(A::Mapping) = A # provided for completeness
Adjoint(A::T) where {T<:Mapping} = Adjoint{T}(A)
Inverse(A::T) where {T<:Mapping} = Inverse{T}(A)
InverseAdjoint(A::T) where {T<:Mapping} = InverseAdjoint{T}(A)
Gram(A::T) where {T<:Mapping} = Gram{T}(A)
Jacobian(A::M, x::T) where {M<:Mapping,T} = Jacobian{M,T}(A, x)
Scaled(α::S, A::T) where {S<:Number,T<:Mapping} = Scaled{T,S}(α, A)
Sum(ops::Mapping...) = Sum(ops)
Sum(ops::T) where {N,T<:NTuple{N,Mapping}} = Sum{N,T}(ops)
Composition(ops::Mapping...) = Composition(ops)
Composition(ops::T) where {N,T<:NTuple{N,Mapping}} = Composition{N,T}(ops)

# Qualified outer constructors to forbid decoration of mappings of specific
# types when, according to the simplification rules, another more simple
# construction should be built instead.  Everything not forbidden is allowed
# except that additional tests may be performed by the inner constructors
# (e.g., Adjoint check that its argument is linear).
#
# FIXME: Some restrictions may be bad ideas like adjoint of sums or
# compositions.

for (func, blacklist) in ((:Adjoint,        (:Identity,
                                             :Adjoint,
                                             :Inverse,
                                             :InverseAdjoint,
                                             :Scaled,
                                             :Sum,
                                             :Composition)),
                          (:Inverse,        (:Identity,
                                             :Adjoint,
                                             :Inverse,
                                             :InverseAdjoint,
                                             :Scaled,
                                             :Composition)),
                          (:InverseAdjoint, (:Identity,
                                             :Adjoint,
                                             :Inverse,
                                             :InverseAdjoint,
                                             :Scaled,
                                             :Composition)),
                          (:Gram,           (:Inverse,
                                             :InverseAdjoint,
                                             :Scaled,)),
                          (:Jacobian,       (:Scaled,)),
                          (:Scaled,         (:Scaled,)))
    for T in blacklist
        if func === :Scaled
            @eval $func{T,S}(α::S, A::T) where {S<:Number,T<:$T} =
                illegal_call_to($func, T)
        elseif func === :Jacobian
            @eval $func{M,T}(A::M, x::T) where {M<:$T,T} =
                illegal_call_to($func, M)
        else
            @eval $func{T}(A::T) where {T<:$T} = illegal_call_to($func, T)
        end
    end
end

@noinline illegal_call_to(::Type{Adjoint}, T::Type) =
    bad_argument("the `Adjoint` constructor cannot be applied to an instance of `",
                 brief(T), "`, use expressions like `A'` or `adjoint(A)`")

@noinline illegal_call_to(::Type{Inverse}, T::Type) =
    bad_argument("the `Inverse` constructor cannot be applied to an instance of `",
                 brief(T), "`, use expressions like `A\\B`, `A/B` or `inv(A)`")

@noinline illegal_call_to(::Type{InverseAdjoint}, T::Type) =
    bad_argument("the `InverseAdjoint` constructor cannot be applied to an instance of `",
                 brief(T), "`, use expressions like `A'\\B`, `A/(B')`, `inv(A')` or `inv(A)'`")

@noinline illegal_call_to(::Type{Gram}, T::Type) =
    bad_argument("the `Gram` constructor cannot be applied to an instance of `",
                 brief(T), "`, use expressions like `A'*A` or `gram(A)`")

@noinline illegal_call_to(::Type{Jacobian}, T::Type) =
    bad_argument("the `Jacobian` constructor cannot be applied to an instance of `",
                 brief(T), "`, use an expression like `∇(A,x)`")

@noinline illegal_call_to(::Type{Scaled}, T::Type) =
    bad_argument("the `Scaled` constructor cannot be applied to an instance of `",
                 brief(T), "`, use expressions like `α*A`")

brief(::Type{<:Adjoint}       ) = "Adjoint"
brief(::Type{<:Inverse}       ) = "Inverse"
brief(::Type{<:InverseAdjoint}) = "InverseAdjoint"
brief(::Type{<:Gram}          ) = "Gram"
brief(::Type{<:Jacobian}      ) = "Jacobian"
brief(::Type{<:Scaled}        ) = "Scaled"
brief(::Type{<:Sum}           ) = "Sum"
brief(::Type{<:Composition}   ) = "Composition"
brief(::Type{<:Identity}      ) = "Identity"
brief(T::Type) = repr(T)

#------------------------------------------------------------------------------
# SCALED TYPE

# Left-multiplication and left-division by a scalar.  The only way to
# right-multiply or right-divide by a scalar is to right multiply or divide by
# the scaled identity.
*(α::S, A::T) where {S<:Number,T<:Mapping} =
    (α == one(α) ? A : isfinite(α) ? Scaled{T,S}(α, A) :
     bad_argument("non-finite multiplier"))
*(α::Number, A::Scaled) = (α*multiplier(A))*unscaled(A)
\(α::Number, A::Mapping) = inv(α)*A
\(α::Number, A::Scaled) = (multiplier(A)/α)*unscaled(A)

#------------------------------------------------------------------------------
# ADJOINT TYPE

# Adjoint for non-specific mappings.
adjoint(A::Mapping) = _adjoint(LinearType(A), A)
adjoint(A::LinearMapping) = _adjoint(Linear(), SelfAdjointType(A), A)
_adjoint(::NonLinear, A::Mapping) =
    throw_forbidden_adjoint_of_non_linear_mapping()
_adjoint(::Linear, A::Mapping) = _adjoint(Linear(), SelfAdjointType(A), A)
_adjoint(::Linear, ::SelfAdjoint, A::Mapping) = A
_adjoint(::Linear, ::NonSelfAdjoint, A::Mapping) = Adjoint(A)

# Adjoint for specific mapping types.
adjoint(A::Identity) = Id
adjoint(A::Scaled{Identity}) = conj(multiplier(A))*Id
adjoint(A::Scaled) = conj(multiplier(A))*adjoint(unscaled(A))
adjoint(A::Adjoint) = unveil(A)
adjoint(A::Inverse) = inv(adjoint(unveil(A)))
adjoint(A::InverseAdjoint) = inv(unveil(A))
adjoint(A::Jacobian) = Jacobian(A)
adjoint(A::Gram) = A
adjoint(A::Composition) =
    # It is assumed that the composition has already been simplified, so we
    # just apply the mathematical formula for the adjoint of a composition.
    Composition(reversemap(adjoint, terms(A)))

function adjoint(A::Sum{N}) where {N}
    # It is assumed that the sum has already been simplified, so we just apply
    # the mathematical formula for the adjoint of a sum and sort the resulting
    # terms.
    B = Vector{Mapping}(undef, N)
    @inbounds for i in 1:N
        B[i] = adjoint(A[i])
    end
    return Sum(to_tuple(sort!(B)))
end

@noinline throw_forbidden_adjoint_of_non_linear_mapping() =
    bad_argument("taking the adjoint of non-linear mappings is not allowed")

#------------------------------------------------------------------------------
# JACOBIAN

"""
    ∇(A, x)

yields a result corresponding to the Jacobian (first partial derivatives) of
the linear mapping `A` for the variables `x`.  If `A` is a linear mapping,
`A` is returned whatever `x`.

The call

    jacobian(A, x)

is an alias for `∇(A,x)`.

"""
∇(A::Mapping, x) = jacobian(A, x)
jacobian(A::Mapping, x) = _jacobian(LinearType(A), A, x)
_jacobian(::Linear, A::Mapping, x) = A
_jacobian(::NonLinear, A::Mapping, x) = Jacobian(A, x)
jacobian(A::Scaled, x) = multiplier(A)*jacobian(unscaled(A), x)

@doc @doc(∇) jacobian

#------------------------------------------------------------------------------
# INVERSE TYPE

# Inverse for non-specific mappings (a simple mapping or a sum or mappings).
inv(A::T) where {T<:Mapping} = Inverse{T}(A)

# Inverse for specific mapping types.
inv(A::Identity) = Id
inv(A::Scaled{Identity}) = inv(multiplier(A))*Id
inv(A::Scaled) = (is_linear(unscaled(A)) ?
                  inv(multiplier(A))*inv(unscaled(A)) :
                  inv(unscaled(A))*(inv(multiplier(A))*Id))
inv(A::Inverse) = unveil(A)
inv(A::AdjointInverse) = adjoint(unveil(A))
inv(A::Adjoint{T}) where {T<:Mapping} = AdjointInverse{T}(unveil(A))
inv(A::Composition) =
    # Even though the composition has already been simplified, taking the
    # inverse may trigger other simplifications, so we must rebuild the
    # composition term by term in reverse order (i.e. applying the mathematical
    # formula for the inverse of a composition).
    _merge_inv_mul(terms(A))

# `_merge_inv_mul([A,i,]B)` is recursively called to build the inverse of a
# composition.  Argument A is a mapping (initially not specified or the
# identity) of the resulting composition, argument `i` is the index of the next
# component to take (initially not specified or set to `N` the number of
# terms), argument `B` is a tuple (initially full) of the remaining terms.
_merge_inv_mul(B::NTuple{N,Mapping}) where {N} =
    # Initialize recursion.
    _merge_inv_mul(inv(last(B)), N - 1, B)

function _merge_inv_mul(A::Mapping, i::Int, B::NTuple{N,Mapping}) where {N}
    # Perform intermediate and last recursion step.
    C = A*inv(B[i])
    return (i > 1 ? _merge_inv_mul(C, i - 1, B) : C)
end

#------------------------------------------------------------------------------
# SUM OF MAPPINGS

# Unary minus and unary plus.
-(A::Mapping) = (-1)*A
-(A::Scaled) = (-multiplier(A))*unscaled(A)
+(A::Mapping) = A

# Subtraction.
-(A::Mapping, B::Mapping) = A + (-B)

# Rules for sums built by `A + B`.
+(A::Mapping, B::Mapping) = Simplify.add(A, B)

#------------------------------------------------------------------------------
# COMPOSITION OF MAPPINGS

# Dot operator (\cdot + tab) involving a mapping acts as the multiply or
# compose operator.
⋅(A::Mapping, B::Mapping) = A*B
⋅(A::Mapping, B) = A*B
⋅(A, B::Mapping) = A*B

# Compose operator (\circ + tab) beween mappings.
∘(A::Mapping, B::Mapping) = A*B

# Rules for the composition of 2 mappings.  Mappings that may behave
# specifically in a composition have type `Identity`, `Scaled` and
# `Composition`; all others have the same behavior.
*(A::Identity,    B::Identity   ) = B
*(A::Identity,    B::Scaled     ) = B
*(A::Identity,    B::Composition) = B
*(A::Identity,    B::Mapping    ) = B
*(A::Scaled,      B::Identity   ) = A
*(A::Scaled,      B::Mapping    ) = multiplier(A)*(unscaled(A)*B)
*(A::Scaled,      B::Scaled     ) =
    is_linear(A) ? (multiplier(A)*multiplier(B))*(unscaled(A)*unscaled(B)) :
    multiplier(A)*(unscaled(A)*B)
*(A::Composition, B::Identity   ) = A
*(A::Composition, B::Composition) = Simplify.compose(A, B)
*(A::Composition, B::Mapping    ) = Simplify.compose(A, B)
*(A::Composition, B::Scaled     ) =
    is_linear(A) ? multiplier(B)*(A*unscaled(B)) : Simplify.compose(A, B)
*(A::Mapping,     B::Identity   ) = A
*(A::Mapping,     B::Scaled     ) =
    is_linear(A) ? multiplier(B)*(A*unscaled(B)) : Composition(A, B)
*(A::Mapping,    B::Composition) = Simplify.compose(A, B)
*(A::Mapping,    B::Mapping    ) = Composition(A, B)

*(A::Inverse{T}, B::T) where {T<:Mapping} =
    is_same_mapping(unveil(A), B) ? Id : Composition(A, B)
*(A::T, B::Inverse{T}) where {T<:Mapping} =
    is_same_mapping(A, unveil(B)) ? Id : Composition(A, B)
*(A::Inverse, B::Inverse) = Composition(A, B)
*(A::InverseAdjoint{T}, B::Adjoint{T}) where {T<:Mapping} =
    is_same_mapping(unveil(A), unveil(B)) ? Id : Composition(A, B)
*(A::Adjoint{T}, B::InverseAdjoint{T}) where {T<:Mapping} =
    is_same_mapping(unveil(A), unveil(B)) ? Id : Composition(A, B)
*(A::InverseAdjoint, B::InverseAdjoint) = Composition(A, B)

# Automatically build Gram operators, Gram(A) ≡ A'*A.  The following automatic
# rules are implemented (for an "allowed" linear mapping A):
#
#     A'*A -> Gram(A)
#     A*A' -> Gram(A')
#     inv(A)*inv(A') -> inv(A'*A) -> inv(Gram(A))
#     inv(A')*inv(A) -> inv(A*A') -> inv(Gram(A'))
#
# other rules implemented elsewhere:
#
#     Gram(inv(A))  -> inv(Gram(A'))
#     Gram(inv(A')) -> inv(Gram(A))
#
# In principle, if forming the adjoint has been allowed, it not needed
# to check whether operands are linear mappings.
*(A::Adjoint{T}, B::T) where {T<:Mapping} =
    is_same_mapping(unveil(A), B) ? Gram(B) : Composition(A, B)
*(A::T, B::Adjoint{T}) where {T<:Mapping} =
    is_same_mapping(A, unveil(B)) ? Gram(B) : Composition(A, B)
*(A::Inverse{T}, B::InverseAdjoint{T}) where {T<:Mapping} =
    is_same_mapping(unveil(A), unveil(B)) ? Inverse(Gram(unveil(A))) :
    Composition(A, B)
*(A::InverseAdjoint{T}, B::Inverse{T}) where {T<:Mapping} =
    is_same_mapping(unveil(A), unveil(B)) ?
    Inverse(Gram(Adjoint(unveil(A)))) : Composition(A, B)

# Left and right divisions.
\(A::Mapping, B::Mapping) = inv(A)*B
/(A::Mapping, B::Mapping) = A*inv(B)

#------------------------------------------------------------------------------
# VCREATE, APPLY AND APPLY!

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
for any supported operations `P` and argument type for `x`.  The result
returned by `vcreate` should be of predictible type to ensure *type-stability*.
Checking the validity (*e.g.* the size) of argument `x` in `vcreate` may be
skipped because this argument will be eventually checked by the `apply!`
method.

See also: [`Mapping`](@ref), [`apply`](@ref).

"""
vcreate(A::Mapping, x, scratch::Bool=false) = vcreate(Direct, A, x, scratch)
vcreate(::Type{P}, A::Mapping, x) where {P<:Operations} =
    vcreate(P, A, x, false)

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

Not all operations may be implemented by the different types of mappings and
`Adjoint` and `InverseAdjoint` may only be applicable for linear mappings.

Optional argument `scratch` indicates whether input argument `x` can be
overwritten by the operation.  This may be exploited to avoid allocating
temporary workspace(s).  The caller should set `scratch = true` if `x` is not
needed after calling `apply`.  If `scratch = true`, then it is possible that
`y` be the same object as `x`; otherwise, `y` is a new object unless applying
the operation yields the same contents as `y` for the result `x` (this is
always true for the identity for instance).  Thus, in general, it should not be
assumed that the result of applying a mapping is different from the input.

Julia methods are provided so that `apply(A', x)` automatically calls
`apply(Adjoint, A, x)` so the shorter syntax may be used without impacting
performances.

See also: [`Mapping`](@ref), [`apply!`](@ref), [`vcreate`](@ref).

"""
apply(A::Mapping, x, scratch::Bool=false) = apply(Direct, A, x, scratch)
apply(P::Type{<:Operations}, A::Mapping, x, scratch::Bool=false) =
    apply!(1, P, A, x, scratch, 0, vcreate(P, A, x, scratch))

*(A::Mapping, x) = apply(Direct, A, x, false)
\(A::Mapping, x) = apply(Inverse, A, x, false)

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
apply!(α::Number, ::Type{P}, A::M, x, scratch::Bool, β::Number, y) -> y
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
#     apply!(α::Number, ::Type{P}, A::MappingType, x::X, scratch::Bool,
#            β::Number, y::Y) where {P<:Operations,X,Y}
#
# has to be implemented (possibly with restrictions on X and Y) by subtypes of
# Mapping so we provide the necessary mechanism to dispatch derived methods.
apply!(A::Mapping, x, y) =
    apply!(1, Direct, A, x, false, 0, y)
apply!(α::Number, A::Mapping, x, y) =
    apply!(α, Direct, A, x, false, 0, y)
apply!(A::Mapping, x, β::Number, y) =
    apply!(1, Direct, A, x, false, β, y)
apply!(α::Number, A::Mapping, x, β::Number, y) =
    apply!(α, Direct, A, x, false, β, y)

apply!(P::Type{<:Operations}, A::Mapping, x, y) =
    apply!(1, P, A, x, false, 0, y)
apply!(α::Number, P::Type{<:Operations}, A::Mapping, x, y) =
    apply!(α, P, A, x, false, 0, y)
apply!(P::Type{<:Operations}, A::Mapping, x, β::Number, y) =
    apply!(1, P, A, x, false, β, y)
apply!(α::Number, P::Type{<:Operations}, A::Mapping, x, β::Number, y) =
    apply!(α, P, A, x, false, β, y)

apply!(A::Mapping, x, scratch::Bool, y) =
    apply!(1, Direct, A, x, scratch, 0, y)
apply!(α::Number, A::Mapping, x, scratch::Bool, y) =
    apply!(α, Direct, A, x, scratch, 0, y)
apply!(A::Mapping, x, scratch::Bool, β::Number, y) =
    apply!(1, Direct, A, x, scratch, β, y)

apply!(P::Type{<:Operations}, A::Mapping, x, scratch::Bool, y) =
    apply!(1, P, A, x, scratch, 0, y)
apply!(α::Number, P::Type{<:Operations}, A::Mapping, x, scratch::Bool, y) =
    apply!(α, P, A, x, scratch, 0, y)
apply!(P::Type{<:Operations}, A::Mapping, x, scratch::Bool, β::Number, y) =
    apply!(1, P, A, x, scratch, β, y)

# Change order of arguments.
apply!(y, A::Mapping, x, scratch::Bool=false) =
    apply!(1, Direct, A, x, scratch, 0, y)
apply!(y, P::Type{<:Operations}, A::Mapping, x, scratch::Bool=false) =
    apply!(1, P, A, x, scratch, 0, y)
apply!(y, α::Number, A::Mapping, x, scratch::Bool=false) =
    apply!(α, Direct, A, x, scratch, 0, y)
apply!(y, α::Number, P::Type{<:Operations}, A::Mapping, x, scratch::Bool=false) =
    apply!(α, P, A, x, scratch, 0, y)
apply!(β::Number, y, A::Mapping, x, scratch::Bool=false) =
    apply!(1, Direct, A, x, scratch, β, y)
apply!(β::Number, y, P::Type{<:Operations}, A::Mapping, x, scratch::Bool=false) =
    apply!(1, P, A, x, scratch, β, y)
apply!(β::Number, y, α::Number, A::Mapping, x, scratch::Bool=false) =
    apply!(α, Direct, A, x, scratch, β, y)
apply!(β::Number, y, α::Number, P::Type{<:Operations}, A::Mapping, x, scratch::Bool=false) =
    apply!(α, P, A, x, scratch, β, y)

# Extend `mul!` so that `A'*x`, `A*B*C*x`, etc. yield the expected result.
mul!(y, A::Mapping, x) = apply!(1, Direct, A, x, false, 0, y)

# Implemention of the `apply!(α,P,A,x,scratch,β,y)` and
# `vcreate(P,A,x,scratch)` methods for a scaled mapping.
for (P, expr) in ((:Direct, :(α*multiplier(A))),
                  (:Adjoint, :(α*conj(multiplier(A)))),
                  (:Inverse, :(α/multiplier(A))),
                  (:InverseAdjoint, :(α/conj(multiplier(A)))))
    @eval begin

        apply!(α::Number, ::Type{$P}, A::Scaled, x, scratch::Bool, β::Number, y) =
            apply!($expr, $P, unscaled(A), x, scratch, β, y)

    end
end

"""
    overwritable(scratch, x, y) -> bool

yields whether the result `y` of applying a mapping to `x` with scratch flag
`scratch` can overwritten.  Arguments `x` and `y` can be reversed.

"""
overwritable(scratch::Bool, x, y) = (scratch || x !== y)

# Implement `apply` for scaled operators to avoid the needs of explicitly
# calling `vcreate` as done by the default implementation of `apply`.  This is
# needed for scaled compositions among others.
function apply(::Type{Direct}, A::Scaled, x, scratch::Bool)
    y = apply(Direct, unscaled(A), x, scratch)
    vscale!((overwritable(scratch, x, y) ? y : vcopy(y)), multiplier(A))
end

function apply(::Type{Adjoint}, A::Scaled, x, scratch::Bool)
    y = apply(Direct, unscaled(A), x, scratch)
    vscale!((overwritable(scratch, x, y) ? y : vcopy(y)), conj(multiplier(A)))
end

function apply(::Type{Inverse}, A::Scaled, x, scratch::Bool)
    y = apply(Direct, unscaled(A), x, scratch)
    vscale!((overwritable(scratch, x, y) ? y : vcopy(y)), 1/multiplier(A))
end

function apply(::Type{InverseAdjoint}, A::Scaled, x, scratch::Bool)
    y = apply(Direct, unscaled(A), x, scratch)
    vscale!((overwritable(scratch, x, y) ? y : vcopy(y)), 1/conj(multiplier(A)))
end

vcreate(P::Type{<:Operations}, A::Scaled, x, scratch::Bool) =
    vcreate(P, unscaled(A), x, scratch)

# Implemention of the `vcreate(P,A,x,scratch)` and
# `apply!(α,P,A,x,scratch,β,y)` methods for the various decorations of a
# mapping so as to automatically unveil the embedded mapping.
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

        vcreate(::Type{$T1}, A::$T2, x, scratch::Bool) =
            vcreate($T3, unveil(A), x, scratch)

        apply!(α::Number, ::Type{$T1}, A::$T2, x, scratch::Bool, β::Number, y) =
            apply!(α, $T3, unveil(A), x, scratch, β, y)

    end
end

# Implementation of the `vcreate(P,A,x,scratch)` and
# `apply!(α,P,A,x,scratch,β,y)` and methods for a sum of mappings.  Note that
# `Sum` instances are warranted to have at least 2 components.

function vcreate(::Type{P}, A::Sum, x,
                 scratch::Bool) where {P<:Union{Direct,Adjoint}}
    # The sum only makes sense if all mappings yields the same kind of result.
    # Hence we just call the vcreate method for the first mapping of the sum.
    vcreate(P, A[1], x, scratch)
end

function apply!(α::Number, P::Type{<:Union{Direct,Adjoint}}, A::Sum{N},
                x, scratch::Bool, β::Number, y) where {N}
    if α == 0
        # Just scale the destination.
        vscale!(y, β)
    else
        # Apply first mapping with β and then other with β=1.  Scratch flag is
        # always false until last mapping because we must preserve x as there
        # is more than one term.
        apply!(α, P, A[1], x, false, β, y)
        for i in 2:N
            apply!(α, P, A[i], x, (scratch && i == N), 1, y)
        end
    end
    return y
end

vcreate(::Type{<:Union{Inverse,InverseAdjoint}}, A::Sum, x, scratch::Bool) =
    throw_unsupported_inverse_of_sum()

apply(::Type{<:Union{Inverse,InverseAdjoint}}, A::Sum, x, scratch::Bool) =
    throw_unsupported_inverse_of_sum()

function apply!(α::Number, ::Type{<:Union{Inverse,InverseAdjoint}}, A::Sum,
                x, scratch::Bool, β::Number, y)
    throw_unsupported_inverse_of_sum()
end

throw_unsupported_inverse_of_sum() =
    error("automatic dispatching of the inverse of a sum of mappings is not supported")

# Implementation of the `apply!(α,P,A,x,scratch,β,y)` method for a composition
# of mappings.  There is no possible `vcreate(P,A,x,scratch)` method for a
# composition so we directly extend the `apply(P,A,x,scratch)` method.  Note
# that `Composition` instances are warranted to have at least 2 components.
#
# The unrolled code (taking care of allowing as few temporaries as possible and
# for the Direct or InverseAdjoint operation) writes:
#
#     w1 = apply(P, A[N], x, scratch)
#     scratch = overwritable(scratch, x, w1)
#     w2 = apply!(1, P, A[N-1], w1, scratch)
#     scratch = overwritable(scratch, w1, w2)
#     w3 = apply!(1, P, A[N-2], w2, scratch)
#     scratch = overwritable(scratch, w2, w3)
#     ...
#     return apply!(α, P, A[1], wNm1, scratch, β, y)
#
# To break the type barrier, this is done by a recursion.  The recursion is
# just done in the other direction for the Adjoint or Inverse operation.

function vcreate(::Type{<:Operations},
                 A::Composition{N}, x, scratch::Bool) where {N}
    error("it is not possible to create the output of a composition of mappings")
end

function apply!(α::Number, ::Type{P}, A::Composition{N}, x, scratch::Bool,
                β::Number, y) where {N,P<:Union{Direct,InverseAdjoint}}
    if α == 0
        # Just scale the destination.
        vscale!(y, β)
    else
        ops = terms(A)
        w = apply(P, *, ops[2:N], x, scratch)
        scratch = overwritable(scratch, w, x)
        apply!(α, P, ops[1], w, scratch, β, y)
    end
    return y
end

function apply(::Type{P}, A::Composition{N}, x,
               scratch::Bool) where {N,P<:Union{Direct,InverseAdjoint}}
    apply(P, *, terms(A), x, scratch)
end

function apply(::Type{P}, ::typeof(*), ops::NTuple{N,Mapping}, x,
               scratch::Bool) where {N,P<:Union{Direct,InverseAdjoint}}
    w = apply(P, ops[N], x, scratch)
    N == 1 && return w
    scratch = overwritable(scratch, w, x)
    apply(P, *, ops[1:N-1], w, scratch)
end

function apply!(α::Number, ::Type{P}, A::Composition{N}, x, scratch::Bool,
                β::Number, y) where {N,P<:Union{Adjoint,Inverse}}
    if α == 0
        # Just scale the destination.
        vscale!(y, β)
    else
        ops = terms(A)
        w = apply(P, *, ops[1:N-1], x, scratch)
        scratch = overwritable(scratch, w, x)
        apply!(α, P, ops[N], w, scratch, β, y)
    end
    return y
end

function apply(::Type{P}, A::Composition{N}, x,
               scratch::Bool) where {N,P<:Union{Adjoint,Inverse}}
    apply(P, *, terms(A), x, scratch)
end

function apply(::Type{P}, ::typeof(*), ops::NTuple{N,Mapping}, x,
               scratch::Bool) where {N,P<:Union{Adjoint,Inverse}}
    w = apply(P, ops[1], x, scratch)
    N == 1 && return w
    scratch = overwritable(scratch, w, x)
    apply(P, *, ops[2:N], w, scratch)
end

# Default rules to apply a Gram operator.  Gram matrices are Hermitian by
# construction which left only 2 cases to deal with.

apply!(α::Number, ::Type{Adjoint}, A::Gram, x, scratch::Bool, β::Number, y) =
    apply!(α, Direct, A, x, scratch, β, y)

apply!(α::Number, ::Type{InverseAdjoint}, A::Gram, x, scratch::Bool, β::Number, y) =
    apply!(α, Inverse, A, x, scratch, β, y)

function apply!(α::Number, ::Type{Direct}, A::Gram, x, scratch::Bool, β::Number, y)
    if α == 0
        vscale!(y, β)
    else
        B = unveil(A) # A ≡ B'*B
        z = apply(Direct, B, x, scratch) # z <- B⋅x
        apply!(α, Adjoint, B, z, (z !== x), β, y) # y <- α⋅B'⋅z + β⋅y
    end
    return y
end

function apply!(α::Number, ::Type{Inverse}, A::Gram, x, scratch::Bool, β::Number, y)
    if α == 0
        vscale!(y, β)
    else
        B = unveil(A) # A ≡ B'⋅B
        # Compute α⋅inv(A)⋅x + β⋅y = α⋅inv(B'⋅B)⋅x + β⋅y
        #                          = α⋅inv(B)⋅inv(B')⋅x + β⋅y
        z = apply(InverseAdjoint, B, x, scratch) # z <- inv(B')⋅x
        apply!(α, Inverse, B, z, (z !== x), β, y) # y <- α⋅inv(B)⋅z + β⋅y
    end
    return y
end
