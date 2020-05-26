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
# Copyright (c) 2017-2019 Éric Thiébaut.
#

const UnsupportedInverseOfSumOfMappings =
    "automatic dispatching of the inverse of a sum of mappings is not supported"

"""

```julia
LinearType(A)
```

yields the *linear* trait of mapping `A` indicating whether `A` is certainly
linear.  The returned value is one of the singletons `Linear()` for linear maps
or `NonLinear()` for other mappings.

See also: [`Trait`](@ref), [`is_linear`](@ref).

"""
LinearType(::LinearMapping) = Linear()
LinearType(::Scaled{<:LinearMapping}) = Linear()
LinearType(A::Union{Scaled,Inverse}) = LinearType(operand(A))
LinearType(::Mapping) = NonLinear() # anything else is non-linear
LinearType(A::Union{Sum,Composition}) =
    (allof(x -> LinearType(x) === Linear(), operands(A)...) ?
     Linear() : NonLinear())
LinearType(A::Scaled{T,S}) where {T,S} =
    # If the multiplier λ of a scaled mapping A = (λ⋅M) is zero, then
    # A behaves linearly even though M is not a linear mapping.
    (multiplier(A) == zero(S) ? Linear() : LinearType(operand(A)))

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
SelfAdjointType(A::Union{Scaled,Inverse}) = SelfAdjointType(operand(A))
SelfAdjointType(A::Sum) =
    (allof(x -> SelfAdjointType(x) === SelfAdjoint(), operands(A)...) ?
     SelfAdjoint() : NonSelfAdjoint())

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
MorphismType(A::Union{Scaled,Inverse}) = MorphismType(operand(A))
MorphismType(A::Union{Sum,Composition}) =
    (allof(x -> MorphismType(x) === Endomorphism(), operands(A)...) ?
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
DiagonalType(A::Union{Scaled,Inverse}) = DiagonalType(operand(A))
DiagonalType(A::Union{Sum,Composition}) =
    (allof(x -> DiagonalType(x) === DiagonalMapping(), operands(A)...) ?
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
#  - Adjoint of a scaled operand is rewritten as a scaled adjoint of the
#    operand.  Similarly, inverse of a scaled operand is rewritten as a scaled
#    inverse of the operand, if the operand is inear, or as the inverse of the
#    operand times a scaled identity otherwise.
#
#  - Adjoint of the inverse is rewritten as inverse of the adjoint.
#
#  - Inner constructors are fully qualified.  Un-qualified outer constructors
#    just call the basic methods.
#
#  - To simplify a sum, the terms corresponding to identical operands (possibly
#    scaled) are first grouped to produce a single operand (possibly scaled)
#    per group, the resulting terms are sorted (so that all equivalent
#    expressions yield the same result) and the "zeros" eliminated (if all
#    terms are "zero", the sum simplifies to the first one).  For now, the
#    sorting is not perfect as it is based on `objectid()` hashing method.
#
#  - To simplify a composition, a fusion algorithm is applied and "ones" are
#    eliminated.  It is assumed that composition is non-commutative so the
#    ordering of terms is left unchanged.  Thanks to this, simplification rules
#    for simple compositions (made of two non-composition operands) can be
#    automatically performed by proper dispatching rules.  Calling the fusion
#    algorithm is only needed for more complex compositions.
#
# The simplication algorithm is not perfect (LazyAlgebra is not intended to be
# for symbolic computations) but do a reasonnable job.  In particular complex
# operands built using the same sequences should be simplified in the same way
# and thus be correctly identified as being identical.

#------------------------------------------------------------------------------
# NEUTRAL ELEMENTS

# The neutral element ("zero") for the addition is zero times an operand of the
# proper type.
zero(A::Mapping) = 0*A

iszero(A::Scaled) = iszero(multiplier(A))
iszero(::Mapping) = false

# The neutral element ("one") for the composition is the identity.
const I = Identity()
one(::Mapping) = I

isone(::Identity) = true
isone(::Mapping) = false

#------------------------------------------------------------------------------
# UNQUALIFIED OUTER CONSTRUCTORS

# Unqualified outer constructors are provided which simply call the appropriate
# basic methods which are extended (elsewhere) to perform all suitable
# simplifications.
Direct(A::Mapping) = A # provided for completeness
Adjoint(A::Mapping) = adjoint(A)
Inverse(A::Mapping) = inv(A)
InverseAdjoint(A::Mapping) = inv(adjoint(A))
Scaled(α::Number, A::Mapping) = α*A
Sum(args::Mapping...) = _simplify_sum(args)
Composition(args::Mapping...) =
    # FIXME: Calling Composition() directly does not simplify expression.
    _merge_sum(args)

#------------------------------------------------------------------------------
# SCALED TYPE

# Left-multiplication and left-division by a scalar.  The only way to
# right-multiply or right-divide by a scalar is to right multiply or divide by
# the scaled identity.
*(α::S, A::T) where {S<:Number,T<:Mapping} =
    (α == one(α) ? A : isfinite(α) ? Scaled{T,S}(α, A) :
     throw(ArgumentError("non-finite multiplier")))
*(α::Number, A::Scaled) = (α*multiplier(A))*operand(A)
\(α::Number, A::Mapping) = inv(α)*A
\(α::Number, A::Scaled) = (multiplier(A)/α)*operand(A)

#------------------------------------------------------------------------------
# ADJOINT TYPE

# Adjoint for non-specific operands.
adjoint(A::LinearMapping) = _adjoint(SelfAdjointType(A), A)
_adjoint(::SelfAdjoint, A::LinearMapping) = A
_adjoint(::NonSelfAdjoint, A::T) where {T<:LinearMapping} = Adjoint{T}(A)
function adjoint(A::T) where {T<:Mapping}
    is_linear(A) ||
        throw(ArgumentError("undefined adjoint of a non-linear operand"))
    return Adjoint{T}(A)
end

# Adjoint for specific operand types.
adjoint(A::Identity) = I
adjoint(A::Scaled{Identity}) = conj(multiplier(A))*I
adjoint(A::Scaled) = conj(multiplier(A))*adjoint(operand(A))
adjoint(A::Adjoint) = operand(A)
adjoint(A::Inverse) = inv(adjoint(operand(A)))
adjoint(A::InverseAdjoint) = inv(operand(A))
adjoint(A::Composition) =
    # It is assumed that the composition has already been simplified, so we
    # just apply the mathematical formula for the adjoint of a composition.
    _merge_mul(reversemap(adjoint, operands(A)))
adjoint(A::Sum) =
    # It is assumed that the sum has already been simplified, so we just apply
    # the mathematical formula for the adjoint of a sum (keeping the same
    # ordering of terms).
    _sort_sum(map(adjoint, operands(A)))

#------------------------------------------------------------------------------
# INVERSE TYPE

# Inverse for non-specific operands (a simple operand or a sum or operands).
inv(A::T) where {T<:Mapping} = Inverse{T}(A)

# Inverse for specific operand types.
inv(A::Identity) = I
inv(A::Scaled{Identity}) = inv(multiplier(A))*I
inv(A::Scaled) = (is_linear(operand(A)) ? inv(multiplier(A))*inv(operand(A)) :
                  inv(operand(A))*(inv(multiplier(A))*I))
inv(A::Inverse) = operand(A)
inv(A::AdjointInverse) = adjoint(operand(A))
inv(A::Adjoint{T}) where {T<:Mapping} = AdjointInverse{T}(operand(A))
inv(A::Composition) =
    # Even though the composition has already been simplified, taking the
    # inverse may trigger other simplifications, so we must rebuild the
    # composition term by term in reverse order (i.e. applying the mathematical
    # formula for the inverse of a composition).
    _merge_inv_mul(operands(A))

# `_merge_inv_mul([A,i,]B)` is recursively called to build the inverse of a
# composition.  Argument A is a mapping (initially not specified or the
# identity) of the resulting composition, argument `i` is the index of the next
# component to take (initially not specified or set to `N` the number of
# operands), argument `B` is a tuple (initially full) of the remaining terms.
_merge_inv_mul(B::NTuple{N,Mapping}) where {N} =
    # Initialize recursion.
    _merge_inv_mul(inv(last(B)), N - 1, B)

function _merge_inv_mul(A::Mapping, i::Int, B::NTuple{N,Mapping}) where {N}
    # Perform intermediate and last recursion step.
    C = A*inv(B[i])
    return (i > 1 ? _merge_inv_mul(C, i - 1, B) : C)
end

#------------------------------------------------------------------------------
# SUM OF OPERANDS

# Unary minus and unary plus.
-(A::Mapping) = (-1)*A
-(A::Scaled) = (-multiplier(A))*operand(A)
+(A::Mapping) = A

# Subtraction.
-(A::Mapping, B::Mapping) = A + (-B)

# Rules for sums built by `A + B`.
+(A::Mapping, B::Mapping) = _simplify_sum((_split_sum(A)..., _split_sum(B)...))

# `_split_sum(A)` yields a tuple of the operands of `A` if it is a sum or just
# `(A,)` otherwise.
_split_sum(A::Sum) = operands(A)
_split_sum(A::Mapping) = (A,)

# `_merge_sum(args...)` constructs a fully qualified sum.  It is assumed that
# the argument(s) have already been simplified.  An empty sum is forbidden
# because there is no universal neutral element ("zero") for the addition.
_merge_sum(arg::Mapping) = arg
_merge_sum(args::Mapping...) = _merge_sum(args)
_merge_sum(args::Tuple{}) = throw(ArgumentError("empty sum"))
_merge_sum(args::Tuple{Mapping}) = args[1]
_merge_sum(args::T) where {N,T<:NTuple{N,Mapping}} = Sum{N,T}(args)

# `_simplify_sum(args...)` simplifies the sum of all operands in `args...` and
# returns a single operand (possibly an instance of `Sum`).  It is assumed that
# the operator `+` is associative and commutative.

# Make a sum out of 0-1 operands.
_simplify_sum(args::Tuple{}) = _merge_sum(args)
_simplify_sum(args::Tuple{Mapping}) = args[1]

# Make a sum out of N operands (with N ≥ 2).
function _simplify_sum(args::NTuple{N,Mapping}) where {N}
    # First group terms corresponding to identical operands (up to an optional
    # multiplier) to produce a single (possibly scaled) operand per group.
    # FIXME: The following algorithm scales as O(N²) which is probably not
    # optimal.  Nevertheless, it never compares twice the same pair of
    # arguments.
    terms = Array{Mapping}(undef, 0)
    flags = fill!(Array{Bool}(undef, N), true)
    i = 1
    while i != 0
        # Push next ungrouped argument.
        push!(terms, args[i])
        flags[i] = false

        # Find any other argument which is identical, possibly scaled, operand.
        k = i + 1
        i = 0
        for j = k:N
            if flags[j]
                if _simplify_sum!(terms, terms[end], args[j])
                    flags[j] = false
                elseif i == 0
                    # Next ungrouped argument to consider.
                    i = j
                end
            end
        end
    end

    # Make a sum out of the terms after having eliminated the zeros.
    return _sort_sum(terms)
end

# `_simplify_sum!(terms, A, B)` is a helper function for the `_simplify_sum`
# method which attempts to make a trivial simplification for `A + B` when
# `A = λ⋅M` and `B = μ⋅M` for any numbers `λ` and `μ` and any operand `M`.
# If such a simplification can be done, the result `(λ + μ)⋅M` is stored as
# the last component of `terms` and `true` is returned; otherwise, `false`
# is returned.
_simplify_sum!(terms::Vector{Mapping}, A::Mapping, B::Mapping) = false

function _simplify_sum!(terms::Vector{Mapping},
                        A::Scaled{T}, B::Scaled{T}) where {T<:Mapping}
    operand(A) === operand(B) || return false
    @inbounds terms[end] = (multiplier(A) + multiplier(B))*operand(A)
    return true
end

function _simplify_sum!(terms::Vector{Mapping},
                        A::Scaled{T}, B::T) where {T<:Mapping}
    operand(A) === B || return false
    @inbounds terms[end] = (multiplier(A) + one(multiplier(A)))*B
    return true
end

function _simplify_sum!(terms::Vector{Mapping},
                        A::T, B::Scaled{T}) where {T<:Mapping}
    A === operand(B) || return false
    @inbounds terms[end] = (one(multiplier(B)) + multiplier(B))*A
    return true
end

function _simplify_sum!(terms::Vector{Mapping},
                        A::T, B::T) where {T<:Mapping}
    A === B || return false
    @inbounds terms[end] = 2*A
    return true
end

# Sort the terms of a sum (so that all equivalent expressions eventually yield
# the same result after simplifications) and eliminate the "zeros"
# (if all terms are "zero", the sum simplifies to the first one).
_sort_sum(args::Mapping...) = _sort_sum(args)
_sort_sum(args::Tuple{Mapping}) = args[1]
function _sort_sum(terms::Union{Tuple{Vararg{Mapping}},Vector{<:Mapping}})
    perms = sortperm([_identifier(terms[i]) for i in 1:length(terms)])
    n = 0
    @inbounds for i in 1:length(perms)
        j = perms[i]
        if ! iszero(terms[j])
            n += 1
            perms[n] = j
        end
    end
    if n ≤ 1
        # All terms are zero or only one term is non-zero, return the first
        # sorted term.
        return terms[perms[1]]
    else
        # Make a sum out of the remaing sorted terms.
        return _merge_sum(ntuple(i -> terms[perms[i]], n))
    end
end

# `_identifier(A)` yields an (almost) unique identifier of operand `A` (of its
# operand component for a scaled operand).  This identifier is suitable for
# sorting terms in a sum of operands.
#
# FIXME: For now, the sorting is not perfect as it is based on objectid()
#        which is a hashing method.
_identifier(A::Mapping) = objectid(A)
_identifier(A::Scaled) = objectid(operand(A))

#------------------------------------------------------------------------------
# COMPOSITION OF OPERANDS

# Dot operator (\cdot + tab) involving a mapping acts as the multiply or
# compose operator.
⋅(A::Mapping, B::Mapping) = A*B
⋅(A::Mapping, B) = A*B
⋅(A, B::Mapping) = A*B

# Compose operator (\circ + tab) beween mappings.
∘(A::Mapping, B::Mapping) = A*B

# Rules for the composition of 2 operands.
*(A::Identity, B::Identity) = I
*(A::Identity, B::Scaled) = B
*(A::Identity, B::Composition) = B
*(A::Identity, B::Mapping) = B
*(A::Scaled, B::Identity) = A
*(A::Scaled, B::Scaled) =
    (is_linear(A) ? (multiplier(A)*multiplier(B))*(operand(A)*operand(B)) :
     multiplier(A)*_merge_mul(operand(A), B))
*(A::Scaled, B::Composition) = multiplier(A)*(operand(A)*B)
*(A::Scaled, B::Mapping) = multiplier(A)*(operand(A)*B)
*(A::Composition, B::Identity) = A
*(A::Composition, B::Scaled) =
    (is_linear(A) ? multiplier(B)*(A*operand(B)) : _simplify_mul(A, B))
*(A::Composition, B::Composition) = _simplify_mul(A, B)
*(A::Composition, B::Mapping) = _simplify_mul(A, B)
*(A::Mapping, B::Identity) = A
*(A::Mapping, B::Scaled) =
    (is_linear(A) ? multiplier(B)*(A*operand(B)) : _merge_mul(A, B))
*(A::Mapping, B::Composition) = _simplify_mul(A, B)
*(A::Mapping, B::Mapping) = _merge_mul(A, B)

*(A::Inverse{T}, B::T) where {T<:Mapping} =
    (operand(A) === B ? I : _merge_mul(A, B))
*(A::T, B::Inverse{T}) where {T<:Mapping} =
    (A === operand(B) ? I : _merge_mul(A, B))
*(A::Inverse, B::Inverse) = _merge_mul(A, B)
*(A::InverseAdjoint{T}, B::Adjoint{T}) where {T<:Mapping} =
    (operand(A) === operand(B) ? I : _merge_mul(A, B))
*(A::Adjoint{T}, B::InverseAdjoint{T}) where {T<:Mapping} =
    (operand(A) === operand(B) ? I : _merge_mul(A, B))
*(A::InverseAdjoint, B::InverseAdjoint) = _merge_mul(A, B)

# Left and right divisions.
\(A::Mapping, B::Mapping) = inv(A)*B
/(A::Mapping, B::Mapping) = A*inv(B)

# `_split_mul(A)` yields a tuple of the operands of `A` if it is a composition
# or just `(A,)` otherwise.
_split_mul(A::Composition) = operands(A)
_split_mul(A::Mapping) = (A,)

# `_merge_mul(args...)` constructs a fully qualified composition.  It is
# assumed that the argument(s) have already been simplified.  An empty
# composition yields the identity which is the universal neutral element
# ("one") for the composition.
_merge_mul(arg::Mapping) = arg
_merge_mul(args::Mapping...) = _merge_mul(args)
_merge_mul(args::Tuple{}) = I
_merge_mul(args::Tuple{Mapping}) = args[1]
_merge_mul(args::T) where {N,T<:NTuple{N,Mapping}} = Composition{N,T}(args)

# `_simplify_mul(A, B)` simplifies the product of `A` and `B`.  The result is a
# tuple `C` of the resulting operands if `A` and `B` are both tuples of
# mappings or an instance of `Mapping` if `A` and `B` are both mappings.  If no
# simplification can be made, the result is just the concatenation of the
# operands in `A` and `B`.  To perform simplifications, it is assumed that, if
# they are compositions, `A` and `B` have already been simplified.  It is also
# assumed that the composition is associative but non-commutative.
_simplify_mul(A::Mapping, B::Mapping) =
    _merge_mul(_simplify_mul(_split_mul(A), _split_mul(B)))

# The following versions of `_simplify_mul` are for `A` and `B` in the form of
# tuples and return a tuple.  The algorithm is recursive and should works for
# any non-commutative binary operator.
_simplify_mul(A::Tuple{}, B::Tuple{}) = (I,)
_simplify_mul(A::Tuple{Vararg{Mapping}}, B::Tuple{}) = A
_simplify_mul(A::Tuple{}, B::Tuple{Vararg{Mapping}}) = B
function _simplify_mul(A::NTuple{M,Mapping},
                       B::NTuple{N,Mapping}) where {M,N}
    # Here M ≥ 1 and N ≥ 1.
    @assert M ≥ 1 && N ≥ 1

    # Attempt to simplify the product of operands at their junction.
    C = _split_mul(A[M]*B[1])
    len = length(C)
    if len == 2
        if C === (A[M], B[1])
            # No simplification, just concatenate the 2 compositions.
            return (A..., B...)
        else
            # There have been some changes, but the result of A[M]*B[1] still
            # have two terms which cannot be further simplified.  So we
            # simplify its head with the remaining leftmost operands and its
            # tail with the remaining rightmost operands.
            L = _simplify_mul(A[1:M-1], C[1]) # simplify leftmost operands
            R = _simplify_mul(C[2], B[2:N])   # simplify rightmost operands
            if L[end] !== C[1] || R[1] !== C[2]
                # At least one of the last of resulting rightmost operands or
                # the first of the resulting leftmost operands has been
                # modified so there may be other possible simplifications.
                return _simplify_mul(L, R)
            else
                # No further simplifications possible.
                return (L..., R...)
            end
        end
    elseif len == 1
        # Simplications have occured resulting in a single operand.  This
        # operand can be simplified whith the remaining leftmost operands
        # and/or with the remaining rightmost operands.  To benefit from the
        # maximum simplifications, we can either do:
        #
        #     _simplify_mul(A[1:end-1], _simplify_mul(C, B[2:end]))
        #
        # that is, simplify right then left, or:
        #
        #     _simplify_mul(_simplify_mul(A[1:end-1], C), B[2:end])
        #
        # that is simplify left then right.  Since we want to propagate
        # multipliers to the right of compositions, the former is the most
        # appropriate.
        return _simplify_mul(A[1:end-1], _simplify_mul(C, B[2:end]))
    else# len == 0
        # The result of A[M]*B[1] is the neutral element for * thus eliminating
        # these operands from the merging.  We just have to repeat the process
        # with the truncated associations in case further simplications are
        # possible.
        return _simplify_mul(A[1:M-1], B[2:N-1])
    end
end

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
            apply!($expr, $P, operand(A), x, scratch, β, y)

    end
end

"""

```julia
overwritable(scratch, x, y) -> bool
```

yields whether the result `y` of applying a mapping to `x` with scratch flag
`scratch` can overwritten.  Arguments `x` and `y` can be reversed.

"""
overwritable(scratch::Bool, x, y) =
    (scratch || ! is_same_mutable_object(x, y))

# Implement `apply` for scaled operators to avoid the needs of explicitly
# calling `vcreate` as done by the default implementation of `apply`.  This is
# needed for scaled compositions among others.
function apply(::Type{Direct}, A::Scaled, x, scratch::Bool)
    y = apply(Direct, operand(A), x, scratch)
    vscale!((overwritable(scratch, x, y) ? y : vcopy(y)), multiplier(A))
end

function apply(::Type{Adjoint}, A::Scaled, x, scratch::Bool)
    y = apply(Direct, operand(A), x, scratch)
    vscale!((overwritable(scratch, x, y) ? y : vcopy(y)), conj(multiplier(A)))
end

function apply(::Type{Inverse}, A::Scaled, x, scratch::Bool)
    y = apply(Direct, operand(A), x, scratch)
    vscale!((overwritable(scratch, x, y) ? y : vcopy(y)), 1/multiplier(A))
end

function apply(::Type{InverseAdjoint}, A::Scaled, x, scratch::Bool)
    y = apply(Direct, operand(A), x, scratch)
    vscale!((overwritable(scratch, x, y) ? y : vcopy(y)), 1/conj(multiplier(A)))
end

vcreate(P::Type{<:Operations}, A::Scaled, x, scratch::Bool) =
    vcreate(P, operand(A), x, scratch)

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

        vcreate(::Type{$T1}, A::$T2, x, scratch::Bool) =
            vcreate($T3, operand(A), x, scratch)

        apply!(α::Number, ::Type{$T1}, A::$T2, x, scratch::Bool, β::Number, y) =
            apply!(α, $T3, operand(A), x, scratch, β, y)

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

vcreate(::Type{<:Union{Inverse,InverseAdjoint}}, A::Sum, x, scratch::Bool) =
    error(UnsupportedInverseOfSumOfMappings)

apply(::Type{<:Union{Inverse,InverseAdjoint}}, A::Sum, x, scratch::Bool) =
    error(UnsupportedInverseOfSumOfMappings)

function apply!(α::Number, ::Type{<:Union{Inverse,InverseAdjoint}}, A::Sum,
                x, scratch::Bool, β::Number, y)
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

function vcreate(P::Type{<:Operations},
                 A::Composition{N}, x, scratch::Bool) where {N}
    error("it is not possible to create the output of a composition of mappings")
end

# Gram matrices are Hermitian by construction.
apply!(α::Number, ::Type{Adjoint}, A::Gram, x, scratch::Bool, β::Number, y) =
    apply!(α, Direct, A, x, scratch, β, y)
apply!(α::Number, ::Type{InverseAdjoint}, A::Gram, x, scratch::Bool, β::Number, y) =
    apply!(α, Inverse, A, x, scratch, β, y)

function apply!(α::Number, P::Type{<:Union{Direct,InverseAdjoint}},
                A::Composition{N}, x, scratch::Bool, β::Number, y) where {N}
    @assert N ≥ 2 "bug in Composition constructor"
    ops = operands(A)
    w = _apply(P, ops[2:N], x, scratch)
    scratch = overwritable(scratch, w, x)
    return apply!(α, P, ops[1], w, scratch, β, y)
end

function apply(P::Type{<:Union{Direct,InverseAdjoint}},
               A::Composition{N}, x, scratch::Bool) where {N}
    @assert N ≥ 2 "bug in Composition constructor"
    return _apply(P, operands(A), x, scratch)
end

function _apply(P::Type{<:Union{Direct,InverseAdjoint}},
                ops::NTuple{N,Mapping}, x, scratch::Bool) where {N}
    w = apply(P, ops[N], x, scratch)
    if N > 1
        scratch = overwritable(scratch, w, x)
        return _apply(P, ops[1:N-1], w, scratch)
    else
        return w
    end
end

function apply!(α::Number, P::Type{<:Union{Adjoint,Inverse}},
                A::Composition{N}, x, scratch::Bool, β::Number, y) where {N}
    @assert N ≥ 2 "bug in Composition constructor"
    ops = operands(A)
    w = _apply(P, ops[1:N-1], x, scratch)
    scratch = overwritable(scratch, w, x)
    return apply!(α, P, ops[N], w, scratch, β, y)
end

function apply(P::Type{<:Union{Adjoint,Inverse}},
               A::Composition{N}, x, scratch::Bool) where {N}
    @assert N ≥ 2 "bug in Composition constructor"
    return _apply(P, operands(A), x, scratch)
end

function _apply(P::Type{<:Union{Adjoint,Inverse}},
                ops::NTuple{N,Mapping}, x, scratch::Bool) where {N}
    w = apply(P, ops[1], x, scratch)
    if N > 1
        scratch = overwritable(scratch, w, x)
        return _apply(P, ops[2:N], w, scratch)
    else
        return w
    end
end
