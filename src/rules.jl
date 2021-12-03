#
# rules.jl -
#
# Implement rules for automatically simplifying expressions involving mappings.
#
#-------------------------------------------------------------------------------
#
# This file is part of LazyAlgebra (https://github.com/emmt/LazyAlgebra.jl)
# released under the MIT "Expat" license.
#
# Copyright (c) 2017-2021 Éric Thiébaut.
#

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
# right-multiply or right-divide a mapping by a scalar is to right multiply or
# divide it by the scaled identity.
*(α::Number, A::Mapping) = (isone(α) ? A : Scaled(α, A))
*(α::Number, A::Scaled) = (α*multiplier(A))*unscaled(A)
\(α::Number, A::Mapping) = inv(α)*A
\(α::Number, A::Scaled) = (multiplier(A)/α)*unscaled(A)
/(α::Number, A::Mapping) = α*inv(A)

#------------------------------------------------------------------------------
# ADJOINT TYPE

# Adjoint for non-specific mappings.
adjoint(A::Mapping) = _adjoint(LinearType(A), A)

_adjoint(::Linear, A::Mapping) = _adjoint(Linear(), SelfAdjointType(A), A)
_adjoint(::Linear, ::SelfAdjoint, A::Mapping) = A
_adjoint(::Linear, ::NonSelfAdjoint, A::Mapping) = Adjoint(A)
_adjoint(::NonLinear, A::Mapping) =
    throw_forbidden_adjoint_of_non_linear_mapping()

# Adjoint for specific mapping types.
adjoint(A::Identity) = Id
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
inv(A::Scaled) = (is_linear(unscaled(A)) ?
                  inv(multiplier(A))*inv(unscaled(A)) :
                  inv(unscaled(A))*(inv(multiplier(A))*Id))
inv(A::Inverse) = unveil(A)
inv(A::InverseAdjoint) = adjoint(unveil(A))
inv(A::Adjoint) = InverseAdjoint(unveil(A))
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

# Simplify the sum of two mappings.
+(A::Mapping, B::Mapping) = add(A, B)

"""
    add(A, B)

performs the final stage of simplifying the sum `A + B` of mappings `A` and
`B`.  This method assumes that any other simplifications than those involving
sum of sums have been performed on `A + B` and that `A` and `B` have already
been simplified (individually).

Trivial simplifications of the composition `A + B` of mappings `A` and `B` must
be done by specializing the operator `+` and `sum(A,B)` may eventually be
called to simplify the sum of `A` and `B` when one of these may be a
sum.  This method just returns `Sum(A,B)` when none of its
arguments is a composition.


yields a simplified sum `A + B` of the mappings `A` and `B`.  This helper
function is intended to be called when at least one of `A` or `B` is a sum.

The ability to perform simplifications relies on implemented specializations of
`A + B` when neither `A` nor `B` are sums.  It is also assumed that `A` and `B`
have already been simplified if they are sums.

"""
add(A::Sum,     B::Mapping) = _add(A, B)
add(A::Mapping, B::Sum    ) = _add(A, B)
add(A::Sum,     B::Sum    ) = _add(A, B)
add(A::Mapping, B::Mapping) = begin
    # Neither `A` nor `B` is a sum.
    if identical(unscaled(A), unscaled(B))
        return (multiplier(A) + multiplier(B))*unscaled(A)
    elseif identifier(A) ≤ identifier(B)
        return Sum(A, B)
    else
        return Sum(B, A)
    end
end

_add(A::Mapping, B::Mapping) = begin
    V = add!(as_vector(+, A), B)
    length(V) == 1 ? V[1] : Sum(to_tuple(V))
end

# Add the terms of a sum one-by-one.  Since terms must be re-ordered, there are
# no obvious better ways to recombine.
function add!(A::Vector{Mapping}, B::Sum{N}) where {N}
    @inbounds for i in 1:N
        add!(A, B[i])
    end
    return A
end

function add!(A::Vector{Mapping}, B::Mapping)
    # Nothing to do if B is zero times anything.
    multiplier(B) == 0 && return A

    # If exact match found, update A in-place and return.
    n = length(A)
    @inbounds for i in 1:n
        if identical(unscaled(A[i]), unscaled(B))
            λ = multiplier(A[i]) + multiplier(B)
            if isone(λ)
                A[i] = unscaled(B)
            elseif !iszero(λ)
                A[i] = λ*unscaled(B)
            else
                # Multiplier is zero. Drop term if there are other terms
                # or keep the single term times zero.
                if n > 1
                    for j in i:n-1
                        A[j] = A[j+1]
                    end
                    resize!(A, n - 1)
                else
                    A[1] = 0*unscaled(A[1])
                end
            end
            return A
        end
    end

    # If no exact match found, insert B in A in order.
    id = identifier(B)
    i = 1
    while i ≤ n && @inbounds(identifier(A[i])) < id
        i += 1
    end
    resize!(A, n + 1)
    @inbounds for j in n:-1:i
        A[j+1] = A[j]
    end
    A[i] = B
    return A
end

#------------------------------------------------------------------------------
# COMPOSITION OF MAPPINGS

# Left and right divisions.
\(A::Mapping, B::Mapping) = inv(A)*B
/(A::Mapping, B::Mapping) = A*inv(B)

# Dot operator (\cdot + tab) involving a mapping acts as the multiply or
# compose operator.
⋅(A::Mapping, B::Mapping) = A*B
⋅(A::Mapping, B::Any    ) = A*B
⋅(A::Any,     B::Mapping) = A*B

# Compose operator (\circ + tab) beween mappings.
∘(A::Mapping, B::Mapping) = A*B

# Rules for the composition of 2 mappings.  Mappings that may behave
# specifically in a composition have type `Identity`, `Scaled` and
# `Composition`; all others have the same behavior.

# Composition with identity.
*(::Identity, ::Identity) = Id
for T in (Scaled, Composition, Sum, Mapping)
    @eval begin
        *(::Identity, A::$T) = A
        *(A::$T, ::Identity) = A
    end
end

# Simplify the composition of two mappings (including compositions).
*(A::Mapping, B::Mapping) = compose(A, B)

# Simplify compositions involving a scaled mapping.
*(A::Scaled, B::Mapping) = multiplier(A)*(unscaled(A)*B)
*(A::Mapping, B::Scaled) =
    if is_linear(A)
        multiplier(B)*(A*unscaled(B))
    else
        compose(A, B)
    end
*(A::Scaled, B::Scaled) =
    if is_linear(A)
        (multiplier(A)*multiplier(B))*(unscaled(A)*unscaled(B))
    else
        multiplier(A)*(unscaled(A)*B)
    end

# Simplify compositions involving an inverse mapping.
*(A::Inverse{T}, B::T) where {T<:Mapping} =
    identical(unveil(A), B) ? Id : compose(A, B)
*(A::T, B::Inverse{T}) where {T<:Mapping} =
    identical(A, unveil(B)) ? Id : compose(A, B)
*(A::Inverse, B::Inverse) = compose(A, B)
*(A::InverseAdjoint{T}, B::Adjoint{T}) where {T<:Mapping} =
    identical(unveil(A), unveil(B)) ? Id : compose(A, B)
*(A::Adjoint{T}, B::InverseAdjoint{T}) where {T<:Mapping} =
    identical(unveil(A), unveil(B)) ? Id : compose(A, B)
*(A::InverseAdjoint, B::InverseAdjoint) = compose(A, B)

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
# In principle, if forming the adjoint has been allowed, it is not needed to
# check whether operands are linear mappings.
*(A::Adjoint{T}, B::T) where {T<:Mapping} =
    identical(unveil(A), B) ? Gram(B) : compose(A, B)
*(A::T, B::Adjoint{T}) where {T<:Mapping} =
    identical(A, unveil(B)) ? Gram(B) : compose(A, B)
*(A::Inverse{T}, B::InverseAdjoint{T}) where {T<:Mapping} =
    identical(unveil(A), unveil(B)) ? Inverse(Gram(unveil(A))) :
    compose(A, B)
*(A::InverseAdjoint{T}, B::Inverse{T}) where {T<:Mapping} =
    identical(unveil(A), unveil(B)) ?
    Inverse(Gram(Adjoint(unveil(A)))) : compose(A, B)

"""
    compose(A,B)

performs the final stage of simplifying the composition `A*B` of mappings `A`
and `B`.  This method assumes that any other simplifications than those
involving composition of compositions have been performed on `A*B` and that `A`
and `B` have already been simplified (individually).

Trivial simplifications of the composition `A*B` of mappings `A` and `B` must
be done by specializing the operator `*` and `compose(A,B)` may eventually be
called to simplify the composition of `A` and `B` when one of these may be a
composition.  This method just returns `Composition(A,B)` when neither `A` nor
`B` is a composition.

""" compose

# Compose two mappings when at least one is a composition or when none is a
# composition.
compose(A::Composition, B::Composition) = _compose(A, B)
compose(A::Composition, B::Mapping    ) = _compose(A, B)
compose(A::Mapping,     B::Composition) = _compose(A, B)
compose(A::Mapping,     B::Mapping    ) = Composition(A, B)

_compose(A::Mapping, B::Mapping) = begin
    C = compose!(as_vector(*, A), B)
    n = length(C)
    return (n == 0 ? Id :
            n == 1 ? C[1] :
            Composition(to_tuple(C)))
end

"""
    compose!(A, B) -> A

overwrites `A` with a simplified composition of a left operand `A` and a right
operand `B`.  The left operand is a composition of (zero or more) mappings
whose terms are stored in the vector of mappings `A` (if `A` is empty, the left
operand is assumed to be the identity).  On return, the vector `A` is modified
to store the terms of the simplified composition of `A` and `B`.  The left
operand `B` may be itself a composition (as an instance of
`LazyAlgebra.Composition` or as a vector of mappings) or any other kind of
mapping.

""" compose!

function compose!(A::Vector{Mapping}, B::Composition{N}) where {N}
    @inbounds for i in 1:N
        # Build the simplified composition A*B[i].
        compose!(A, B[i])
        if identical(last(A), B[i])
            # The last term of the simplified composition A*B[i] is still B[i],
            # which indicates that composing A with B[i] did not yield any
            # simplifications.  It is sufficient to append all the other terms
            # of B to A as no further simplifications are expected.
            return append_terms!(A, B, (i+1):N)
        end
    end
    return A
end

function compose!(A::Vector{Mapping}, B::Mapping)
    # Compute the simplified composition of the last term of A with B.  The
    # result is either a simple mapping or a simplified composition.
    m = length(A); @certify m > 0
    C = A[m]*B

    # Replace the last term of A with C.
    if C isa Composition && identical(C[1], A[m])
        # Nothing has changed at the tail of the composition A.  No further
        # simplifications are expected.  Push all terms of C to A, but the
        # first term of C which is identical to the last term of A.
        append_terms!(A, C, 2:length(C))
    elseif m > 1
        # Drop the last term of A and compose the remaining terms with C.  This
        # may trigger further simplifications
        compose!(resize!(A, m - 1), C)
    elseif C isa Composition
        # Replace the only term of A by all the terms of the composition C.
        # This is the same as above but avoids calling `resize!` as a small
        # optimization.
        A[1] = C[1]
        append_terms!(A, C, 2:length(C))
    else
        # Replace the only term of A by the simple mapping C.
        A[1] = C
    end
    return A
end

#------------------------------------------------------------------------------
# UTILITIES FOR BUILDING SUMS AND COMPOSITIONS

"""
    as_vector(op, A)

yields a vector of mappings with the terms of the mapping `A`.  Argument `op`
is `+` or `*`.  If `op` is `+` (resp. `*`) and `A` is a sum (resp. a
composition) of mappings, the terms of `A` are extracted in the returned
vector; otherwise, the returned vector has just one element which is `A`.

"""
function as_vector(::Union{typeof(+),typeof(*)}, A::Mapping)
    V = Vector{Mapping}(undef, 1)
    V[1] = A
    return V
end

as_vector(::typeof(+), A::Sum        ) = collect_terms(A)
as_vector(::typeof(*), A::Composition) = collect_terms(A)

"""
    collect_terms(A)

collects the terms of the mapping `A` into a vector.  This is similar to
`collect(A)` except that the element type of the result is forced to be
`Mapping`.

"""
function collect_terms(A::Union{Sum{N},Composition{N}}) where {N}
    V = Vector{Mapping}(undef, N)
    @inbounds for i in 1:N
        V[i] = A[i]
    end
    return V
end

"""
    append_terms!(A, B, I=1:length(B)) -> A

pushes all terms `B[i]` for all `i ∈ I` to `A` and returns `A`.

"""
function append_terms!(A::Vector{Mapping},
                       B::Union{Vector{Mapping},Composition},
                       I::AbstractUnitRange{<:Integer} = Base.OneTo(length(B)))

    imin, imax = Int(first(I)), Int(last(I))
    (1 ≤ imin && imax ≤ length(B)) ||
        bad_argument("out of bounds indices in given range")
    if imin ≤ imax
        m = length(A)
        n = imax - imin + 1
        resize!(A, m + n)
        k = m + 1 - imin
        @inbounds for i in I
            A[k+i] = B[i]
        end
    end
    return A
end
