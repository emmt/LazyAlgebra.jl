#
# simplify.jl -
#
# Implement automatic simplifications for sums and compositions of mappings.
#
#-------------------------------------------------------------------------------
#
# This file is part of LazyAlgebra (https://github.com/emmt/LazyAlgebra.jl)
# released under the MIT "Expat" license.
#
# Copyright (c) 2017-2020 Éric Thiébaut.
#

module Simplify

using ..LazyAlgebra

using ..LazyAlgebra:
    Adjoint,
    Composition,
    Composition,
    Direct,
    Gram,
    Inverse,
    InverseAdjoint,
    LinearMapping,
    Mapping,
    Scaled,
    Sum,
    identifier,
    is_same_mapping,
    multiplier,
    terms,
    to_tuple,
    unscaled,
    unveil

#------------------------------------------------------------------------------
# RULES FOR SUMS OF MAPPINGS

"""
    Simplify.add(A, B)

yields a simplied sum of mappings `A + B`.

""" add

# Simplify the sum of two mappings none of which being a sum.
function add(A::Mapping, B::Mapping)
    if is_same_mapping(unscaled(A), unscaled(B))
        return (multiplier(A) + multiplier(B))*unscaled(A)
    elseif identifier(A) ≤ identifier(B)
        return Sum(A, B)
    else
        return Sum(B, A)
    end
end

# Simplify the sum of two mappings at least one of which being a sum.
add(A::Sum,     B::Mapping) = _add(A, B)
add(A::Mapping, B::Sum    ) = _add(A, B)
add(A::Sum,     B::Sum    ) = _add(A, B)
function _add(A::Mapping, B::Mapping)
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
        if is_same_mapping(unscaled(A[i]), unscaled(B))
            λ = multiplier(A[i]) + multiplier(B)
            if λ == 1
                A[i] = unscaled(B)
            elseif λ != 0
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
# RULES FOR COMPOSITIONS OF MAPPINGS

"""
    Simplify.compose(A, B)

yields a simplified composition of mappings `A*B` when at least one of `A` or
`B` are a composition.  Just yields the composition `A` by `B` without
simplifications if none are a composition.

The ability to perform simplifications relies on implemented specializations of
`A*B` when neither `A` nor `B` are compositions.  It is also assumed that `A`
and `B` have already been simplified if they are compositions.

"""
compose(A::Mapping,     B::Mapping    ) = Composition(A, B)
compose(A::Composition, B::Mapping    ) = _compose(A, B)
compose(A::Mapping,     B::Composition) = _compose(A, B)
compose(A::Composition, B::Composition) = _compose(A, B)

function _compose(A::Mapping, B::Mapping)
    C = compose!(as_vector(*, A), B)
    n = length(C)
    return (n == 0 ? Id :
            n == 1 ? C[1] :
            Composition(to_tuple(C)))
end

"""
    compose!(A, B) -> A

overwrites `A` with a simplified composition of a left operand `A` and a right
operand `B`.  The left operand is a composition of (0, 1, or more) mappings
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
        if is_same_mapping(last(A), B[i])
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
    m = length(A); @assert m > 0
    C = A[m]*B

    # Replace the last term of A with C.
    if C isa Composition && is_same_mapping(C[1], A[m])
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
# UTILITIES

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

end # module
