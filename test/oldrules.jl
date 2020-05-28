#
# oldrules.jl --
#
# Implement simplification rules for LazyAlgebra.  This is the old version,
# kept for comparisons and benchmarking.  The new version is simpler and
# faster.
#
module OldRules

using LazyAlgebra

using LazyAlgebra:
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
    multiplier,
    unveil,
    terms,
    to_tuple

#------------------------------------------------------------------------------
# RULES FOR SUMS OF MAPPINGS

"""
    OldRules.add(A, B)

yields a simplied sum of mappings `A + B`.

"""
add(A::Mapping, B::Mapping) =
    simplify_sum((split_sum(A)..., split_sum(B)...))

# `split_sum(A)` yields a tuple of the terms of `A` if it is a sum or just
# `(A,)` otherwise.
split_sum(A::Sum) = terms(A)
split_sum(A::Mapping) = (A,)

# `merge_sum(args...)` constructs a fully qualified sum.  It is assumed that
# the argument(s) have already been simplified.  An empty sum is forbidden
# because there is no universal neutral element ("zero") for the addition.
merge_sum(arg::Mapping) = arg
merge_sum(args::Mapping...) = merge_sum(args)
merge_sum(args::Tuple{}) = throw(ArgumentError("empty sum"))
merge_sum(args::Tuple{Mapping}) = args[1]
merge_sum(args::T) where {N,T<:NTuple{N,Mapping}} = Sum{N,T}(args)

# `simplify_sum(args...)` simplifies the sum of all terms in `args...` and
# returns a single term (possibly an instance of `Sum`).  It is assumed that
# the operator `+` is associative and commutative.

# Make a sum out of 0-1 terms.
simplify_sum(args::Tuple{}) = merge_sum(args)
simplify_sum(args::Tuple{Mapping}) = args[1]

# Make a sum out of N terms (with N ≥ 2).
function simplify_sum(args::NTuple{N,Mapping}) where {N}
    # First group terms corresponding to identical mappings (up to an optional
    # multiplier) to produce a single (possibly scaled) term per group.  FIXME:
    # The following algorithm scales as O(N²) which is probably not optimal.
    # Nevertheless, it never compares twice the same pair of arguments.
    terms = Array{Mapping}(undef, 0)
    flags = fill!(Array{Bool}(undef, N), true)
    i = 1
    while i != 0
        # Push next ungrouped argument.
        push!(terms, args[i])
        flags[i] = false

        # Find any other argument which is identical, possibly scaled, term.
        k = i + 1
        i = 0
        for j = k:N
            if flags[j]
                if simplify_sum!(terms, terms[end], args[j])
                    flags[j] = false
                elseif i == 0
                    # Next ungrouped argument to consider.
                    i = j
                end
            end
        end
    end

    # Make a sum out of the terms after having eliminated the zeros.
    return sort_sum(terms)
end

# `simplify_sum!(terms, A, B)` is a helper function for the `simplify_sum`
# method which attempts to make a trivial simplification for `A + B` when `A =
# λ⋅M` and `B = μ⋅M` for any numbers `λ` and `μ` and any mapping `M`.  If such
# a simplification can be done, the result `(λ + μ)⋅M` is stored as the last
# component of `terms` and `true` is returned; otherwise, `false` is returned.
simplify_sum!(terms::Vector{Mapping}, A::Mapping, B::Mapping) = false

function simplify_sum!(terms::Vector{Mapping},
                       A::Scaled{T}, B::Scaled{T}) where {T<:Mapping}
    unscaled(A) === unscaled(B) || return false
    @inbounds terms[end] = (multiplier(A) + multiplier(B))*unscaled(A)
    return true
end

function simplify_sum!(terms::Vector{Mapping},
                       A::Scaled{T}, B::T) where {T<:Mapping}
    unscaled(A) === B || return false
    @inbounds terms[end] = (multiplier(A) + one(multiplier(A)))*B
    return true
end

function simplify_sum!(terms::Vector{Mapping},
                       A::T, B::Scaled{T}) where {T<:Mapping}
    A === unscaled(B) || return false
    @inbounds terms[end] = (one(multiplier(B)) + multiplier(B))*A
    return true
end

function simplify_sum!(terms::Vector{Mapping},
                       A::T, B::T) where {T<:Mapping}
    A === B || return false
    @inbounds terms[end] = 2*A
    return true
end

# Sort the terms of a sum (so that all equivalent expressions eventually yield
# the same result after simplifications) and eliminate the "zeros"
# (if all terms are "zero", the sum simplifies to the first one).
sort_sum(args::Mapping...) = sort_sum(args)
sort_sum(args::Tuple{Mapping}) = args[1]
function sort_sum(terms::Union{Tuple{Vararg{Mapping}},Vector{<:Mapping}})
    perms = sortperm([identifier(terms[i]) for i in 1:length(terms)])
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
        return merge_sum(ntuple(i -> terms[perms[i]], n))
    end
end

#------------------------------------------------------------------------------
# RULES FOR COMPOSITIONS OF MAPPINGS

"""
    OldRules.compose(A, B)

yields a simplified composition of mappings `A*B`.

"""
compose(A::Mapping, B::Mapping) = simplify_mul(A, B)

# `split_mul(A)` yields a tuple of the terms of `A` if it is a composition or
# just `(A,)` otherwise.
split_mul(A::Composition) = terms(A)
split_mul(A::Mapping) = (A,)

# `merge_mul(args...)` constructs a fully qualified composition.  It is assumed
# that the argument(s) have already been simplified.  An empty composition
# yields the identity which is the universal neutral element ("one") for the
# composition.
merge_mul(arg::Mapping) = arg
merge_mul(args::Mapping...) = merge_mul(args)
merge_mul(args::Tuple{}) = Id
merge_mul(args::Tuple{Mapping}) = args[1]
merge_mul(args::T) where {N,T<:NTuple{N,Mapping}} = Composition{N,T}(args)

# `simplify_mul(A,B)` simplifies the product of `A` and `B`.  The result is a
# tuple `C` of the resulting terms if `A` and `B` are both tuples of mappings
# or an instance of `Mapping` if `A` and `B` are both mappings.  If no
# simplification can be made, the result is just the concatenation of the terms
# in `A` and `B`.  To perform simplifications, it is assumed that, if they are
# compositions, `A` and `B` have already been simplified.  It is also assumed
# that the composition is associative but non-commutative.
simplify_mul(A::Mapping, B::Mapping) =
    merge_mul(simplify_mul(split_mul(A), split_mul(B)))

# The following versions of `simplify_mul` are for `A` and `B` in the form of
# tuples and return a tuple.  The algorithm is recursive and should works for
# any non-commutative binary operator.
simplify_mul(A::Tuple{}, B::Tuple{}) = (Id,)
simplify_mul(A::Tuple{Vararg{Mapping}}, B::Tuple{}) = A
simplify_mul(A::Tuple{}, B::Tuple{Vararg{Mapping}}) = B
function simplify_mul(A::NTuple{M,Mapping},
                       B::NTuple{N,Mapping}) where {M,N}
    # Here M ≥ 1 and N ≥ 1.
    @assert M ≥ 1 && N ≥ 1

    # Attempt to simplify the product of the terms at the junction.
    C = split_mul(A[M]*B[1])
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
            L = simplify_mul(A[1:M-1], C[1]) # simplify leftmost operands
            R = simplify_mul(C[2], B[2:N])   # simplify rightmost operands
            if L[end] !== C[1] || R[1] !== C[2]
                # At least one of the last of resulting rightmost operands or
                # the first of the resulting leftmost operands has been
                # modified so there may be other possible simplifications.
                return simplify_mul(L, R)
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
        #     simplify_mul(A[1:end-1], simplify_mul(C, B[2:end]))
        #
        # that is, simplify right then left, or:
        #
        #     simplify_mul(simplify_mul(A[1:end-1], C), B[2:end])
        #
        # that is simplify left then right.  Since we want to propagate
        # multipliers to the right of compositions, the former is the most
        # appropriate.
        return simplify_mul(A[1:end-1], simplify_mul(C, B[2:end]))
    else
        # len == 0 The result of A[M]*B[1] is the neutral element for * thus
        # eliminating these terms from the merging.  We just have to repeat the
        # process with the truncated associations in case further simplications
        # are possible.
        return simplify_mul(A[1:M-1], B[2:N-1])
    end
end

end # module
