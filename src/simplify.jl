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
    multiplier,
    unveil,
    terms,
    to_tuple,
    unscaled

#------------------------------------------------------------------------------
# RULES FOR SUMS OF MAPPINGS

"""
    Simplify.add(A, B)

yields a simplied sum of mappings `A + B`.

"""
add(A::Mapping, B::Mapping) = simplify_sum(A, B)

function simplify_sum(A::Sum, B::Mapping)
    V = build_sum!(as_vector(A), B)
    length(V) == 1 ? V[1] : Sum(to_tuple(V))
end

function simplify_sum(A::Mapping, B::Sum)
    V = build_sum!(as_vector(A), B)
    length(V) == 1 ? V[1] : Sum(to_tuple(V))
end

function simplify_sum(A::Mapping, B::Mapping)
    if unscaled(A) === unscaled(B)
        return (multiplier(A) + multiplier(B))*unscaled(A)
    elseif identifier(A) ≤ identifier(B)
        return Sum((A, B))
    else
        return Sum((B, A))
    end
end

function as_vector(A::Mapping)
    V = Vector{Mapping}(undef, 1)
    V[1] = A
    return V
end

function as_vector(A::Sum{N}) where {N}
    V = Vector{Mapping}(undef, N)
    @inbounds for i in 1:N
        V[i] = A[i]
    end
    return V
end

# Add the terms of a sum one-by-one.
function build_sum!(A::Vector{Mapping}, B::Sum{N}) where {N}
    @inbounds for i in 1:N
        build_sum!(A, B[i])
    end
    return A
end

function build_sum!(A::Vector{Mapping}, B::Mapping)
    # Nothing to do if B is zero times anything.
    multiplier(B) == 0 && return A

    # If exact match found, update A in-place and return.
    n = length(A)
    @inbounds for i in 1:n
        if unscaled(A[i]) === unscaled(B)
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
