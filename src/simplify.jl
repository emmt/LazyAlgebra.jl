# Implement simplification rules for sums and compositions of LazyAlgebra operators.
#
# Contrarily to automatic rules applied at construction time, the result returned by
# `simplify` and `try_simplify` may not be type-stable. One of the difficulty is
# to avoid re-trying to simplify (sub-)expressions that have been already simplified and
# yet not forget to apply all implemented simplifications.

is_nothing(::Nothing) = true
is_nothing(::Any) = false
is_something(x) = !is_nothing(x)

order_in_sum(A::Operator) = hash(A)
order_in_sum(A::Prod{<:Number}) = hash(A[2])

# Structure to protect a sum from further simplifications.
struct Marked{T<:Sum} <: Operator
    parent::T
    # Inner constructor to forbid specifying the type parameter and restrict possible
    # parent.
    Marked(A::T) where {T<:Sum} = new{T}(A)
end
Base.parent(A::Marked) = A.parent
for cmp in (:(==), :isequal)
    @eval begin
        Base.$cmp(A::Marked, B::Marked) = $cmp(parent(A), parent(B))
    end
end

# Mark sum term(s) in an operator leaving other terms unchanged so that a marked scaled
# operator remains a scaled operator and a marked composition of operators remains a
# composition of operators.
mark(A::Sum) = Marked(A)
mark(A::Prod{<:Number}) = A[1]*mark(A)
mark(A::Prod{<:Operator}) = mark(A[1])*mark(A[2])
mark(A::Operator) = A

unmark(A::Marked) = parent(A)
unmark(A::Prod{<:Number}) = A[1]*unmark(A)
unmark(A::Prod{<:Operator}) = unmark(A[1])*unmark(A[2])
unmark(A::Operator) = A

function unmark!(A::AbstractVector{Operator})
    for i in eachindex(A)
        A[i] = unmark(A[i])
    end
    return A
end

"""
    LazyAlgebra.simplify(A::Operator) -> B::Operator

yields an operator `B` which is a simplified version of operator `A` and such that `A*x ≈
B*x` holds for any acceptable argument `x` (the `≈` accounts for possible rounding
errors). If no simplifications are possible, `A` itself may be returned.

The method [`LazyAlgebra.try_simplify`](@ref) shall be extended to implement the
simplification rules applied by `LazyAlgebra.simplify`.

"""
simplify(A::Operator) = something(try_simplify(A), A)

# Simplify a composition of operators.
function simplify(A::Prod{<:Operator})
    # Fist try to simplify the whole composition. If this fails, apply a hierarchical
    # strategy.
    B = try_simplify(A)
    is_something(B) ? B : simplify_prod(flatten_prod!(𝟙, Operator[], A)..., false)
end

flatten_prod!(λ::Number, A::AbstractVector{Operator}, B::Prod{<:Number}) =
    flatten_prod!(λ*B[1], A, B[2])
flatten_prod!(λ::Number, A::AbstractVector{Operator}, B::Prod{<:Operator}) =
    flatten_prod!(flatten_prod!(λ, A, B[1])..., B[2])
flatten_prod!(λ::Number, A::AbstractVector{Operator}, B::Operator) =
    λ, push!(A, B)
flatten_prod!(λ::Number, A::AbstractVector{Operator}, B::Sum) =
    # Since any sum remains a single term in a product of operator, it is convenient to
    # simplify the sum before pushing it to the list of terms. However, any sum in the
    # simplified sum must be marked to avoid repeated attempts to simplify it when the
    # product itself is simplified. This also avoids infinite recursion of
    # `flatten_prod!`.
    flatten_prod!(λ, A, mark(simplify(B)))

# Simplify a flattened product of operators by trying to simplify all possible
# sub-expressions of decreasing lengths.
function simplify_prod(λ::Number, A::AbstractVector{Operator}, whole::Bool)
    n = length(A) # length of sub-expressions to consider
    if !whole
        n -= 1
    end
    while n ≥ 1
        # If any simplification of a sub-expression of length `n` is possible, substitute
        # the sub-expression by its simplified version and repeat the process from the
        # beginning. Sums in the simplified expression, if any, are marked to not simplify
        # them again.
        for i in firstindex(A):(lastindex(A) - n + 1)
            B = try_simplify(foldr(Prod, view(A, i:i+n-1)))
            is_something(B) && return simplify_prod(
                λ, view(A, firstindex(A):i-1), mark(B), view(A, i+n:lastindex(A)))
        end
        n -= 1
    end
    # The product cannot be further simplified, rebuild a product with the marks removed
    # and return this product times the multiplier if not equal to 1.
    B = foldr(Prod, unmark!(A))
    return isone(λ) ? B : λ*B
end

function simplify_prod(λ::Number, A::AbstractVector{Operator}, B::Operator,
                       C::AbstractVector{Operator})
    A′ = Operator[]
    for Aᵢ in A
        push!(A′, Aᵢ)
    end
    λ′, _ = flatten_prod!(λ, A′, B)
    for Cᵢ in C
        push!(A′, Cᵢ)
    end
    return simplify_prod(λ′, A′, true)
end

# Simplify a sum of 2 terms. This is an optimized version for speed-up.
simplify(A::Sum{<:Operator,<:Operator}) = simplify_sum(simplify(A[1]), simplify(A[2]))

# To simplify the sum of 2 terms (that have been separately simplified), first try a more
# simple expression, otherwise return the sum of the sorted terms.
function simplify_sum(A::Operator, B::Operator) # operands assumed to have been simplified
    C = try_simplify(A + B)
    if is_something(C)
        return C
    elseif order_in_sum(B) < order_in_sum(A)
        return B + A
    else
        return A + B
    end
end

# Simplify a sum of any number of terms. For speed-up, a sum of 2 terms is simplified by
# another specialized method.
simplify(A::Sum{<:Operator,<:Sum}) = simplify_sum!(flatten_sum!(Operator[], A))

flatten_sum!(A::AbstractVector{Operator}, B::Sum) =
    flatten_sum!(flatten_sum!(A, B[1]), B[2])
flatten_sum!(A::AbstractVector{Operator}, (λ,B)::Prod{<:Number,<:Sum}) =
    # Distribute multiplication by a scalar over the terms of a sum.
    isone(λ) ? flatten_sum!(A, B) : flatten_sum!(flatten_sum!(A, λ*B[1]), λ*B[2])
flatten_sum!(A::AbstractVector{Operator}, B::Operator) =
    flatten_sum!(Stage(1), A, simplify(B))
function flatten_sum!(::Stage{1}, A::AbstractVector{Operator}, B::Operator)
    # This version is called when `B` is not a sum and has been simplified. First, Attempt
    # to combine `B` with any preceding terms of the sum; if this fails, `B` is appended
    # to the list of terms.
    for i in eachindex(A)
        C = try_simplify(A[i] + B)
        if is_something(C)
            A[i] = C
            return A
        end
    end
    return push!(A, B)
end

# Given a sum flattened by `flatten_sum!`, simplify the sum of terms.
function simplify_sum!(A::AbstractVector{Operator})
    # Eliminate zeros.
    rng = firstindex(A):lastindex(A)
    j = first(rng) # writing index
    for i in rng
        if !iszero(A[i])
            if j < i
                A[j] = A[i]
            end
            j += 1
        end
    end
    n = max(1, j - first(rng)) # number of remaining terms

    # If only one term remains, return this term as it represents a simplification of the
    # sum which initially had at least two terms.
    n == 1 && return first(A)

    # Restrict the list to the non-zero terms.
    if n < length(A)
        resize!(A, n)
    end

    # Rebuild simplified sum according to right-associativity and sorting terms according
    # to their hash-value.
    I = sortperm(map(order_in_sum, A))
    if I == 1:n
        return foldr(Sum, A)
    else
        return foldr(Sum, A[I])
    end
end

"""
    LazyAlgebra.try_simplify(A::Operator) -> Union{Operator,Nothing}

attempts to simplify the operator `A` and yields the resulting operator if a
simplification is possible and `nothing` otherwise.

This method is the cornerstone of the higher level method [`LazyAlgebra.simplify`](@ref)
and `LazyAlgebra.try_simplify(A)` shall be extended to perform specific simplifications
based on the type of `A`. It is not expected that the value returned by
`LazyAlgebra.try_simplify` be inferable but, for the simplification rules to combine
correctly (in particular to avoid infinite recursions), this value must not be `A` or an
equivalent construction. For example, if `A = B + C`, the result shall be neither `B + C`
nor `C + B` which are respectively represented by `LazyAlgebra.Sum(B, C)` and
`LazyAlgebra.Sum(C, B)`.

"""
try_simplify(A::Operator) = nothing

# Never simplify a marked expression.
try_simplify(A::Marked) = nothing

# When trying to simplify a sum, the work is divided in stages where `try_simplify` is
# called to simplify the sum of 2 terms which have been separately simplified. Calling
# `try_simplify(A+B)` and not a more specialized method, say `try_simplify_sum(A,B)`, is
# to let other simplifications of sums of specific operators or combination of operators
# to be implemented. Hence, the only simplification considered for a sum of 2 terms is to
# replace it by a scaled operator if the two operands are equal up to a multiplier.
function try_simplify((A,B)::Sum) # operands assumed to have been simplified
    C = unscaled(A)
    if isequal(unscaled(B), C)
        λ = multiplier(A) + multiplier(B)
        return isone(λ) ? C : λ*C
    else
        return nothing
    end
end

# Try to simplify a scaled operator. It is assumed that the whole expression cannot be
# simplified by another more specific rule (otherwise this method would not have been
# called).
function try_simplify(A::Prod{<:Number})
    # Try to simplify the right-hand side and to eliminate the multiplier.
    λ = multiplier(A)
    B = try_simplify(unscaled(A))
    if is_something(B)
        return isone(λ) ? B : λ*B
    else
        return isone(λ) ? unscaled(A) : nothing
    end
end

# When simplifying a product of an operator and its inverse, return shaped identity if
# possible.
try_simplify((A,B)::Prod{<:Inverse,<:Inverse}) = nothing
try_simplify((A,B)::Prod{<:Operator,<:Inverse}) = try_simplify_ratio(A, parent(B))
try_simplify((A,B)::Prod{<:Inverse,<:Operator}) = try_simplify_ratio(parent(A), B)
try_simplify_ratio(A::Operator, B::Operator) =
    !isequal(A, B) ? nothing :
    InputShape(A) isa HasInputShape ? Identity(input_shape(A)) :
    OutputShape(A) isa HasOutputShape ? Identity(output_shape(A)) : Id

# For the adjoint (resp. inverse) of an operator, first attempt to simplify the parent
# operator and, if this succeeds, return the simplification of the adjoint (resp. inverse)
# of the simplified parent; otherwise, return nothing.
try_simplify(A::Adjoint) =
    (B = try_simplify(A')) isa Nothing ? nothing : simplify(B')

try_simplify(A::Inverse) =
    (B = try_simplify(inv(A))) isa Nothing ? nothing : simplify(inv(B))

try_simplify(A::InverseAdjoint) =
    !((B = try_simplify(A')) isa Nothing) ? simplify(B') :
    !((B = try_simplify(inv(A))) isa Nothing) ? simplify(inv(B)) : nothing

is_complex(::Type{T}) where {T<:Number} = is_complex(bare_type(T))
is_complex(::Type{<:Complex}) = true
is_complex(::Type{<:Any}) = false

const DiagonalOperator = Union{Diag,Adjoint{<:Diag},Inverse{<:Diag},InverseAdjoint{<:Diag}}

# Simplification rules for diagonal operators.
function try_simplify((A,B)::Prod{<:DiagonalOperator,<:DiagonalOperator})
    input_axes(A) == input_axes(B) || return nothing
    if false
        a = diag(A)
        b = diag(B)
        c = similar(a, prod_type(eltype(a), eltype(b)))
        @. c = a*b
        return Diag(c)
    else
        return Diag(map(*, diag(A), diag(B)))
    end
end

function try_simplify((A,B)::Sum{<:DiagonalOperator,<:DiagonalOperator})
    input_axes(A) == input_axes(B) || return nothing
    if false
        a = diag(A)
        b = diag(B)
        c = similar(a, sum_type(eltype(a), eltype(b)))
        @. c = a + b
        return Diag(c)
    else
        return Diag(map(+, diag(A), diag(B)))
    end
end

function try_simplify((λ,A)::Prod{<:Number,<:DiagonalOperator})
    isone(λ) && return simplify(A)
    f = Base.Fix1(*, convert_multiplier(λ, eltype(A)))
    return Diag(map(f, diag(A)))
end

function try_simplify(A::DiagonalOperator)
    A isa Diag && return nothing
    A isa Adjoint{<:Diag} && !(bare_type(eltype(A)) <: Complex) && return A'
    return Diag(copy(diag(A)))
end

try_simplify((A,B)::Prod{Identity,Identity}) =
    A isa UniversalIdentity ? B :
    B isa UniversalIdentity ? A :
    input_axes(A) != output_axes(B) ? nothing :
    B isa Identity{<:Dims} ? B : A
