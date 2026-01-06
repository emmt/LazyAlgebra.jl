# Implement simplification rules for sums and compositions of LazyAlgebra operators.
#
# Contrarily to automatic rules applied at construction time, the result returned by
# `simplify` and `try_simplify` may not be type-stable. One of the difficulty is to avoid
# re-trying to simplify (sub-)expressions that have been already simplified and yet not
# forget to apply all implemented simplifications.

is_nothing(::Nothing) = true
is_nothing(::Any) = false
is_something(x) = !is_nothing(x)

order_in_sum(A::Operator) = hash(A)
order_in_sum(A::Scaled) = hash(A[2])

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
mark(A::Scaled) = A[1]*mark(A[2])
mark(A::Prod) = mark(A[1])*mark(A[2])
mark(A::Operator) = A

# Revert the effects of `mark`.
unmark(A::Marked) = parent(A)
unmark(A::Scaled) = A[1]*unmark(A)
unmark(A::Prod) = unmark(A[1])*unmark(A[2])
unmark(A::Operator) = A

function unmark!(A::AbstractVector{Operator})
    @inbounds for i in eachindex(A)
        A[i] = unmark(A[i])
    end
    return A
end

"""
    LazyAlgebra.simplify(A::Operator) -> B::Operator

Return an operator `B` which is a simplified version of operator `A` and such that `A*x ≈
B*x` holds for any acceptable argument `x` (the `≈` accounts for possible rounding errors).
If no simplifications are possible, `A` itself may be returned.

The method [`LazyAlgebra.try_simplify`](@ref) shall be extended to implement the
simplification rules applied by `LazyAlgebra.simplify`.

"""
simplify(A::Operator) = something(try_simplify(A), A)

# Simplify a composition of operators.
function simplify(A::Prod)
    # Fist try to simplify the whole composition. If this fails, attempt to simplify
    # sub-expressions of decreasing lengths.
    B = try_simplify(A)
    if is_something(B)
        return B
    else
        return simplify_prod(flatten_prod!(𝟙, Operator[], A)..., false)
    end
end

flatten_prod!(λ::Number, A::AbstractVector{Operator}, B::Scaled) =
    flatten_prod!(λ*B[1], A, B[2])
flatten_prod!(λ::Number, A::AbstractVector{Operator}, B::Prod) =
    flatten_prod!(flatten_prod!(λ, A, B[1])..., B[2])
flatten_prod!(λ::Number, A::AbstractVector{Operator}, B::Operator) =
    λ, push!(A, B)
flatten_prod!(λ::Number, A::AbstractVector{Operator}, B::Sum) =
    # Since any sum remains a single term in a product of operator, it is convenient to
    # simplify the sum before pushing it to the list of terms. However, any sum in the
    # simplified sum must be marked to avoid repeated attempts to simplify it when the
    # product itself is simplified. This also avoids infinite recursion of `flatten_prod!`.
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
            if is_something(B)
                return simplify_prod(λ, view(A, firstindex(A):i-1),
                                     mark(B), view(A, i+n:lastindex(A)))
            end
        end
        n -= 1
    end
    # The product cannot be further simplified, rebuild a product with the marks removed and
    # return this product times the multiplier if not equal to 1.
    B = foldr(Prod, unmark!(A))
    return isone(λ) ? B : λ*B
end

function simplify_prod(λ::Number, A::AbstractVector{Operator}, B::Operator,
                       C::AbstractVector{Operator})
    # Accumulate all operators of the composition `λ*prod(A)*B*prod(C)` in `R` and the
    # product of multipliers in `μ`.
    R = Operator[]
    for Aᵢ in A
        push!(R, Aᵢ)
    end
    μ, _ = flatten_prod!(λ, R, B)
    for Cᵢ in C
        push!(R, Cᵢ)
    end
    return simplify_prod(μ, R, true)
end

# Simplify a sum of any number of terms.
simplify(A::Sum) = simplify_sum!(flatten_sum!(Operator[], A))

flatten_sum!(A::AbstractVector{Operator}, B::Sum) =
    flatten_sum!(flatten_sum!(A, B[1]), B[2])
flatten_sum!(A::AbstractVector{Operator}, (λ,B)::Scaled{<:Number,<:Sum}) =
    # Distribute multiplication by a scalar over the terms of a sum.
    isone(λ) ? flatten_sum!(A, B) : flatten_sum!(flatten_sum!(A, λ*B[1]), λ*B[2])
flatten_sum!(A::AbstractVector{Operator}, B::Operator) =
    flatten_sum!(Val(1), A, simplify(B))
function flatten_sum!(::Val{1}, A::AbstractVector{Operator}, B::Operator)
    # This version is called when `B` is not a sum and has been simplified. First, attempt
    # to combine `B` with any preceding terms of the sum; if this fails, `B` is appended to
    # the list of terms.
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
    n = j - first(rng) # number of remaining terms

    # Return a sum of the remaining terms sorted according to their hash-value. For a small
    # number of remaining terms, bypass sorting to speed-up the process.
    if n ≤ 1
        # Having less than 1 remaining terms means that the sum simplifies to zero, it is
        # still valid to return the first term.
        return first(A)
    elseif n ≤ 3
        i = firstindex(A)
        if n == 2
            return sorted_sum(A[i], A[i+1])
        else
            return sorted_sum(A[i], A[i+1], A[i+2])
        end
    else
        if n < length(A)
            # Restrict the list to the non-zero terms.
            resize!(A, n)
        end
        I = sortperm(map(order_in_sum, A))
        if I == 1:n
            return foldr(Sum, A)
        else
            return foldr(Sum, A[I])
        end
    end
end

sorted_sum(A::Operator, B::Operator) =
    order_in_sum(B) < order_in_sum(A) ? B + A : A + B

function sorted_sum(A::Operator, B::Operator, C::Operator)
    A_order = order_in_sum(A)
    B_order = order_in_sum(B)
    C_order = order_in_sum(C)
    if B_order < A_order
        if C_order < B_order
            return C + B + A
        elseif C_order < A_order
           return B + C + A
        else
           return B + A + C
        end
    else
        if C_order < A_order
            return C + A + B
        elseif C_order < B_order
           return A + C + B
        else
           return A + B + C
        end
    end
end

"""
    LazyAlgebra.try_simplify(A::Operator) -> Union{Operator,Nothing}

Attempt to simplify the operator `A` and yields the resulting operator if a simplification
is possible and `nothing` otherwise.

This method is the cornerstone of the higher level method [`LazyAlgebra.simplify`](@ref) and
`LazyAlgebra.try_simplify(A)` shall be extended to perform specific simplifications based on
the type of `A`. It is not expected that the value returned by `LazyAlgebra.try_simplify` be
inferable but, for the simplification rules to combine correctly (in particular to avoid
infinite recursions), this value must not be `A` or an equivalent construction. For example,
if `A = B + C`, the result shall be neither `B + C` nor `C + B` which are respectively
represented by `LazyAlgebra.Sum(B, C)` and `LazyAlgebra.Sum(C, B)`.

"""
try_simplify(A::Operator) = nothing

# Never simplify a marked expression.
try_simplify(A::Marked) = nothing

# When trying to simplify a sum, the work is divided in stages where `try_simplify` is
# called to simplify the sum of 2 terms which have been separately simplified. Calling
# `try_simplify(A+B)` and not a more specialized method, say `try_simplify_sum(A,B)`, is to
# let other simplifications of sums of specific operators or combination of operators to be
# implemented. Hence, the only simplification considered for a sum of 2 terms is to replace
# it by a scaled operator if the two operands are equal up to a multiplier.
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
function try_simplify((λ, A)::Scaled)
    # Try to simplify the right-hand side and to eliminate the multiplier.
    B = try_simplify(A)
    if is_something(B) # FIXME also check for 0 and -1?
        return isone(λ) ? B : λ*B
    else
        return isone(λ) ? A : nothing
    end
end

# When simplifying a product of an operator and its inverse, return shaped identity if
# possible.
try_simplify(A::Prod{<:Inverse,<:Inverse}) = nothing
try_simplify(A::Prod{<:Operator,<:Inverse}) = isequal(A[1], A[2][]) ? Id : nothing
try_simplify(A::Prod{<:Inverse,<:Operator}) = isequal(A[1][], A[2]) ? Id : nothing

# For the adjoint (resp. transpose or inverse) of an operator, first attempt to simplify the
# parent operator and, if this succeeds, return the simplification of the adjoint (resp.
# transpose or inverse) of the simplified parent; otherwise, return nothing.
for (f, T) in (:adjoint   => :Adjoint,
               :transpose => :Transpose,
               :conj      => :Conjugate,
               :inv       => :Inverse)
    @eval begin
        try_simplify(A::$T) =
            (B = try_simplify($f(A))) isa Nothing ? nothing : simplify($f(B))
    end
    T !== :Inverse && @eval begin
        try_simplify(A::$(Symbol("Inverse",T))) =
            !((B = try_simplify($f(A))) isa Nothing) ? simplify($f(B)) :
            !((B = try_simplify(inv(A))) isa Nothing) ? simplify(inv(B)) : nothing
    end
end

# Simplification rules for diagonal operators. In products, the identity has been
# automatically suppressed at construction time, so only sums of diagonal operators and
# (scaled) identity have to be considered.
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

function try_simplify((λ,A)::Scaled{<:Number,<:DiagonalOperator})
    isone(λ) && return simplify(A)
    f = Base.Fix1(*, convert_multiplier(λ, eltype(A)))
    return Diag(map(f, diag(A)))
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

function try_simplify((A,B)::Sum{<:MaybeScaled{<:DiagonalOperator},<:MaybeScaled{<:Identity}})
    if B isa Union{ShapedIdentity,Scaled{<:Number,<:ShapedIdentity}}
        input_axes(A) == input_axes(B) || return nothing
    end
    a = diag(A)
    λ = convert_multiplier(multiplier(B), eltype(a))
    c = similar(a, sum_type(eltype(a), typeof(λ)))
    c .= a .+ λ
    return Diag(c)
end

function try_simplify((A,B)::Sum{<:MaybeScaled{<:Identity},<:MaybeScaled{<:DiagonalOperator}})
    try_simplify(B + A)
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

# Complex rules for:
#
#     μ*inv(B)*C*B + λ*Id -> inv(B)*(μ*C + λ*Id)*B
#     μ*B*C*inv(B) + λ*Id -> B*(μ*C + λ*Id)*inv(B)

function try_simplify(A::Sum{<:MaybeScaled{<:Prod{<:Inverse{<:T},<:Prod{<:Operator,<:T}}},
                             <:MaybeScaled{<:Identity}}) where {T<:Operator}
    # `A = μ*inv(B)*C*D + λ*Id` with `B` and `D` having the same type.
    Q = unscaled(A[1])
    μ = multiplier(A[1])
    B  = inv(Q[1])
    C  = Q[2][1]
    D  = Q[2][2]
    λI = A[2]
    if !isequal(B, D)
        nothing
    elseif isone(μ)
        inv(B)*simplify(C + λI)*B
    else
        inv(B)*simplify(μ*C + λI)*B
    end
end

function try_simplify(A::Sum{<:MaybeScaled{<:Identity},
                             <:MaybeScaled{<:Prod{<:Inverse{<:T},<:Prod{<:Operator,<:T}}}}) where {T<:Operator}
    # Permute the terms
    return try_simplify(A[2] + A[1])
end

function try_simplify(A::Sum{<:MaybeScaled{<:Prod{<:T,<:Prod{<:Operator,<:Inverse{<:T}}}},
                             <:MaybeScaled{<:Identity}}) where {T<:Operator}
    # `A = μ*B*C*inv(D) + λ*Id` with `B` and `D` having the same type
    Q = unscaled(A[1])
    μ = multiplier(A[1])
    B  = Q[1]
    C  = Q[2][1]
    D  = inv(Q[2][2])
    λI = A[2]
    if !isequal(B, D)
        nothing
    elseif isone(μ)
        B*simplify(C + λI)*inv(B)
    else
        B*simplify(μ*C + λI)*inv(B)
    end
end

function try_simplify(A::Sum{<:MaybeScaled{<:Identity},
                             <:MaybeScaled{<:Prod{<:T,<:Prod{<:Operator,<:Inverse{<:T}}}}}) where {T<:Operator}
    # Permute the terms
    return try_simplify(A[2] + A[1])
end
