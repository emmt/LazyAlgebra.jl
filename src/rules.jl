#
# rules.jl -
#
# Implement arithmetic rules for building associations (sum and composition) of linear
# operators and their variants (adjoint, inverse, etc.). Automatic simplification rules are
# implemented by specializing the constructors.
#
#-------------------------------------------------------------------------------------------

# Accessors and base methods for `Adjoint`, `Transpose`, `Conjugate`, and `Inverse`
# wrappers.
for (f, W) in (:adjoint   => :Adjoint,
               :transpose => :Transpose,
               :conj      => :Conjugate,
               :inv       => :Inverse)
    @eval begin
        Base.parent(A::$W) = getfield(A, :parent)
        Base.parent(::Type{$W{T}}) where {T} = T # parent is also applicable to type
        Base.getindex(A::$W) = parent(A)
        Base.$f(A::Operator) = $W(A)
        $W(A::$W) = parent(A) # wrapper constructor is also the unwrapper
    end
end

"""
    LazyAlgebra.terms(A::Operator) -> tup

Return a tuple of the terms involved in the operator `A`. If `A` is neither a sum of
operators, a composition of operators, nor a scaled operator, `(A,)` is returned.

"""
terms(A::Union{Sum,Prod,Scaled}) = getfield(A, :terms)
terms(A::Operator) = (A,)

"""
    LazyAlgebra.terms(typeof(A)) -> tup

Return a tuple of the types of the terms involved in the operator `A`. This is like
`map(typeof, terms(A))` but directly applicable to the type of `A`.

"""
terms(::Type{<:Union{Sum{T},Prod{T}}}) where {T} = fieldtypes(T)
terms(::Type{Scaled{α,A}}) where {α,A} = (α,A,)
terms(::Type{A}) where {A<:Operator} = (A,)

"""
    LazyAlgebra.terms(op, A::Operator) -> tup::Tuple{Vararg{Operator}}

Return a tuple of operators such that `op(tup...)` is equivalent to `A` with `op` the
addition `+` or the multiplication `*`. If `op` is `+` and `A` is a sum or if `op` is `*`
and `A` is a product, the result is the tuple of the operators involved in `A`; otherwise,
the result is the tuple `(A,)`.

"""
terms(::typeof(+), A::Sum) = terms(A)
terms(::typeof(*), A::Prod) = terms(A)
terms(::Union{Function,Type}, A::Operator) = (A,)

# `Sum`, `Prod`, and `Scaled` can be used as iterators of their terms. See `base/tuple.i` for
# the iterators implementation for tuples.
Base.Tuple(A::Union{Sum,Prod,Scaled}) = terms(A) # TODO remove?
Base.IteratorEltype(::Type{<:Union{Sum,Prod}}) = Base.HasEltype()
Base.eltype(::Type{<:Union{Sum{T},Prod{T}}}) where {T} = eltype(T)
Base.IteratorSize(::Type{<:Union{Sum,Prod,Scaled}}) = Base.HasLength()
Base.length(A::Union{Sum{T},Prod{T}}) where {T} = fieldcount(T)
Base.length(A::Scaled) = 2
@inline Base.iterate(A::Union{Sum,Prod,Scaled}, i::Int = 1) = iterate(terms(A), i)
@propagate_inbounds Base.getindex(A::Union{Sum,Prod,Scaled}, i::Integer) = getindex(terms(A), i)

# For `first` and `last`, we know that `Sum`, `Prod`, and `Scaled` have a non-empty list of
# terms.
Base.first(A::Union{Sum,Prod,Scaled}) = @inbounds A[1]
Base.last(A::Union{Sum,Prod,Scaled}) = @inbounds A[length(A)]

Base.firstindex(A::Union{Sum,Prod,Scaled}) = 1
Base.lastindex(A::Union{Sum,Prod,Scaled}) = length(A)

# Extend `Base.tail` and `Base.front` for sums and compositions.
for f in (:tail, :front)
    for S in (:Sum, :Prod)
        @eval Base.$f(A::$S) = $S(Base.$f(terms(A)))
    end
end

# Propagate the adjoint in scaled operators, sums and compositions.
Adjoint((α,A)::Scaled) = conj(α) * Adjoint(A)
Adjoint(A::Sum) = Sum(map(Adjoint, terms(A)))
Adjoint(A::Prod) = Prod(reversemap(Adjoint, terms(A)))

# Propagate the transpose in scaled operators, sums and compositions.
Transpose((α,A)::Scaled) = α * Transpose(A)
Transpose(A::Sum) = Sum(map(Transpose, terms(A)))
Transpose(A::Prod) = Prod(reversemap(Transpose, terms(A)))

# Propagate the conjugate in scaled operators, sums and compositions.
Conjugate((α,A)::Scaled) = conj(α) * Conjugate(A)
Conjugate(A::Sum) = Sum(map(Conjugate, terms(A)))
Conjugate(A::Prod) = Prod(map(Conjugate, terms(A)))

# Propagate the inverse in products.
Inverse((α,A)::Scaled) = α \ Inverse(A)
Inverse(A::Prod) = Prod(reversemap(Inverse, terms(A)))

# Maintain inverse on top of adjoint, transpose, and conjugate.
for W in (:Adjoint, :Transpose, :Conjugate)
    @eval begin
        $W(A::Inverse{<:Any}) = Inverse($W(parent(A)))
        $W(A::Inverse{<:$W}) = inv(parent(parent(A)))
    end
end

# Other automatic simplification rules between adjoint, transpose, and conjugate.
Adjoint(A::Conjugate) = Transpose(parent(A))
Conjugate(A::Adjoint) = Transpose(parent(A))
Transpose(A::Conjugate) = Adjoint(parent(A))
Conjugate(A::Transpose) = Adjoint(parent(A))
Adjoint(A::Transpose) = Conjugate(parent(A))
Transpose(A::Adjoint) = Conjugate(parent(A))

# Unary plus and minus of operators.
Base.:(+)(A::Operator) = A
#
Base.:(-)(A::Scaled) = (-A[1]) * A[2]
Base.:(-)(A::Operator) = (-𝟙) * A

# Addition and composition of operators shall work like mathematical operators ∑ and ∏ (\sum
# and \prod in LaTeX). This is the purpose of the following rules. As a result of these
# rules, `Sum` and `Prod` instances are guaranteed to have at least 2 terms.
#
# 1. Extend Julia's `+` and `*` of two operators to call the constructors with a variable
#    number of operators consisting in the splatted operands. More specialized methods are
#    specified elsewhere for automatic simplifications.
#
for (op, constructor) in (:(+) => :Sum, :(*) => :Prod)
    @eval Base.$op(A::Operator, B::Operator) =
        $constructor(terms($op, A)..., terms($op, B)...,)
end
#
# 2. Consider the case of the sum and composition of an empty list of operators. Refuse to
#    build an empty sum because there is no universal null operator but an empty composition
#    of operators yields the universal identity.
Sum(::Tuple{}) = throw_bad_argument("cannot build an empty sum of operators")
Prod(::Tuple{}) = Id
#
# 3. A sum or composition of a single term automatically simplifies to this term.
for constructor in (:Sum, :Prod)
    @eval $constructor(A::Operator) = A
    @eval $constructor((A,)::Tuple{Operator}) = A
end
#
# 4. Otherwise, call the constructor with a tuple of terms.
for constructor in (:Sum, :Prod)
    @eval $constructor(terms::Operator...) = $constructor(terms)
end

# Subtraction of operators is rewritten as an addition.
Base.:(-)(A::Operator, B::Operator) = A + (-B)

# `∘` is an alias for `*` when both operands are operators.
Base.:(∘)(A::Operator, B::Operator) = A * B

# Multiplication of an operator by a scalar yields a scaled operator.
Base.:(*)(A::Operator,   β::Number  ) = β * A
Base.:(*)(α::Number,     B::Operator) = Scaled(α, B)
Base.:(*)(α::Neutral{1}, B::Operator) = B

# Factorize multiplier to the left of a composition.
Base.:(*)((α,A)::Scaled, B::Operator) = α * (A * B)
Base.:(*)(A::Operator, (β,B)::Scaled) = β * (A * B)
Base.:(*)((α,A)::Scaled, (β,B)::Scaled) = (α * β) * (A * B)

# Extend left or right division by '/' or '\' when at least one operand is an operator and
# the other is a scalar or an operator. Any simplifications of the multiplication are
# automatically done by the `Prod` constructor. Hence, divisions are re-expressed as
# multiplications.
#
Base.:(/)(A::Operator, β::Number) = inverse(β) * A
Base.:(/)((α,A)::Scaled, β::Number) = divide(α, β) * A
#
Base.:(/)(A::Operator, B::Operator) = A * inv(B)
Base.:(/)(α::Number,   B::Operator) = error(
    "`A\\β` and `β/A` for a linear operator `A` and a number `β` intentionally not supported, write `β*inv(A)` or `β*Id/A` if that is the intention")
#
# Default rule for left-division in base Julia is: x\y -> adjoint(adjoint(y)/adjoint(x))
# which, in LazyAlgebra, simplifies to: x\y -> inv(x)*y.
Base.:(\)(A::Operator, B::Operator) = inv(A) * B
Base.:(\)(α::Number,   B::Operator) = B / α
Base.:(\)(A::Operator, β::Number  ) = β / A

# Equality.
#
# If no more specific rules exist, consider that two operators are different by default
# unless they are the same object. This can be overridden for more specific operator types.
Base.:(==)(A::T, B::T) where {T<:Operator} = A === B
Base.:(==)(A::Operator, B::Operator) = false
Base.isequal(A::Operator, B::Operator) = A == B
#
# For sums, compositions, adjoint, inverse, etc., `isequal` is mostly used to simplify
# expressions like sums or compositions of operators. Hence, comparisons can be implemented
# by very simple rules. The only restriction is that the result shall only be accurate after
# full simplification rules have been applied to both operands.
for eq in (:(==), :isequal)
    @eval begin
        # Equality for sums of operators.
        #
        # For comparing two sums, due to commutativity of addition all possible permutations
        # should be compared but would scale as O(n!) with n the number of terms or would
        # require first sorting the terms of A and B. This is too long, so equality is only
        # tested without permutations. This is sufficient if A and B have been "simplified"
        # (and thus their terms sorted).
        Base.$eq(A::Sum, B::Sum) = $eq(terms(A), terms(B))
        #
        # Equality involving scaled operators.
        Base.$eq((α,A)::Scaled, (β,B)::Scaled) = $eq(α, β) && (iszero(β) || $eq(A, B))
        Base.$eq((α,A)::Scaled, B::Operator) = isone(α) && $eq(A, B)
        Base.$eq(A::Operator, B::Scaled) = $eq(B, A)
        #
        # Equality for compositions of operators.
        Base.$eq(A::Prod, B::Prod) = $eq(terms(A), terms(B))
        #
        # Equality for adjoint, transpose, and inverse (accounting for inverse-adjoint
        # results from these rules).
        Base.$eq(A::Adjoint,   B::Adjoint  ) = $eq(parent(A), parent(B))
        Base.$eq(A::Transpose, B::Transpose) = $eq(parent(A), parent(B))
        Base.$eq(A::Conjugate, B::Conjugate) = $eq(parent(A), parent(B))
        Base.$eq(A::Inverse,   B::Inverse  ) = $eq(parent(A), parent(B))
    end
end

# Simplification rules for products and sums.
#
# - We do not distribute multiplication by a scalar among the terms of a sum to not
#   prevent the left-factorization of multipliers at construction time. This is done by
#   `simplify`.
#
# - Scalar operands are moved to the leftmost part of products and factorized.
Prod(A::Operator, (β,B)::Scaled) = β * (A * B)
Prod((α,A)::Scaled, B::Operator) = α * (A * B)
Prod((α,A)::Scaled, (β,B)::Scaled) = (α * β) * (A * B)

# The following constructor insures that the right operand is always an unscaled operator.
Scaled(α::Number, (β,B)::Scaled) = (α * β) * B

#------------------------------------------------------------------------ Neutral elements -

# The neutral element ("zero") for the addition is zero times a mapping of the proper type.
Base.zero(A::Operator) = 𝟘 * A

Base.iszero(A::Scaled) = iszero(A[1])
Base.iszero(::Operator) = false

# The neutral element ("one") for the composition is the identity even though it is "too
# universal".
Base.one(::Union{Operator,Type{<:Operator}}) = Id

Base.isone(::Identity) = true
Base.isone(A::Scaled{<:Number,<:Identity}) = isone(multiplier(A))
Base.isone(::Operator) = false

#------------------------------------------------------------------------------- Precision -

# Precision for adjoint, and inverse. Thanks to recursion, this also works for
# inverse-adjoint.
for W in (:Adjoint, :Transpose, :Inverse)
    @eval begin
        TypeUtils.get_precision(::Type{$W{A}}) where {A} = get_precision(A)
        TypeUtils.adapt_precision(::Type{T}, A::$W) where {T<:TypeUtils.Precision} =
            $W(adapt_precision(T, parent(A)))
    end
end

# Precision for sums and compositions
for S in (:Sum, :Prod)
    @eval begin
        TypeUtils.get_precision(::Type{$S{T}}) where {T} = get_precision(T)
        TypeUtils.adapt_precision(::Type{T}, A::$S) where {T<:TypeUtils.Precision} =
            $S(map(adapt_precision(T), terms(A)))
    end
end

# Precision of a scaled operator does not depend on the multiplier.
TypeUtils.get_precision(::Type{Scaled{<:Number,A}}) where {A} = get_precision(A)
TypeUtils.adapt_precision(::Type{T}, (λ,A)::Scaled) where {T<:TypeUtils.Precision} =
    adapt_precision(T, λ) * adapt_precision(T, A)
