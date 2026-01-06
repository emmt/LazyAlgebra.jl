#
# rules.jl -
#
# Implement arithmetic rules for building associations (sum and composition) of linear
# operators and their variants (adjoint, inverse, etc.).
#
#-------------------------------------------------------------------------------------------

# Accessors and base methods for Adjoint, Transpose, Conjugate, and Inverse wrappers.
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

# Accessors for Sum, Prod, and Scaled and make them iterable.
for S in (:Sum, :Prod, :Scaled)
    @eval begin
        Base.Tuple(A::$S) = getfield(A, :operands)
        Base.first(A::$S) = @inbounds A[1]
        Base.last( A::$S) = @inbounds A[2]
        Base.firstindex(A::$S) = 1
        Base.lastindex( A::$S) = 2
        Base.length(A::$S) = 2
        Base.IteratorSize(::Type{<:$S}) = Base.HasLength()
        @inline Base.iterate(A::$S, i::Int = 1) =
            1 ≤ i ≤ 2 ? ((@inbounds Tuple(A)[i]), i + 1) : nothing
        @inline Base.getindex(A::$S, i::Integer) =
            1 ≤ i ≤ 2 ? (@inbounds Tuple(A)[i]) : throw_bounds_error(A, i)
    end
end

# Propagate the adjoint in products and in sums.
Adjoint((α,B)::Scaled) = conj(α) * Adjoint(B)
Adjoint((A,B)::Prod) = Adjoint(B) * Adjoint(A)
Adjoint((A,B)::Sum) = Adjoint(A) + Adjoint(B)

# Propagate the transpose in products and in sums.
Transpose((α,B)::Scaled) = α * Transpose(B)
Transpose((A,B)::Prod) = Transpose(B) * Transpose(A)
Transpose((A,B)::Sum) = Transpose(A) + Transpose(B)

# Propagate the conjugate in products and in sums.
Conjugate((α,B)::Scaled) = conj(α) * Conjugate(B)
Conjugate((A,B)::Prod) =  Conjugate(A) * Conjugate(B)
Conjugate((A,B)::Sum) = Conjugate(A) + Conjugate(B)

# Propagate the inverse in products.
Inverse((α,B)::Scaled) = α \ Inverse(B)
Inverse((A,B)::Prod) = Inverse(B) * Inverse(A)

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

# Addition (+) and subtraction (-) of operators yield a Sum.
Base.:(+)(A::Operator, B::Operator) = Sum(A, B)
Base.:(-)(A::Operator, B::Operator) = A + (-B)

# Extend multiplication by `*` and left or right division by '/' or '\' when at least one
# operand is an operator and the other is a scalar or an operator. Any simplifications of
# the multiplication are automatically done by the `Prod` constructor. Hence, divisions are
# re-expressed as multiplications.
Base.:(∘)(A::Operator, B::Operator) = A * B
Base.:(*)(A::Operator, B::Operator) = Prod(A, B)
Base.:(*)(A::Operator, β::Number  ) = β * A
Base.:(*)(α::Number,   B::Operator) = Scaled(α, B)
#
Base.:(/)(A::Operator, β::Number) = inverse(β) * A
Base.:(/)(A::Scaled,   β::Number) = divide(A[1], β) * A[2]
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
# expressions like sums of operators and products of an operator and an inverse operator.
# Hence, comparisons can be implemented by very simple rules. The only restriction is that
# the result shall only be accurate after full simplification rules have been applied to
# both operands.
for eq in (:(==), :isequal)
    @eval begin
        # Equality for sums of operators.
        #
        # For comparing two sums, due to commutativity of addition all possible permutations
        # should be compared but would scale as O(n!) with n the number of terms or would
        # require first sorting the terms of A and B. This is too long, so equality is only
        # tested without permutations. This is sufficient if A and B have been "simplified"
        # (and thus their terms sorted).
        Base.$eq(A::Sum, B::Sum) = ($eq(A[1], B[1]) && $eq(A[2], B[2]))
        #
        # Equality for scaled operators.
        Base.$eq(A::Scaled, B::Scaled) = $eq(A[1], B[1]) && (iszero(A[1]) || $eq(A[2], B[2]))
        #
        # Equality for compositions of operators.
        Base.$eq(A::Prod, B::Prod) = $eq(A[1], B[1]) && $eq(A[2], B[2])
        #
        # Equality for adjoint, transpose, and inverse (accounting for inverse-adjoint
        # results from these rules).
        Base.$eq(A::Adjoint,   B::Adjoint) = $eq(parent(A), parent(B))
        Base.$eq(A::Transpose, B::Transpose) = $eq(parent(A), parent(B))
        Base.$eq(A::Conjugate, B::Conjugate) = $eq(parent(A), parent(B))
        Base.$eq(A::Inverse,   B::Inverse) = $eq(parent(A), parent(B))
        #
        # Comparing operators of mixed kinds is delegated to an auxiliary function to reduce
        # the cases to handle. We consider `Scaled` to be the most specific, then `Sum`,
        # then others.
        Base.$eq(A::Sum,      B::Scaled  ) = $eq(B, A)
        Base.$eq(A::Prod,     B::Scaled  ) = $eq(B, A)
        Base.$eq(A::Operator, B::Scaled  ) = $eq(B, A)
        Base.$eq(A::Scaled,   B::Sum     ) = compare_with($eq, A, B)
        Base.$eq(A::Scaled,   B::Prod    ) = compare_with($eq, A, B)
        Base.$eq(A::Scaled,   B::Operator) = compare_with($eq, A, B)
        #
        Base.$eq(A::Prod,     B::Sum     ) = $eq(B, A)
        Base.$eq(A::Operator, B::Sum     ) = $eq(B, A)
        Base.$eq(A::Sum,      B::Prod    ) = compare_with($eq, A, B)
        Base.$eq(A::Sum,      B::Operator) = compare_with($eq, A, B)
    end
end

# Comparing scaled with others.
compare_with(f::Union{typeof(==),typeof(isequal)}, A::Scaled, B::Operator) =
    isone(A[1]) && f(A[2], B)

# For comparing a sum and another operator for equality, it is lazily assumed that the i/o
# sizes of the terms of the sum are compatible. Again, the number of considered cases are
# not meant to be exhaustive, just to be sufficient if A and B have been simplified.
compare_with(f::Union{typeof(==),typeof(isequal)}, A::Sum, B::Operator) =
    (iszero(A[1]) && f(A[2], B)) || (iszero(A[2]) && f(A[1], B))

compare_with(f::Union{typeof(==),typeof(isequal)}, A::Operator, B::Operator) = false

# Simplification rules for products and sums.
#
# - NOTE Do not distribute multiplication by a scalar among the terms of a sum to not
#   prevent the left-factorization of multipliers at construction time. This is done by
#   `simplify`.
#
# - Number operands are moved to the leftmost part of products and factorized.
Prod(A::Operator, (β,B)::Scaled) = β * (A * B)
Prod((α,A)::Scaled, B::Operator) = α * (A * B)
Prod((α,A)::Scaled, (β,B)::Scaled) = (α * β) * (A * B)
#
# - Right-associativity is applied to keep product and sum of operators in the expected
#   order for applying these constructions to an argument. See `unsafe_vmul!` method for
#   these constructions. As a result, the left-hand side of a `Sum` (resp. a `Prod`) shall
#   never be a `Sum` (resp. a `Prod`).
Sum((A,B)::Sum,  C::Operator) = A + (B + C)
Prod((A,B)::Prod, C::Operator) = A * (B * C)

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

# Precision for sums and compositions.
TypeUtils.get_precision(::Type{Sum{A,B}}) where {A,B} = get_precision(A, B)
TypeUtils.get_precision(::Type{Prod{A,B}}) where {A,B} = get_precision(A, B)
TypeUtils.adapt_precision(::Type{T}, (A,B)::Sum) where {T<:TypeUtils.Precision} =
    adapt_precision(T, A) + adapt_precision(T, B)
TypeUtils.adapt_precision(::Type{T}, (A,B)::Prod) where {T<:TypeUtils.Precision} =
    adapt_precision(T, A) * adapt_precision(T, B)

# Precision of a scaled operator does not depend on the multiplier.
TypeUtils.get_precision(::Type{Scaled{<:Number,A}}) where {A} = get_precision(A)
TypeUtils.adapt_precision(::Type{T}, (λ,A)::Scaled) where {T<:TypeUtils.Precision} =
    adapt_precision(T, λ) * adapt_precision(T, A)
