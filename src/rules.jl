#
# rules.jl -
#
# Implement arithmetic rules for building associations (sum and composition) of linear
# operators and their variants (adjoint, inverse, etc.).
#
#-----------------------------------------------------------------------------------------

# Accessors for Adjoint, Inverse, Sum, and Prod.
Base.parent(A::Union{Adjoint,Inverse}) = getfield(A, :parent)
Base.getindex(A::Union{Adjoint,Inverse}) = parent(A)
Base.Tuple( A::Union{Sum,Prod}) = getfield(A, :operands)
for Wrapper in (:Adjoint, :Inverse)
    @eval begin
        # Make `parent` also applicable to types of wrapped operators.
        Base.parent(::Type{$Wrapper{T}}) where {T} = T
    end
end

# Make Sum and Prod iterable.
Base.first(A::Union{Sum,Prod}) = @inbounds A[1]
Base.last( A::Union{Sum,Prod}) = @inbounds A[2]
Base.firstindex(A::Union{Sum,Prod}) = 1
Base.lastindex( A::Union{Sum,Prod}) = 2
Base.length(A::Union{Sum,Prod}) = 2
Base.IteratorSize(::Type{<:Union{Sum,Prod}}) = Base.HasLength()
@inline Base.iterate(A::Union{Sum,Prod}, i::Int = 1) =
    1 ≤ i ≤ 2 ? (unsafe_getindex(A, i), i + 1) : nothing
@inline Base.getindex(A::Union{Sum,Prod}, i::Integer) =
    1 ≤ i ≤ 2 ? unsafe_getindex(A, i) : throw(BoundsError(A, i))

@inline unsafe_getindex(A::Union{Sum,Prod}, i::Integer) =
    @inbounds getindex(Tuple(A), Int(i))

# Extend `A'` to call `Adjoint(A)` for any operator `A`, automatically simplify taking the
# adjoint of the adjoint of an operator and propagate the adjoint in products and in sums.
Base.adjoint(A::Operator ) = Adjoint(A)
Adjoint(A::Adjoint       ) = parent(A)
Adjoint(A::Prod{<:Number}) = conj(A[1]) * Adjoint(A[2])
Adjoint(A::Prod          ) = Adjoint(A[2]) * Adjoint(A[1])
Adjoint(A::Sum           ) = Adjoint(A[1]) + Adjoint(A[2])
Adjoint(α::Number        ) = conj(α)

# Maintain inverse on top of adjoint.
Adjoint(A::Inverse           ) = Inverse(Adjoint(parent(A)))
Adjoint(A::Inverse{<:Adjoint}) = inv(parent(parent(A)))

# Extend `inv(A)` to call `Inverse(A)` for any operator `A`. Automatically simplify taking
# the inverse of the inverse of an operator and propagate the inverse in products.
Base.inv(A::Operator     ) = Inverse(A)
Inverse(A::Inverse       ) = parent(A)
Inverse(A::Prod{<:Number}) = A[1] \ Inverse(A[2])
Inverse(A::Prod          ) = Inverse(A[2]) * Inverse(A[1])
Inverse(α::Number        ) = is_rationalizable(α) ? one(α)//α : inv(α)
Inverse(α::Neutral       ) = inv(α)

# Unary plus and minus of operators.
Base.:(+)(A::Operator) = A
#
Base.:(-)(A::Prod{<:Number}) = (-A[1]) * A[2]
Base.:(-)(A::Operator) = (-𝟙) * A

# Addition (+) and subtraction (-) of operators yield a Sum.
Base.:(+)(A::Operator, B::Operator) = Sum(A, B)
Base.:(-)(A::Operator, B::Operator) = A + (-B)

# Extend multiplication by `*` and left or right division by '/' or '\' when at least one
# operand is an operator and the other is a scalar or an operator. Any simplifications of
# the multiplication are automatically done by the `Prod` constructor. Hence, divisions
# are re-expressed as multiplications.
Base.:(∘)(A::Operator, B::Operator) = A * B
Base.:(*)(A::Operator, β::Number  ) = β * A
Base.:(*)(A::Operand,  B::Operator) = Prod(A, B)
Base.:(*)(α::Number,   B::Operator) = Prod(α, B)
#
Base.:(/)(A::Operator,       β::Number) = Inverse(β) * A
Base.:(/)(A::Prod{<:Number}, β::Number) = divide(A[1], β) * A[2]
Base.:(/)(A::Operator,       B::Operator) = A * inv(B)
Base.:(/)(α::Number,         B::Operator) = error(
    "`A\\β` and `β/A` for a linear operator `A` and a number `β` intentionally not supported, write `β*inv(A)` or `β*Id/A` if that is the intention")
#
# Default rule for left-division in base Julia is: x\y -> adjoint(adjoint(y)/adjoint(x))
# which, in LazyAlgebra, simplifies to: x\y -> inv(x)*y.
Base.:(\)(A::Operator, B::Operator) = inv(A) * B
Base.:(\)(α::Number,   B::Operator) = B / α
Base.:(\)(A::Operator, β::Number  ) = β / A

# Equality. If no more specific rules exist, consider that two operators are different by
# default unless they are the same object.
Base.:(==)(A::T, B::T) where {T<:Operator} = A === B
Base.:(==)(A::Operator, B::Operator) = false
Base.isequal(A::Operator, B::Operator) = A == B
for eq in (:(==), :isequal)
    @eval begin
        # Equality for sums of operators.
        #
        # For comparing two sums, due to commutativity of addition all possible
        # permutations should be compared but would scale as O(n!) with n the number of
        # terms or would require first sorting the terms of A and B. This is too long, so
        # equality is only tested without permutations. This is sufficient if A and B have
        # been "simplified" (and thus their terms sorted).
        Base.$eq(A::Sum, B::Sum) = ($eq(A[1], B[1]) && $eq(A[2], B[2]))
        #
        # For comparing a sum and another operator, it is lazily assumed that the i/o
        # sizes of the terms of the sum are compatible. Again, the number of considered
        # cases are not meant to be exhaustive, just to be sufficient if A and B have been
        # simplified.
        Base.$eq(A::Operator, B::Sum) = $eq(B, A)
        Base.$eq(A::Sum, B::Operator) =
            (iszero(A[1]) && $eq(A[2], B)) || (iszero(A[2]) && $eq(A[1], B))

        # Equality for scaled operators and compositions of operators.
        Base.$eq(A::Prod, B::Prod) = $eq(A[1], B[1]) && $eq(A[2], B[2])
        Base.$eq(A::Prod, B::Operator) = isone(A[1]) && $eq(A[2], B)
        Base.$eq(A::Operator, B::Prod) = $eq(B, A)

        # Equality between sums and products.
        Base.$eq(A::Sum, B::Prod) = $eq(B, A)
        Base.$eq(A::Prod, B::Sum) = isone(A[1]) && $eq(A[2], B)

        # Equality for adjoint and inverse (accounting for inverse-adjoint results from
        # these rules).
        Base.$eq(A::Adjoint, B::Adjoint) = $eq(parent(A), parent(B))
        Base.$eq(A::Inverse, B::Inverse) = $eq(parent(A), parent(B))
    end
end

# Simplification rules for products and sums.
#
# - NOTE Do not distribute multiplication by a scalar among the terms of a sum to not
#   prevent the left-factorization of multipliers at construction time. This is done by
#   `simplify`.
#
# - Number operands are moved to the leftmost part of products and factorized.
Prod(α::Number,           β::Number        ) = α * β
Prod(A::Operator,         β::Number        ) = Prod(β, A)
Prod(α::Number,           B::Prod{<:Number}) = (α * B[1]) * B[2]
Prod(A::Operator,         B::Prod{<:Number}) = B[1] * (A * B[2])
Prod(A::Prod{<:Operator}, B::Prod{<:Number}) = B[1] * (A * B[2])
Prod(A::Prod{<:Number},   B::Prod{<:Number}) = (A[1] * B[1]) * (A[2] * B[2])
#
# - Right-associativity is applied to keep product and sum of operators in the expected
#   order for applying these constructions to an argument. See `unsafe_vmul!` method for
#   these constructions. As a result, the left-hand side of a `Sum` (resp. a `Prod`) shall
#   never be a `Sum` (resp. a `Prod`).
Sum( A::Sum,  B::Operator) = A[1] + (A[2] + B)
Prod(A::Prod, B::Operator) = A[1] * (A[2] * B)

#---------------------------------------------------------------------- NEUTRAL ELEMENTS -

# The neutral element ("zero") for the addition is zero times a mapping of the
# proper type.
Base.zero(A::Operator) = 𝟘 * A
Base.zero(A::Prod{<:Number}) = zero(A[1]) * A[2]

Base.iszero(A::Prod{<:Number}) = iszero(A[1])
Base.iszero(::Operator) = false

# The neutral element ("one") for the composition is the identity.
Base.one(::Union{Operator,Type{<:Operator}}) = Id

Base.isone(::Identity) = true
Base.isone(::Operator) = false

#----------------------------------------------------------------------------- PRECISION -

# Precision for adjoint, and inverse. Thanks to recursion, this also works for
# inverse-adjoint.
for W in (:Adjoint, :Inverse)
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
