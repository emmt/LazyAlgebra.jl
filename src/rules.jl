# rules.jl -
#
# Rules for operations on mappings.
#

# Unary plus and minus.
+(A::AbstractMapping) = A
-(A::AbstractMapping) = -1*A

# Left multiplication by a scalar.
*(α::Number, B::AbstractMapping) = Scaled(α, B)
*(α::Number, B::Scaled) = Scaled(α*multiplier(B), unscaled(B))

# Left division by a scalar.
\(α::Number, B::AbstractMapping) = Scaled(inv(α), B)
\(α::Number, B::Scaled) = Scaled(α\multiplier(B), unscaled(B))

# Right multiplication by a scalar.
*(A::AbstractMapping, β::Number) =
    if is_linear(typeof(A)) # NOTE use is_linear on type for type-stability
        β*A
    else
        A*(β*Identity(input_domain(A)))
    end

# Right division by a scalar.
/(A::AbstractMapping, β::Number) =
    if is_linear(typeof(A)) # NOTE use is_linear on type for type-stability
         β\A
    else
        A*(β\Identity(input_domain(A)))
    end

# Right multiplication by a non-scalar calls `apply`.
*(A::AbstractMapping, b) = apply(A, b)

# Other rules.
\(A::AbstractMapping, b) = inv(A)*b

# Adjoint.
adjoint(A::AbstractMapping) = Adjoint(A)
adjoint(A::Adjoint) = parent(A)
adjoint(A::Scaled) = conj(multiplier(A))*adjoint(unscaled(A))
adjoint(A::Identity) = A
adjoint(A::Null) = Null(output_domain(A) => input_domain(A))
function adjoint(A::Sum)
    src_terms = terms(A)
    dst_terms = similar(src_terms)
    @inbounds for i in eachindex(src_terms, dst_terms)
        dst_terms[i] = adjoint(src_terms[i])
    end
    return Sum(output_domain(A) => input_domain(A), dst_terms)
end
adjoint(A::Composition) =
    Composition(map_with_eltype(AbstractMapping, adjoint, Iterators.reverse(terms(A))))

# Inverse.
inv(A::AbstractMapping) = Inverse(A)
inv(A::Inverse) = parent(A)
inv(A::Adjoint) = inv(parent(A))' # NOTE always compose adjoint and inverse as inv(A)' (i.e. Adjoint{Inverse{...}})
inv(A::Scaled) = inv(unscaled(A))/multiplier(A)
# FIXME    if is_linear(typeof(A)) # NOTE use is_linear on type for type-stability
# FIXME        multiplier(A)\inv(unscaled(A))
# FIXME    else
# FIXME        inv(unscaled(A))/multiplier(A)
# FIXME    end
inv(A::Identity) = A
inv(A::Composition) =
    Composition(map_with_eltype(AbstractMapping, inv, Iterators.reverse(terms(A))))

"""
    LazyAlgebra.is_linear(A)

yields whether mapping (resp. mapping type) `A` is a linear mapping (resp. a
linear mapping type).

When directly used on a type rather than on an instance, the result is always
type-stable but may be inaccurate in the sense that there may be false
negatives for constructions like sums or compositions. At least there are never
false positives. This feature is used by automatic (type-stable)
simplifications.

"""
is_linear(::Type{T}) where {T<:AbstractMapping} = false
is_linear(::Type{T}) where {T<:AbstractLinearMapping} = true
is_linear(::Type{T}) where {T<:Scaled} = is_linear(unscaled(T))
is_linear(::Type{T}) where {T<:Adjoint} = is_linear(parent(T))
is_linear(::Type{T}) where {T<:Inverse} = is_linear(parent(T))
is_linear(A::Union{Sum,Composition}) = all(is_linear, terms(A))
is_linear(A::AbstractMapping) = is_linear(typeof(A))

#------------------------------------------------------------------------------
# Addition of mappings.

function +(A::AbstractMapping, B::AbstractMapping)
    inp = input_domain(A) ∩ output_domain(B)
    out = output_domain(A) + output_domain(B)
    return unsafe_add(inp => out, A, B)
end

function unsafe_add(io::Pair{AbstractDomain,AbstractDomain},
                    A::AbstractMapping, B::AbstractMapping)
    return Sum(io, concat(AbstractMapping, A, B))
end
function unsafe_add(io::Pair{AbstractDomain,AbstractDomain},
                    A::Sum, B::Sum)
    return Sum(io, concat(AbstractMapping, terms(A), terms(B)))
end
function unsafe_add(io::Pair{AbstractDomain,AbstractDomain},
                    A::AbstractMapping, B::Sum)
    return Sum(io, concat(AbstractMapping, A, terms(B)))
end
function unsafe_add(io::Pair{AbstractDomain,AbstractDomain},
                    A::Sum, B::AbstractMapping)
    return Sum(io, concat(AbstractMapping, terms(A), B))
end

#------------------------------------------------------------------------------
# Composition of mappings.

function *(A::AbstractMapping, B::AbstractMapping)
    input_domain(A) ⊆ output_domain(B) || throw(ArgumentError(
        "`input_domain(A) ⊆ output_domain(B)` must hold for the composition `A*B`"))
    return unsafe_compose(A, B)
end

unsafe_compose(A::AbstractMapping, B::AbstractMapping) =
    Composition(concat(AbstractMapping, A, B))
unsafe_compose(A::Composition, B::Composition) =
    Composition(concat(AbstractMapping, terms(A), terms(B)))
unsafe_compose(A::AbstractMapping, B::Composition) =
    Composition(concat(AbstractMapping, A, terms(B)))
unsafe_compose(A::Composition, B::AbstractMapping) =
    Composition(concat(AbstractMapping, terms(A), B))
