# Implement rank-1 operators in LazyAlgebra.

"""
    LazyAlgebra.AbstractRankOneOperator{U,V}

is the parent type of *rank-1* linear operators of the form `A = u*v'`. Type parameters
`U` and `V` are the respective types of the left-hand side *vector* `u` and of the
right-hand side *vector* `v`.

For an instance `A` of this type, `first(A)` and `last(A)` respectively yield `u` and `v`.

There exist two concrete implementations: [`SymmetricRankOneOperator`](@ref) and
[`RankOneOperator`](@ref).

""" AbstractRankOneOperator

"""
    RankOneOperator(u, v) -> A

yields the *rank-1* linear operator `A = u*v'` defined by the two *vectors* `u` and `v`
and behaving as:

    A*x  -> vdot(v, x) * u
    A'*x -> vdot(u, x) * v

See also [`SymmetricRankOneOperator`](@ref), [`Operator`](@ref), [`vmul`](@ref).

""" RankOneOperator

"""
    SymmetricRankOneOperator(u) -> A

yields the *symmetric rank-1* operator `A = u*u'` defined by the *vector* `u` and behaving
as follows:

    A'*x -> A*x
    A*x  -> vscale(vdot(u, x)), u)

See also: [`RankOneOperator`](@ref), [`Operator`](@ref),
          [`Trait`](@ref) [`vmul!`](@ref), [`vcreate`](@ref).

""" SymmetricRankOneOperator

Base.show(io::IO, A::RankOneOperator) = print(io, "RankOneOperator(…)")
Base.show(io::IO, A::SymmetricRankOneOperator) = print(io, "SymmetricRankOneOperator(…)")

# Accessors.
Base.first(A::RankOneOperator) = getfield(A, :u)
Base.last( A::RankOneOperator) = getfield(A, :v)
Base.first(A::SymmetricRankOneOperator) = getfield(A, :u)
Base.last( A::SymmetricRankOneOperator) = first(A)

# NOTE It is so simple to re-build a rank-1 operator that taking the adjoint is directly
#      simplified as follows (this is type-stable):
Adjoint(A::RankOneOperator) = RankOneOperator(last(A), first(A))
Adjoint(A::SymmetricRankOneOperator) = A
#      If this simplification is not applied, un-comment the following 2 lines:
# Base.first(A::Adjoint{<:AbstractRankOneOperator}) = last(parent(A))
# Base.last( A::Adjoint{<:AbstractRankOneOperator}) = first(parent(A))

# Operator API for rank-1 operators.
Base.eltype(::Type{<:AbstractRankOneOperator{U,V}}) where {U,V} =
    float(prod_type(eltype(U), eltype(V)))

OutputShape(::Type{<:AbstractRankOneOperator{U,V}}) where {U,V} = HasOutputShape{ndims(U)}()
output_axes(A::AbstractRankOneOperator) = axes(first(A))

InputShape(::Type{<:AbstractRankOneOperator{U,V}}) where {U,V} = HasInputShape{ndims(V)}()
input_axes(A::AbstractRankOneOperator) = axes(last(A))

function unsafe_vmul!(α::Number, A::Union{K,Adjoint{K}}, x::AbstractArray,
                      β::Number, y::AbstractArray) where {K <: AbstractRankOneOperator}
    # Call `vcombine!` at stage 1 to dispatch on the values of `α` and `β` because array
    # axes have already been checked.
    v = last(A)
    λ = α*vdot(v, x)
    u = first(A)
    vcombine!(λ, u, β, y, _Stage(1))
end

# Set precision for rank-1 operators.
_with_precision(::Type{T}, A::RankOneOperator) where {T<:AbstractFloat} =
    RankOneOperator(_with_precision(T, first(A)), _with_precision(T, last(A)))
_with_precision(::Type{T}, A::SymmetricRankOneOperator) where {T<:AbstractFloat} =
    SymmetricRankOneOperator(_with_precision(T, first(A)))
