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

# Testing for equality.
for cmp in (:(==), :isequal)
    @eval begin
        Base.$cmp(A::RankOneOperator, B::RankOneOperator) =
            A === B || ($cmp(first(A), first(B)) && $cmp(last(A), last(B)))
        Base.$cmp(A::SymmetricRankOneOperator, B::SymmetricRankOneOperator) =
            A === B || $cmp(first(A), first(B))
        Base.$cmp(A::AbstractRankOneOperator, B::AbstractRankOneOperator) =
            $cmp(first(A), first(B)) && $cmp(last(A), last(B))
    end
end

# NOTE It is so simple to re-build a rank-1 operator that taking the adjoint is directly
#      simplified as follows (this is type-stable):
Adjoint(A::RankOneOperator) = RankOneOperator(last(A), first(A))
Adjoint(A::SymmetricRankOneOperator) = A
#      If this simplification is not applied, un-comment the following 2 lines:
# Base.first(A::Adjoint{<:AbstractRankOneOperator}) = last(parent(A))
# Base.last( A::Adjoint{<:AbstractRankOneOperator}) = first(parent(A))

# Operator API for rank-1 operators.
Base.eltype(::Type{<:AbstractRankOneOperator{U,V}}) where {U,V} =
    prod_type(eltype(U), eltype(V))

OutputShape(::Type{<:AbstractRankOneOperator{U,V}}) where {U,V} = HasOutputShape{ndims(U)}()
output_axes(A::AbstractRankOneOperator) = axes(first(A))

InputShape(::Type{<:AbstractRankOneOperator{U,V}}) where {U,V} = HasInputShape{ndims(V)}()
input_axes(A::AbstractRankOneOperator) = axes(last(A))

function unsafe_vmul!(α::Number, A::Union{K,Adjoint{K}}, x::AbstractArray,
                      β::Number, y::AbstractArray) where {K <: AbstractRankOneOperator}
    # Call `vcombine!` knowing that indices have been checked and `β` converted and
    # dispatched, so it just remains to converted `α` and dispatch on its value.
    unsafe_vcombine!(Val(:alpha), α*vdot(last(A), x), first(A), β, y)
    return y
end

# Precision for rank-1 operators.
TypeUtils.get_precision(::Type{A}) where {A<:AbstractRankOneOperator} = get_precision(eltype(A))
TypeUtils.adapt_precision(::Type{T}, A::RankOneOperator) where {T<:TypeUtils.Precision} =
    RankOneOperator(adapt_precision(T, first(A)), adapt_precision(T, last(A)))
TypeUtils.adapt_precision(::Type{T}, A::SymmetricRankOneOperator) where {T<:TypeUtils.Precision} =
    SymmetricRankOneOperator(adapt_precision(T, first(A)))
