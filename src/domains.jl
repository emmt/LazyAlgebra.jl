# Constructors and operations on domains.

ArrayDomain{T}(dims::Integer...) where {T} = ArrayDomain{T}(dims)
ArrayDomain{T}(dims::Tuple{Vararg{Integer}}) where {T} = ArrayDomain{T}(to_size(dims))
ArrayDomain{T}(dims::Dims{N}) where {T,N} = ArrayDomain{T,N}(prod(dims), dims)
ArrayDomain(A::AbstractArray{T,N}) where {T,N} = ArrayDomain{T}(A)
function ArrayDomain{T}(A::AbstractArray{<:Any,N}) where {T,N}
    Base.has_offset_axes(A) && throw(ArgumentError(
        "array must has 1-based indices, use `OffsetArrayDomain` instead"))
    # FIXME: check that A ∈ D holds?
    return ArrayDomain{T,N}(length(A), size(A))
end

function Base.show(io::IO, A::ArrayDomain{T,N}) where {T,N}
    print(io, "ArrayDomain{", T, "}(")
    join(io, element_size(A), ", ")
    print(io, ")")
    nothing
end

OffsetArrayDomain{T}(inds::AcceptableArrayAxis...) where {T} = OffsetArrayDomain{T}(inds)
OffsetArrayDomain{T}(inds::AcceptableArrayAxes) where {T} = OffsetArrayDomain{T}(to_axes(inds))
OffsetArrayDomain{T}(inds::I) where {T,N,I<:ArrayAxes{N}} =
    OffsetArrayDomain{T,N,I}(mapreduce(length, *, inds), inds)
OffsetArrayDomain(A::AbstractArray{T,N}) where {T,N} = OffsetArrayDomain{T}(A)
function OffsetArrayDomain{T}(A::AbstractArray{<:Any,N}) where {T,N}
    inds = axes(A)
    # FIXME: check that A ∈ D holds?
    return AbstractArrayDomain{T,N,typeof(inds)}(length(A), inds)
end

function Base.show(io::IO, A::OffsetArrayDomain{T,N}) where {T,N}
    print(io, "OffsetArrayDomain{", T, "}(")
    join(io, element_axes(A), ", ")
    print(io, ")")
    nothing
end

"""
    element_ndims(A)

yields the number of dimensions of the elements of the domain or domain type
`A`. This is only implemented for domains whose elements are arrays.

"""
element_ndims(A::AbstractArrayDomain) = element_ndims(typeof(A))
element_ndims(::Type{<:AbstractArrayDomain{T,N}}) where {T,N} = N

"""
    element_eltype(A)

yields the type of the entries of the elements of the domain or domain type
`A`. This is only implemented for domains whose elements are arrays.

"""
element_eltype(A::AbstractArrayDomain) = element_eltype(typeof(A))
element_eltype(::Type{<:AbstractArrayDomain{T,N}}) where {T,N} = T

"""
    element_length(A)

yields the number of entries of the elements of the domain or domain type `A`.
This is only implemented for domains whose elements are arrays.

"""
element_length(A::ArrayDomain) = getfield(A, :length)
element_length(A::OffsetArrayDomain) = getfield(A, :length)

"""
    element_size(A)
    element_size(A, i)

yield the dimensions or the `i`-th dimension of the elements of the domain or
domain type `A`. This is only implemented for domains whose elements are
arrays.

"""
element_size(A::ArrayDomain) = getfield(A, :size)
element_size(A::ArrayDomain, i::Integer) = element_size(A)[i]
element_size(A::OffsetArrayDomain) = map(length, element_axes(A))
element_size(A::OffsetArrayDomain, i::Integer) = length(element_axes(A, i))

"""
    element_axes(A)
    element_axes(A, i)

yield the index ranges or the `i`-th index range of the elements of the domain
or domain type `A`. This is only implemented for domains whose elements are
arrays.

"""
element_axes(A::ArrayDomain) = map(Base.OneTo, element_size(A))
element_axes(A::ArrayDomain, i::Integer) = Base.OneTo(element_size(A, i))
element_axes(A::OffsetArrayDomain) = getfield(A, :axes)
element_axes(A::OffsetArrayDomain, i::Integer) = element_axes(A)[i]

Base.in(A, B::AbstractDomain) = false
Base.in(A::AbstractArray{S,N}, B::ArrayDomain{T,N}) where {S,T,N} =
    is_subtype(S, T) && !Base.has_offset_axes(A) && size(A) == element_size(B)
Base.in(A::AbstractArray{S,N}, B::OffsetArrayDomain{T,N}) where {S,T,N} =
    is_subtype(S, T) && axes(A) == element_axes(B)

Base.issubset(A::AbstractDomain, B::AbstractDomain) = false
Base.issubset(A::AbstractArrayDomain{Ta,N}, B::AbstractArrayDomain{Tb,N}) where {Ta,Tb,N} =
    is_subtype(Ta, Tb) && element_axes(A) == element_axes(B)

Base.:(+)(A::AbstractDomain) = A
Base.:(-)(A::AbstractDomain) = -1*A
Base.:(-)(A::AbstractDomain, B::AbstractDomain) = A + (-B)

Base.:(*)(α::Number, B::AbstractDomain) = B   # FIXME only for reals?
Base.:(*)(A::AbstractDomain, β::Number) = β*A # FIXME really support this syntax?
Base.:(\)(α::Number, B::AbstractDomain) = B   # FIXME only for reals?
Base.:(/)(A::AbstractDomain, β::Number) = β\A # FIXME really support this syntax?

"""
    LazyAlgebra.EmptyDomain()
    LazyAlgebra.∅

is the singleton representing an empty domain.

""" EmptyDomain
@doc EmptyDomain ∅
Base.show(io::IO, ::EmptyDomain) = print(io, "∅")

Base.union(A::EmptyDomain, B::AbstractDomain) = B
Base.union(A::AbstractDomain, B::EmptyDomain) = A
Base.union(A::EmptyDomain, B::EmptyDomain) = ∅

Base.intersect(A::EmptyDomain, B::AbstractDomain) = ∅
Base.intersect(A::AbstractDomain, B::EmptyDomain) = ∅
Base.intersect(A::EmptyDomain, B::EmptyDomain) = ∅

# NOTE: May not be type-stable.
Base.:(+)(A::AbstractDomain, B::AbstractDomain) =
    A ⊆ B ? B :
    B ⊆ A ? A : throw(ArgumentError("addition of domains `A + B` is not implemented"))

# NOTE: May not be type-stable.
Base.union(A::AbstractDomain, B::AbstractDomain) =
    A ⊆ B ? B :
    B ⊆ A ? A : throw(ArgumentError("union of domains `A ∪ B` is not implemented"))

# NOTE: May not be type-stable.
Base.intersect(A::AbstractDomain, B::AbstractDomain) =
    A ⊆ B ? A :
    B ⊆ A ? B : throw(ArgumentError("intersection of domains `A ∩ B` is not implemented"))
