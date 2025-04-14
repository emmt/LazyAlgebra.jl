to_axis(x::Integer) = Base.OneTo{Int}(x)
to_axis(x::AbstractUnitRange{Int}) = x
to_axis(x::AbstractUnitRange{<:Integer}) = map(Int, x)
to_axis(x::Base.OneTo{<:Integer}) = Base.OneTo{Int}(length(x))

to_axes(x::Tuple{Vararg{Union{Integer,AbstractUnitRange{<:Integer}}}}) = map(to_axis, x)
to_axes(x::Tuple{Vararg{AbstractUnitRange{Int}}}) = x

to_size(x::Dims) = x
to_size(x::Tuple{Vararg{Integer}}) = map(Int, x)

eltype_in(A::AbstractArray, ::Type{T}) where {T} = eltype_in(typeof(A), T)
eltype_in(::Type{<:AbstractArray{S}}, ::Type{T}) where {T,S<:T} = true
eltype_in(::Type{<:AbstractArray{S}}, ::Type{T}) where {T,S} = promote_type(S, T) <: T

"""
    LazyAlgebra.is_subtype(A, B)

yields whether type `A` is a subtype of type `B`. By default, this yields
`A <: B` but this method may be extended to implement other behavior for
specific types.

"""
is_subtype(::Type{A}, ::Type{B}) where {A,B} = A <: B

"""
    LazyAlgebra.concrete_float(T)

yields a concrete floating-point type based on type `T`. Only the bare
numerical type of `T` may be changed. Units, if any, are preserved. If `T` is
real, the result is real; if `T` is complex, the result is complex.

"""
@inline function concrete_float(::Type{T}) where {T}
    F = float(T)
    return isconcretetype(F) ? F : convert_bare_type(Float64, T)
end

"""
    LazyAlgebra.is_integer(x)

yields whether the bare type of `x` is integer. `x` may be a number or a number type.

"""
is_integer(x) = is_integer(typeof(x))
is_integer(::Type{T}) where {T<:Number} = bare_type(T) <: Integer
@noinline is_integer(::Type{T}) where {T} = throw(ArgumentError("`is_integer($T)` not implmented"))

"""
    LazyAlgebra.is_real(x)

yields whether the bare type of `x` is real. `x` may be a number or a number type.

"""
is_real(x) = is_real(typeof(x))
is_real(::Type{T}) where {T<:Number} = bare_type(T) <: Real
@noinline is_real(::Type{T}) where {T} = throw(ArgumentError("`is_real($T)` not implmented"))

"""
    LazyAlgebra.is_complex(x)

yields whether the bare type of `x` is complex. `x` may be a number or a number type.

"""
is_complex(x) = is_complex(typeof(x))
is_complex(::Type{T}) where {T<:Number} = bare_type(T) <: Complex
@noinline is_complex(::Type{T}) where {T} = throw(ArgumentError("`is_complex($T)` not implmented"))

abstract type NumericSet end
struct IntegerSet     <: NumericSet end # FIXME NumericIntegers
struct RealSet        <: NumericSet end # FIXME NumericReals
struct ComplexSet     <: NumericSet end # FIXME NumericComplexes
struct RealComplexSet <: NumericSet end
const ℤ = IntegerSet()
const ℝ = RealSet()
const ℂ = ComplexSet()
const 𝕂 = RealComplexSet()
Base.in(x, ::IntegerSet) = is_integer(x)
Base.in(x, ::RealSet) = is_real(x)
Base.in(x, ::ComplexSet) = is_complex(x)
Base.in(x, ::RealComplexSet) = is_real(x) || is_complex(x)

"""
    LazyAlgebra.concat!(A, B) -> A

concatenates `A` and `B` into the vector `A`.

"""
function concat!(A::AbstractVector, B::AbstractVector)
    i = lastindex(A)
    resize!(A, length(A) + length(B))
    @inbounds for x in B
        A[i += 1] = x
    end
    return A
end
function concat!(A::AbstractVector, B)
    resize!(A, length(A) + 1)
    @inbounds A[lastindex(A)] = B
    return A
end

"""
    LazyAlgebra.concat([T,] A, B)

concatenates `A` and `B` into a vector whose element type is `T`. `A` and `B`
may be (abstract) vectors, otherwise they are assumed to be single entries in
the result. Element type `T` is automatically guessed for arguments if not
specified.

"""
concat(A::AbstractVector, B::AbstractVector) =
    concat(promote_type(eltype(A), eltype(B)), A, B)
concat(A, B::AbstractVector) =
    concat(promote_type(typeof(A), eltype(B)), A, B)
concat(A::AbstractVector, B) =
    concat(promote_type(eltype(A), typeof(B)), A, B)
concat(A, B) =
    concat(promote_type(typeof(A), typeof(B)), A, B)

function concat(::Type{T}, A::AbstractVector, B::AbstractVector) where {T}
    dest = Vector{T}(undef, length(A) + length(B))
    i = firstindex(dest)
    @inbounds for x in A
        dest[i] = x
        i += 1
    end
    @inbounds for x in B
        dest[i] = x
        i += 1
    end
    return dest
end

function concat(::Type{T}, A, B::AbstractVector) where {T}
    dest = Vector{T}(undef, 1 + length(B))
    i = firstindex(dest)
    @inbounds dest[i] = A
    @inbounds for x in B
        i += 1
        dest[i] = x
    end
    return dest
end

function concat(::Type{T}, A::AbstractVector, B) where {T}
    dest = Vector{T}(undef, length(A) + 1)
    i = firstindex(dest)
    @inbounds for x in A
        dest[i] = x
        i += 1
    end
    @inbounds dest[i] = B
    return dest
end

function concat(::Type{T}, A, B) where {T}
    dest = Vector{T}(undef, 2)
    i = firstindex(dest)
    @inbounds dest[i] = A
    @inbounds dest[i + 1] = B
    return dest
end

"""
    LazyAlgebra.map_with_eltype(T, f, A) -> B

yields a vector `B` with elements of type `T` set with `f.(A)`.

"""
function map_with_eltype(::Type{T}, f, A::AbstractVector) where {T}
    B = similar(A, T)
    @inbounds for i in eachindex(A, B)
        B[i] = f(A[i])
    end
    return B
end
