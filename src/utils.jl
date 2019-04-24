#
# utils.jl -
#
# General purpose methods.
#
#-------------------------------------------------------------------------------
#
# This file is part of LazyAlgebra (https://github.com/emmt/LazyAlgebra.jl)
# released under the MIT "Expat" license.
#
# Copyright (c) 2017-2019 Éric Thiébaut.
#

"""
```julia
StorageType(A)
```

yields the type of storage of the elements of argument `A`.  If `A` is a *flat*
array, that is an array with contiguous elements in column-major order and
first element at index 1, the singleton `FlatStorage()` is returned; otherwise,
the singleton `AnyStorage()` is returned.

This method can be extended for custom array types to quickly return the
correct answer.

See also [`isflatarray`](@ref), [`flatarray`](@ref).

"""
StorageType(::Array) = FlatStorage()
StorageType() = AnyStorage()
StorageType(::Any) = AnyStorage()
function StorageType(A::StridedVector)::StorageType
    (first(axes(A,1)) == 1 && stride(A,1) == 1) ? FlatStorage() : AnyStorage()
end
function StorageType(A::StridedMatrix)
    inds, dims, stds = axes(A), size(A), strides(A)
    (first(inds[1]) == 1 && stds[1] == 1 &&
     first(inds[2]) == 1 && stds[2] == dims[1]) ?
     FlatStorage() : AnyStorage()
end
function StorageType(A::StridedArray{T,3})::StorageType where {T}
    inds, dims, stds = axes(A), size(A), strides(A)
    (first(inds[1]) == 1 && stds[1] == 1 &&
     first(inds[2]) == 1 && stds[2] == dims[1] &&
     first(inds[3]) == 1 && stds[3] == dims[1]*dims[2]) ?
     FlatStorage() : AnyStorage()
end
function StorageType(A::StridedArray{T,4})::StorageType where {T}
    inds, dims, stds = axes(A), size(A), strides(A)
    (first(inds[1]) == 1 && stds[1] == 1 &&
     first(inds[2]) == 1 && stds[2] == dims[1] &&
     first(inds[3]) == 1 && stds[3] == dims[1]*dims[2] &&
     first(inds[4]) == 1 && stds[4] == dims[1]*dims[2]*dims[3]) ?
     FlatStorage() : AnyStorage()
end
function StorageType(A::StridedArray{T,N})::StorageType where {T,N}
    inds, dims, stds = axes(A), size(A), strides(A)
    n = 1
    @inbounds for d in 1:N
        if first(inds[d]) != 1 || stds[d] != n
            return AnyStorage()
        end
        n *= dims[d]
    end
    return FlatStorage()
end

@doc @doc(StorageType) FlatStorage
@doc @doc(StorageType) AnyStorage


"""
```julia
isflatarray(A) -> boolean
```

yields whether array `A` can be indexed as a *flat* array, that is an array
with contiguous elements in column-major order and first element at index 1.
This also means that `A` has 1-based indices along all its dimensions.

Several arguments can be checked in a single call:

```julia
isflatarray(A, B, C, ...)
```

is the same as:

```julia
isflatarray(A) && isflatarray(B) && isflatarray(C) && ...
```

See also [`StorageType`](@ref), [`flatarray`](@ref),
[`has_standard_indexing`](@ref).

"""
isflatarray(::Array) = true
isflatarray(A::AbstractArray) = (StorageType(A) === FlatStorage())
isflatarray() = false
isflatarray(::Any) = false
isflatarray(args...) = allof(isflatarray, args...)
#
# Above version could be:
#
#     isflatarray(args...) = all(isflatarray, args)
#
# but using `all(isflatarray, A, x, y)` for `A`, `x` and `y` flat arrays of
# sizes (3,4,5,6), (5,6) and (3,4) takes 9.0ns (with Julia 1.0, 29.1ns with
# Julia 0.6) while using `allof` takes 0.02ns (i.e. is eliminated by the
# compiler).
#

"""
```julia
has_standard_indexing(A)
has_standard_indexing(A, B, ...)
```

Return `true` if the indices of `A` start with 1 along all axes.  If multiple
arguments are passed, equivalent to `has_standard_indexing(A) && has_standard_indexing(B) &&
...`.

Opposite of `Base.has_offset_axes` which is not available in version of Julia
older than 0.7.

"""
has_standard_indexing(arg) = allof(x -> first(x) == 1, axes(arg)...)
has_standard_indexing(args...) = allof(has_standard_indexing, args...)

"""
```julia
flatarray([T=eltype(A),] A)
```

lazily yields a *flat* array based on `A`, that is an array with contiguous
elements in column-major order and first element at index 1.  Optional argument
`T` is to specify the element type of the result.  Argument `A` is returned if
it is already a flat array with the requested element type; otherwise,
[`convert`](@ref) is called to produce the result (an `Array{T}` in that case).

Similarly:

```julia
flatvector([T=eltype(V),] V)
flatmatrix([T=eltype(M),] M)
```

respectively yield a *flat* vector from `V` and a *flat* matrix from `M`.

See also [`isflatarray`](@ref), [`convert`](@ref).
"""
flatarray(A::Array) = A
flatarray(::Type{T}, A::Array{T,N}) where {T,N} = A
flatarray(A::AbstractArray{T,N}) where {T,N} =
    _flatarray(StorageType(A), A)
flatarray(::Type{T}, A::AbstractArray{<:Any,N}) where {T,N} =
    convert(Array{T,N}, A)

_flatarray(::FlatStorage, A::AbstractArray) = A
_flatarray(::StorageType, A::AbstractArray{T,N}) where {T,N} =
    convert(Array{T,N}, A)

flatvector(V::AbstractVector{T}) where {T} = flatarray(T, V)
flatvector(::Type{T}, V::AbstractVector) where {T} = flatarray(T, V)
@doc @doc(flatarray) flatvector

flatmatrix(M::AbstractMatrix{T}) where {T} = flatarray(T, M)
flatmatrix(::Type{T}, M::AbstractMatrix) where {T} = flatarray(T, M)
@doc @doc(flatarray) flatmatrix

"""
Any of the following calls:

```julia
allindices(A)
allindices((n1, n2, ...))
allindices((i1:j1, i2:j2, ...))
allindices(CartesianIndex(i1, i2, ...), CartesianIndex(j1, j2, ...))
allindices(R)
```

yields an instance of `CartesianIndices` or `CartesianRange` (whichever is the
most efficient depending on the version of Julia) for multi-dimensional
indexing of all the elements of array `A`, a multi-dimensional array of
dimensions `(n1,n2,...)`, a multi-dimensional region whose first and last
indices are `(i1,i2,...)` and `(j1,j2,...)` or a Cartesian region defined by
`R`, an instance of `CartesianIndices` or of `CartesianRange`.

"""
allindices(A::AbstractArray) = allindices(axes(A))
allindices(dim::Int) = Base.OneTo(dim)
allindices(dim::Integer) = Base.OneTo(Int(dim))
allindices(rng::AbstractUnitRange{Int}) = rng
allindices(rng::AbstractUnitRange{<:Integer}) = convert(UnitRange{Int}, rng)
@static if isdefined(Base, :CartesianIndices)
    allindices(R::CartesianIndices) = R
    allindices(start::CartesianIndex{N}, stop::CartesianIndex{N}) where {N} =
        CartesianIndices(map((i,j) -> i:j, start.I, stop.I))
    allindices(dims::Tuple{Vararg{Integer}}) =
        CartesianIndices(map(allindices, dims))
    allindices(rngs::NTuple{N,AbstractUnitRange{<:Integer}}) where {N} =
        CartesianIndices(rngs)
else
    allindices(R::CartesianIndices) = allindices(R.indices)
    allindices(R::CartesianRange) = R
    allindices(start::CartesianIndex{N}, stop::CartesianIndex{N}) where {N} =
        CartesianRange(start, stop)
    allindices(dims::NTuple{N,Integer}) where {N} =
        CartesianRange(one(CartesianIndex{N}), CartesianIndex(dims))
    allindices(rngs::NTuple{N,AbstractUnitRange{<:Integer}}) where {N} =
        CartesianRange(CartesianIndex(map(first, rngs)),
                       CartesianIndex(map(last,  rngs)))
end

@deprecate fastrange allindices

"""
```julia
allof(p, args...) -> Bool
```

checks whether predicate `p` returns `true` for all arguments `args...`,
returning `false` as soon as possible (short-circuiting).

```julia
allof(args...) -> Bool
```

checks whether all arguments `args...` are `true`, returning `false` as soon as
possible (short-circuiting).  Arguments can be booleans or arrays of booleans.
The latter are considered as `true` if all their elements are `true` and are
considered as `false` otherwise (if any of their elements are `false`).
Arguments can also be iterables to check whether all their values are `true`.
As a consequence, an empty iterable is considered as `true`.

This method can be much faster than `all(p, args)` or `all(args)` because its
result may be determined at compile time.  However, `missing` values are not
considered as special.

See also: [`all`](@ref), [`anyof`](@ref), [`noneof`](@ref).

"""
allof(p::Function, a) = p(a)::Bool
allof(p::Function, a, b...) = p(a) && allof(p, b...)
allof(a, b...) = allof(a) && allof(b...)
allof(a::Bool) = a
function allof(a::AbstractArray{Bool})
    @inbounds for i in eachindex(a)
        a[i] || return false
    end
    return true
end
function allof(itr)
    for val in itr
        allof(val) || return false
    end
    return true
end

"""
```julia
anyof(p, args...) -> Bool
```

checks whether predicate `p` returns `true` for any argument `args...`,
returning `true` as soon as possible (short-circuiting).

```julia
anyof(args...) -> Bool
```

checks whether all arguments `args...` are `true`, returning `false` as soon as
possible (short-circuiting).  Arguments can be booleans or arrays of booleans.
The latter are considered as `true` if any of their elements are `true` and are
considered as `false` otherwise (if all their elements are `false`).  Arguments
can also be iterables to check whether any of their values are `true`.  As a
consequence, an empty iterable is considered as `false`.

This method can be much faster than `any(p, args)` or `any(args)` because its
result may be determined at compile time.  However, `missing` values are not
considered as special.

To check whether predicate `p` returns `false` for all argument `args...`
or whether all argument `args...` are false, repectively call:

```julia
noneof(p, args...) -> Bool
```

or

```julia
noneof(args...) -> Bool
```

which are the same as `!anyof(p, args...)` and `!anyof(args...)`.

See also: [`any`](@ref), [`allof`](@ref).

"""
anyof(p::Function, a) = p(a)::Bool
anyof(p::Function, a, b...) = p(a) || anyof(p, b...)
anyof(a, b...) = anyof(a) || anyof(b...)
anyof(a::Bool) = a
function anyof(a::AbstractArray{Bool})
    @inbounds for i in eachindex(a)
        a[i] && return true
    end
    return false
end
function anyof(itr)
    for val in itr
        anyof(val) && return true
    end
    return false
end

noneof(args...) = ! anyof(args...)
@doc @doc(anyof) noneof

"""
```julia
reversemap(f, args)
```

applies the function `f` to arguments `args` in reverse order and return the
result.  For now, the arguments `args` must be in the form of a simple tuple
and the result is the tuple: `(f(args[end]),f(args[end-1]),...,f(args[1])`.

Also see: [`map`](@ref), [`ntuple`](@ref).

"""
reversemap(f::Function, args::NTuple{N,Any}) where {N} =
    ntuple(i -> f(args[(N + 1) - i]), Val(N))

"""

```julia
convert_multiplier(λ, T [, S=T])
```

yields multiplier `λ` converted to a suitable type for multiplying array whose
elements have type `T` and for storage in a destination array whose elements
have type `S`.

The following rules are applied:

1. Convert `λ` to the same floating-point precision as `T`.

2. The result is a real if `λ` is a real or both `T` and `S` are real types;
   otherwise (that is if `λ` is complex and at least one of `T` or `S` is a
   complex type), the result is a complex.

Result can be a real if imaginary part of `λ` is zero but this would break the
rule of type-stability at compilation time.

"""
convert_multiplier(λ::Real, ::Type{T}) where {T<:Floats} =
    (isconcretetype(T) ? convert(real(T), λ) : operand_type_not_concrete(T))
convert_multiplier(λ::Complex, ::Type{T}) where {T<:Floats} =
    # Call to `convert` will clash if `T` is real and `imag(λ)` is non-zero
    # (this is what we want).
    (isconcretetype(T) ? convert(T, λ) : operand_type_not_concrete(T))

convert_multiplier(λ::Real, ::Type{T}, ::Type{<:Number}) where {T<:Floats} =
    (isconcretetype(T) ? convert(real(T), λ) : operand_type_not_concrete(T))
convert_multiplier(λ::Complex, ::Type{T}, ::Type{<:Real}) where {T<:Reals} =
    # Call to `convert` will clash if `imag(λ)` is non-zero (this is what we
    # want).
    (isconcretetype(T) ? convert(T, λ) : operand_type_not_concrete(T))
convert_multiplier(λ::Complex, ::Type{T}, ::Type{<:Complex}) where {T<:Floats} =
    (isconcretetype(T) ? convert(Complex{real(T)}, λ) : operand_type_not_concrete(T))

convert_multiplier(λ::L, ::Type{T}) where {L<:Number,T} =
    (isconcretetype(T) ? unsupported_multiplier_conversion(L, T, T) :
     operand_type_not_concrete(T))

convert_multiplier(λ::L, ::Type{T}, ::Type{S}) where {L<:Number,T,S} =
    (isconcretetype(T) ? unsupported_multiplier_conversion(L, T, S) :
     operand_type_not_concrete(T))

@noinline unsupported_multiplier_conversion(::Type{L}, ::Type{O}, ::Type{S}) where {L<:Number,O,S} =
    error("unsupported conversion of multiplier with type $L for operand with element type $O and storage with element type $S")

# Note: the only direct sub-types of `Number` are abstract types `Real` and
# `Complex`.
@noinline operand_type_not_concrete(::Type{T}) where {T} =
    error("operand type $T is not a concrete type")
