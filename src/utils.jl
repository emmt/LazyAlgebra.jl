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
# Copyright (c) 2017-2018 Éric Thiébaut.
#

"""
```julia
is_flat_array(A) -> boolean
```

yields whether array `A` can be indexed as a *flat* array, that is an array
with contiguous elements and first element at index 1.  This also means that
`A` has 1-based indices along all its dimensions.

Several arguments can be checked in a single call:

```julia
is_flat_array(A, B, C, ...)
```

is the same as:

```julia
is_flat_array(A) && is_flat_array(B) && is_flat_array(C) && ...
```

"""
is_flat_array(A::DenseArray) = true

function is_flat_array(A::AbstractArray{T,N}) where {T,N}
    Base.has_offset_axes(A) && return false
    n = 1
    @inbounds for d in 1:N
        stride(A, d) == n || return false
        n *= size(A, d)
    end
    return true
end

is_flat_array() = false
is_flat_array(::Any) = false
is_flat_array(args...) = allof(is_flat_array, args...)
#
# Above version could be:
#
#     is_flat_array(args...) = all(is_flat_array, args)
#
# but using `all(is_flat_array, A, x, y)` for `A`, `x` and `y` flat arrays of
# sizes (3,4,5,6), (5,6) and (3,4) takes 9.0ns (with Julia 1.0, 29.1ns with
# Julia 0.6) while using `allof` takes 0.02ns (i.e. is eliminated by the
# compiler).
#

"""
```julia
densearray([T=eltype(A),] A)
```

lazyly yields a dense array based on `A`.  Optional argument `T` is to specify
the element type of the result.  Argument `A` is returned if it is already a
dense array with the requested element type; otherwise, [`convert`](@ref) is
called to produce the result.

Similarly:

```julia
densevector([T=eltype(V),] V)
densematrix([T=eltype(M),] M)
```

respectively yield a dense vector from `V` and a dense matrix from `M`.

"""
densearray(A::DenseArray) = A
densearray(::Type{T}, A::DenseArray{T,N}) where {T,N} = A
densearray(A::AbstractArray{T,N}) where {T,N} = densearray(T, A)
densearray(::Type{T}, A::AbstractArray{<:Any,N}) where {T,N} =
    convert(Array{T,N}, A)

densevector(V::AbstractVector{T}) where {T} = densearray(T, V)
densevector(::Type{T}, V::AbstractVector) where {T} = densearray(T, V)
@doc @doc(densearray) densevector

densematrix(M::AbstractMatrix{T}) where {T} = densearray(T, M)
densematrix(::Type{T}, M::AbstractMatrix) where {T} = densearray(T, M)
@doc @doc(densearray) densematrix

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
    import Base: axes
    allindices(R::CartesianIndices) = R
    allindices(start::CartesianIndex{N}, stop::CartesianIndex{N}) where {N} =
        CartesianIndices(map((i,j) -> i:j, start.I, stop.I))
    allindices(dims::Tuple{Vararg{Integer}}) =
        CartesianIndices(map(allindices, dims))
    allindices(rngs::NTuple{N,AbstractUnitRange{<:Integer}}) where {N} =
        CartesianIndices(rngs)
else
    import Base: indices
    const axes = indices
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
function convert_multiplier(λ::Number, ::Type{T}) where {T<:Reals}
    return convert(T, λ)
end

function convert_multiplier(λ::Real, ::Type{Complex{T}}) where {T<:Reals}
    return convert(T, λ)
end

function convert_multiplier(λ::Complex, ::Type{T}) where {T<:Complexes}
    return convert(T, λ)
end

function convert_multiplier(λ::Number, ::Type{T},
                            ::Type{<:Reals}) where {T<:Reals}
    return convert(T, λ)
end

function convert_multiplier(λ::Real, ::Type{Complex{T}},
                            ::Type{<:Union{Reals,Complexes}}) where {T<:Reals}
    return convert(T, λ)
end

function convert_multiplier(λ::Complex, ::Type{T},
                            ::Type{Complexes}) where {T<:Reals}
    return convert(Complex{T}, λ)
end

function convert_multiplier(λ::Complex, ::Type{T},
                            ::Type{Complexes}) where {T<:Complexes}
    return convert(T, λ)
end
