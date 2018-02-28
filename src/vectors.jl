#
# vectors.jl -
#
# Implement basic operations for *vectors*.  Here arrays of any rank are
# considered as *vectors*, the only requirements are that, when combining
# *vectors*, they have the same type and dimensions.  These methods are
# intended to be used for numerical optimization and thus, for now,
# elements must be real (not complex).
#
#-------------------------------------------------------------------------------
#
# This file is part of the MockAlgebra package released under the MIT "Expat"
# license.
#
# Copyright (c) 2017-2018 Éric Thiébaut.
#

"""
```julia
vnorm2([T,] v)
```

yields the Euclidean (L2) norm of `v`.  The floating point type of the result
can be imposed by optional argument `T`.  Also see [`vnorm1`](@ref) and
[`vnorminf`](@ref).

"""
function vnorm2(::Type{T},
                v::AbstractArray{<:Real,N})::T where {T<:AbstractFloat,N}
    local s::T = zero(T)
    @inbounds @simd for i in eachindex(v)
        s += v[i]*v[i]
    end
    return sqrt(s)
end

vnorm2(v::AbstractArray{T,N}) where {T<:AbstractFloat,N} = vnorm2(T, v)
vnorm2(v::AbstractArray{T,N}) where {T<:Real,N} = vnorm2(float(T), x)
vnorm2(x) = sqrt(vdot(x, x))
vnorm2(::Type{T}, x) where {T<:AbstractFloat} = sqrt(vdot(T, x, x))

"""
```julia
vnorm1([T,] v)
```

yields the L1 norm of `v`, that is the sum of the absolute values of its
elements.  The floating point type of the result can be imposed by optional
argument `T`.  Also see [`vnorm2`](@ref) and [`vnorminf`](@ref).

"""
function vnorm1(::Type{T},
                v::AbstractArray{<:Real,N})::T where {T<:AbstractFloat,N}
    local s::T = zero(T)
    @inbounds @simd for i in eachindex(v)
        s += abs(v[i])
    end
    return s
end

vnorm1(v::AbstractArray{T,N}) where {T<:AbstractFloat,N} = vnorm1(T, v)
vnorm1(v::AbstractArray{T,N}) where {T<:Real,N} = vnorm1(float(T), x)


"""
```julia
vnorminf([T,] v)
```

yields the infinite norm of `v`, that is the maximum absolute value of its
elements.  The floating point type of the result can be imposed by optional
argument `T`.  Also see [`vnorm1`](@ref) and [`vnorm2`](@ref).

"""
function vnorminf(::Type{T},
                  v::AbstractArray{<:Real,N})::T where {T<:AbstractFloat,N}
    local s::T = zero(T)
    @inbounds @simd for i in eachindex(v)
        s = max(s, abs(v[i]))
    end
    return s
end
function vnorminf(::Type{T},
                  v::AbstractArray{<:Unsigned,N})::T where {T<:AbstractFloat,N}
    local s::T = zero(T)
    @inbounds @simd for i in eachindex(v)
        s = max(s, v[i])
    end
    return s
end

vnorminf(v::AbstractArray{T,N}) where {T<:AbstractFloat,N} = vnorminf(T, v)
vnorminf(v::AbstractArray{T,N}) where {T<:Real,N} = vnorminf(float(T), x)

#------------------------------------------------------------------------------

"""
```julia
vcreate(x)
```

yields a new variable instance similar to `x`.  If `x` is an array, the
element type of the result is a floating-point type.

Also see [`similar`](@ref).

"""
vcreate(x::AbstractArray{T,N}) where {T<:AbstractFloat,N} = similar(x, T)
vcreate(x::AbstractArray{T,N}) where {T,N} = similar(x, float(T))

#------------------------------------------------------------------------------

"""
```julia
vcopy!(dst, src) -> dst
```

copies the contents of `src` into `dst` and returns `dst`.  This function
checks that the copy makes sense (for instance, the `copy!` operation does not
check that the source and destination have the same dimensions).

Also see [`copy!`](@ref), [`vcopy`](@ref), [`vswap!`](@ref).

"""
function vcopy!(dst::AbstractArray{Td,N},
                src::AbstractArray{Ts,N}) where {Td, Ts, N}
    @assert indices(dst) == indices(src)
    copy!(dst, src)
end

"""
```julia
vcopy(x)
```

yields a fresh copy of the *vector* `x`.  If `x` is is an array, the element
type of the result is a floating-point type.

Also see [`copy`](@ref), [`vcopy!`](@ref), [`vcreate!`](@ref).

"""
vcopy(x) = vcopy!(vcreate(x), x)

"""
```julia
vswap!(x, y)
```

exchanges the contents of `x` and `y` (which must have the same type and size
if they are arrays).

Also see [`vcopy!`](@ref).

"""
vswap!(x::DenseArray{T,N}, y::DenseArray{T,N}) where {T,N} =
    pointer(x) != pointer(y) && _vswap!(x, y)

vswap!(x::AbstractArray{T,N}, y::AbstractArray{T,N}) where {T,N} =
    _vswap!(x, y)

function _vswap!(x::AbstractArray{T,N}, y::AbstractArray{T,N}) where {T,N}
    if indices(x) != indices(y)
        throw(DimensionMismatch("`x` and `y` must have the same indices"))
    end
    @inbounds @simd for i in eachindex(x, y)
        temp = x[i]
        x[i] = y[i]
        y[i] = temp
    end
    return nothing
end

#------------------------------------------------------------------------------

"""
```julia
vfill!(x, α) -> x
```

sets all elements of `x` with the scalar value `α` and return `x`.

Also see [`vzero!`](@ref).

"""
function vfill!(x::AbstractArray{T,N}, alpha::T) where {T<:AbstractFloat,N}
    @inbounds @simd for i in eachindex(x)
        x[i] = alpha
    end
    return x
end

vfill!(x::DenseArray{T,N}, alpha::Real) where {T<:AbstractFloat,N} =
    vfill!(x, T(alpha))

"""

```julia
vzero!(A) -> A
```

fills `A` with zeros and returns it.

Also see [`vfill!`](@ref).

"""
vzero!(A::AbstractArray{T,N}) where {T,N} = fill!(A, zero(T))

#------------------------------------------------------------------------------

"""
```julia
vscale!(dst, α, src) -> dst
```

overwrites `dst` with `α*src` and returns `dst`.  Computations are done at the
numerical precision of `src`.  The source argument may be omitted to perform
*in-place* scaling:

```julia
vscale!(x, α) -> x
```

which overwrites `x` with `α*x` and returns `x`.

Also see [`vscale`](@ref).

"""
function vscale!(dst::AbstractArray{Td,N},
                 alpha::Real,
                 src::AbstractArray{Ts,N}) where {Td<:AbstractFloat,
                                                  Ts<:AbstractFloat,N}
    if indices(dst) != indices(src)
        throw(DimensionMismatch("`dst` and `src` must have the same indices"))
    end
    if alpha == zero(alpha)
        vzero!(dst)
    elseif alpha == -one(alpha)
        @inbounds @simd for i in eachindex(dst, src)
            dst[i] = -src[i]
        end
    elseif alpha == one(alpha)
        copy!(dst, src)
    else
        const α = convert(Ts, alpha)
        @inbounds @simd for i in eachindex(dst, src)
            dst[i] = α*src[i]
        end
    end
    return dst
end

function vscale!(x::AbstractArray{T,N}, alpha::Real) where {T<:AbstractFloat,N}
    if alpha == zero(alpha)
        vzero!(x)
    elseif alpha == -one(alpha)
        @inbounds @simd for i in eachindex(x)
            x[i] = -x[i]
        end
    elseif alpha != one(alpha)
        const α = T(alpha)
        @inbounds @simd for i in eachindex(x)
            x[i] *= α
        end
    end
    return x
end

# In place scaling for other *vector* types.
vscale!(x, alpha::Real) = vscale!(x, alpha, x)

"""
```julia
vscale(α, x)
```

yields a new *vector* whose elements are those of `x` multiplied by the scalar
`α`.

Also see [`vscale!`](@ref), [`vcopy`](@ref).

"""
vscale(alpha::Real, x) =
    alpha == one(alpha) ? vcopy(x) : vscale!(vcreate(x), alpha, x)

#------------------------------------------------------------------------------

"""
### Elementwise multiplication

```julia
vproduct(x, y) -> z
```

yields the elementwise multiplication of `x` by `y`.  To avoid allocating the
result, the destination array `dst` can be specified with the in-place version
of the method:

```julia
vproduct!(x, y) -> x
```

which overwrites `x` with the elementwise multiplication of `x` by `y`.

Another destination than `x` can be provided:

```julia
vproduct!(dst, [sel,] x, y) -> dst
```

where `sel` is an optional selection of indices to which apply the operation.


"""
vproduct(x::V, y::V) where {V} = vproduct!(vcreate(x), x, y)

vproduct!(dst::V, src::V) where {V} = vproduct!(dst, dst, src)

@doc @doc(vproduct) vproduct!

function vproduct!(dst::AbstractArray{<:AbstractFloat,N},
                   x::AbstractArray{<:AbstractFloat,N},
                   y::AbstractArray{<:AbstractFloat,N}) where {N}
    @assert indices(dst) == indices(x) == indices(y)
    @inbounds @simd for i in eachindex(dst, x, y)
        dst[i] = x[i]*y[i]
    end
    return dst
end

function vproduct!(dst::DenseArray{<:AbstractFloat,N},
                   sel::AbstractVector{Int},
                   x::DenseArray{<:AbstractFloat,N},
                   y::DenseArray{<:AbstractFloat,N}) where {N}
    @assert size(dst) == size(x) == size(y)
    const n = length(dst)
    @inbounds @simd for i in eachindex(sel)
        j = sel[i]
        1 ≤ j ≤ n || throw(BoundsError())
        dst[j] = x[j]*y[j]
    end
    return dst
end

#------------------------------------------------------------------------------

"""
### Linear combination of arrays

```julia
vcombine(α, x [, β, y]) -> dst
```

yields the linear combination `dst = α*x` or `dst = α*x + β*y`.

To avoid allocating the result, the destination array `dst` can be specified
with the in-place version of the method:

```julia
vcombine!(dst, α, x [, β, y]) -> dst
```

The code is optimized for some specific values of the coefficients `α` and `β`.
For instance, if `α` (resp. `β`) is zero, then the contents of `x` (resp. `y`)
is not used.

The source(s) and the destination can be the same.  For instance, the two
following lines of code produce the same result:

```julia
vcombine!(dst, 1, dst, α, x)
vupdate!(dst, α, x)
```

and the following statements also yield the same result:

 ```julia
vcombine!(dst, α, x)
vscale!(dst, α, x)
```

"""
vcombine(alpha::Real, x) = vscale(alpha, x)

vcombine(alpha::Real, x::V, beta::Real, y::V) where {V} =
    vcombine!(vcreate(x), alpha, x, beta, y)

vcombine!(dst::V, alpha::Real, x::V) where {V} = vscale!(dst, alpha, x)

@doc @doc(vcombine) vcombine!

function vcombine!(dst::AbstractArray{<:AbstractFloat,N},
                   alpha::Real,
                   x::AbstractArray{Tx,N},
                   beta::Real,
                   y::AbstractArray{Ty,N}) where {Tx<:AbstractFloat,
                                                  Ty<:AbstractFloat,N}
    @assert indices(dst) == indices(x) == indices(y)
    if alpha == zero(alpha)
        vscale!(dst, beta, y)
    elseif beta == zero(beta)
        vscale!(dst, alpha, x)
    elseif alpha == one(alpha)
        if beta == one(beta)
            @inbounds @simd for i in eachindex(dst, x, y)
                dst[i] = x[i] + y[i]
            end
        elseif beta == -one(beta)
            @inbounds @simd for i in eachindex(dst, x, y)
                dst[i] = x[i] - y[i]
            end
        else
            const β = Ty(beta)
            @inbounds @simd for i in eachindex(dst, x, y)
                dst[i] = x[i] + β*y[i]
            end
        end
    elseif alpha == -one(alpha)
        if beta == one(beta)
            @inbounds @simd for i in eachindex(dst, x, y)
                dst[i] = y[i] - x[i]
            end
        elseif beta == -one(beta)
            @inbounds @simd for i in eachindex(dst, x, y)
                dst[i] = -x[i] - y[i]
            end
        else
            const β = Ty(beta)
            @inbounds @simd for i in eachindex(dst, x, y)
                dst[i] = β*y[i] - x[i]
            end
        end
    else
        const α = Tx(alpha)
        if beta == one(beta)
            @inbounds @simd for i in eachindex(dst, x, y)
                dst[i] = α*x[i] + y[i]
            end
        elseif beta == -one(beta)
            @inbounds @simd for i in eachindex(dst, x, y)
                dst[i] = α*x[i] - y[i]
            end
        else
            const β = Ty(beta)
            @inbounds @simd for i in eachindex(dst, x, y)
                dst[i] = α*x[i] + β*y[i]
            end
        end
    end
    return dst
end

#--- INNER PRODUCT ------------------------------------------------------------

"""
### Inner product

```julia
vdot([T,] [w,] x, y)
```

yields the inner product of `x` and `y`; that is, the sum of `x[i]*y[i]` or, if
`w` is specified, the sum of `w[i]*x[i]*y[i]`, for all indices `i`.  Optional
argument `T` is the floating point type of the result.

Another possibility is:

```julia
vdot([T,] sel, x, y)
```

with `sel` a selection of indices to restrict the computation of the inner
product to some selected elements.  This yields the sum of `x[i]*y[i]` for all
`i ∈ sel`.

If the arguments are complex, they are considered as vectors of pairs of reals
and the result is:

```julia
vdot(x, y) = x[1].re*y[1].re + x[1].im*y[1].im +
             x[2].re*y[2].re + x[2].im*y[2].im + ...
```

"""
function vdot(::Type{T},
              x::AbstractArray{<:Real,N},
              y::AbstractArray{<:Real,N}) where {T<:AbstractFloat,N}
    return _vdot(T, x, y)
end

function vdot(x::AbstractArray{Tx,N},
              y::AbstractArray{Ty,N}) where {Tx<:Real,Ty<:Real,N}
    return vdot(float(promote_type(Tx, Ty)), x, y)
end

function vdot(::Type{T},
              x::AbstractArray{Complex{Tx},N},
              y::AbstractArray{Complex{Ty},N}) where {T<:AbstractFloat,
                                                      Tx<:Real,Ty<:Real,N}
    return _vdot(T, x, y)
end

# Note that we cannot use union here because we want that both be real or both
# be complex.
function vdot(x::AbstractArray{Complex{Tx},N},
              y::AbstractArray{Complex{Ty},N}) where {Tx<:Real,Ty<:Real,N}
    return vdot(float(promote_type(Tx, Ty)), x, y)
end

function vdot(::Type{T},
              w::AbstractArray{<:Real,N},
              x::AbstractArray{<:Real,N},
              y::AbstractArray{<:Real,N})::T where {T<:AbstractFloat,N}
    if !(indices(w) == indices(x) == indices(y))
        throw(DimensionMismatch("`w`, `x` and `y` must have the same indices"))
    end
    local s::T = zero(T)
    @inbounds @simd for i in eachindex(w, x, y)
        s += w[i]*x[i]*y[i]
    end
    return s
end

function vdot(w::AbstractArray{Tw,N},
              x::AbstractArray{Tx,N},
              y::AbstractArray{Ty,N}) where {Tw<:Real,Tx<:Real,Ty<:Real,N}
    return vdot(float(promote_type(Tw, Tx, Ty)), w, x, y)
end

function vdot(::Type{T},
              w::AbstractArray{<:Real,N},
              x::AbstractArray{Complex{Tx},N},
              y::AbstractArray{Complex{Ty},N})::T where {T<:AbstractFloat,
                                                         Tx<:Real,Ty<:Real,N}
    if !(indices(w) == indices(x) == indices(y))
        throw(DimensionMismatch("`w`, `x` and `y` must have the same indices"))
    end
    local s::T = zero(T)
    @inbounds @simd for i in eachindex(w, x, y)
        s += (x[i].re*y[i].re + x[i].im*y[i].im)*w[i]
    end
    return s
end

function vdot(w::AbstractArray{Tw,N},
              x::AbstractArray{Complex{Tx},N},
              y::AbstractArray{Complex{Ty},N}) where {Tw<:Real,
                                                      Tx<:Real,Ty<:Real,N}
    return vdot(float(promote_type(Tw, Tx, Ty)), w, x, y)
end

# FIXME: extend to other types of arrays and Cartesian indices.

function vdot(::Type{T},
              sel::AbstractVector{Int},
              x::DenseArray{<:Real,N},
              y::DenseArray{<:Real,N})::T where {T<:AbstractFloat,N}
    if size(y) != size(x)
        throw(DimensionMismatch("`x` and `y` must have same dimensions"))
    end
    local s::T = zero(T)
    const n = length(x)
    @inbounds @simd for i in eachindex(sel)
        j = sel[i]
        1 ≤ j ≤ n || throw(BoundsError())
        s += x[j]*y[j]
    end
    return s
end

function vdot(sel::AbstractVector{Int},
              x::DenseArray{Tx,N},
              y::DenseArray{Ty,N}) where {Tx<:Real,Ty<:Real,N}
    return vdot(float(promote_type(Tx, Ty)), sel, x, y)
end

function vdot(::Type{T},
              sel::AbstractVector{Int},
              x::AbstractArray{Complex{Tx},N},
              y::AbstractArray{Complex{Ty},N})::T where {T<:AbstractFloat,
                                                         Tx<:Real,Ty<:Real,N}
    if size(y) != size(x)
        throw(DimensionMismatch("`x` and `y` must have same dimensions"))
    end
    local s::T = zero(T)
    const n = length(x)
    @inbounds @simd for i in eachindex(sel)
        j = sel[i]
        1 ≤ j ≤ n || throw(BoundsError())
        s += x[j].re*y[j].re + x[j].im*y[j].im
    end
    return s
end

function vdot(sel::AbstractVector{Int},
              x::DenseArray{Complex{Tx},N},
              y::DenseArray{Complex{Ty},N}) where {Tx<:Real,Ty<:Real,N}
    return vdot(float(promote_type(Tx, Ty)), sel, x, y)
end


# Pure Julia implementations (there may exist a faster BLAS counterpart).

function _vdot(::Type{T},
               x::AbstractArray{<:Real,N},
               y::AbstractArray{<:Real,N})::T where {T<:AbstractFloat, N}
    if indices(x) != indices(y)
        throw(DimensionMismatch("`x` and `y` must have the same indices"))
    end
    local s::T = zero(T)
    @inbounds @simd for i in eachindex(x, y)
        s += x[i]*y[i]
    end
    return s
end

function _vdot(x::AbstractArray{Tx,N},
               y::AbstractArray{Ty,N}) where {Tx<:Real, Ty<:Real, N}
    return _vdot(float(promote_type(Tx, Ty)), x, y)
end

function _vdot(::Type{T},
               x::AbstractArray{Complex{Tx},N},
               y::AbstractArray{Complex{Ty},N})::T where {T<:AbstractFloat,
                                                          Tx<:Real,Ty<:Real,N}
    if indices(x) != indices(y)
        throw(DimensionMismatch("`x` and `y` must have the same indices"))
    end
    local s::T = zero(T)
    @inbounds @simd for i in eachindex(x, y)
        s += x[i].re*y[i].re + x[i].im*y[i].im
    end
    return s
end

function _vdot(x::AbstractArray{Complex{Tx},N},
               y::AbstractArray{Complex{Ty},N}) where {Tx<:Real, Ty<:Real, N}
    return _vdot(float(promote_type(Tx, Ty)), x, y)
end

#--- VECTOR UPDATE ------------------------------------------------------------

"""
```julia
vupdate!(y, [sel,] α, x) -> y
```

overwrites `y` with `α*x + y` and returns `y`.  The code is optimized for some
specific values of the multiplier `α`.  For instance, if `α` is zero, then `y`
is left unchanged without using `x`.  Computations are performed at the
numerical precision of `x`.

Optional argument `sel` is a selection of indices to which apply the operation.
Note that if an index is repeated, the operation will be performed several
times at this location.

"""
function vupdate!(y::AbstractArray{Ty,N},
                  alpha::Real,
                  x::AbstractArray{Tx,N}) where {Ty<:AbstractFloat,
                                                 Tx<:AbstractFloat,N}
    return _vupdate!(y, convert(Tx, alpha), x)
end

# Pure Julia implementation.
function _vupdate!(y::AbstractArray{Ty,N},
                   alpha::Real,
                   x::AbstractArray{Tx,N}) where {Ty<:AbstractFloat,
                                                  Tx<:AbstractFloat,N}
    if indices(x) != indices(y)
        throw(DimensionMismatch("`x` and `y` must have the same indices"))
    end
    if alpha == one(alpha)
        @inbounds @simd for i in eachindex(y, x)
            y[i] += x[i]
        end
    elseif alpha == -one(alpha)
        @inbounds @simd for i in eachindex(y, x)
            y[i] -= x[i]
        end
    elseif alpha != zero(alpha)
        const α = Tx(alpha)
        @inbounds @simd for i in eachindex(y, x)
            y[i] += α*x[i]
        end
    end
    return y
end

function vupdate!(y::DenseArray{Ty,N},
                  sel::AbstractVector{Int},
                  alpha::Real,
                  x::DenseArray{Tx,N}) where {Ty<:AbstractFloat,
                                              Tx<:AbstractFloat,N}
    if size(x) != size(y)
        throw(DimensionMismatch("`x` and `y` must have the same dimensions"))
    end
    if alpha == one(alpha)
        const n = length(x)
        @inbounds @simd for i in eachindex(sel)
            j = sel[i]
            1 ≤ j ≤ n || throw(BoundsError())
            y[j] += x[j]
        end
    elseif alpha == -one(alpha)
        const n = length(x)
        @inbounds @simd for i in eachindex(sel)
            j = sel[i]
            1 ≤ j ≤ n || throw(BoundsError())
            y[j] -= x[j]
        end
    elseif alpha != zero(alpha)
        const n = length(x)
        const α = Tx(alpha)
        @inbounds @simd for i in eachindex(sel)
            j = sel[i]
            1 ≤ j ≤ n || throw(BoundsError())
            y[j] += α*x[j]
        end
    end
    return y
end

#--- MATRIX-VECTOR PRODUCT ----------------------------------------------------

function apply(A::AbstractArray{<:Real},
               x::AbstractArray{<:Real})
    return apply(Direct, A, x)
end

function apply(::Type{Op},
               A::AbstractArray{<:Real},
               x::AbstractArray{<:Real}) where {Op<:Operations}
    return apply!(vcreate(Op, A, x), Op, A, x)
end

# By default, use pure Julia code for the generalized matrix-vector product.
function apply!(y::AbstractArray{<:Real},
                ::Type{Op},
                A::AbstractArray{<:Real},
                x::AbstractArray{<:Real}) where {Op<:Union{Adjoint,Direct}}
    return _apply!(y, Op, A, x)
end

function vcreate(::Type{Direct},
                 A::AbstractArray{Ta,Na},
                 x::AbstractArray{Tx,Nx}) where {Ta<:AbstractFloat, Na,
                                                 Tx<:AbstractFloat, Nx}
    inds = indices(A)
    Ny = Na - Nx
    if Nx ≥ Na || indices(x) != inds[Ny+1:end]
        throw(DimensionMismatch("the dimensions of `x` do not match the trailing dimensions of `A`"))
    end
    Ty = promote_type(Ta, Tx)
    return similar(Array{Ty}, inds[1:Ny])
end

function vcreate(::Type{Adjoint},
                 A::AbstractArray{Ta,Na},
                 x::AbstractArray{Tx,Nx}) where {Ta<:AbstractFloat, Na,
                                                 Tx<:AbstractFloat, Nx}
    inds = indices(A)
    Ny = Na - Nx
    if Nx ≥ Na || indices(x) != inds[1:Nx]
        throw(DimensionMismatch("the dimensions of `x` do not match the leading dimensions of `A`"))
    end
    Ty = promote_type(Ta, Tx)
    return similar(Array{Ty}, inds[Ny+1:end])
end


# Pure Julia code implementations.

function _apply!(y::AbstractArray{<:Real},
                 ::Type{Direct},
                 A::AbstractArray{<:Real},
                 x::AbstractArray{<:Real},
                 overwrite::Bool = true)
    if indices(A) != (indices(y)..., indices(x)...)
        throw(DimensionMismatch("`x` and/or `y` have indices incompatible with `A`"))
    end
    # Loop through the coefficients of A assuming column-major storage order.
    overwrite && vzero!(y)
    I, J = CartesianRange(indices(y)), CartesianRange(indices(x))
    @inbounds for j in J
        @simd for i in I
            y[i] += A[i,j]*x[j]
        end
    end
    return y
end

function _apply!(y::AbstractArray{Ty},
                 ::Type{Adjoint},
                 A::AbstractArray{Ta},
                 x::AbstractArray{Tx}) where {Ta<:Real, Tx<:Real, Ty<:Real}
    return _apply!(promote_type(Ty, Ta, Tx), y, Adjoint, A, x)
end

function _apply!(::Type{T},
                 y::AbstractArray{<:Real},
                 ::Type{Adjoint},
                 A::AbstractArray{<:Real},
                 x::AbstractArray{<:Real}) where {T<:Real}
    if indices(A) != (indices(x)..., indices(y)...)
        throw(DimensionMismatch("`x` and/or `y` have indices incompatible with `A`"))
    end
    # Loop through the coefficients of A assuming column-major storage order.
    I, J = CartesianRange(indices(x)), CartesianRange(indices(y))
    @inbounds for j in J
        local s::T = zero(T)
        @simd for i in I
            s += A[i,j]*x[i]
        end
        y[j] = s
    end
    return y
end
