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
# This file is part of the LazyAlgebra package released under the MIT "Expat"
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

for norm in (:vnorm2, :vnorm1, :vnorminf)
    @eval begin
        $norm(v::AbstractArray{T,N}) where {T<:AbstractFloat,N} = $norm(T, v)
        $norm(v::AbstractArray{T,N}) where {T<:Real,N} = $norm(float(T), x)
    end
end

#------------------------------------------------------------------------------

"""
```julia
vcreate(x)
```

yields a new variable instance similar to `x`.  If `x` is an array, the
element type of the result is a floating-point type.

Also see [`similar`](@ref).

"""
vcreate(x::AbstractArray{T,N}) where {T<:Union{Reals,Complexes},N} =
    similar(x, T)

vcreate(x::AbstractArray{T,N}) where {T,N} =
    similar(x, float(T))

#------------------------------------------------------------------------------

"""
```julia
vcopy!(dst, src) -> dst
```

copies the contents of `src` into `dst` and returns `dst`.  This function
checks that the copy makes sense (for instance, for array arguments, the
`copy!` operation does not check that the source and destination have the same
dimensions).

Also see [`copy!`](@ref), [`vcopy`](@ref), [`vswap!`](@ref).

"""
function vcopy!(dst::AbstractArray{Td,N},
                src::AbstractArray{Ts,N}) where {Td, Ts, N}
    indices(dst) == indices(src) ||
        _errdims("`dst` and `src` must have the same indices")
    copy!(dst, src)
end

@inline _errdims(msg::AbstractString) = throw(DimensionMismatch(msg))

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
function vswap!(x::AbstractArray{T,N}, y::AbstractArray{T,N}) where {T,N}
    indices(x) == indices(y) ||
        _errdims("`x` and `y` must have the same indices")
    __mayswap!(x, y)
end

__mayswap!(x::DenseArray{T,N}, y::DenseArray{T,N}) where {T,N} =
    pointer(x) == pointer(y) || __swap!(x, y)

__mayswap!(x::AbstractArray{T,N}, y::AbstractArray{T,N}) where {T,N} =
    __swap!(x, y)

function __swap!(x::AbstractArray{T,N}, y::AbstractArray{T,N}) where {T,N}
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
vfill!(x::AbstractArray{T,N}, α::Real) where {T,N} =
    _vfill!(x, convert(T, α))

function _vfill!(x::AbstractArray{T,N}, α::T) where {T,N}
    @inbounds @simd for i in eachindex(x)
        x[i] = α
    end
    return x
end

"""

```julia
vzero!(x) -> x
```

fills `x` with zeros and returns it.

Also see [`vfill!`](@ref).

"""
vzero!(A::AbstractArray{T,N}) where {T,N} = fill!(A, zero(T))
vzero!(x) = fill!(x, 0)

"""

```julia
vzeros(x)
```

yields a *vector* like `x` filled with zeros.

Also see [`vones`](@ref), [`vcreate`](@ref), [`vfill!`](@ref).

"""
vzeros(x) = vzero!(vcreate(x))

"""

```julia
vones(x)
```

yields a *vector* like `x` filled with ones.

Also see [`vzeros`](@ref), [`vcreate`](@ref), [`vfill!`](@ref).

"""
vones(x) = vfill!(vcreate(x), 1)

#------------------------------------------------------------------------------

"""
```julia
vscale!(dst, α, src) -> dst
```

overwrites `dst` with `α*src` and returns `dst`.  Computations are done at the
numerical precision of `promote_type(eltype(src),eltype(dst))`.  The source
argument may be omitted to perform *in-place* scaling:

```julia
vscale!(x, α) -> x
```

overwrites `x` with `α*x` and returns `x`.  The convention is that the result
is zero-filled if `α=0` (whatever the values in the source).

Methods are provided by default so that the order of the factor `α` and the
source vector may be reversed:

```julia
vscale!(dst, src, α) -> dst
vscale!(α, x) -> x
```

Also see [`vscale`](@ref).

"""
vscale!(α::Real, x::T) where {T} = vscale!(x, α)
vscale!(dst::D, src::S, α::Real) where {D,S} = vscale!(dst, α, src)

function vscale!(dst::AbstractArray{Td,N},
                 α::Real,
                 src::AbstractArray{Ts,N}) where {Td<:AbstractFloat,
                                                  Ts<:AbstractFloat,N}
    return _vscale!(dst, promote_scalar(Td, Ts, α), src)
end

function vscale!(dst::AbstractArray{Complex{Td},N},
                 α::Real,
                 src::AbstractArray{Complex{Ts},N}) where {Td<:AbstractFloat,
                                                           Ts<:AbstractFloat,N}
    return _vscale!(dst, promote_scalar(Td, Ts, α), src)
end

function _vscale!(dst::AbstractArray{Td,N},
                  α::AbstractFloat,
                  src::AbstractArray{Ts,N}) where {Td,Ts,N}
    if α == 0
        vzero!(dst)
    elseif α == 1
        vcopy!(dst, src)
    else
        indices(dst) == indices(src) ||
            _errdims("`dst` and `src` must have the same indices")
        if α == -1
            @inbounds @simd for i in eachindex(dst, src)
                dst[i] = -src[i]
            end
        else
            @inbounds @simd for i in eachindex(dst, src)
                dst[i] = α*src[i]
            end
        end
    end
    return dst
end

function vscale!(x::AbstractArray{<:Union{T,Complex{T}},N},
                 α::Real) where {T<:AbstractFloat,N}
    if α == 0
        vzero!(x)
    elseif α == -1
        @inbounds @simd for i in eachindex(x)
            x[i] = -x[i]
        end
    elseif α != 1
        const alpha = convert(T, α)
        @inbounds @simd for i in eachindex(x)
            x[i] *= alpha
        end
    end
    return x
end

# In place scaling for other *vector* types.
vscale!(x::T, α::Real) where {T} = vscale!(x, α, x)

"""
```julia
vscale(α, x)
```
or
```julia
vscale(x, α)
```

yield a new *vector* whose elements are those of `x` multiplied by the scalar
`α`.

Also see [`vscale!`](@ref), [`vcreate`](@ref).

"""
vscale(α::Real, x) = vscale!(vcreate(x), α, x)
vscale(x, α::Real) = vscale(α, x)

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
vproduct!(dst, [sel,] x, y) -> dst
```

which overwrites `dst` with the elementwise multiplication of `x` by `y`.
Optional argument `sel` is a selection of indices to which apply the operation.

"""
vproduct(x::V, y::V) where {V} = vproduct!(vcreate(x), x, y)

@doc @doc(vproduct) vproduct!


for (Td, Tx, Ty) in ((:Td,            :Tx,            :Ty),
                     (:(Complex{Td}), :Tx,            :Ty),
                     (:(Complex{Td}), :(Complex{Tx}), :Ty),
                     (:(Complex{Td}), :Tx,            :(Complex{Ty})),
                     (:(Complex{Td}), :(Complex{Tx}), :(Complex{Ty})))

    @eval function vproduct!(dst::AbstractArray{$Td,N},
                             x::AbstractArray{$Tx,N},
                             y::AbstractArray{$Ty,N}) where {Td<:AbstractFloat,
                                                             Tx<:AbstractFloat,
                                                             Ty<:AbstractFloat,
                                                             N}
        @assert indices(dst) == indices(x) == indices(y)
        @inbounds @simd for i in eachindex(dst, x, y)
            dst[i] = x[i]*y[i]
        end
        return dst
    end

    @eval function vproduct!(dst::DenseArray{$Td,N},
                             sel::AbstractVector{Int},
                             x::DenseArray{$Tx,N},
                             y::DenseArray{$Ty,N}) where {Td<:AbstractFloat,
                                                          Tx<:AbstractFloat,
                                                          Ty<:AbstractFloat,
                                                          N}
        @assert size(dst) == size(x) == size(y)
        jmin, jmax = extrema(sel)
        1 ≤ jmin ≤ jmax ≤ length(dst) || throw(BoundsError())
        @inbounds @simd for i in eachindex(sel)
            j = sel[i]
            dst[j] = x[j]*y[j]
        end
        return dst
    end

end

#--- VECTOR UPDATE ------------------------------------------------------------

"""
```julia
vupdate!(y, [sel,] α, x) -> y
```

overwrites `y` with `α*x + y` and returns `y`.  The code is optimized for some
specific values of the multiplier `α`.  For instance, if `α` is zero, then `y`
is left unchanged without using `x`.  Computations are performed at the
numerical precision of `promote_type(eltype(x),eltype(y))`.

Optional argument `sel` is a selection of indices to which apply the operation.
Note that if an index is repeated, the operation will be performed several
times at this location.

See also: [`vscale!`](@ref), [`vcombine!](@ref).

"""
function vupdate!(y::AbstractArray{Ty,N},
                  α::Real,
                  x::AbstractArray{Tx,N}) where {Ty<:AbstractFloat,
                                                 Tx<:AbstractFloat,N}
    return _vupdate!(y, promote_scalar(Tx, Ty, α), x)
end

function vupdate!(y::AbstractArray{Complex{Ty},N},
                  α::Real,
                  x::AbstractArray{Complex{Tx},N}) where {Ty<:AbstractFloat,
                                                          Tx<:AbstractFloat,N}
    return _vupdate!(y, promote_scalar(Tx, Ty, α), x)
end

# Pure Julia implementation.
function _vupdate!(y::AbstractArray{Ty,N},
                   α::AbstractFloat,
                   x::AbstractArray{Tx,N}) where {Ty,Tx,N}
    indices(x) == indices(y) ||
        _errdims("`x` and `y` must have the same indices")
    if α == 1
        @inbounds @simd for i in eachindex(y, x)
            y[i] += x[i]
        end
    elseif α == -1
        @inbounds @simd for i in eachindex(y, x)
            y[i] -= x[i]
        end
    elseif α != 0
        @inbounds @simd for i in eachindex(y, x)
            y[i] += α*x[i]
        end
    end
    return y
end

function vupdate!(y::DenseArray{Ty,N},
                  sel::AbstractVector{Int},
                  α::Real,
                  x::DenseArray{Tx,N}) where {Ty<:AbstractFloat,
                                              Tx<:AbstractFloat,N}
    return _vupdate!(y, sel, promote_scalar(Tx, Ty, α), x)
end

function vupdate!(y::DenseArray{Complex{Ty},N},
                  sel::AbstractVector{Int},
                  α::Real,
                  x::DenseArray{Complex{Tx},N}) where {Ty<:AbstractFloat,
                                                       Tx<:AbstractFloat,N}
    return _vupdate!(y, sel, promote_scalar(Tx, Ty, α), x)
end

function _vupdate!(y::DenseArray{Ty,N},
                   sel::AbstractVector{Int},
                   α::AbstractFloat,
                   x::DenseArray{Tx,N}) where {Ty,Tx,N}
    size(x) == size(y) ||
        _errdims("`x` and `y` must have the same dimensions")
    jmin, jmax = extrema(sel)
    1 ≤ jmin ≤ jmax ≤ length(x) || throw(BoundsError())
    if α == 1
        @inbounds @simd for i in eachindex(sel)
            j = sel[i]
            y[j] += x[j]
        end
    elseif α == -1
        @inbounds @simd for i in eachindex(sel)
            j = sel[i]
            y[j] -= x[j]
        end
    elseif α != 0
        @inbounds @simd for i in eachindex(sel)
            j = sel[i]
            y[j] += α*x[j]
        end
    end
    return y
end

#------------------------------------------------------------------------------
# LINEAR COMBINATION

"""
### Linear combination of arrays

```julia
vcombine(α, x, β, y) -> dst
```

yields the linear combination `dst = α*x + β*y`.

To avoid allocating the result, the destination array `dst` can be specified
with the in-place version of the method:

```julia
vcombine!(dst, α, x, β, y) -> dst
```

The code is optimized for some specific values of the coefficients `α` and `β`.
For instance, if `α` (resp. `β`) is zero, then the prior contents of `x`
(resp. `y`) is not used.

The source(s) and the destination can be the same.  For instance, the two
following lines of code produce the same result:

```julia
vcombine!(dst, 1, dst, α, x)
vupdate!(dst, α, x)
```

See also: [`vscale!`](@ref), [`vupdate!](@ref).

"""
vcombine(α::Real, x::V, β::Real, y::V) where {V} =
    vcombine!(vcreate(x), α, x, β, y)

# Deal with real/complex arrays.
for (Td, Tx, Ty) in ((:Td,            :Tx,            :Ty),
                     (:(Complex{Td}), :Tx,            :Ty),
                     (:(Complex{Td}), :(Complex{Tx}), :Ty),
                     (:(Complex{Td}), :Tx,            :(Complex{Ty})),
                     (:(Complex{Td}), :(Complex{Tx}), :(Complex{Ty})))

    @eval begin
        function vcombine!(dst::AbstractArray{$Td,N},
                           α::Real,
                           x::AbstractArray{$Tx,N},
                           β::Real,
                           y::AbstractArray{$Ty,N}) where {Td<:AbstractFloat,
                                                           Tx<:AbstractFloat,
                                                           Ty<:AbstractFloat,N}
            _vcombine!(dst,
                       promote_scalar(Td, Tx, Ty, α), x,
                       promote_scalar(Td, Tx, Ty, β), y)
        end
    end
end

function _vcombine!(dst::AbstractArray{Td,N},
                    α::AbstractFloat,
                    x::AbstractArray{Tx,N},
                    β::Real,
                    y::AbstractArray{Ty,N}) where {Td,Tx,Ty,N}
    @assert indices(dst) == indices(x) == indices(y)
    if α == 0
        _vscale!(dst, β, y)
    elseif β == 0
        _vscale!(dst, α, x)
    elseif α == 1
        if β == 1
            @inbounds @simd for i in eachindex(dst, x, y)
                dst[i] = x[i] + y[i]
            end
        elseif β == -1
            @inbounds @simd for i in eachindex(dst, x, y)
                dst[i] = x[i] - y[i]
            end
        else
            @inbounds @simd for i in eachindex(dst, x, y)
                dst[i] = x[i] + β*y[i]
            end
        end
    elseif α == -1
        if β == 1
            @inbounds @simd for i in eachindex(dst, x, y)
                dst[i] = y[i] - x[i]
            end
        elseif β == -1
            @inbounds @simd for i in eachindex(dst, x, y)
                dst[i] = -x[i] - y[i]
            end
        else
            @inbounds @simd for i in eachindex(dst, x, y)
                dst[i] = β*y[i] - x[i]
            end
        end
    else
        if β == 1
            @inbounds @simd for i in eachindex(dst, x, y)
                dst[i] = α*x[i] + y[i]
            end
        elseif β == -1
            @inbounds @simd for i in eachindex(dst, x, y)
                dst[i] = α*x[i] - y[i]
            end
        else
            @inbounds @simd for i in eachindex(dst, x, y)
                dst[i] = α*x[i] + β*y[i]
            end
        end
    end
    return dst
end

@doc @doc(vcombine) vcombine!

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
    indices(w) == indices(x) == indices(y) ||
        _errdims("`w`, `x` and `y` must have the same indices")
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
    indices(w) == indices(x) == indices(y) ||
        _errdims("`w`, `x` and `y` must have the same indices")
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
    size(y) == size(x) ||
        _errdims("`x` and `y` must have same dimensions")
    jmin, jmax = extrema(sel)
    1 ≤ jmin ≤ jmax ≤ length(x) || throw(BoundsError())
    local s::T = zero(T)
    @inbounds @simd for i in eachindex(sel)
        j = sel[i]
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
    size(y) == size(x) ||
        _errdims("`x` and `y` must have same dimensions")
    jmin, jmax = extrema(sel)
    1 ≤ jmin ≤ jmax ≤ length(x) || throw(BoundsError())
    local s::T = zero(T)
    @inbounds @simd for i in eachindex(sel)
        j = sel[i]
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
    indices(x) == indices(y) ||
        _errdims("`x` and `y` must have the same indices")
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
    indices(x) == indices(y) ||
        _errdims("`x` and `y` must have the same indices")
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
