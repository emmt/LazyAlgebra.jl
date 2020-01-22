#
# vectors.jl -
#
# Implement basic operations for *vectors*.  Here arrays of any rank are
# considered as *vectors*, the only requirements are that, when combining
# *vectors*, they have the same list of axes (i.e. the same dimensions for
# most arrays).  These methods are intended to be used for numerical
# optimization.
#
#-------------------------------------------------------------------------------
#
# This file is part of LazyAlgebra (https://github.com/emmt/LazyAlgebra.jl)
# released under the MIT "Expat" license.
#
# Copyright (c) 2017-2020 Éric Thiébaut.
#

# To simplify the code, we rely on the fact that converting `x::T` to type `T`
# does nothing.  According to Julia code (e.g. number.jl):
#
#     convert(::Type{T}, x::T)      where {T<:Number} = x
#     convert(::Type{T}, x::Number) where {T<:Number} = T(x)
#

"""
```julia
vnorm2([T,] v)
```

yields the Euclidean (L2) norm of `v`.  The floating point type of the result
can be imposed by optional argument `T`.  Also see [`vnorm1`](@ref) and
[`vnorminf`](@ref).

"""
function vnorm2(::Type{T}, v::AbstractArray{<:Real}) where {T<:AbstractFloat}
    s = zero(T)
    @inbounds @simd for i in eachindex(v)
        s += T(v[i])^2
    end
    return sqrt(s)
end

function vnorm2(::Type{T},
                v::AbstractArray{<:Complex{<:Real}}) where {T<:AbstractFloat}
    s = zero(T)
    @inbounds @simd for i in eachindex(v)
        s += abs2(Complex{T}(v[i]))
    end
    return sqrt(s)
end


"""
```julia
vnorm1([T,] v)
```

yields the L1 norm of `v`, that is the sum of the absolute values of its
elements.  The floating point type of the result can be imposed by optional
argument `T`.  For a complex valued argument, the result is the sum of the
absolute values of the real part and of the imaginary part of the elements
(like BLAS `asum`).

See also [`vnorm2`](@ref) and [`vnorminf`](@ref).

"""
function vnorm1(::Type{T}, v::AbstractArray{<:Real}) where {T<:AbstractFloat}
    s = zero(T)
    @inbounds @simd for i in eachindex(v)
        s += abs(T(v[i]))
    end
    return s
end

function vnorm1(::Type{T},
                v::AbstractArray{<:Complex{<:Real}}) where {T<:AbstractFloat}
    s = zero(T)
    @inbounds @simd for i in eachindex(v)
        z = Complex{T}(v[i])
        s += abs(real(z)) + abs(imag(z))
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
function vnorminf(::Type{T}, v::AbstractArray{T}) where {T<:AbstractFloat}
    amax = zero(T)
    @inbounds @simd for i in eachindex(v)
        amax = max(amax, abs(v[i]))
    end
    return amax
end

function vnorminf(::Type{T},
                  v::AbstractArray{Complex{T}}) where {T<:AbstractFloat}
    amax = zero(T)
    @inbounds @simd for i in eachindex(v)
        amax = max(amax, abs2(v[i]))
    end
    return sqrt(amax)
end

function vnorminf(::Type{T},
                  v::AbstractArray{E}) where {T<:AbstractFloat,R<:Real,
                                              E<:Union{R,Complex{R}}}
    T(vnorminf(R, v))
end

for norm in (:vnorm2, :vnorm1, :vnorminf)
    @eval begin
        $norm(v::AbstractArray{<:Union{R,Complex{R}}}) where {R<:AbstractFloat} =
            $norm(R, v)
        $norm(v::AbstractArray{<:Union{R,Complex{R}}}) where {R<:Real} =
            $norm(float(R), v)
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
vcreate(x::AbstractArray{T,N}) where {R<:Real,T<:Union{R,Complex{R}},N} =
    similar(x, float(T))

#------------------------------------------------------------------------------

"""
```julia
vcopy!(dst, src) -> dst
```

copies the contents of `src` into `dst` and returns `dst`.  This function
checks that the copy makes sense (for instance, for array arguments, the
`copyto!` operation does not check that the source and destination have the
same dimensions).

Also see [`copyto!`](@ref), [`vcopy`](@ref), [`vswap!`](@ref).

"""
function vcopy!(dst::AbstractArray{<:Real,N},
                src::AbstractArray{<:Real,N}) where {N}
    axes(dst) == axes(src) || throw_dimensions_mismatch()
    copyto!(dst, src)
end

function vcopy!(dst::AbstractArray{<:Complex{<:Real},N},
                src::AbstractArray{<:Complex{<:Real},N}) where {N}
    axes(dst) == axes(src) || throw_dimensions_mismatch()
    copyto!(dst, src)
end

@noinline throw_dimensions_mismatch() =
    throw_dimensions_mismatch("arguments have incompatible dimensions/indices")

@noinline throw_dimensions_mismatch(mesg::AbstractString) =
    throw(DimensionMismatch(mesg))

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
function vswap!(x::AbstractArray{T,N},
                y::AbstractArray{T,N}) where {R<:Real,T<:Union{R,Complex{R}},N}
    axes(x) == axes(y) || throw_dimensions_mismatch()
    __mayswap!(x, y)
end

__mayswap!(x::Array{T,N}, y::Array{T,N}) where {T,N} =
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

Also see [`vzero!`](@ref), [`fill!`](@ref).

"""
vfill!(x, α) = fill!(x, α)
vfill!(x::AbstractArray{T}, α) where {T} = vfill!(x, T(α))
function vfill!(x::AbstractArray{T}, α::T) where {T}
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
vzero!(x) = vfill!(x, 0)
vzero!(x::AbstractArray{T}) where {T} = vfill!(x, zero(T))

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

Also see [`vscale`](@ref), [`LinearAlgebra.rmul!](@ref).

"""
function vscale!(dst::AbstractArray{<:Floats,N},
                 α::Number,
                 src::AbstractArray{<:Floats,N}) where {N}
    axes(dst) == axes(src) || throw_dimensions_mismatch()
    return _vscale!(dst, α, src)
end

# This *private* method assumes that arguments have same indices.
function _vscale!(dst::AbstractArray{<:Floats,N},
                  α::Number,
                  src::AbstractArray{T,N}) where {T<:Floats,N}
    if α == 1
        copyto!(dst, src)
    elseif α == 0
        vzero!(dst)
    elseif α == -1
        @inbounds @simd for i in eachindex(dst, src)
            dst[i] = -src[i]
        end
    else
        a = promote_multiplier(α, T)
        @inbounds @simd for i in eachindex(dst, src)
            dst[i] = a*src[i]
        end
    end
    return dst
end

# In-place scaling.
function vscale!(x::AbstractArray{T}, α::Number) where {T<:Floats}
    if α == 0
        vzero!(x)
    elseif α == -1
        @inbounds @simd for i in eachindex(x)
            x[i] = -x[i]
        end
    elseif α != 1
        alpha = promote_multiplier(α, T)
        @inbounds @simd for i in eachindex(x)
            x[i] *= alpha
        end
    end
    return x
end

# In-place scaling with reverse order of arguments.
vscale!(α::Number, x) = vscale!(x, α)

# Scaling for other *vector* types.
vscale!(x, α::Number) = vscale!(x, α, x)
vscale!(dst, src, α::Number) = vscale!(dst, α, src)

# The following methods are needed to avoid looping forever.
vscale!(::Number, ::Number) = error("bad argument types")
vscale!(::Any, ::Number, ::Number) = error("bad argument types")
vscale!(::Number, ::Any, ::Number) = error("bad argument types")
vscale!(::Number, ::Number, ::Any) = error("bad argument types")
vscale!(::Number, ::Number, ::Number) = error("bad argument types")

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
vscale(α::Number, x) = _vscale!(vcreate(x), α, x)
vscale(x, α::Number) = vscale(α, x)

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

vproduct(x::AbstractArray{Tx,N}, y::AbstractArray{Ty,N}) where {Tx,Ty,N} =
    vproduct!(similar(x, promote_type(Tx,Ty)), x, y)

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
        axes(dst) == axes(x) == axes(y) || throw_dimensions_mismatch()
        @inbounds @simd for i in eachindex(dst, x, y)
            dst[i] = x[i]*y[i]
        end
        return dst
    end

    @eval function vproduct!(dst::Array{$Td,N},
                             sel::AbstractVector{Int},
                             x::Array{$Tx,N},
                             y::Array{$Ty,N}) where {Td<:AbstractFloat,
                                                     Tx<:AbstractFloat,
                                                     Ty<:AbstractFloat,
                                                     N}
        size(dst) == size(x) == size(y) || throw_dimensions_mismatch()
        jmin, jmax = extrema(sel)
        1 ≤ jmin ≤ jmax ≤ length(dst) || throw(BoundsError())
        @inbounds @simd for i in eachindex(sel)
            j = sel[i]
            dst[j] = x[j]*y[j]
        end
        return dst
    end

end

@doc @doc(vproduct) vproduct!

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
function vupdate!(y::AbstractArray{<:Floats,N},
                  α::Number,
                  x::AbstractArray{T,N}) where {T<:Floats,N}
    axes(x) == axes(y) || throw_dimensions_mismatch()
    if α == 1
        @inbounds @simd for i in eachindex(y, x)
            y[i] += x[i]
        end
    elseif α == -1
        @inbounds @simd for i in eachindex(y, x)
            y[i] -= x[i]
        end
    elseif α != 0
        a = promote_multiplier(α, T)
        @inbounds @simd for i in eachindex(y, x)
            y[i] += a*x[i]
        end
    end
    return y
end

function vupdate!(y::AbstractArray{<:Floats,N},
                  sel::AbstractVector{Int},
                  α::Number,
                  x::AbstractArray{T,N}) where {T<:Floats,N}
    @assert IndexStyle(y) == IndexLinear()
    @assert IndexStyle(x) == IndexLinear()
    axes(x) == axes(y) || throw_dimensions_mismatch()
    jmin, jmax = extrema(sel)
    I = eachindex(IndexLinear(), x)
    first(I) ≤ jmin ≤ jmax ≤ last(I) || throw(BoundsError())
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
        a = promote_multiplier(α, T)
        @inbounds @simd for i in eachindex(sel)
            j = sel[i]
            y[j] += a*x[j]
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

The code is optimized for some specific values of the multipliers `α` and `β`.
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
vcombine(α::Number, x::V, β::Number, y::V) where {V} =
    vcombine!(vcreate(x), α, x, β, y)

function vcombine!(dst::AbstractArray{<:Floats,N},
                   α::Number,
                   x::AbstractArray{Tx,N},
                   β::Number,
                   y::AbstractArray{Ty,N}) where {Tx<:Floats,
                                                  Ty<:Floats,N}
    axes(dst) == axes(x) == axes(y) || throw_dimensions_mismatch()
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
            b = promote_multiplier(β, Ty)
            @inbounds @simd for i in eachindex(dst, x, y)
                dst[i] = x[i] + b*y[i]
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
            b = promote_multiplier(β, Ty)
            @inbounds @simd for i in eachindex(dst, x, y)
                dst[i] = b*y[i] - x[i]
            end
        end
    else
        a = promote_multiplier(α, Tx)
        if β == 1
            @inbounds @simd for i in eachindex(dst, x, y)
                dst[i] = a*x[i] + y[i]
            end
        elseif β == -1
            @inbounds @simd for i in eachindex(dst, x, y)
                dst[i] = a*x[i] - y[i]
            end
        else
            b = promote_multiplier(β, Ty)
            @inbounds @simd for i in eachindex(dst, x, y)
                dst[i] = a*x[i] + b*y[i]
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

yields the inner product of `x` and `y`; that is, the sum of `conj(x[i])*y[i]`
or, if `w` is specified, the sum of `w[i]*conj(x[i])*y[i]` (`w` must have
real-valued elements), for all indices `i`.  Optional argument `T` is the
floating point type of the result.

Another possibility is:

```julia
vdot([T,] sel, x, y)
```

with `sel` a selection of indices to restrict the computation of the inner
product to some selected elements.  This yields the sum of `x[i]*y[i]` for all
`i ∈ sel`.

If the arguments have complex-valued elements and `T` is specified as a
floating-point type, complexes are considered as vectors of pairs of reals and
the result is:

```julia
vdot(T::Type{AbstractFloat}, x, y) = x[1].re*y[1].re + x[1].im*y[1].im +
                                     x[2].re*y[2].re + x[2].im*y[2].im + ...
```

"""
function vdot(::Type{T},
              x::AbstractArray{<:Real,N},
              y::AbstractArray{<:Real,N})::T where {T<:AbstractFloat,N}
    axes(x) == axes(y) || throw_dimensions_mismatch()
    s = zero(T)
    @inbounds @simd for i in eachindex(x, y)
        s += convert(T, x[i]*y[i])
    end
    return s
end

function vdot(x::AbstractArray{Tx,N},
              y::AbstractArray{Ty,N}) where {Tx<:Real,Ty<:Real,N}
    return vdot(float(promote_type(Tx, Ty)), x, y)
end

function vdot(::Type{Complex{T}},
              x::AbstractArray{Complex{Tx},N},
              y::AbstractArray{Complex{Ty},N}
              )::Complex{T} where {T<:AbstractFloat,Tx<:Real,Ty<:Real,N}
    axes(x) == axes(y) || throw_dimensions_mismatch()
    s = zero(Complex{T})
    @inbounds @simd for i in eachindex(x, y)
        xi = convert(Complex{T}, x[i])
        yi = convert(Complex{T}, y[i])
        s += conj(xi)*yi
    end
    return s
end

# This one yields the real part of the dot product, just as if complexes
# were pairs of reals.
function vdot(::Type{T},
              x::AbstractArray{Complex{Tx},N},
              y::AbstractArray{Complex{Ty},N})::T where {T<:AbstractFloat,
                                                         Tx<:Real,Ty<:Real,N}
    axes(x) == axes(y) || throw_dimensions_mismatch()
    s = zero(T)
    @inbounds @simd for i in eachindex(x, y)
        xi = convert(Complex{T}, x[i])
        yi = convert(Complex{T}, y[i])
        s += real(xi)*real(yi) + imag(xi)*imag(yi)
    end
    return s
end

# Note that we cannot use union here because we want that both be real or both
# be complex.
function vdot(x::AbstractArray{Complex{Tx},N},
              y::AbstractArray{Complex{Ty},N}) where {Tx<:Real,Ty<:Real,N}
    return vdot(Complex{float(promote_type(Tx, Ty))}, x, y)
end

function vdot(::Type{T},
              w::AbstractArray{<:Real,N},
              x::AbstractArray{<:Real,N},
              y::AbstractArray{<:Real,N})::T where {T<:AbstractFloat,N}
    axes(w) == axes(x) == axes(y) || throw_dimensions_mismatch()
    s = zero(T)
    @inbounds @simd for i in eachindex(w, x, y)
        wi = convert(T, w[i])
        xi = convert(T, x[i])
        yi = convert(T, y[i])
        s += wi*xi*yi
    end
    return s
end

function vdot(w::AbstractArray{Tw,N},
              x::AbstractArray{Tx,N},
              y::AbstractArray{Ty,N}) where {Tw<:Real,Tx<:Real,Ty<:Real,N}
    return vdot(float(promote_type(Tw, Tx, Ty)), w, x, y)
end

function vdot(::Type{Complex{T}},
              w::AbstractArray{<:Real,N},
              x::AbstractArray{Complex{Tx},N},
              y::AbstractArray{Complex{Ty},N}
              )::Complex{T} where {T<:AbstractFloat,Tx<:Real,Ty<:Real,N}
    axes(w) == axes(x) == axes(y) || throw_dimensions_mismatch()
    s = zero(Complex{T})
    @inbounds @simd for i in eachindex(w, x, y)
        wi = convert(T, w[i])
        xi = convert(Complex{T}, x[i])
        yi = convert(Complex{T}, y[i])
        s += wi*conj(xi)*yi
    end
    return s
end

function vdot(::Type{T},
              w::AbstractArray{<:Real,N},
              x::AbstractArray{Complex{Tx},N},
              y::AbstractArray{Complex{Ty},N})::T where {T<:AbstractFloat,
                                                         Tx<:Real,Ty<:Real,N}
    axes(w) == axes(x) == axes(y) || throw_dimensions_mismatch()
    s = zero(T)
    @inbounds @simd for i in eachindex(w, x, y)
        wi = convert(T, w[i])
        xi = convert(Complex{T}, x[i])
        yi = convert(Complex{T}, y[i])
        s += (real(xi)*real(yi) + imag(xi)*imag(yi))*wi
    end
    return s
end

function vdot(w::AbstractArray{Tw,N},
              x::AbstractArray{Complex{Tx},N},
              y::AbstractArray{Complex{Ty},N}) where {Tw<:Real,Tx<:Real,
                                                      Ty<:Real,N}
    return vdot(Complex{float(promote_type(Tw, Tx, Ty))}, w, x, y)
end

function vdot(::Type{T},
              sel::AbstractVector{Int},
              x::Array{<:Real,N},
              y::Array{<:Real,N})::T where {T<:AbstractFloat,N}
    size(y) == size(x) || throw_dimensions_mismatch()
    jmin, jmax = extrema(sel)
    1 ≤ jmin ≤ jmax ≤ length(x) || throw(BoundsError())
    s = zero(T)
    @inbounds @simd for i in eachindex(sel)
        j = sel[i]
        xj = convert(T, x[j])
        yj = convert(T, y[j])
        s += xj*yj
    end
    return s
end

function vdot(sel::AbstractVector{Int},
              x::Array{Tx,N},
              y::Array{Ty,N}) where {Tx<:Real,Ty<:Real,N}
    return vdot(float(promote_type(Tx, Ty)), sel, x, y)
end

function vdot(::Type{Complex{T}},
              sel::AbstractVector{Int},
              x::Array{Complex{Tx},N},
              y::Array{Complex{Ty},N})::Complex{T} where {T<:AbstractFloat,
                                                          Tx<:Real,Ty<:Real,N}
    size(y) == size(x) || throw_dimensions_mismatch()
    jmin, jmax = extrema(sel)
    1 ≤ jmin ≤ jmax ≤ length(x) || throw(BoundsError())
    s = zero(Complex{T})
    @inbounds @simd for i in eachindex(sel)
        j = sel[i]
        xj = convert(Complex{T}, x[j])
        yj = convert(Complex{T}, y[j])
        s += conj(xj)*yj
    end
    return s
end

function vdot(::Type{T},
              sel::AbstractVector{Int},
              x::Array{Complex{Tx},N},
              y::Array{Complex{Ty},N})::T where {T<:AbstractFloat,
                                                 Tx<:Real,Ty<:Real,N}
    size(y) == size(x) || throw_dimensions_mismatch()
    jmin, jmax = extrema(sel)
    1 ≤ jmin ≤ jmax ≤ length(x) || throw(BoundsError())
    s = zero(T)
    @inbounds @simd for i in eachindex(sel)
        j = sel[i]
        xj = convert(Complex{T}, x[j])
        yj = convert(Complex{T}, y[j])
        s += real(xj)*real(yj) + imag(xj)*imag(yj)
    end
    return s
end

function vdot(sel::AbstractVector{Int},
              x::Array{Complex{Tx},N},
              y::Array{Complex{Ty},N}) where {Tx<:Real,Ty<:Real,N}
    return vdot(Complex{float(promote_type(Tx, Ty))}, sel, x, y)
end
