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
# Copyright (c) 2017-2018 Éric Thiébaut.
#

# FIXME: To simplify the code, we rely on the fact that converting `x::T` to
# type `T` does nothing.  According to julia/base/essentials.jl:
#
#     convert(::Type{T}, x::T) where {T} = x
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
                v::AbstractArray{<:Real})::T where {T<:AbstractFloat}
    s = zero(T)
    @inbounds @simd for i in eachindex(v)
        x = convert(T, v[i])
        s += x*x
    end
    return sqrt(s)
end

function vnorm2(::Type{T},
                v::AbstractArray{<:Complex{<:Real}})::T where {T<:AbstractFloat}
    s = zero(T)
    @inbounds @simd for i in eachindex(v)
        z = convert(Complex{T}, v[i])
        s += abs2(z)
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
function vnorm1(::Type{T},
                v::AbstractArray{<:Real})::T where {T<:AbstractFloat}
    s = zero(T)
    @inbounds @simd for i in eachindex(v)
        x = convert(T, v[i])
        s += abs(x)
    end
    return s
end

function vnorm1(::Type{T},
                v::AbstractArray{<:Complex{<:Real}})::T where {T<:AbstractFloat}
    s = zero(T)
    @inbounds @simd for i in eachindex(v)
        z = convert(Complex{T}, v[i])
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
function vnorminf(::Type{T}, v::AbstractArray{T})::T where {T<:AbstractFloat}
    amax = zero(T)
    @inbounds @simd for i in eachindex(v)
        x = v[i]
        amax = max(amax, abs(x))
    end
    return amax
end

function vnorminf(::Type{T}, v::AbstractArray{R})::T where {T<:AbstractFloat,
                                                            R<:Real}
    return convert(T, vnorminf(R, v))
end

# FIXME: avoid overflows?
function vnorminf(::Type{T},
                  v::AbstractArray{<:Complex{<:Real}})::T where {T<:AbstractFloat}
    amax = zero(T)
    @inbounds @simd for i in eachindex(v)
        z = convert(Complex{T}, v[i])
        amax = max(amax, abs2(z))
    end
    return sqrt(amax)
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
    axes(dst) == axes(src) ||
        __baddims("`dst` and `src` must have the same axes")
    copyto!(dst, src)
end

function vcopy!(dst::AbstractArray{<:Complex{<:Real},N},
                src::AbstractArray{<:Complex{<:Real},N}) where {N}
    axes(dst) == axes(src) ||
        __baddims("`dst` and `src` must have the same axes")
    copyto!(dst, src)
end

@inline __baddims(msg::AbstractString) = throw(DimensionMismatch(msg))

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
    axes(x) == axes(y) ||
        __baddims("`x` and `y` must have the same axes")
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
vfill!(x::AbstractArray{T}, α::Real) where {T<:AbstractFloat} =
    __vfill!(x, convert(T, α))

function vfill!(x::AbstractArray{Complex{R}},
                α::Union{Real,Complex{<:Real}}) where {R<:AbstractFloat}
    __vfill!(x, convert(Complex{R}, α))
end

function __vfill!(x::AbstractArray{T}, α::T) where {T}
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

""" vscale!

for (Td, Ts) in ((:Td,            :Ts),
                 (:(Complex{Td}), :(Complex{Ts})))
    @eval begin

        function vscale!(dst::AbstractArray{$Td,N},
                         α::Real,
                         src::AbstractArray{$Ts,N}) where {Td<:AbstractFloat,
                                                           Ts<:AbstractFloat,N}
            axes(dst) == axes(src) ||
                __baddims("`dst` and `src` must have the same axes")
            if α == 1
                copyto!(dst, src)
            elseif α == 0
                fill!(dst, 0)
            elseif α == -1
                @inbounds @simd for i in eachindex(dst, src)
                    dst[i] = -src[i]
                end
            else
                a = promote_scalar(Td, Ts, α)
                @inbounds @simd for i in eachindex(dst, src)
                    dst[i] = a*src[i]
                end
            end
            return dst
        end

    end
end

# In-place scaling.
function vscale!(x::AbstractArray{<:Union{T,Complex{T}}},
                 α::Real) where {T<:AbstractFloat}
    if α == 0
        vzero!(x)
    elseif α == -1
        @inbounds @simd for i in eachindex(x)
            x[i] = -x[i]
        end
    elseif α != 1
        alpha = convert(T, α)
        @inbounds @simd for i in eachindex(x)
            x[i] *= alpha
        end
    end
    return x
end

# In-place scaling with reverse order of arguments.
vscale!(α::Real, x) = vscale!(x, α)

# Scaling for other *vector* types.
vscale!(x, α::Real) = vscale!(x, α, x)
vscale!(dst, src, α::Real) = vscale!(dst, α, src)

# The following methods are needed to avoid looping forever.
vscale!(::Real, ::Real) = error("bad argument types")
vscale!(::Any, ::Real, ::Real) = error("bad argument types")
vscale!(::Real, ::Any, ::Real) = error("bad argument types")
vscale!(::Real, ::Real, ::Any) = error("bad argument types")
vscale!(::Real, ::Real, ::Real) = error("bad argument types")

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
        axes(dst) == axes(x) == axes(y) ||
            __baddims("`x` and `y` must have the same axes as `dst`")
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
        size(dst) == size(x) == size(y) ||
            __baddims("`x` and `y` must have the same dimensions as `dst`")
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

""" vupdate!

for (Tx, Ty) in ((:Tx,            :Ty),
                 (:(Complex{Tx}), :(Complex{Ty})))

    @eval function vupdate!(y::AbstractArray{$Ty,N},
                            α::Real,
                            x::AbstractArray{$Tx,N}) where {Ty<:AbstractFloat,
                                                            Tx<:AbstractFloat,
                                                            N}
        axes(x) == axes(y) ||
            __baddims("`x` and `y` must have the same axes")
        if α == 1
            @inbounds @simd for i in eachindex(y, x)
                y[i] += x[i]
            end
        elseif α == -1
            @inbounds @simd for i in eachindex(y, x)
                y[i] -= x[i]
            end
        elseif α != 0
            a = promote_scalar(Tx, Ty, α)
            @inbounds @simd for i in eachindex(y, x)
                y[i] += a*x[i]
            end
        end
        return y
    end

    @eval function vupdate!(y::DenseArray{$Ty,N},
                            sel::AbstractVector{Int},
                            α::Real,
                            x::DenseArray{$Tx,N}) where {Ty<:AbstractFloat,
                                                         Tx<:AbstractFloat,N}
        size(x) == size(y) ||
            __baddims("`x` and `y` must have the same dimensions")
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
            a = promote_scalar(Tx, Ty, α)
            @inbounds @simd for i in eachindex(sel)
                j = sel[i]
                y[j] += a*x[j]
            end
        end
        return y
    end

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
            axes(dst) == axes(x) == axes(y) ||
                __baddims("`x` and `y` must have the same axes as `dst`")
            if α == 0
                vscale!(dst, β, y)
            elseif β == 0
                vscale!(dst, α, x)
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
                    b = promote_scalar(Td, Tx, Ty, β)
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
                    b = promote_scalar(Td, Tx, Ty, β)
                    @inbounds @simd for i in eachindex(dst, x, y)
                        dst[i] = b*y[i] - x[i]
                    end
                end
            else
                a = promote_scalar(Td, Tx, Ty, α)
                if β == 1
                    @inbounds @simd for i in eachindex(dst, x, y)
                        dst[i] = a*x[i] + y[i]
                    end
                elseif β == -1
                    @inbounds @simd for i in eachindex(dst, x, y)
                        dst[i] = a*x[i] - y[i]
                    end
                else
                    b = promote_scalar(Td, Tx, Ty, β)
                    @inbounds @simd for i in eachindex(dst, x, y)
                        dst[i] = a*x[i] + b*y[i]
                    end
                end
            end
            return dst
        end

    end
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
    axes(x) == axes(y) ||
        __baddims("`x` and `y` must have the same axes")
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
    axes(x) == axes(y) ||
        __baddims("`x` and `y` must have the same axes")
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
    axes(x) == axes(y) ||
        __baddims("`x` and `y` must have the same axes")
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
    axes(w) == axes(x) == axes(y) ||
        __baddims("`w`, `x` and `y` must have the same axes")
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
    axes(w) == axes(x) == axes(y) ||
        __baddims("`w`, `x` and `y` must have the same axes")
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
    axes(w) == axes(x) == axes(y) ||
        __baddims("`w`, `x` and `y` must have the same axes")
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
              x::DenseArray{<:Real,N},
              y::DenseArray{<:Real,N})::T where {T<:AbstractFloat,N}
    size(y) == size(x) ||
        __baddims("`x` and `y` must have same dimensions")
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
              x::DenseArray{Tx,N},
              y::DenseArray{Ty,N}) where {Tx<:Real,Ty<:Real,N}
    return vdot(float(promote_type(Tx, Ty)), sel, x, y)
end

function vdot(::Type{Complex{T}},
              sel::AbstractVector{Int},
              x::DenseArray{Complex{Tx},N},
              y::DenseArray{Complex{Ty},N}
              )::Complex{T} where {T<:AbstractFloat,Tx<:Real,Ty<:Real,N}
    size(y) == size(x) ||
        __baddims("`x` and `y` must have same dimensions")
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
              x::DenseArray{Complex{Tx},N},
              y::DenseArray{Complex{Ty},N})::T where {T<:AbstractFloat,
                                                      Tx<:Real,Ty<:Real,N}
    size(y) == size(x) ||
        __baddims("`x` and `y` must have same dimensions")
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
              x::DenseArray{Complex{Tx},N},
              y::DenseArray{Complex{Ty},N}) where {Tx<:Real,Ty<:Real,N}
    return vdot(Complex{float(promote_type(Tx, Ty))}, sel, x, y)
end
