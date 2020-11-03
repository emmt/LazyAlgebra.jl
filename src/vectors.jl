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

"""
    vnorm2([T,] v)

yields the Euclidean (L2) norm of `v`.  The floating point type of the result
can be imposed by optional argument `T`.  Also see [`vnorm1`](@ref) and
[`vnorminf`](@ref).

"""
function vnorm2(v::AbstractArray{<:Union{T,Complex{T}}}) where {T<:AbstractFloat}
    s = zero(T)
    @inbounds @simd for i in eachindex(v)
        s += abs2(v[i])
    end
    return sqrt(s)
end

"""
    vnorm1([T,] v)

yields the L1 norm of `v`, that is the sum of the absolute values of its
elements.  The floating point type of the result can be imposed by optional
argument `T`.  For a complex valued argument, the result is the sum of the
absolute values of the real part and of the imaginary part of the elements
(like BLAS `asum`).

See also [`vnorm2`](@ref) and [`vnorminf`](@ref).

"""
function vnorm1(v::AbstractArray{T}) where {T<:AbstractFloat}
    s = zero(T)
    @inbounds @simd for i in eachindex(v)
        s += abs(v[i])
    end
    return s
end

function vnorm1(v::AbstractArray{Complex{T}}) where {T<:AbstractFloat}
    sr = zero(T)
    si = zero(T)
    @inbounds @simd for i in eachindex(v)
        z = v[i]
        sr += abs(real(z))
        si += abs(imag(z))
    end
    return sr + si
end

"""
    vnorminf([T,] v)

yields the infinite norm of `v`, that is the maximum absolute value of its
elements.  The floating point type of the result can be imposed by optional
argument `T`.  Also see [`vnorm1`](@ref) and [`vnorm2`](@ref).

"""
function vnorminf(v::AbstractArray{T}) where {T<:AbstractFloat}
    absmax = zero(T)
    @inbounds @simd for i in eachindex(v)
        absmax = max(absmax, abs(v[i]))
    end
    return absmax
end

function vnorminf(v::AbstractArray{Complex{T}}) where {T<:AbstractFloat}
    abs2max = zero(T)
    @inbounds @simd for i in eachindex(v)
        abs2max = max(abs2max, abs2(v[i]))
    end
    return sqrt(abs2max)
end

# Versions with forced type of output result.
for func in (:vnorm2, :vnorm1, :vnorminf)
    @eval $func(::Type{T}, v) where {T<:AbstractFloat} = T($func(v))
end

#------------------------------------------------------------------------------

"""
    vcreate(x)

yields a new variable instance similar to `x`.  If `x` is an array, the
element type of the result is a floating-point type.

Also see [`similar`](@ref).

"""
vcreate(x::AbstractArray{T,N}) where {R<:Real,T<:Union{R,Complex{R}},N} =
    similar(x, float(T))

#------------------------------------------------------------------------------

"""
    vcopy!(dst, src) -> dst

copies the contents of `src` into `dst` and returns `dst`.  This function
checks that the copy makes sense (for instance, for array arguments, the
`copyto!` operation does not check that the source and destination have the
same dimensions).

Also see [`copyto!`](@ref), [`vcopy`](@ref), [`vswap!`](@ref).

"""
function vcopy!(dst::AbstractArray{<:Real,N},
                src::AbstractArray{<:Real,N}) where {N}
    if dst !== src
        axes(dst) == axes(src) || arguments_have_incompatible_axes()
        copyto!(dst, src)
    end
    return dst
end

function vcopy!(dst::AbstractArray{<:Complex{<:Real},N},
                src::AbstractArray{<:Complex{<:Real},N}) where {N}
    if dst !== src
        axes(dst) == axes(src) || arguments_have_incompatible_axes()
        copyto!(dst, src)
    end
    return dst
end

"""
    vcopy(x)

yields a fresh copy of the *vector* `x`.  If `x` is is an array, the element
type of the result is a floating-point type.

Also see [`copy`](@ref), [`vcopy!`](@ref), [`vcreate!`](@ref).

"""
vcopy(x) = vcopy!(vcreate(x), x)

"""
    vswap!(x, y)

exchanges the contents of `x` and `y` (which must have the same element type
and axes if they are arrays).

Also see [`vcopy!`](@ref).

"""
vswap!(x::T, y::T) where {T<:AbstractArray} =
    x === y || _swap!(x, y)

vswap!(x::AbstractArray{T,N}, y::AbstractArray{T,N}) where {T,N} =
    _swap!(x, y)

# Forced swapping.
_swap!(x::AbstractArray{T,N}, y::AbstractArray{T,N}) where {T,N} =
    @inbounds @simd for i in all_indices(x, y)
        temp = x[i]
        x[i] = y[i]
        y[i] = temp
    end

#------------------------------------------------------------------------------

"""
    vfill!(x, α) -> x

sets all elements of `x` with the scalar value `α` and return `x`.

Also see [`vzero!`](@ref), [`fill!`](@ref).

"""
vfill!(x, α) = fill!(x, α)
vfill!(x::AbstractArray{T}, α) where {T} = vfill!(x, T(α))
vfill!(x::AbstractArray{T}, α::T) where {T} = begin
    @inbounds @simd for i in eachindex(x)
        x[i] = α
    end
    return x
end

"""
    vzero!(x) -> x

fills `x` with zeros and returns it.

Also see [`vfill!`](@ref).

"""
vzero!(x) = vfill!(x, 0)
vzero!(x::AbstractArray{T}) where {T} = vfill!(x, zero(T))

"""
    vzeros(x)

yields a *vector* like `x` filled with zeros.

Also see [`vones`](@ref), [`vcreate`](@ref), [`vfill!`](@ref).

"""
vzeros(x) = vzero!(vcreate(x))

"""
    vones(x)

yields a *vector* like `x` filled with ones.

Also see [`vzeros`](@ref), [`vcreate`](@ref), [`vfill!`](@ref).

"""
vones(x) = vfill!(vcreate(x), 1)

#------------------------------------------------------------------------------

"""
    vscale!(dst, α, src) -> dst

overwrites `dst` with `α*src` and returns `dst`.  Computations are done at the
numerical precision of `promote_eltype(src,dst)`.  The source argument may be
omitted to perform *in-place* scaling:

    vscale!(x, α) -> x

overwrites `x` with `α*x` and returns `x`.  The convention is that the result
is zero-filled if `α=0` (whatever the values in the source).

Methods are provided by default so that the order of the factor `α` and the
source vector may be reversed:

    vscale!(dst, src, α) -> dst
    vscale!(α, x) -> x

Also see [`vscale`](@ref), [`LinearAlgebra.rmul!](@ref).

"""
function vscale!(dst::AbstractArray{<:Floats,N},
                 α::Number,
                 src::AbstractArray{<:Floats,N}) where {N}
    if α == 1
        vcopy!(dst, src)
    elseif α == 0
        axes(dst) == axes(src) || arguments_have_incompatible_axes()
        vzero!(dst)
    elseif α == -1
        @inbounds @simd for i in all_indices(dst, src)
            dst[i] = -src[i]
        end
    else
        alpha = promote_multiplier(α, src)
        @inbounds @simd for i in all_indices(dst, src)
            dst[i] = alpha*src[i]
        end
    end
    return dst
end

# In-place scaling.
function vscale!(x::AbstractArray{<:Floats}, α::Number)
    if α == 0
        vzero!(x)
    elseif α == -1
        @inbounds @simd for i in eachindex(x)
            x[i] = -x[i]
        end
    elseif α != 1
        alpha = promote_multiplier(α, x)
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
    vscale(α, x)

or

    vscale(x, α)

yield a new *vector* whose elements are those of `x` multiplied by the scalar
`α`.

Also see [`vscale!`](@ref), [`vcreate`](@ref).

"""
vscale(α::Number, x) = vscale!(vcreate(x), α, x)
vscale(x, α::Number) = vscale(α, x)

#------------------------------------------------------------------------------
# ELEMENT-WISE MULTIPLICATION

"""
    vproduct(x, y) -> z

yields the element-wise multiplication of `x` by `y`.  To avoid allocating the
result, the destination array `dst` can be specified with the in-place version
of the method:

    vproduct!(dst, [sel,] x, y) -> dst

which overwrites `dst` with the elementwise multiplication of `x` by `y`.
Optional argument `sel` is a selection of indices to which apply the operation.

"""
vproduct(x::V, y::V) where {V} = vproduct!(vcreate(x), x, y)

vproduct(x::AbstractArray{<:Any,N}, y::AbstractArray{<:Any,N}) where {N} =
    vproduct!(similar(x, promote_eltype(x,y)), x, y)

for Td in (AbstractFloat, Complex{<:AbstractFloat}),
    Tx in (AbstractFloat, Complex{<:AbstractFloat}),
    Ty in (AbstractFloat, Complex{<:AbstractFloat})

    if Td <: Complex || (Tx <: Real && Ty <: Real)

        @eval function vproduct!(dst::AbstractArray{<:$Td,N},
                                 x::AbstractArray{<:$Tx,N},
                                 y::AbstractArray{<:$Ty,N}) where {N}
            @inbounds @simd for i in all_indices(dst, x, y)
                dst[i] = x[i]*y[i]
            end
            return dst
        end

        @eval function vproduct!(dst::AbstractArray{<:$Td,N},
                                 sel::AbstractVector{Int},
                                 x::AbstractArray{<:$Tx,N},
                                 y::AbstractArray{<:$Ty,N}) where {N}
            if checkselection(sel, dst, x, y)
                @inbounds @simd for j in eachindex(sel)
                    i = sel[j]
                    dst[i] = x[i]*y[i]
                end
            end
            return dst
        end

    end

end

@doc @doc(vproduct) vproduct!

#------------------------------------------------------------------------------
# VECTOR UPDATE

"""
    vupdate!(y, [sel,] α, x) -> y

overwrites `y` with `α*x + y` and returns `y`.  The code is optimized for some
specific values of the multiplier `α`.  For instance, if `α` is zero, then `y`
is left unchanged without using `x`.  Computations are performed at the
numerical precision of `promote_eltype(x,y)`.

Optional argument `sel` is a selection of indices to which apply the operation.
Note that if an index is repeated, the operation will be performed several
times at this location.

See also: [`vscale!`](@ref), [`vcombine!](@ref).

"""
function vupdate!(y::AbstractArray{<:Floats,N},
                  α::Number,
                  x::AbstractArray{<:Floats,N}) where {N}
    I = all_indices(x, y)
    if α == 1
        @inbounds @simd for i in I
            y[i] += x[i]
        end
    elseif α == -1
        @inbounds @simd for i in I
            y[i] -= x[i]
        end
    elseif α != 0
        alpha = promote_multiplier(α, x)
        @inbounds @simd for i in I
            y[i] += alpha*x[i]
        end
    end
    return y
end

function vupdate!(y::AbstractArray{<:Floats,N},
                  sel::AbstractVector{Int},
                  α::Number,
                  x::AbstractArray{<:Floats,N}) where {N}
    if checkselection(sel, x, y)
        if α == 1
            @inbounds @simd for j in eachindex(sel)
                i = sel[j]
                y[i] += x[i]
            end
        elseif α == -1
            @inbounds @simd for j in eachindex(sel)
                i = sel[j]
                y[i] -= x[i]
            end
        elseif α != 0
            alpha = promote_multiplier(α, x)
            @inbounds @simd for j in eachindex(sel)
                i = sel[j]
                y[i] += alpha*x[i]
            end
        end
    end
    return y
end


#------------------------------------------------------------------------------
# LINEAR COMBINATION

"""
    vcombine(α, x, β, y) -> dst

yields the linear combination `dst = α*x + β*y`.

To avoid allocating the result, the destination array `dst` can be specified
with the in-place version of the method:

    vcombine!(dst, α, x, β, y) -> dst

The code is optimized for some specific values of the multipliers `α` and `β`.
For instance, if `α` (resp. `β`) is zero, then the prior contents of `x`
(resp. `y`) is not used.

The source(s) and the destination can be the same.  For instance, the two
following lines of code produce the same result:

    vcombine!(dst, 1, dst, α, x)
    vupdate!(dst, α, x)

See also: [`vscale!`](@ref), [`vupdate!](@ref).

"""
vcombine(α::Number, x::V, β::Number, y::V) where {V} =
    vcombine!(vcreate(x), α, x, β, y)

function vcombine!(dst::AbstractArray{<:Floats,N},
                   α::Number,
                   x::AbstractArray{<:Floats,N},
                   β::Number,
                   y::AbstractArray{<:Floats,N}) where {N}
    if α == 0
        axes(x) == axes(dst) || arguments_have_incompatible_axes()
        vscale!(dst, β, y)
    elseif β == 0
        axes(y) == axes(dst) || arguments_have_incompatible_axes()
        vscale!(dst, α, x)
    else
        I = all_indices(dst, x, y)
        if α == 1
            if β == 1
                @inbounds @simd for i in I
                    dst[i] = x[i] + y[i]
                end
            elseif β == -1
                @inbounds @simd for i in I
                    dst[i] = x[i] - y[i]
                end
            else
                beta = promote_multiplier(β, y)
                @inbounds @simd for i in I
                    dst[i] = x[i] + beta*y[i]
                end
            end
        elseif α == -1
            if β == 1
                @inbounds @simd for i in I
                    dst[i] = y[i] - x[i]
                end
            elseif β == -1
                @inbounds @simd for i in I
                    dst[i] = -x[i] - y[i]
                end
            else
                beta = promote_multiplier(β, y)
                @inbounds @simd for i in I
                    dst[i] = beta*y[i] - x[i]
                end
            end
        else
            alpha = promote_multiplier(α, x)
            if β == 1
                @inbounds @simd for i in I
                    dst[i] = alpha*x[i] + y[i]
                end
            elseif β == -1
                @inbounds @simd for i in I
                    dst[i] = alpha*x[i] - y[i]
                end
            else
                beta = promote_multiplier(β, y)
                @inbounds @simd for i in I
                    dst[i] = alpha*x[i] + beta*y[i]
                end
            end
        end
    end
    return dst
end

@doc @doc(vcombine) vcombine!

#------------------------------------------------------------------------------
# INNER PRODUCT

"""
    vdot([T,] [w,] x, y)

yields the inner product of `x` and `y`; that is, the sum of `conj(x[i])*y[i]`
or, if `w` is specified, the sum of `w[i]*conj(x[i])*y[i]` (`w` must have
real-valued elements), for all indices `i`.  Optional argument `T` is the
floating point type of the result.

Another possibility is:

    vdot([T,] sel, x, y)

with `sel` a selection of indices to restrict the computation of the inner
product to some selected elements.  This yields the sum of `x[i]*y[i]` for all
`i ∈ sel`.

If the arguments have complex-valued elements and `T` is specified as a
floating-point type, complexes are considered as vectors of pairs of reals and
the result is:

    vdot(T::Type{AbstractFloat}, x, y)
    -> ((x[1].re*y[1].re + x[1].im*y[1].im) +
        (x[2].re*y[2].re + x[2].im*y[2].im) + ...)

"""
vdot(::Type{T}, x, y) where {T<:AbstractFloat} = T(vdot(x,y))
vdot(::Type{T}, w, x, y) where {T<:AbstractFloat} = T(vdot(w,x,y))

function vdot(x::AbstractArray{<:AbstractFloat,N},
              y::AbstractArray{<:AbstractFloat,N}) where {N}
    s = zero(promote_eltype(x, y))
    @inbounds @simd for i in all_indices(x, y)
        s += x[i]*y[i]
    end
    return s
end

function vdot(x::AbstractArray{<:Complex{<:AbstractFloat},N},
              y::AbstractArray{<:Complex{<:AbstractFloat},N}) where {N}
    s = zero(promote_eltype(x, y))
    @inbounds @simd for i in all_indices(x, y)
        s += conj(x[i])*y[i]
    end
    return s
end

# This one yields the real part of the dot product, just as if complexes were
# pairs of reals.
function vdot(T::Type{<:AbstractFloat},
              x::AbstractArray{<:Complex{<:AbstractFloat},N},
              y::AbstractArray{<:Complex{<:AbstractFloat},N}) where {N}
    s = zero(real(promote_eltype(x, y)))
    @inbounds @simd for i in all_indices(x, y)
        xi = x[i]
        yi = y[i]
        s += real(xi)*real(yi) + imag(xi)*imag(yi)
    end
    return T(s)
end

function vdot(T::Type{Complex{<:AbstractFloat}},
              x::AbstractArray{<:Complex{<:AbstractFloat},N},
              y::AbstractArray{<:Complex{<:AbstractFloat},N}) where {N}
    return T(vdot(x, y))
end

function vdot(w::AbstractArray{<:AbstractFloat,N},
              x::AbstractArray{<:AbstractFloat,N},
              y::AbstractArray{<:AbstractFloat,N}) where {N}
    s = zero(promote_eltype(w, x, y))
    @inbounds @simd for i in all_indices(w, x, y)
        s += w[i]*x[i]*y[i]
    end
    return s
end

function vdot(T::Type{<:AbstractFloat},
              w::AbstractArray{<:AbstractFloat,N},
              x::AbstractArray{<:AbstractFloat,N},
              y::AbstractArray{<:AbstractFloat,N}) where {N}
    return T(vdot(w, x, y))
end

function vdot(w::AbstractArray{<:AbstractFloat,N},
              x::AbstractArray{<:Complex{<:AbstractFloat},N},
              y::AbstractArray{<:Complex{<:AbstractFloat},N}) where {N}
    s = zero(promote_eltype(w, x, y))
    @inbounds @simd for i in all_indices(w, x, y)
        s += w[i]*conj(x[i])*y[i]
    end
    return s
end

function vdot(T::Type{<:AbstractFloat},
              w::AbstractArray{<:AbstractFloat,N},
              x::AbstractArray{<:Complex{<:AbstractFloat},N},
              y::AbstractArray{<:Complex{<:AbstractFloat},N}) where {N}
    s = zero(real(promote_eltype(w, x, y)))
    @inbounds @simd for i in all_indices(w, x, y)
        xi = x[i]
        yi = y[i]
        s += (real(xi)*real(yi) + imag(xi)*imag(yi))*w[i]
    end
    return T(s)
end

function vdot(T::Type{Complex{<:AbstractFloat}},
              w::AbstractArray{<:AbstractFloat,N},
              x::AbstractArray{<:Complex{<:AbstractFloat},N},
              y::AbstractArray{<:Complex{<:AbstractFloat},N}) where {N}
    return T(vdot(w, x, y))
end

function vdot(sel::AbstractVector{Int},
              x::AbstractArray{<:AbstractFloat,N},
              y::AbstractArray{<:AbstractFloat,N}) where {N}
    s = zero(promote_eltype(x, y))
    if checkselection(sel, x, y)
        @inbounds @simd for j in eachindex(sel)
            i = sel[j]
            s += x[i]*y[i]
        end
    end
    return s
end

function vdot(T::Type{<:AbstractFloat},
              sel::AbstractVector{Int},
              x::AbstractArray{<:AbstractFloat,N},
              y::AbstractArray{<:AbstractFloat,N}) where {N}
    return T(vdot(sel, x, y))
end

function vdot(sel::AbstractVector{Int},
              x::AbstractArray{<:Complex{<:AbstractFloat},N},
              y::AbstractArray{<:Complex{<:AbstractFloat},N}) where {N}
    s = zero(promote_eltype(x, y))
    if checkselection(sel, x, y)
        @inbounds @simd for j in eachindex(sel)
            i = sel[j]
            s += conj(x[i])*y[i]
        end
    end
    return s
end

function vdot(T::Type{<:AbstractFloat},
              sel::AbstractVector{Int},
              x::AbstractArray{<:Complex{<:AbstractFloat},N},
              y::AbstractArray{<:Complex{<:AbstractFloat},N}) where {N}
    s = zero(real(promote_eltype(x, y)))
    if checkselection(sel, x, y)
        @inbounds @simd for j in eachindex(sel)
            i = sel[j]
            xi = x[i]
            yi = y[i]
            s += real(xi)*real(yi) + imag(xi)*imag(yi)
        end
    end
    return T(s)
end

function vdot(T::Type{Complex{<:AbstractFloat}},
              sel::AbstractVector{Int},
              x::AbstractArray{<:Complex{<:AbstractFloat},N},
              y::AbstractArray{<:Complex{<:AbstractFloat},N}) where {N}
    return T(vdot(sel, x, y))
end

# Check compatibility os selected indices with other specifed array(s) and
# return whether the selection is non-empty.
@inline function checkselection(sel::AbstractVector{Int},
                                A::AbstractArray{<:Any,N}) where {N}
    @assert IndexStyle(sel) === IndexLinear()
    @assert IndexStyle(A) === IndexLinear()
    flag = !isempty(sel)
    if flag
        imin, imax = extrema(sel)
        I = eachindex(IndexLinear(), A)
        ((first(I) ≤ imin) & (imax ≤ last(I))) || out_of_range_selection()
    end
    return flag
end

@inline function checkselection(sel::AbstractVector{Int},
                                A::AbstractArray{<:Any,N},
                                B::AbstractArray{<:Any,N}) where {N}
    @assert IndexStyle(B) === IndexLinear()
    axes(A) == axes(B) || arguments_have_incompatible_axes()
    checkselection(sel, A)
end

@inline function checkselection(sel::AbstractVector{Int},
                                A::AbstractArray{<:Any,N},
                                B::AbstractArray{<:Any,N},
                                C::AbstractArray{<:Any,N}) where {N}
    @assert IndexStyle(B) === IndexLinear()
    @assert IndexStyle(C) === IndexLinear()
    axes(A) == axes(B) == axes(C) || arguments_have_incompatible_axes()
    checkselection(sel, A)
end

@noinline out_of_range_selection() =
    bad_argument("some selected indices are out of range")
