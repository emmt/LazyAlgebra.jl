#
# vectors.jl -
#
# Implement basic operations for *vectors*. In `LazyAlgebra`, arrays of any number of
# dimensions are considered as *vectors*, the only requirements are that, when combining
# *vectors*, they have the same axes (i.e., for most arrays, the same dimensions).
#
#-----------------------------------------------------------------------------------------
#
# This file is part of LazyAlgebra (https://github.com/emmt/LazyAlgebra.jl)
# released under the MIT "Expat" license.
#
# Copyright (c) 2017-2025, Éric Thiébaut.
#

"""
    vnorm1([T::Type,] x)

yields the 1-norm of `x` treated as a *vector*, that is the sum of the absolute values of
the elements of `x`. An equivalent formulation is:

    mapreduce(abs, +, x)

The floating-point type of the result can be imposed by optional argument `T`.

See also [`vnorm2`](@ref) and [`vnorminf`](@ref).

"""
function vnorm1(x::AbstractArray)
    s = zero(sum_type(real(eltype(x))))
    @inbounds @fastmath @simd for i in eachindex(x)
        s += abs(x[i])
    end
    return s
end

vmorm1(x::Number) = abs(x)

"""
    vnorm2([T::Type,] x)

yields the Euclidean norm of `x` treated as a *vector*, that is the square root of the sum
of the squared absolute values of the elements of `x`. An equivalent formulation is:

    sqrt(mapreduce(abs2, +, x))

The floating-point type of the result can be imposed by optional argument `T`.

See also [`vnorm1`](@ref) and [`vnorminf`](@ref).

"""
function vnorm2(x::AbstractArray)
    R = real(eltype(x))
    s = zero(sumprod_type(R, R))
    @inbounds @fastmath @simd for i in eachindex(x)
        s += abs2(x[i])
    end
    return sqrt(s)
end

vmorm2(x::Number) = abs(x)

"""
    vnorminf([T::Type,] x)

yields the infinite-norm of `x` treated as a *vector*, that is the maximum absolute value
of the elements of `x`. An equivalent formulation is:

    mapreduce(abs, max, x)

Optional argument `T` is to specify the floating-point type of the result.

See also [`vnorm1`](@ref) and [`vnorm2`](@ref).

"""
function vnorminf(x::AbstractArray)
    s = abs(zero(eltype(x)))
    @inbounds @simd for i in eachindex(x) # do not use @fastmath for isnan to work correctly
        a = abs(x[i])
        s = (isnan(a) | (a > s)) ? a : s
    end
    return s
end

vmorminf(x::Number) = abs(x)

# Versions with forced floating-point type of output result.
for func in (:vnorm2, :vnorm1, :vnorminf)
    @eval $func(::Type{T}, x) where {T<:AbstractFloat} =
        convert_floating_point_type(T, $func(x))
end

#-----------------------------------------------------------------------------------------
# INNER PRODUCT

"""
     vdot([T::Type,] [w::AbstractArray,] x::AbstractArray, y::AbstractArray)

yields the inner product of `w`, `x`, and `y` treated as *vectors*; that is, the sum of
`conj(x[i])*y[i]` or, if `w` is specified, the sum of `w[i]*conj(x[i])*y[i]` (`w` shall
have real-valued elements), for all indices `i`. Optional argument `T` is to impose the
floating-point type of the result.

See also [`LazyAlgebra.unsafe_vdot`](@ref).

"""
function vdot(x::AbstractArray, y::AbstractArray)
    @assert_same_axes x y
    return unsafe_vdot(x, y)
end

function vdot(w::AbstractArray, x::AbstractArray, y::AbstractArray)
    @assert_same_axes w, x y
    real(eltype(w)) === eltype(w) || throw(ArgumentError("`w` shall have real-valued entries"))
    return unsafe_vdot(w, x, y)
end

@inline vdot(::Type{T}, args...) where {T<:AbstractFloat} =
    convert_floating_point_type(T, vdot(args...))

"""
    vdot([w::Real,] x::Union{Real,Complex}, y::Union{Real,Complex})

yields the inner product of `w`, `x`, and `y` both treated as 1-element *vectors*; that
is, `conj(x)*y` or, if `w` is specified, the `w*conj(x)*y` (`w` shall have real-valued
elements). This method is intended to be called by [`LazyAlgebra.unsafe_vdot`](@ref) on
the entries of its input *vectors*. This method may be extended for specific number types.

See also [`LazyAlgebra.vdot`](@ref).

"""
vdot(x::Real,    y::Real   ) = x*y
vdot(x::Real,    y::Complex) = x*real(y)
vdot(x::Complex, y::Real   ) = real(x)*y
vdot(x::Complex, y::Complex) = conj(x)*y

vdot(w::Real, x::Real,    y::Real   ) = w*x*y
vdot(w::Real, x::Real,    y::Complex) = w*x*real(y)
vdot(w::Real, x::Complex, y::Real   ) = w*real(x)*y
vdot(w::Real, x::Complex, y::Complex) = w*conj(x)*y

"""
    LazyAlgebra.unsafe_vdot([w::AbstractArray,] x::AbstractArray, y::AbstractArray)

yields the scalar product of `x` by `y` both treated as *vectors*. This method is called
by [`LazyAlgebra.vdot`](@ref) when `axes(x) == axes(y)` holds. This method may be extended
for specific array types.

"""
function unsafe_vdot(x::AbstractArray, y::AbstractArray)
    s = 0*vdot(zero(eltype(x)), zero(eltype(y)))
    @inbounds @fastmath for i in eachindex(x, y)
        s += vdot(x[i], y[i])
    end
    return s
end

function unsafe_vdot(w::AbstractArray, x::AbstractArray, y::AbstractArray)
    s = 0*vdot(zero(eltype(w)), zero(eltype(x)), zero(eltype(y)))
    @inbounds @fastmath for i in eachindex(w, x, y)
        s += vdot(w[i], x[i], y[i])
    end
    return s
end

"""
    vdot([T,] sel::AbstractVector{Int}, x::AbstractArray, y::AbstractArray)

yields the inner product of `x` and `y` restricted to the indices in `sel`; that is, the
sum of `vdot(x[i], y[i])` for all `i ∈ sel`.

"""
function vdot(sel::AbstractVector{Int}, x::AbstractArray, y::AbstractArray)
    @assert_same_axes x y
    imin, imax = extrema(sel)
    ((firstindex(x) ≤ imin) & (imax ≤ lastindex(x))) || out_of_range_selection()
    return unsafe_vdot(sel, x, y)
end

function unsafe_vdot(sel::AbstractVector{Int}, x::AbstractArray, y::AbstractArray)
    s = 0*vdot(zero(eltype(x)), zero(eltype(y)))
    if IndexStyle(x, y) == IndexLinear()
        @inbound @fastmath for j in eachindex(sel)
            i = sel[j]
            s += vdot(x[i], y[i])
        end
    else
        I = CartesianIndices(axes(x))
        @inbound @fastmath for j in eachindex(sel)
            i = I[sel[j]]
            s += vdot(x[i], y[i])
        end
    end
    return s
end

@noinline out_of_range_selection() =
    bad_argument("some selected indices are out of range")

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

sets all elements of `x` with the scalar value `α` and return `x`.  The default
implementation just calls `fill!(x,α)` but this method may be specialized for
specific types of variables `x`.

Also see [`vzero!`](@ref), [`fill!`](@ref).

"""
vfill!(x, α) = fill!(x, α)

"""
    vzero!(x) -> x

fills `x` with zeros and returns it.  The default implementation just calls
`fill!(x,zero(eltype(x)))` but this method may be specialized for specific
types of variables `x`.

Also see [`vfill!`](@ref).

"""
vzero!(x) = vfill!(x, zero(eltype(x)))

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
vones(x::AbstractArray{T}) where {T} = vfill!(vcreate(x), one(T))

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
function vupdate!(y::AbstractArray{<:Number,N},
                  α::Number,
                  x::AbstractArray{<:Number,N}) where {N}
    axes(x) == axes(y) || arguments_have_incompatible_axes()
    if α == 1
        @inbounds @simd for i in eachindex(x, y)
            y[i] += x[i]
        end
    elseif α == -1
        @inbounds @simd for i in eachindex(x, y)
            y[i] -= x[i]
        end
    elseif α != 0
        alpha = promote_multiplier(α, x)
        @inbounds @simd for i in eachindex(x, y)
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

----

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

function vcombine!(dst::AbstractArray{<:Number,N},
                   α::Number, x::AbstractArray{<:Number,N},
                   β::Number, y::AbstractArray{<:Number,N}) where {N}
    axes(dst) == axes(x) == axes(y) || arguments_have_incompatible_axes()
    if α == 0
        if β == 0
            vzero!(dst)
        elseif β == 1
            _vcombine!(dst, axpby_yields_y,     0,x, 1,y)
        elseif β == -1
            _vcombine!(dst, axpby_yields_my,    0,x,-1,y)
        else
            b = promote_multiplier(β, y)
            _vcombine!(dst, axpby_yields_by,    0,x, b,y)
        end
    elseif α == 1
        if β == 0
            _vcombine!(dst, axpby_yields_x,     1,x, 0,y)
        elseif β == 1
            _vcombine!(dst, axpby_yields_xpy,   1,x, 1,y)
        elseif β == -1
            _vcombine!(dst, axpby_yields_xmy,   1,x,-1,y)
        else
            b = promote_multiplier(β, y)
            _vcombine!(dst, axpby_yields_xpby,  1,x, b,y)
        end
    elseif α == -1
        if β == 0
            _vcombine!(dst, axpby_yields_mx,   -1,x, 0,y)
        elseif β == 1
            _vcombine!(dst, axpby_yields_ymx,  -1,x, 1,y)
        elseif β == -1
            _vcombine!(dst, axpby_yields_mxmy, -1,x,-1,y)
        else
            b = promote_multiplier(β, y)
            _vcombine!(dst, axpby_yields_bymx, -1,x, b,y)
        end
    else
        a = promote_multiplier(α, x)
        if β == 0
            _vcombine!(dst, axpby_yields_ax,    a,x, 0,y)
        elseif β == 1
            _vcombine!(dst, axpby_yields_axpy,  a,x, 1,y)
        elseif β == -1
            _vcombine!(dst, axpby_yields_axmy,  a,x,-1,y)
        else
            b = promote_multiplier(β, y)
            _vcombine!(dst, axpby_yields_axpby, a,x, b,y)
        end
    end
    return dst
end

function _vcombine!(dst::AbstractArray{<:Number,N},
                    f::Function,
                    α::Number, x::AbstractArray{<:Number,N},
                    β::Number, y::AbstractArray{<:Number,N}) where {N}
    @inbounds @simd for i in eachindex(dst, x, y)
        dst[i] = f(α, x[i], β, y[i])
    end
end


@doc @doc(vcombine) vcombine!
