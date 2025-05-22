# vectors.jl -
#
# Implement basic operations for *vectors*. In `LazyAlgebra`, arrays of any number of
# dimensions are considered as *vectors*, the only requirements are that, when combining
# *vectors*, they have the same axes (i.e., for most arrays, the same dimensions).

#--------------------------------------------------------------------------------- VNORM -

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

vnorm1(x::Number) = abs(x)

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

vnorm2(x::Number) = abs(x)

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

vnorminf(x::Number) = abs(x)

# Versions with forced floating-point type of output result.
for func in (:vnorm2, :vnorm1, :vnorminf)
    @eval $func(::Type{T}, x) where {T<:AbstractFloat} =
        convert_floating_point_type(T, $func(x))
end

#---------------------------------------------------------------------------------- VDOT -

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
    @assert_same_axes w x y
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

yields the scalar product of `x` by `y` both treated as *vectors*. This method shall only
be called after having asserted that `axes(x) == axes(y)` holds. This method may be
extended for specific array types.

See also [`LazyAlgebra.vdot`](@ref).

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
    # NOTE We cannot use `@simd` here due to scattering.
    s = 0*vdot(zero(eltype(x)), zero(eltype(y)))
    if IndexStyle(x, y) == IndexLinear()
        @inbounds @fastmath for j in eachindex(sel)
            i = sel[j]
            s += vdot(x[i], y[i])
        end
    else
        I = CartesianIndices(axes(x))
        @inbounds @fastmath for j in eachindex(sel)
            i = I[sel[j]]
            s += vdot(x[i], y[i])
        end
    end
    return s
end

@noinline out_of_range_selection() =
    bad_argument("some selected indices are out of range")

#--------------------------------------------------------------------------------- VCOPY -

"""
    vcopy(x::AbstractArray)

yields a fresh copy of `x`. Compared to `copy(x)`, the element type of the result is
guaranteed to be floating-point.

See also [`vcopy!`](@ref), [`vcreate`](@ref), and [`LazyAlgebra.unsafe_vcopy!`](@ref).

"""
vcopy(x) = unsafe_vcopy!(vcreate(x), x)

"""
    vcopy!(dst, src) -> dst

copies the contents of `src` into `dst` and returns `dst`. An exception is thrown if `dst`
and `src` do not have the same axes.

See also [`copyto!`](@ref), [`vcopy`](@ref), [`vswap!`](@ref), and
[`LazyAlgebra.unsafe_vcopy!`](@ref).

"""
function vcopy!(dst::AbstractArray, src::AbstractArray)
    if dst !== src
        @assert_same_axes dst src
        unsafe_vcopy!(dst, src)
    end
    return dst
end

"""
    LazyAlgebra.unsafe_vcopy!(dst::AbstractArray, src::AbstractArray)

copies the values of `src` into `dst`.

!!! warning
    This function shall only be called if `dst` and `src` are different objects with the
    the same axes.

See also [`vcopy!](@ref).

"""
unsafe_vcopy!(dst::AbstractArray, src::AbstractArray) =
    copyto!(dst, firstindex(dst), src, firstindex(src), length(dst))

#------------------------------------------------------------------------------- VCREATE -

"""
    vcreate(x::AbstractArray)

yields an array similar to `x` and with elements of floating-point type.

See also [`vcopy`](@ref).

"""
vcreate(x::AbstractArray) = similar(x, float(eltype(x)))

#--------------------------------------------------------------------------------- VSWAP -

"""
    vswap!(x, y)

exchanges the contents of `x` and `y`. An exception is thrown if `x` and `y` do not have
the same axes.

See also [`vcopy!`](@ref) and [`LazyAlgebra.unsafe_vswap!`](@ref).

"""
function vswap!(x::AbstractArray, y::AbstractArray)
    if x !== y
        @assert_same_axes x y
        unsafe_vswap!(x, y)
    end
    return nothing
end

"""
    LazyAlgebra.unsafe_vswap!(x::AbstractArray, y::AbstractArray)

swaps the values of `x` and `y`.

!!! warning
    This function shall only be called by if `x` and `y` are different objects with the
    same axes.

See also [`vswap!](@ref).

"""
function unsafe_vswap!(x::AbstractArray, y::AbstractArray)
    @inbounds @simd for i in eachindex(x, y)
        temp = x[i]
        x[i] = y[i]
        y[i] = temp
    end
end

#--------------------------------------------------------------------------------- VFILL -

"""
    vfill!(x, α) -> x

sets all elements of `x` with the scalar value `α` and return `x`. The default
implementation just calls `fill!` with `α` convereted to `eltype(x)` but this method may
be specialized for specific types of variables `x`.

See also [`vzeros!`](@ref), and [`vzeros`](@ref).

"""
vfill!(x::AbstractArray, α::Number) = fill!(x, as(eltype(x), α))

#------------------------------------------------------------------ VZEROS, VONES, VNANS -

"""
    vzeros!(x) -> x

fills `x` with zeros and returns it. The default implementation just calls
`fill!(x, zero(eltype(x)))` but this method may be specialized for specific types of
variables `x`.

See also [`vfill!`](@ref) and [`vzeros`](@ref).

"""
vzeros!(x::AbstractArray) = vfill!(x, zero(eltype(x)))

"""
    vzeros(x)

yields an array similar to `x` but filled with zeros and of floating-point type.

See also [`vones`](@ref), [`vfill!`](@ref), and [`vcreate`](@ref).

"""
vzeros(x::AbstractArray) = vzeros!(vcreate(x))

"""
    vones!(x) -> x

fills `x` with ones and returns it. The default implementation just calls
`fill!(x, one(eltype(x)))` but this method may be specialized for specific types of
variables `x`.

See also [`vfill!`](@ref) and [`vones`](@ref).

"""
vones!(x::AbstractArray) = vfill!(x, one(eltype(x)))

"""
    vones(x)

yields an array similar to `x` but filled with ones and of floating-point type.

See also [`vzeros`](@ref) and [`vfill!`](@ref).

"""
vones(x::AbstractArray) = vones!(vcreate(x))

"""
    vnans(x)

yields an array similar to `x` but filled with NaNs and of floating-point type.

See also [`vones`](@ref), [`vfill!`](@ref), and [`vcreate`](@ref).

"""
vnans(x::AbstractArray) = vnans!(vcreate(x))

"""
    vnans!(x) -> x

fills `x` with NaNs and returns it. The default implementation just calls `fill!(x,
NaN*zero(eltype(x)))` but this method may be specialized for specific types of variables
`x`.

See also [`vfill!`](@ref) and [`vnans`](@ref).

"""
vnans!(x::AbstractArray) = vfill!(x, NaN*zero(eltype(x)))

#-------------------------------------------------------------------------------- VSCALE -

"""
    y = vscale(α::Number, x::AbstractArray)
    y = vscale(x::AbstractArray, α::Number)

yield a new *vector* `y` whose elements are those of `x` multiplied by the scalar `α`.

See also [`vscale!`](@ref).

"""
vscale(x::AbstractArray, α::Number) = vscale(α, x)
function vscale(α::Number, x::AbstractArray)
    # NOTE The following method to infer the element type `T` of the result should be
    # inline with the one implemented by `output_eltype`.
    α = convert_multiplier(α, x)
    T = prod_type(typeof(α), eltype(x))
    return dispatch_vscale!(similar(x, T), α, x)
end

"""
    vscale!(x, α) -> x
    vscale!(α, x) -> x

overwrite `x` with `α*x` and returns `x`. The convention is that `x` is zero-filled if
`iszero(α)` holds (whatever the values of `x`) and that nothing is done if `isone(α)`
holds. Multiplier `α` shall not have units.

See also [`vscale`](@ref), [`vzeros!`](@ref), [`LinearAlgebra.rmul!](@ref), and
[`LazyAlgebra.dispatch_vscale!`](@ref).

"""
vscale!(α::Number, x::AbstractArray) = vscale!(x, α)

vscale!(x::AbstractArray, α::Number) =
    dispatch_vscale!(x, convert_multiplier(α, eltype(x)))

"""
    LazyAlgebra.dispatch_vscale!(x, α) -> x

overwrites `x` with `α*x` and returns `x`.

This method calls [`vzeros!(x)`](@ref vzeros!) if `iszero(α)` holds and [`unsafe_vscale!(x,
α)`](@ref LazyAlgebra.unsafe_vscale!) if neither `iszero(α)` nor `isone(α)` hold.

!!! warning
    This method shall be called with the multiplier `α` converted to a suitable
    floating-point type.

See also [`vscale!`](@ref).

"""
function dispatch_vscale!(x::AbstractArray, α::Number)
    if iszero(α)
        vzeros!(x)
    elseif !isone(α)
        unsafe_vscale!(x, α)
    end
    return x
end

"""
    LazyAlgebra.unsafe_vscale!(x::AbstractArray, α::Number) -> x

scales in-place the values of `x` by the scalar `α` and returns `x`.

!!! warning
    This function shall be called with `α` converted to a suitable floating-point type and
    only when neither `iszero(α)` nor `isone(α)` hold.

See also [`vscale!`](@ref) and [`LazyAlgebra.dispatch_vscale!`](@ref).

"""
function unsafe_vscale!(x::AbstractArray, α::Number)
    @inbounds @simd for i in eachindex(x)
        x[i] *= α
    end
    return x
end

"""
    vscale!(dst, α, src) -> dst

overwrites `dst` with `α*src` and returns `dst`.

See also [`vscale`](@ref), [`vcopy!`](@ref), [`LinearAlgebra.rmul!](@ref), and
[`LazyAlgebra.dispatch_vscale!`](@ref).

"""
function vscale!(dst::AbstractArray, α::Number, src::AbstractArray)
    dst === src && return vscale!(dst, α)
    @assert_same_axes dst src
    return dispatch_vscale!(dst, convert_multiplier(α, src), src)
end

"""
    LazyAlgebra.dispatch_vscale!(dst, α, x) -> dst

overwrites `dst` with `α*x` and returns `dst`.

This method calls [`vzeros!(x)`](@ref vzeros!) if `iszero(α)` holds, [`unsafe_vcopy!(dst,
src)`](@ref LazyAlgebra.unsafe_vcopy!) if `isone(α)` holds, and [`unsafe_vscale!(dst, α,
src)`](@ref LazyAlgebra.unsafe_vscale!) if neither `iszero(α)` nor `isone(α)` hold.

!!! warning
    This method shall only be called after having checked that `dst` and `x` have the same
    axes and with the multiplier `α` converted to a suitable floating-point type.

See also [`vscale!`](@ref).

"""
function dispatch_vscale!(dst::AbstractArray, α::Number, src::AbstractArray)
    if iszero(α)
        vzeros!(dst)
    elseif isone(α)
        unsafe_vcopy!(dst, src)
    else
        unsafe_vscale!(dst, α, src)
    end
    return dst
end

"""
    LazyAlgebra.unsafe_vscale!(dst::AbstractArray, α::Number, src::AbstractArray) -> dst

overwrites `dst` with `α*src` and returns `dst`.

!!! warning
    This function shall only be called after having checked that `dst` and `x` have the
    same axes, with the multiplier `α` converted to a suitable floating-point type, and if
    neither `iszero(α)` nor `isone(α)` hold.

See also [`vscale!`](@ref) and [`LazyAlgebra.dispatch_vscale!`](@ref)..

"""
function unsafe_vscale!(dst::AbstractArray, α::Number, src::AbstractArray)
    @inbounds @simd for i in eachindex(dst, src)
        dst[i] = α*src[i]
    end
    return dst
end

#------------------------------------------------------------------------------ VPRODUCT -

"""
    vproduct(x, y) -> z

yields the element-wise multiplication (Hadamar product) of `x` by `y`.

See also [`vproduct!`](@ref) and [`LazyAlgebra.unsafe_vproduct`](@ref).

"""
function vproduct(x::AbstractArray{<:Any,N},
                  y::AbstractArray{<:Any,N}) where {N}
    @assert_same_axes x y
    T = prod_type(eltype(x), eltype(y))
    dst = similar(x, T)
    unsafe_vproduct!(dst, x, y)
    return dst
end

"""
    vproduct!(dst, [sel,] x, y) -> dst

overwrites `dst` with the elementwise multiplication (Hadamar product) of `x` by `y`.

Optional argument `sel` is a selection of indices to which apply the operation. The
destination is left unchanged for indices not in `sel`. The behavior is unpredictable if
the indices in `sel` are not all unique.

See also [`vproduct`](@ref) and [`LazyAlgebra.unsafe_vproduct`](@ref).

"""
function vproduct!(dst::AbstractArray{<:Any,N},
                   x::AbstractArray{<:Any,N},
                   y::AbstractArray{<:Any,N}) where {N}
    @assert_same_axes dst x y
    unsafe_vproduct!(dst, x, y)
    return dst
end

function vproduct!(dst::AbstractArray{<:Any,N},
                   sel::AbstractVector{Int},
                   x::AbstractArray{<:Any,N},
                   y::AbstractArray{<:Any,N}) where {N}
    @assert_same_axes dst x y
    imin, imax = extrema(sel)
    ((firstindex(dst) ≤ imin) & (imax ≤ lastindex(dst))) || out_of_range_selection()
    unsafe_vproduct!(dst, sel, x, y)
    return dst
end

"""
    LazyAlgebra.unsafe_vproduct!(dst, [sel,] x, y)

overwrites `dst` with the elementwise multiplication (Hadamar product) of `x` by `y`. This
method is called by [`vproduct!`](@ref) and [`vproduct`](@ref) after checking all
arguments so that `@inbounds` can be assumed for performing the operation.

"""
function unsafe_vproduct!(dst::AbstractArray{<:Any,N},
                          x::AbstractArray{<:Any,N},
                          y::AbstractArray{<:Any,N}) where {N}
    @inbounds @fastmath @simd for i in eachindex(dst, x, y)
        dst[i] = x[i]*y[i]
    end
    nothing
end

function unsafe_vproduct!(dst::AbstractArray{<:Any,N},
                          sel::AbstractVector{Int},
                          x::AbstractArray{<:Any,N},
                          y::AbstractArray{<:Any,N}) where {N}
    # NOTE We cannot use `@simd` here due to scattering.
    if IndexStyle(dst, x, y) == IndexLinear()
        @inbounds @fastmath for i in sel
            dst[i] = x[i]*y[i]
        end
    else
        I = CartesianIndices(axes(dst))
        @inbounds @fastmath for j in sel
            i = I[j]
            dst[i] = x[i]*y[i]
        end
    end
    nothing
end

#------------------------------------------------------------------------------- VUPDATE -

"""
    vupdate!(y, [sel,] α, x) -> y

overwrites `y` with `α*x + y` and returns `y`. The code is optimized for some specific
values of the multiplier `α`. For instance, if `α` is zero, then `y` is left unchanged
without using `x`.

Optional argument `sel` is a selection of indices to which apply the operation. Note that
if an index is repeated, the operation will be performed several times at this location.

See also [`vscale!`](@ref), [`vcombine!](@ref), and [`LazyAlgebra.unsafe_vupdate!](@ref).

""" vupdate!

# Stage 0: check axes.

function vupdate!(y::AbstractArray{Ty,N},
                  α::Number, x::AbstractArray{Tx,N},
                  ::Stage{0} = _Stage(0)) where {Tx,Ty,N}
    @assert_same_axes x y
    return vupdate!(y, convert_multiplier(α, Tx), x, _Stage(1))
end

function vupdate!(y::AbstractArray{Ty,N}, sel::AbstractVector{Int},
                  α::Number, x::AbstractArray{Tx,N},
                  ::Stage{0} = _Stage(0)) where {Tx,Ty,N}
    @assert_same_axes x y
    imin, imax = extrema(sel)
    ((firstindex(x) ≤ imin) & (imax ≤ lastindex(x))) || out_of_range_selection()
    return vupdate!(y, sel, convert_multiplier(α, Tx), x, _Stage(1))
end

# Stage 1: dispatch on the value of `α`.

function vupdate!(y::AbstractArray{Ty,N},
                  α::Number, x::AbstractArray{Tx,N},
                  ::Stage{1}) where {Tx,Ty,N}
    α == ZERO || unsafe_vupdate!(y, convert_multiplier(α, Tx), x)
    return y
end

function vupdate!(y::AbstractArray{Ty,N}, sel::AbstractVector{Int},
                  α::Number, x::AbstractArray{Tx,N},
                  ::Stage{1}) where {Ty,Tx,N}
    α == ZERO || unsafe_vupdate!(y, sel, convert_multiplier(α, Tx), x)
    return y
end

"""
    LazyAlgebra.unsafe_vupdate!(y, [sel,] α, x) -> y

This method is called by [`vupdate!`](@ref) to overwrites `y` with `α*x + y` if and only
if `iszero(α)` does not hold. This method can assume `@inbounds` in its computations.

!!! note
    This function may be extended to support specific array types.

!!! warning
    This function shall only be called after having checked that `x` and `y` have the same
    axes and with `α` converted to a suitable type.

See also [`vupdate!`](@ref).

"""
function unsafe_vupdate!(y::AbstractArray{Ty,N},
                         α::Number, x::AbstractArray{Ty,N}) where {Ty,Tx,N}
    @inbounds @inbounds @simd for i in eachindex(x, y)
        y[i] += α*x[i]
    end
    return y
end

function unsafe_vupdate!(y::AbstractArray{<:Any,N},
                         sel::AbstractVector{Int},
                         α::Number,
                         x::AbstractArray{<:Any,N}) where {N}
    # NOTE We cannot use `@simd` here due to scattering.
    if IndexStyle(x, y) == IndexLinear()
        @inbounds @fastmath for i in sel
            y[i] += α*x[i]
        end
    else
        I = CartesianIndices(axes(x))
        @inbounds @fastmath for j in sel
            i = I[j]
            y[i] += α*x[i]
        end
    end
    return y
end

#------------------------------------------------------------------------------ VCOMBINE -

"""
    vcombine(α, x, β, y) -> z

yields the linear combination `z = α*x + β*y` throwing an exception if `x` and `y` do not
have the same axes.

See also [`vcombine!`](@ref), [`vscale!`](@ref), [`vupdate!](@ref), and
[`LazyAlgebra.convert_multiplier](@ref).

"""
function vcombine(α::Number, x::AbstractArray{Tx,N},
                  β::Number, y::AbstractArray{Ty,N}) where {Tx,Ty,N}
    # Array arguments must have the same axes.
    @assert_same_axes x y

    # Convert multipliers to infer the element type of the result. The extra cost of
    # converting the multipliers twice (if any, since further conversions should leave the
    # multipliers unchanged) is certainly negligible compared to the allocation and
    # computation times.
    α = convert_multiplier(α, Tx)
    β = convert_multiplier(β, Ty)
    Tz = sum_type(prod_type(typeof(α), Tx), prod_type(typeof(β), Ty))
    z = similar(x, Tz) # FIXME type of array does not depend on y

    # Call in-place method at stage 1 to dispatch on the values of `α` and `β` because
    # array axes have already been checked.
    return vcombine!(z, α, x, β, y, _Stage(1))
end

"""
    vcombine!(z=y, α, x, β, y) -> z

overwrites `z` with the linear combination `α*x + β*y` and returns `z`. An exception is
thrown if `x`, `y`, and `z`, do mot have the same axes. If `z` is omitted, `z = y` is
assumed.

The code is optimized for some specific values of the multipliers `α` and `β`. For
instance, if `α` (resp. `β`) is zero, then the prior contents of `x` (resp. `y`) is not
used.

The source(s) and the destination can be the same. For instance, the following lines
of code all produce the same result (stored in `y`):

    vcombine!(y, α, x, 1, y)
    vcombine!(α, x, 1, y)
    vupdate!(y, α, x)

The [`LazyAlgebra.unsafe_vcombine!](@ref) may be extended to implement specific array
types.

See also [`vcombine`](@ref), [`vscale!`](@ref), [`vupdate!](@ref),
[`LazyAlgebra.vcombine!](@ref), and [`LazyAlgebra.unsafe_vcombine!](@ref).

""" vcombine!

# Stage 0: check axes.

function vcombine!(α::Number, x::AbstractArray{Tx,N},
                   β::Number, y::AbstractArray{Ty,N},
                   ::Stage{0} = _Stage(0)) where {Tx,Ty,N}
    @assert_same_axes x y
    return vcombine!(α, x, β, y, _Stage(1))
end

function vcombine!(z::AbstractArray{Tz,N},
                   α::Number, x::AbstractArray{Tx,N},
                   β::Number, y::AbstractArray{Ty,N},
                   ::Stage{0} = _Stage(0)) where {Tz,Tx,Ty,N}
    @assert_same_axes x y z
    return vcombine!(z, α, x, β, y, _Stage(1))
end

# Stage 1: dispatch on the value of `α`.

function vcombine!(α::Number, x::AbstractArray{Tx,N},
                   β::Number, y::AbstractArray{Ty,N},
                   ::Stage{1}) where {Tx,Ty,N}
    if α isa StaticMultiplier
        vcombine!(α, x, β, y, _Stage(2))
    elseif iszero(α)
        vcombine!(ZERO*unit(α), x, β, y, _Stage(2))
    elseif α == oneunit(α)
        vcombine!(ONE*unit(α), x, β, y, _Stage(2))
    elseif is_signed(α) && α == -oneunit(α)
        vcombine!(-ONE*unit(α), x, β, y, _Stage(2))
    else
        vcombine!(convert_multiplier(α, Tx), x, β, y, _Stage(2))
    end
    return y
end

function vcombine!(z::AbstractArray{Tz,N},
                   α::Number, x::AbstractArray{Tx,N},
                   β::Number, y::AbstractArray{Ty,N},
                   ::Stage{1}) where {Tz,Tx,Ty,N}
    if α isa StaticMultiplier
        vcombine!(z, α, x, β, y, _Stage(2))
    elseif iszero(α)
        vcombine!(z, ZERO*unit(α), x, β, y, _Stage(2))
    elseif α == oneunit(α)
        vcombine!(z, ONE*unit(α), x, β, y, _Stage(2))
    elseif is_signed(α) && α == -oneunit(α)
        vcombine!(z, -ONE*unit(α), x, β, y, _Stage(2))
    else
        vcombine!(z, convert_multiplier(α, Tx), x, β, y, _Stage(2))
    end
    return z
end

# Stage 2: dispatch on the value of `β`.

function vcombine!(α::Number, x::AbstractArray{Tx,N},
                   β::Number, y::AbstractArray{Ty,N},
                   ::Stage{2}) where {Tx,Ty,N}
    if β isa StaticMultiplier
        unsafe_vcombine!(α, x, β, y)
    elseif iszero(β)
        unsafe_vcombine!(α, x, ZERO*unit(β), y)
    elseif β == oneunit(β)
        unsafe_vcombine!(α, x, ONE*unit(β), y)
    elseif is_signed(β) && β == -oneunit(β)
        unsafe_vcombine!(α, x, -ONE*unit(β), y)
    else
        unsafe_vcombine!(α, x, convert_multiplier(β, Ty), y)
    end
    return y
end

function vcombine!(z::AbstractArray{Tz,N},
                   α::Number, x::AbstractArray{Tx,N},
                   β::Number, y::AbstractArray{Ty,N},
                   ::Stage{2}) where {Tz,Tx,Ty,N}
    if β isa StaticMultiplier
        unsafe_vcombine!(z, α, x, β, y)
    elseif iszero(β)
        unsafe_vcombine!(z, α, x, ZERO*unit(β), y)
    elseif β == oneunit(β)
        unsafe_vcombine!(z, α, x, ONE*unit(β), y)
    elseif is_signed(β) && β == -oneunit(β)
        unsafe_vcombine!(z, α, x, -ONE*unit(β), y)
    else
        unsafe_vcombine!(z, α, x, convert_multiplier(β, Ty), y)
    end
    return z
end

# Last stage: the unsafe one.

"""
    LazyAlgebra.unsafe_vcombine!(α, x, β, y) -> y

overwrites `y` with `α*x + β*y` and returns `y`.

!!! note
    This function may be extended to support specific array types.

!!! warning
    This function shall only be called after having checked that `x` and `y` have the same
    axes and with `α` and `β` converted to suitable types.

See also [`vcombine!`](@ref).

"""
function unsafe_vcombine!(α::Number, x::AbstractArray,
                          β::Number, y::AbstractArray)
    @inbounds @fastmath @simd for i in eachindex(x, y)
        y[i] = α*x[i] + β*y[i]
    end
    return y
end

"""
    LazyAlgebra.unsafe_vcombine!(z, α, x, β, y) -> z

overwrites `z` with `α*x + β*y` and returns `z`.

!!! note
    This function may be extended to support specific array types.

!!! warning
    This function shall only be called after having checked that `x`, `y`, and `z` have
    the same axes and with `α` and `β` converted to suitable types.

See also [`vcombine!`](@ref).

"""
function unsafe_vcombine!(z::AbstractArray,
                          α::Number, x::AbstractArray,
                          β::Number, y::AbstractArray)
    @inbounds @fastmath @simd for i in eachindex(x, y, z)
        z[i] = α*x[i] + β*y[i]
    end
    return z
end

function unsafe_vcombine!(α::Number, x::AbstractArray{Tx,N},
                          β::Number, y::AbstractArray{Ty,N}) where {Tx,Ty,N}
    # We know that neither `α` nor `β` is zero and that `β` is not one.
    if α == one(α)
        if β == -one(β)
            @inbounds @fastmath @simd for i in eachindex(x, y)
                y[i] = x[i] - y[i]
            end
        else
            @inbounds @fastmath @simd for i in eachindex(x, y)
                y[i] = β*y[i] + x[i]
            end
        end
    elseif α == -one(α)
        @inbounds @fastmath @simd for i in eachindex(x, y)
            y[i] = β*y[i] - x[i]
        end
    else
        if β == -one(β)
            @inbounds @fastmath @simd for i in eachindex(x, y)
                y[i] = α*x[i] - y[i]
            end
        else
            @inbounds @fastmath @simd for i in eachindex(x, y)
                y[i] = α*x[i] + β*y[i]
            end
        end
    end
    return y
end
