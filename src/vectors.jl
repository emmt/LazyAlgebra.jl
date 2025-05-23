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
    y = vscale(α, x)
    y = vscale(x, α)

yield a new array `y` whose elements are those of array `x` multiplied by the scalar `α`
following the conventions:

- The floating-point type of the result only depends on the type of the elements of `x`.

- If `α == 𝟘` holds, `y` is zero-filled.

See also [`vscale!`](@ref).

"""
vscale(x::AbstractArray, α::Number) = vscale(α, x)
function vscale(α::Number, x::AbstractArray)
    # Convert the multiplier to infer the element type of the result. The extra cost of
    # converting the multiplier twice (if any, since further conversions should leave the
    # multiplier unchanged) is certainly negligible compared to the allocation and
    # computation times.
    α = convert_multiplier(α, eltype(x))
    T = prod_type(typeof(α), eltype(x))
    y = similar(x, T)

    # Call in-place method at stage 1 to dispatch on the value of `α` because array axes
    # are guaranteed to be the same.
    return vscale!(y, α, x, _Stage(1))
end

"""
    vscale!(x, α) -> x
    vscale!(α, x) -> x

overwrite `x` with `α*x` and returns `x`. Another possibility is:

    vscale!(y, α, x) -> y

to overwrite `y` with `α*x` and returns `y`.

Multiplier `α` shall be dimensionless. The convention is that the destination is
zero-filled if `α == 𝟘` holds (whatever the values of `x`) and that nothing is done if `y`
is unspecified (topmost cases) or if `y` is `x` and if `α == 𝟙` holds.

See also [`vscale`](@ref), [`vzeros!`](@ref), [`LinearAlgebra.rmul!](@ref), and
[`LazyAlgebra.unsafe_vscale!`](@ref).

"""
vscale!(α::Number, x::AbstractArray) = vscale!(x, α)

# Stages of in-place scaling:
#   0. Convert multiplier.
#   1. Dispatch on multiplier.
#   2. Call `unsafe_vscale!` if necessary.
function vscale!(x::AbstractArray, α::Number)
    α′ = convert_inplace_multiplier(α, eltype(x))
    return vscale!(x, α′, _Stage(1))
end
function vscale!(x::AbstractArray, α::Number, ::Stage{1})
    @dispatch_on_multiplier α vscale!(x, α, _Stage(2))
    return x
end
function vscale!(x::AbstractArray, α::Number, ::Stage{2})
    α == 𝟙 || unsafe_vscale!(x, α)
    return x
end

# Stages of out-of-place scaling:
#   0. Call in-place scaling if `x` and `y` are the same thing; otherwise, check axes and
#      proceed with stage 1.
#   1. Convert multiplier.
#   2. Dispatch on multiplier to call `unsafe_vscale!`.
function vscale!(y::AbstractArray, α::Number, x::AbstractArray)
    y === x && return vscale!(x, α)
    @assert_same_axes x y
    return vscale!(y, α, x, _Stage(1))
end
function vscale!(y::AbstractArray, α::Number, x::AbstractArray, ::Stage{1})
    α′ = convert_multiplier(α, eltype(x))
    return vscale!(y, α′, x, _Stage(2))
end
function vscale!(y::AbstractArray, α::Number, x::AbstractArray, ::Stage{2})
    @dispatch_on_multiplier α unsafe_vscale!(y, α, x)
    return y
end

"""
    LazyAlgebra.unsafe_vscale!(x::AbstractArray, α::Number) -> x

scales in-place the values of `x` by the scalar `α` and returns `x`. Another possibility
is:

    LazyAlgebra.unsafe_vscale!(y::AbstractArray, α::Number, x::AbstractArray) -> y

to overwrite `y` with `α*x`.

The statement:

    isone(β) || unsafe_vscale!(y, β)

may be used in an *unsafe method* like [`LazyAlgebraunsafe_vmul!`](@ref) to pre-scale an
output array `y` by its multiplier `β` provided `β` has been converted to an efficient
type as should be the case at this stage.

!!! warning
    This function shall be called with `α` converted to a efficient type and, if `y` is
    specified, after having checked that `y` and `x` have the same axes.

See also [`vscale!`](@ref).

"""
function unsafe_vscale!(x::AbstractArray, α::Number)
    @inbounds @fastmath @simd for i in eachindex(x)
        x[i] *= α
    end
    return x
end

function unsafe_vscale!(y::AbstractArray, α::Number, x::AbstractArray)
    @inbounds @fastmath @simd for i in eachindex(x, y)
        y[i] = α*x[i]
    end
    return y
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

# Stages of updating:
#   0. Check axes.
#   1. Convert multiplier `α`.
#   2. Dispatch on the value of the multiplier `α`.
#   3. Call unsafe method if multiplier `α` is non-zero.

function vupdate!(y::AbstractArray, α::Number, x::AbstractArray)
    @assert_same_axes x y
    return vupdate!(y, α, x, _Stage(1))
end
function vupdate!(y::AbstractArray{Ty,N}, α::Number, x::AbstractArray{Tx,N},
                  ::Stage{1}) where {Tx,Ty,N}
    α′ = convert_multiplier(α, eltype(x))
    return vupdate!(y, α′, x, _Stage(2))
end
function vupdate!(y::AbstractArray{Ty,N}, α::Number, x::AbstractArray{Tx,N},
                  ::Stage{2}) where {Tx,Ty,N}
    @dispatch_on_multiplier α vupdate!(y, α, x, _Stage(3))
    return y
end
function vupdate!(y::AbstractArray{Ty,N}, α::Number, x::AbstractArray{Tx,N},
                  ::Stage{3}) where {Tx,Ty,N}
    α isa StaticMultiplier{0} || unsafe_vupdate!(y, α, x)
    return y
end

# Idem with a selection of indices.

function vupdate!(y::AbstractArray, sel::AbstractVector{Int},
                  α::Number, x::AbstractArray)
    @assert_same_axes x y
    imin, imax = extrema(sel)
    ((firstindex(x) ≤ imin) & (imax ≤ lastindex(x))) || out_of_range_selection()
    return vupdate!(y, sel, α, x, _Stage(1))
end
function vupdate!(y::AbstractArray{Ty,N}, sel::AbstractVector{Int},
                  α::Number, x::AbstractArray{Tx,N}, ::Stage{1}) where {Tx,Ty,N}
    α′ = convert_multiplier(α, eltype(x))
    return vupdate!(y, sel, α′, x, _Stage(2))
end
function vupdate!(y::AbstractArray{Ty,N}, sel::AbstractVector{Int},
                  α::Number, x::AbstractArray{Tx,N}, ::Stage{2}) where {Tx,Ty,N}
    @dispatch_on_multiplier α vupdate!(y, sel, α, x, _Stage(3))
    return y
end
function vupdate!(y::AbstractArray{Ty,N}, sel::AbstractVector{Int},
                  α::Number, x::AbstractArray{Tx,N}, ::Stage{3}) where {Tx,Ty,N}
    α isa StaticMultiplier{0} || unsafe_vupdate!(y, sel, α, x)
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
    axes and with `α` converted to an efficient type.

See also [`vupdate!`](@ref).

"""
function unsafe_vupdate!(y::AbstractArray{Ty,N},
                         α::Number, x::AbstractArray{Tx,N}) where {Tx,Ty,N}
    @inbounds @inbounds @simd for i in eachindex(x, y)
        y[i] += α*x[i]
    end
    return y
end

function unsafe_vupdate!(y::AbstractArray{Ty,N}, sel::AbstractVector{Int},
                         α::Number, x::AbstractArray{Tx,N}) where {Tx,Ty,N}
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
function vcombine(α::Number, x::AbstractArray, β::Number, y::AbstractArray)
    # Array arguments must have the same axes.
    @assert_same_axes x y

    # Convert multipliers to infer the element type of the result. The extra cost of
    # converting the multipliers twice (if any, since further conversions should leave the
    # multipliers unchanged) is certainly negligible compared to the allocation and
    # computation times.
    α = convert_multiplier(α, eltype(x))
    β = convert_multiplier(β, eltype(y))
    Tz = sum_type(prod_type(typeof(α), eltype(x)), prod_type(typeof(β), eltype(y)))
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

# Stages for `vcombine!(α,x,β,y)`:
#   0. Check axes.
#   1. Convert `α`.
#   2. Dispatch on `α`.
#   3. Call `vscal!` if `α` is zero; convert `β` and proceed with next stage otherwise.
#   4. Dispatch on `β` and call the unsafe method.

function vcombine!(α::Number, x::AbstractArray,
                   β::Number, y::AbstractArray)
    @assert_same_axes x y
    return vcombine!(α, x, β, y, _Stage(1))
end
function vcombine!(α::Number, x::AbstractArray{Tx,N},
                   β::Number, y::AbstractArray{Ty,N}, ::Stage{1}) where {Tx,Ty,N}
    α′ = convert_multiplier(α, eltype(x))
    return vcombine!(α′, x, β, y, _Stage(2))
end
function vcombine!(α::Number, x::AbstractArray{Tx,N},
                   β::Number, y::AbstractArray{Ty,N}, ::Stage{2}) where {Tx,Ty,N}
    @dispatch_on_multiplier α vcombine!(α, x, β, y, _Stage(3))
    return y
end
function vcombine!(α::Number, x::AbstractArray{Tx,N},
                   β::Number, y::AbstractArray{Ty,N}, ::Stage{3}) where {Tx,Ty,N}
    if α isa StaticMultiplier{0}
        vscale!(y, β)
    else
        β′ = convert_inplace_multiplier(β, eltype(y))
        vcombine!(α, x, β′, y, _Stage(4))
    end
    return y
end
function vcombine!(α::Number, x::AbstractArray{Tx,N},
                   β::Number, y::AbstractArray{Ty,N}, ::Stage{4}) where {Tx,Ty,N}
    @dispatch_on_multiplier β unsafe_vcombine!(α, x, β, y)
    return y
end

# Idem for `vcombine!(z, α,x,β,y)`:

function vcombine!(z::AbstractArray,
                   α::Number, x::AbstractArray,
                   β::Number, y::AbstractArray)
    @assert_same_axes x y z
    return vcombine!(z, α, x, β, y, _Stage(1))
end
function vcombine!(z::AbstractArray{Tz,N},
                   α::Number, x::AbstractArray{Tx,N},
                   β::Number, y::AbstractArray{Ty,N}, ::Stage{1}) where {Tx,Ty,Tz,N}
    α′ = convert_multiplier(α, eltype(x))
    return vcombine!(z, α′, x, β, y, _Stage(2))
end
function vcombine!(z::AbstractArray{Tz,N},
                   α::Number, x::AbstractArray{Tx,N},
                   β::Number, y::AbstractArray{Ty,N}, ::Stage{2}) where {Tx,Ty,Tz,N}
    @dispatch_on_multiplier α vcombine!(z, α, x, β, y, _Stage(3))
    return z
end
function vcombine!(z::AbstractArray{Tz,N},
                   α::Number, x::AbstractArray{Tx,N},
                   β::Number, y::AbstractArray{Ty,N}, ::Stage{3}) where {Tx,Ty,Tz,N}
    if α isa StaticMultiplier{0}
        vscale!(z, β, y, _Stage(1))
    else
        β′ = convert_multiplier(β, eltype(y))
        vcombine!(z, α, x, β′, y, _Stage(4))
    end
    return z
end
function vcombine!(z::AbstractArray{Tz,N},
                   α::Number, x::AbstractArray{Tx,N},
                   β::Number, y::AbstractArray{Ty,N}, ::Stage{4}) where {Tx,Ty,Tz,N}
    @dispatch_on_multiplier β unsafe_vcombine!(z, α, x, β, y)
    return z
end

"""
    LazyAlgebra.unsafe_vcombine!(α, x, β, y) -> y

overwrites `y` with `α*x + β*y` and returns `y`.

!!! note
    This function may be extended to support specific array types.

!!! warning
    This function shall only be called after having checked that `x` and `y` have the same
    axes and with `α` and `β` converted to efficient types.

See also [`vcombine!`](@ref).

"""
function unsafe_vcombine!(α::Number, x::AbstractArray{Tx,N},
                          β::Number, y::AbstractArray{Ty,N}) where {Tx,Ty,N}
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
    the same axes and with `α` and `β` converted to efficient types.

See also [`vcombine!`](@ref).

"""
function unsafe_vcombine!(z::AbstractArray{Tz,N},
                          α::Number, x::AbstractArray{Tx,N},
                          β::Number, y::AbstractArray{Ty,N}) where {Tx,Ty,Tz,N}
    @inbounds @fastmath @simd for i in eachindex(x, y, z)
        z[i] = α*x[i] + β*y[i]
    end
    return z
end
