# vectors.jl -
#
# Implement basic operations for *vectors*. In `LazyAlgebra`, arrays of any number of
# dimensions are considered as *vectors*, the only requirements are that, when combining
# *vectors*, they have the same axes (i.e., for most arrays, the same dimensions).

#--------------------------------------------------------------------------------- VNORM -

"""
    vnorm1([T::Type,] x)

Return the 1-norm of `x` treated as a *vector*, that is the sum of the absolute values of
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

Return the Euclidean norm of `x` treated as a *vector*, that is the square root of the sum
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

Return the infinite-norm of `x` treated as a *vector*, that is the maximum absolute value
of the elements of `x`. An equivalent formulation is:

    mapreduce(abs, max, x)

Optional argument `T` is to specify the floating-point type of the result.

See also [`vnorm1`](@ref) and [`vnorm2`](@ref).

"""
function vnorminf(x::AbstractArray)
    s = abs(zero(eltype(x)))
    @inbounds @simd for i in eachindex(x) # do not use @fastmath for isnan to work correctly
        s = fast_max(s, abs(x[i]))
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

Return the inner product of `w`, `x`, and `y` treated as *vectors*; that is, the sum of
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

Return the inner product of `w`, `x`, and `y` both treated as 1-element *vectors*; that
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

Return the scalar product of `x` by `y` both treated as *vectors*. This method shall only
be called after having asserted that `axes(x) == axes(y)` holds. This method may be
extended for specific array types.

See also [`LazyAlgebra.vdot`](@ref).

"""
function unsafe_vdot(x::AbstractArray, y::AbstractArray)
    T = typeof(vdot(zero(eltype(x)), zero(eltype(y)))*0)
    s = zero(T)
    @inbounds @fastmath for i in eachindex(x, y)
        s += vdot(x[i], y[i])
    end
    return s
end

function unsafe_vdot(w::AbstractArray, x::AbstractArray, y::AbstractArray)
    T = typeof(vdot(zero(eltype(w)), zero(eltype(x)), zero(eltype(y)))*0)
    s = zero(T)
    @inbounds @fastmath for i in eachindex(w, x, y)
        s += vdot(w[i], x[i], y[i])
    end
    return s
end

"""
    vdot([T,] sel::AbstractVector{Int}, x::AbstractArray, y::AbstractArray)

Return the inner product of `x` and `y` restricted to the indices in `sel`; that is, the
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
    throw_bad_argument("some selected indices are out of range")

#--------------------------------------------------------------------------------- VCOPY -

"""
    vcopy(x::AbstractArray)

Return a fresh copy of `x`. Compared to `copy(x)`, the element type of the result is
guaranteed to be floating-point.

See also [`vcopy!`](@ref), [`vcreate`](@ref), and [`LazyAlgebra.unsafe_vcopy!`](@ref).

"""
function vcopy(x)
    y = vcreate(x)
    unsafe_vcopy!(y, x)
    return y
end

"""
    vcopy!(dst, src) -> dst

Copy the content of `src` into `dst` and return `dst` throwing  an exception if `dst`
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

Copy the values of `src` into `dst`.

!!! warning
    This function shall only be called if `dst` and `src` are different objects with the
    the same axes.

See also [`vcopy!](@ref).

"""
function unsafe_vcopy!(dst::AbstractArray, src::AbstractArray)
    copyto!(dst, firstindex(dst), src, firstindex(src), length(dst))
    return nothing
end

#------------------------------------------------------------------------------- VCREATE -

"""
    vcreate(x::AbstractArray)

Create an array similar to `x` and with elements of floating-point type.

See also [`vcopy`](@ref).

"""
vcreate(x::AbstractArray) = similar(x, float(eltype(x)))

#--------------------------------------------------------------------------------- VSWAP -

"""
    vswap!(x, y)

Exchange the contents of `x` and `y` throwing an exception is if `x` and `y` do not have
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

Swap the values of `x` and `y`.

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
    return nothing
end

#--------------------------------------------------------------------------------- VFILL -

"""
    vfill!(x, α) -> x

Set all elements of `x` with the scalar value `α` and return `x`. The default
implementation just calls `fill!` with `α` convereted to `eltype(x)` but this method may
be specialized for specific types of variables `x`.

See also [`vzeros!`](@ref), and [`vzeros`](@ref).

"""
vfill!(x::AbstractArray, α::Number) = fill!(x, as(eltype(x), α))

#------------------------------------------------------------------ VZEROS, VONES, VNANS -

"""
    vzeros!(x) -> x

Fill `x` with zeros and return it. The default implementation just calls `fill!(x,
zero(eltype(x)))` but this method may be specialized for specific types of variables `x`.

See also [`vfill!`](@ref) and [`vzeros`](@ref).

"""
vzeros!(x::AbstractArray) = vfill!(x, zero(eltype(x)))

"""
    vzeros(x)

Return an array similar to `x` but filled with zeros and of floating-point type.

See also [`vones`](@ref), [`vfill!`](@ref), and [`vcreate`](@ref).

"""
vzeros(x::AbstractArray) = vzeros!(vcreate(x))

"""
    vones!(x) -> x

Fill `x` with ones and return it. The default implementation just calls `fill!(x,
oneunit(eltype(x)))` but this method may be specialized for specific types of variables
`x`.

See also [`vfill!`](@ref) and [`vones`](@ref).

"""
vones!(x::AbstractArray) = vfill!(x, oneunit(eltype(x)))

"""
    vones(x)

Return an array similar to `x` but filled with ones and of floating-point type.

See also [`vzeros`](@ref) and [`vfill!`](@ref).

"""
vones(x::AbstractArray) = vones!(vcreate(x))

"""
    vnans(x)

Return an array similar to `x` but filled with NaNs and of floating-point type.

See also [`vones`](@ref), [`vfill!`](@ref), and [`vcreate`](@ref).

"""
vnans(x::AbstractArray) = vnans!(vcreate(x))

"""
    vnans!(x) -> x

Fill `x` with NaNs and returns it. The default implementation just calls `fill!(x,
NaN*unit(eltype(x)))` but this method may be specialized for specific types of variables
`x`.

See also [`vfill!`](@ref) and [`vnans`](@ref).

"""
vnans!(x::AbstractArray) = vfill!(x, NaN*unit(eltype(x)))

#-------------------------------------------------------------------------------- VSCALE -

"""
    y = vscale(α, x)
    y = vscale(x, α)

Return a new array `y` whose elements are those of array `x` multiplied by the scalar `α`
following the conventions:

- The floating-point type of the result only depends on the type of the elements of `x`.

- If `α == 𝟘` holds, `y` is zero-filled.

See also [`vscale!`](@ref).

"""
vscale(x::AbstractArray, α::Number) = vscale(α, x)
function vscale(α::Number, x::AbstractArray)
    # Convert the multiplier to infer the element type `T` of the result. The extra cost
    # of converting the multiplier twice (if any, since further conversions should leave
    # the multiplier unchanged) is certainly negligible compared to the allocation and
    # computation times.
    α = convert_multiplier(α, eltype(x))
    T = output_eltype(α, x)
    y = similar(x, T)

    # Call in-place method at a stage to dispatch on the value of `α` because array axes
    # are guaranteed to be the same.
    unsafe_vscale!(Val(:alpha), y, α, x)
    return y
end

"""
    vscale!(x, α) -> x
    vscale!(α, x) -> x

Overwrite `x` with `α*x` and return `x`. Another possibility is:

    vscale!(y, α, x) -> y

to overwrite `y` with `α*x` and returns `y`.

Multiplier `α` shall be dimensionless. The convention is that the destination is
zero-filled if `α == 𝟘` holds (whatever the values of `x`) and that nothing is done if `y`
is unspecified (topmost cases) or if `y` is `x` and if `α == 𝟙` holds.

See also [`vscale`](@ref), [`vzeros!`](@ref), [`LinearAlgebra.rmul!](@ref), and
[`LazyAlgebra.unsafe_vscale!`](@ref).

"""
vscale!(α::Number, x::AbstractArray) = vscale!(x, α)

function vscale!(x::AbstractArray, α::Number)
    # Check arguments types and units.
    _ = convert(eltype(x), zero(α)*zero(eltype(x)))::eltype(x)
    # Deal with the multiplier.
    unsafe_vscale!(Val(:alpha), x, α)
    return x
end

function unsafe_vscale!(::Val{:alpha},
                        x::AbstractArray, α::Number)
    α = convert_multiplier(α, eltype(x))
    @dispatch_on_multiplier α unsafe_vscale!(x, α)
    return nothing
end

function vscale!(y::AbstractArray, α::Number, x::AbstractArray)
    # Check whether `x` and `y` are the same object.
    y === x && return vscale!(x, α)
    # Check arguments indices, types, and units.
    @assert_same_axes x y
    _ = convert(eltype(y), zero(α)*zero(eltype(x)))::eltype(y)
    # Deal with the multiplier.
    unsafe_vscale!(Val(:alpha), y, α, x)
    return y
end

function unsafe_vscale!(::Val{:alpha},
                        y::AbstractArray, α::Number, x::AbstractArray)
    α = convert_multiplier(α, eltype(x))
    @dispatch_on_multiplier α unsafe_vscale!(y, α, x)
    return nothing
end

"""
    LazyAlgebra.unsafe_vscale!(x::AbstractArray, α::Number)

scales in-place the values of `x` by the scalar `α`. Another possibility is:

    LazyAlgebra.unsafe_vscale!(y::AbstractArray, α::Number, x::AbstractArray)

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
    return nothing
end

function unsafe_vscale!(y::AbstractArray, α::Number, x::AbstractArray)
    @inbounds @fastmath @simd for i in eachindex(x, y)
        y[i] = α*x[i]
    end
    return nothing
end

#------------------------------------------------------------------------------ VPRODUCT -

"""
    vproduct(x, y) -> z

Return the element-wise multiplication (Hadamar product) of `x` by `y`.

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

Overwrite `dst` with the elementwise multiplication (Hadamar product) of `x` by `y`.

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

Overwrite `dst` with the elementwise multiplication (Hadamar product) of `x` by `y`. This
method is called by [`vproduct!`](@ref) and [`vproduct`](@ref) after checking all
arguments so that `@inbounds` can be assumed for performing the operation.

"""
function unsafe_vproduct!(dst::AbstractArray{<:Any,N},
                          x::AbstractArray{<:Any,N},
                          y::AbstractArray{<:Any,N}) where {N}
    @inbounds @fastmath @simd for i in eachindex(dst, x, y)
        dst[i] = x[i]*y[i]
    end
    return nothing
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
    return nothing
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

"""
function vupdate!(y::AbstractArray, α::Number, x::AbstractArray)
    # Check arguments indices, types, and units.
    @assert_same_axes x y
    _ = convert(eltype(y), zero(α)*zero(eltype(x)))::eltype(y)
    # Deal with multiplier.
    unsafe_vupdate!(Val(:alpha), y, α, x)
    return y
end

function unsafe_vupdate!(::Val{:alpha},
                         y::AbstractArray, α::Number, x::AbstractArray)
    α = convert_multiplier(α, eltype(x))
    iszero(α) && return # skip computations if `α` is zero
    @dispatch_on_multiplier α unsafe_vupdate!(y, α, x)
    return nothing
end

# Idem with a selection of indices.
function vupdate!(y::AbstractArray, sel::AbstractVector{Int},
                  α::Number, x::AbstractArray)
    # Check arguments indices, types, and units.
    @assert_same_axes x y
    _ = convert(eltype(y), zero(α)*zero(eltype(x)))::eltype(y)
    imin, imax = extrema(sel)
    ((firstindex(x) ≤ imin) & (imax ≤ lastindex(x))) || out_of_range_selection()
    # Deal with multiplier.
    unsafe_vupdate!(Val(:alpha), y, sel, α, x)
    return y
end

function unsafe_vupdate!(::Val{:alpha},
                         y::AbstractArray, sel::AbstractVector{Int},
                         α::Number, x::AbstractArray)
    α = convert_multiplier(α, eltype(x))
    iszero(α) && return # skip computations if `α` is zero
    @dispatch_on_multiplier α unsafe_vupdate!(y, sel, α, x)
    return nothing
end

"""
    LazyAlgebra.unsafe_vupdate!(y, [sel,] α, x)

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
    return nothing
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
    return nothing
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
    # Check arguments indices, types, and units.
    @assert_same_axes x y
    _ = convert(eltype(y), zero(α)*zero(eltype(x)) + zero(β)*zero(eltype(y)))::eltype(y)

    # Convert multipliers to infer the element type of the result.
    α = convert_multiplier(α, eltype(x))
    β = convert_multiplier(β, eltype(y))
    T = sum_type(output_eltype(α, x), output_eltype(β, y))
    z = similar(x, T)

    # Call unsafe method to dispatch on the values of `α` and `β` because indices, types,
    # and units have been checked,
    unsafe_vcombine!(Val(:alpha_beta), z, α, x, β, y)
    return z
end

"""
    vcombine!(z=y, α, x, β, y) -> z

Overwrite `z` with the linear combination `α*x + β*y` and return `z`. If `z` is omitted,
`z = y` is assumed.

An exception is thrown if `x`, `y`, and `z`, do mot have the same axes or if argument
types or units are incompatible.

The code is optimized for some specific values of the multipliers `α` and `β`. For
instance, if `α` (resp. `β`) is zero, then the prior contents of `x` (resp. `y`) is not
used.

The source(s) and the destination can be the same. For instance, the following lines
of code all produce the same result (stored in `y`):

    vcombine!(y, α, x, 𝟙, y)
    vcombine!(α, x, 𝟙, y)
    vupdate!(y, α, x)

The [`LazyAlgebra.unsafe_vcombine!](@ref) may be extended to implement specific array
types.

See also [`vcombine`](@ref), [`vscale!`](@ref), [`vupdate!](@ref),
[`LazyAlgebra.vcombine!](@ref), and [`LazyAlgebra.unsafe_vcombine!](@ref).

"""
function vcombine!(α::Number, x::AbstractArray, β::Number, y::AbstractArray)
    # Check arguments indices, types, and units.
    @assert_same_axes x y
    _ = convert(eltype(y), zero(α)*zero(eltype(x)) + zero(β)*zero(eltype(y)))::eltype(y)
    # Deal with multipliers.
    unsafe_vcombine!(Val(:alpha_beta), α, x, β, y)
    return y
end

function unsafe_vcombine!(::Val{:alpha_beta},
                          α::Number, x::AbstractArray, β::Number, y::AbstractArray)
    α = convert_multiplier(α, eltype(x))
    @dispatch_on_multiplier α unsafe_vcombine!(Val(:beta), α, x, β, y)
    return nothing
end

function unsafe_vcombine!(::Val{:beta},
                          α::Number, x::AbstractArray, β::Number, y::AbstractArray)
    β = convert_multiplier(β, eltype(y))
    @dispatch_on_multiplier β unsafe_vcombine!(α, x, β, y)
    return nothing
end

function unsafe_vcombine!(::Val{:alpha},
                          α::Number, x::AbstractArray, β::Number, y::AbstractArray)
    α = convert_multiplier(α, eltype(x))
    @dispatch_on_multiplier α unsafe_vcombine!(α, x, β, y)
    return nothing
end

# Idem for `vcombine!(z, α,x,β,y)`:
function vcombine!(z::AbstractArray,
                   α::Number, x::AbstractArray,
                   β::Number, y::AbstractArray)
    # Check arguments indices, types, and units.
    @assert_same_axes x y z
    _ = convert(eltype(z), zero(α)*zero(eltype(x)) + zero(β)*zero(eltype(y)))::eltype(z)
    # Deal with multipliers.
    unsafe_vcombine!(Val(:alpha_beta), z, α, x, β, y)
    return z
end

function unsafe_vcombine!(::Val{:alpha_beta},
                          z::AbstractArray,
                          α::Number, x::AbstractArray,
                          β::Number, y::AbstractArray)
    α = convert_multiplier(α, eltype(x))
    @dispatch_on_multiplier α unsafe_vcombine!(Val(:beta), z, α, x, β, y)
    return nothing
end

function unsafe_vcombine!(::Val{:alpha},
                          z::AbstractArray,
                          α::Number, x::AbstractArray,
                          β::Number, y::AbstractArray)
    α = convert_multiplier(α, eltype(x))
    @dispatch_on_multiplier α unsafe_vcombine!(z, α, x, β, y)
    return nothing
end

function unsafe_vcombine!(::Val{:beta},
                          z::AbstractArray,
                          α::Number, x::AbstractArray,
                          β::Number, y::AbstractArray)
    β = convert_multiplier(β, eltype(y))
    @dispatch_on_multiplier β unsafe_vcombine!(z, α, x, β, y)
    return nothing
end

"""
    LazyAlgebra.unsafe_vcombine!(α, x, β, y)

Overwrite `y` with `α*x + β*y`.

!!! note
    This function may be extended to support specific array types.

!!! warning
    This function shall only be called after having checked that `x` and `y` have the same
    axes and with `α` and `β` converted to efficient types.

See also [`vcombine!`](@ref).

"""
function unsafe_vcombine!(α::Number, x::AbstractArray,
                          β::Number, y::AbstractArray)
    @inbounds @fastmath @simd for i in eachindex(x, y)
        y[i] = α*x[i] + β*y[i]
    end
    return nothing
end

"""
    LazyAlgebra.unsafe_vcombine!(z, α, x, β, y)

Overwrite `z` with `α*x + β*y`.

!!! note
    This function may be extended to support specific array types.

!!! warning
    This function shall only be called after having checked that `x`, `y`, and `z` have
    the same axes and with `α` and `β` converted to efficient types.

See also [`vcombine!`](@ref).

"""
function unsafe_vcombine!(z::AbstractArray,
                          α::Number, x::AbstractArray,
                          β::Number, y::AbstractArray)
    @inbounds @fastmath @simd for i in eachindex(x, y, z)
        z[i] = α*x[i] + β*y[i]
    end
    return nothing
end

#---------------------------------------------------------------------------------- VMAP -

"""
    LazyAlgebra.vmap!(y, α, f, w, x) -> y

overwrites `y` with `α*f.(w, x)` and returns `y`. Other possibility:

    LazyAlgebra.vmap!(α, f, w, x, β, y) -> y

to overwrite `y` with `α*f.(w, x) + β*y`. An exception is thrown if `w`, `x`, and `y` do
not have the same axes.

See also [`LazyAlgebra.unsafe_vmap!`](@ref).

"""
vmap!(y::AbstractArray, α::Number, f::Function, w::AbstractArray, x::AbstractArray) =
    vmap!(α, f, w, x, 𝟘, y)

function vmap!(α::Number, f::Function, w::AbstractArray, x::AbstractArray,
               β::Number, y::AbstractArray)
    # Check arguments indices, types, and units.
    @assert_same_axes w x y
    _ = convert(eltype(y), zero(α)*zero(Base.promote_op(f, prod_type(w, x)))
                + zero(β)*zero(eltype(y)))::eltype(y)
    # Deal with multipliers.
    unsafe_vmap!(Val(:alpha_beta), α, f, w, x, β, y)
    return z
end

function unsafe_vmap!(::Val{:alpha_beta},
                      α::Number, f::Function, w::AbstractArray, x::AbstractArray,
                      β::Number, y::AbstractArray)
    # Deal with `β` than `α`.
    β = convert_multiplier(β, eltype(y))
    @dispatch_on_multiplier β unsafe_vmap!(Val(:alpha), α, f, w, x, β, y)
    return nothing
end

function unsafe_vmap!(::Val{:alpha},
                      α::Number, f::Function, w::AbstractArray, x::AbstractArray,
                      β::Number, y::AbstractArray)
    α = convert_multiplier(α, Base.promote_op(f, eltype(w), eltype(x)))
    if iszero(α)
        # Skip computing `α*f(w[i]*x[i])`.
        unsafe_vscale!(y, β)
    else
        @dispatch_on_multiplier α unsafe_vmap!(α, f, w, x, β, y)
    end
    return nothing
end

function unsafe_vmap!(::Val{:beta},
                      α::Number, f::Function, w::AbstractArray, x::AbstractArray,
                      β::Number, y::AbstractArray)
    β = convert_multiplier(β, eltype(y))
    if iszero(α)
        # Skip computing `α*f(w[i]*x[i])`.
        @dispatch_on_multiplier β unsafe_vscale!(y, β)
    else
        @dispatch_on_multiplier β unsafe_vmap!(α, f, w, x, β, y)
    end
    return nothing
end

"""
    LazyAlgebra.unsafe_vmap!(α, f, w, x, β, y)

Overwrite `y` with `y[i] = α*f(w[i], x[i]) + β*y[i])`.

!!! warning
    This method assumes that `w`, `x`, and `y` have the same axes, and that multipliers
    `α` and `β` have efficient types.

"""
function unsafe_vmap!(α::Number, f::Function, w::AbstractArray{Tw,N}, x::AbstractArray{Tx,N},
                      β::Number, y::AbstractArray{Ty,N}) where {Tw,Tx,Ty,N}
    @inbounds @simd for i in eachindex(w, x, y)
        y[i] = α*f(w[i], x[i]) + β*y[i]
    end
    return nothing
end
