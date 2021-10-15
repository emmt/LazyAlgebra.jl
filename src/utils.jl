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
# Copyright (c) 2017-2020 Éric Thiébaut.
#

@noinline bad_argument(args...) = bad_argument(string(args...))
bad_argument(mesg::String) = throw(ArgumentError(mesg))

@noinline bad_size(args...) = bad_size(string(args...))
bad_size(mesg::String) = throw(DimensionMismatch(mesg))

arguments_have_incompatible_axes() =
    bad_size("arguments have incompatible dimensions/indices")

operands_have_incompatible_axes() =
    bad_size("operands have incompatible dimensions/indices")

"""
    message([io=stdout,] header, args...; color=:blue)

prints a message on `io` with `header` text in bold followed by a space,
`args...` and a newline.  Keyword `color` can be used to specify the text color
of the message.

"""
message(header::String, args...; kwds...) =
    message(stdout, header, args...; kwds...)

@noinline function message(io::IO, header::String, args...;
                           color::Symbol=:blue)
    printstyled(io, header; color=color, bold=true)
    printstyled(io, " ", args...; color=color, bold=false)
    println(io)
end

"""
    warn([io=stdout,] args...)

prints a warning message in yellow on `io` with `"Warning: "` in bold followed
by `args...` and a newline.

"""
warn(args...) = warn(stderr, args...)
warn(io::IO, args...) = message(io, "Warning:", args...; color=:yellow)

#inform(args...) = inform(stderr, args...)
#inform(io::IO, args...) = message(io, "Info:", args...; color=:blue)

"""
    promote_multiplier(λ, T)

yields multiplier `λ` converted to a suitable floating-point type for
multiplying values or expressions of type `T`.  This method is *type stable*.
The result has the same floating-point precision as `T` and is a real if `λ` is
real or a complex if `λ` is complex.

Multiple arguments can be specified after the multiplier `λ`:

    promote_multiplier(λ, args...)

to have `T` the promoted type of all types in `args...` or all element types of
arrays in `args...`.

See methods [`LazyAlgebra.multiplier_type`](@ref) and
[`LazyAlgebra.multiplier_floatingpoint_type`](@ref).

""" promote_multiplier

# Note taht the only direct sub-types of `Number` are abstract types `Real` and
# `Complex`.  Also see discussion here
# (https://github.com/emmt/LinearInterpolators.jl/issues/7) for details about
# the following implementation.

@inline function promote_multiplier(λ::Real, args...)
    T = multiplier_floatingpoint_type(args...)
    return convert(T, λ)::T
end

@inline function promote_multiplier(λ::Complex{<:Real}, args...)
    T = multiplier_floatingpoint_type(args...)
    return convert(Complex{T}, λ)::Complex{T}
end

"""
    multiplier_floatingpoint_type(args...) -> T::AbstractFloat

yields the multiplier floating-point type for the arguments `args...` of the
multiplier.  Each argument may be anything acceptable for
[`LazyAlgebra.multiplier_type`](@ref).  The result is guaranteed to be a
concrete floating-point type.

See methods [`LazyAlgebra.promote_multiplier`](@ref) and
[`LazyAlgebra.multiplier_type`](@ref).

""" multiplier_floatingpoint_type

multiplier_floatingpoint_type(::Tuple{}) =
    throw(ArgumentError("at least one other argument must be specified"))

@inline function multiplier_floatingpoint_type(args...)
    T = promote_type(map(multiplier_type, args)...)
    (T <: Number && isconcretetype(T)) || error(
        "resulting multiplier type ", T, " is not a concrete real type")
    return float(real(T))
end

"""
    multiplier_type(x) -> T::Number

yields the *element* type to be imposed to multipliers of `x`.  The result must
be a concrete number type.  Argument `x` may be an array, a number, or a data
type.  Other packages are however encouraged to specialize this method for
their needs.

See methods [`LazyAlgebra.promote_multiplier`](@ref) and
[`LazyAlgebra.multiplier_floatingpoint_type`](@ref).

"""
multiplier_type(::Type{T}) where  {T<:Number} = T
multiplier_type(::AbstractArray{T}) where {T<:Number} = T
multiplier_type(::T) where  {T<:Number} = T

"""
    to_tuple(arg)

converts `arg` into an `N`-tuple where `N` is the number of elements of `arg`.
This is equivalent to `Tuple(arg)` or `(arg...,)` for a vector but it is much
faster for small vectors.

""" to_tuple

to_tuple(x::Tuple) = x

# The cutoff at n = 10 below reflects what is used by `ntuple`.  This value is
# somewhat arbitrary, on the machines where I tested the code, the explicit
# unrolled expression for n = 10 is still about 44 times faster than `(x...,)`.
# Calling `ntuple` for n ≤ 10 is about twice slower; for n > 10, `ntuple` is
# slower than `(x...,)`.
function to_tuple(x::AbstractVector)
    n = length(x)
    @inbounds begin
        n == 0 ? () :
        n > 10 || firstindex(x) != 1 ? (x...,) :
        n == 1 ? (x[1],) :
        n == 2 ? (x[1], x[2]) :
        n == 3 ? (x[1], x[2], x[3]) :
        n == 4 ? (x[1], x[2], x[3], x[4]) :
        n == 5 ? (x[1], x[2], x[3], x[4], x[5]) :
        n == 6 ? (x[1], x[2], x[3], x[4], x[5], x[6]) :
        n == 7 ? (x[1], x[2], x[3], x[4], x[5], x[6], x[7]) :
        n == 8 ? (x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8]) :
        n == 9 ? (x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9]) :
        (x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9], x[10])
    end
end

"""
    @certify expr [mesg]

asserts that expression `expr` is true; otherwise, throws an `AssertionError`
exception with message `mesg`.  If unspecified, `mesg` is `expr` converted into
a string.  Compared to `@assert`, the assertion made by `@certify` may never be
disabled whatever the optimization level.

"""
macro certify(expr)
    _certify(expr, string(expr))
end
macro certify(expr, mesg::Union{Expr,Symbol})
    _certify(expr, :(string($(esc(mesg)))))
end
macro certify(expr, mesg::AbstractString)
    _certify(expr, mesg)
end
macro certify(expr, mesg)
    _certify(expr, string(mesg))
end
_certify(expr, mesg) = :($(esc(expr)) ? nothing : throw(AssertionError($mesg)))
