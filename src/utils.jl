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

incompatible_axes() = bad_size("arguments have incompatible dimensions/indices")

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

```julia
promote_multiplier(λ, T)
```

yields multiplier `λ` converted to a suitable floating-point type for
multiplying values or expressions of type `T`.

Multiple arguments can be specified after the multiplier `λ`:

```julia
promote_multiplier(λ, args...)
```

to have `T` the promoted type of all types in `args...` or all element types of
arrays in `args...`.

This method is *type stable*.  The result has the same floating-point precision
as `T` and is a real if `λ` is real or a complex if `λ` is complex.

"""
promote_multiplier(λ::Real, ::Type{T}) where {T<:Floats} = begin
    # If λ ∈ ℝ, the returned multiplier is real with the same floating-point
    # precision as `T`.
    isconcretetype(T) || operand_type_not_concrete(T)
    convert(real(T), λ)
end

promote_multiplier(λ::Complex, ::Type{T}) where {T<:Floats} = begin
    # If λ ∈ ℂ, the returned multiplier is complex with the same floating-point
    # precision as `T`.
    isconcretetype(T) || operand_type_not_concrete(T)
    convert(Complex{real(T)}, λ)
end

@noinline promote_multiplier(λ::L, ::Type{T}) where {L<:Number,T} = begin
    # Other possible cases throw errors.
    isconcretetype(T) || operand_type_not_concrete(T)
    error(string("unsupported conversion of multiplier with type ", L,
                 " for operand with element type ",T))
end

promote_multiplier(λ::Number, ::AbstractArray{T}) where {T} =
    promote_multiplier(λ, T)
promote_multiplier(λ::Number, args::AbstractArray...) =
    promote_multiplier(λ, map(eltype, args)...)
promote_multiplier(λ::Number, args::Type...) =
    promote_multiplier(λ, promote_type(args...))

# Note: the only direct sub-types of `Number` are abstract types `Real` and
# `Complex`.
@noinline operand_type_not_concrete(::Type{T}) where {T} =
    error("operand type ", T, " is not a concrete type")

"""
    to_tuple(arg)

converts `arg` into an `N`-tuple where `N` is the number of elements of `arg`.
This is equivallent to `Tuple(arg)` or `(arg...,)` for a vector but it is much
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
