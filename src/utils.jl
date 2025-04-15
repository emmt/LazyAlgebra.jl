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
