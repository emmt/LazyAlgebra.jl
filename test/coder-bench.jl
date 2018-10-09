#
# coder-bench.jl -
#
# Benchmark code produced by the coder.
#
#-------------------------------------------------------------------------------
#
# This file is part of LazyAlgebra (https://github.com/emmt/LazyAlgebra.jl)
# released under the MIT "Expat" license.
#
# Copyright (c) 2017-2018 Éric Thiébaut.
#

module CoderBench

using Compat
using Compat.Printf
using BenchmarkTools
include("../src/coder.jl")
import .Coder

# This version is basic (no SIMD and bound checking).
gensum1(x) = gensum1(typeof(x))
gensum1(::Type{T}) where {T} = error("unsupported argument type $T")
gensum1(::Type{<:AbstractArray{T,N}}) where {T,N} =
    Coder.encode(
        :(s = zero(T)),
        :(n = length(x)),
        :for, :(i = 1:n), # use "=" here
        (
            :(s += x[i]),
        ),
        :( return s )
    )
@generated sum1(x::AbstractArray{T,N}) where {T,N} = gensum1(x)

# This version is similar to above but keyword "in" is used instead of "=".
gensum2(x) = gensum2(typeof(x))
gensum2(::Type{T}) where {T} = error("unsupported argument type $T")
gensum2(::Type{<:AbstractArray{T,N}}) where {T,N} =
    Coder.encode(
        :(s = zero(T)),
        :(n = length(x)),
        :for, :(i in 1:n), # use "in" here
        (
            :(s += x[i]),
        ),
        :( return s )
    )
@generated sum2(x::AbstractArray{T,N}) where {T,N} = gensum2(x)

# This version is without bound checking and without SIMD.
gensum3(x) = gensum3(typeof(x))
gensum3(::Type{T}) where {T} = error("unsupported argument type $T")
gensum3(::Type{<:AbstractArray{T,N}}) where {T,N} =
    Coder.encode(
        :(s = zero(T)),
        :(n = length(x)),
        :inbounds,
        (
            :for, :(i in 1:n),
            (
                :(s += x[i]),
            )
        ),
        :( return s )
    )
@generated sum3(x::AbstractArray{T,N}) where {T,N} = gensum3(x)

# This version is without bound checking and with SIMD.
gensum4(x) = gensum4(typeof(x))
gensum4(::Type{T}) where {T} = error("unsupported argument type $T")
gensum4(::Type{<:AbstractArray{T,N}}) where {T,N} =
    Coder.encode(
        :(s = zero(T)),
        :(n = length(x)),
        :inbounds,
        (
            :simd_for, :(i in 1:n),
            (
                :(s += x[i]),
            )
        ),
        :( return s )
    )
@generated sum4(x::AbstractArray{T,N}) where {T,N} = gensum4(x)

function summary(prefix::AbstractString, s::Real, s0::Real, t, n::Integer)
    @printf("%10s %6.3f Gflops,  relative error = %.3g\n",
            prefix, n/minimum(t.times),
            abs(s - s0)/abs(s0))
end

# Benchmark results:
function benchmark(n::Integer=10_000)
    x = randn(n)
    s0 = sum(x)
    s1 = sum1(x)
    s2 = sum2(x)
    s3 = sum3(x)
    s4 = sum4(x)
    t0 = @benchmark sum($x)
    summary("sum:", s0, s0, t0, n)
    t1 = @benchmark sum1($x)
    summary("sum1:", s1, s0, t1, n)
    t2 = @benchmark sum2($x)
    summary("sum2:", s2, s0, t2, n)
    t3 = @benchmark sum3($x)
    summary("sum3:", s3, s0, t3, n)
    t4 = @benchmark sum4($x)
    summary("sum4:", s4, s0, t4, n)
end

end #module
