#
# utils-tests.jl -
#
# Test utility functions.
#
module TestingLazyAlgebraUtilities

using LazyAlgebra
using Neutrals
using Unitful
using Unitful: km, cm, mm, μm, °, s
using Random
using Test

using LazyAlgebra: convert_multiplier
using LazyAlgebra: fast_min, fast_max

@testset "Inference     " begin
    x, y, z = ("x" => 1.0f0, 1.0 + 2.0im, ('α', 0x01, -2.0f0))
    @test @inferred(LazyAlgebra.fieldtypes(       x )) === (String, Float32)
    @test @inferred(LazyAlgebra.fieldtypes(typeof(x))) === (String, Float32)
    @test @inferred(LazyAlgebra.fieldtypes(       y )) === (Float64, Float64)
    @test @inferred(LazyAlgebra.fieldtypes(typeof(y))) === (Float64, Float64)
    @test @inferred(LazyAlgebra.fieldtypes(       z )) === (Char, UInt8, Float32)
    @test @inferred(LazyAlgebra.fieldtypes(typeof(z))) === (Char, UInt8, Float32)
    if VERSION ≥ v"1.1.0"
        @test LazyAlgebra.fieldtypes(x) === Base.fieldtypes(typeof(x))
        @test LazyAlgebra.fieldtypes(y) === Base.fieldtypes(typeof(y))
        @test LazyAlgebra.fieldtypes(z) === Base.fieldtypes(typeof(z))
    end
end

@testset "Multipliers   " begin
    #
    # Tests for `LazyAlgebra.convert_multiplier`.
    #
    for T in (Float32, Float16, BigFloat, Float64, ComplexF32, ComplexF64, typeof(pi))
        R = float(real(T))
        for λ in (1.0, -1, π, 2 - 1im)
            λc = isa(λ, Complex) ? Complex{R}(λ) : R(λ)
            @test convert_multiplier(λ, T) == λc
            @test typeof(convert_multiplier(λ, T)) == typeof(λc)
        end
    end
    for α in instances(Neutral), T in (Float16, Float32, Float64)
        x = rand(T, 2, 3)
        A = Id
        @test @inferred(convert_multiplier(α, T)) === α
        @test @inferred(convert_multiplier(T, α)) === α
        @test @inferred(convert_multiplier(α*μm, T)) === α*μm
        @test @inferred(convert_multiplier(T, α*(°/s))) === α*(°/s)
        @test @inferred(convert_multiplier(α, x)) === α
        @test @inferred(convert_multiplier(α*cm, x)) === α*cm
        @test @inferred(convert_multiplier(α, A, x)) === α
        @test @inferred(convert_multiplier(α*(cm/s), A, x)) === α*(cm/s)
    end
end # testset

@testset "Fast min./max." begin
    u = cm/s
    for T in (Int16, Float64, Float32, Float16, BigFloat)
        # NOTE == does not work for NaNs, === does not work for BigFloat,
        #      but isequal works for both.
        x, y, z = T(1), 2, 0
        # fast_min
        @test isequal(@inferred(fast_min(y, x)), x)
        @test isequal(@inferred(fast_min(x, y)), x)
        @test isequal(@inferred(fast_min(y*u, x*u)), x*u)
        @test isequal(@inferred(fast_min(x*u, y*u)), x*u)
        # fast_max
        @test isequal(@inferred(fast_max(z, x)), x)
        @test isequal(@inferred(fast_max(x, z)), x)
        @test isequal(@inferred(fast_max(z*u, x*u)), x*u)
        @test isequal(@inferred(fast_max(x*u, z*u)), x*u)
        if T <: AbstractFloat
            x, y, z = T(NaN), T(-Inf), T(Inf)
            # fast_min
            @test isequal(@inferred(fast_min(y, x)), x)
            @test isequal(@inferred(fast_min(x, y)), x)
            @test isequal(@inferred(fast_min(x, x)), x)
            @test isequal(@inferred(fast_min(y*u, x*u)), x*u)
            @test isequal(@inferred(fast_min(x*u, y*u)), x*u)
            @test isequal(@inferred(fast_min(x*u, x*u)), x*u)
            # fast_max
            @test isequal(@inferred(fast_max(z, x)), x)
            @test isequal(@inferred(fast_max(x, z)), x)
            @test isequal(@inferred(fast_max(x, x)), x)
            @test isequal(@inferred(fast_max(z*u, x*u)), x*u)
            @test isequal(@inferred(fast_max(x*u, z*u)), x*u)
            @test isequal(@inferred(fast_max(x*u, x*u)), T(NaN)*u)
        end
    end
end # testset

nothing

end # module
