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

nothing

end # module
