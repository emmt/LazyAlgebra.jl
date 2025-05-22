#
# utils-tests.jl -
#
# Test utility functions.
#
module TestingLazyAlgebraUtilities

using LazyAlgebra
using Neutrals
using Random
using Test

@testset "Multipliers  " begin
    #
    # Tests for `LazyAlgebra.convert_multiplier`.
    #
    let convert_multiplier = LazyAlgebra.convert_multiplier
        # FIXME add tests with Unitful quantities
        for T in  (Float32, Float16, BigFloat, Float64, ComplexF32, ComplexF64, typeof(pi))
            R = float(real(T))
            for λ in (1.0, -1, π, 2 - 1im)
                λc = isa(λ, Complex) ? Complex{R}(λ) : R(λ)
                @test convert_multiplier(λ, T) == λc
                @test typeof(convert_multiplier(λ, T)) == typeof(λc)
            end
        end
        @test convert_multiplier(ZERO, Float16) === ZERO
        @test convert_multiplier(ONE, Float32) === ONE
        @test convert_multiplier(-ONE, Float64) === -ONE
    end
end # testset

nothing

end # module
