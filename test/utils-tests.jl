#
# utils-tests.jl -
#
# Test utility functions.
#

using Random
using Test
using LazyAlgebra

@testset "Utilities" begin
    #
    # Tests for `convert_multiplier`.
    #
    convert_multiplier = LazyAlgebra.convert_multiplier
    R = (Float32, Float16, BigFloat, Float64)
    C = (ComplexF32, ComplexF64)
    for i in randperm(length(R)) # prevent compilation-time optimization
        Tr = R[i]
        Trp = R[i < length(R) ? i + 1 : 1]
        j = rand(1:length(C))
        Tc = C[j]
        Tcp = C[length(C) + 1 - j]
        @test convert_multiplier(1, Tr) == convert(Tr, 1)
        @test isa(convert_multiplier(π, Tc), AbstractFloat)
        @test (v = convert_multiplier(2.0, Tr)) == 2 && isa(v, Tr)
        @test (v = convert_multiplier(2.0, Tr, Trp)) == 2 && isa(v, Tr)
        @test (v = convert_multiplier(2.0, Tr, Tc)) == 2 && isa(v, Tr)
        @test (v = convert_multiplier(2.0, Tc, Tc)) == 2 && isa(v, real(Tc))
        @test convert_multiplier(1+0im, Tr) == 1
        @test convert_multiplier(1+0im, Tr, Trp) == 1
        @test_throws InexactError convert_multiplier(1+2im, Tr)
        @test convert_multiplier(1+2im, Tr, Tc) == 1+2im
        @test convert_multiplier(1+2im, Tc, Tcp) == 1+2im
    end
    for T in (AbstractFloat, Complex, Number)[randperm(3)]
        @test_throws ErrorException convert_multiplier(1, T)
    end
end
nothing
