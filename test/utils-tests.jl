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
    # Tests for `promote_multiplier`.
    #
    identical(a::T, b::T) where {T} = (a == b)
    identical(a, b) = false
    promote_multiplier = LazyAlgebra.promote_multiplier
    types = (Float32, Float16, BigFloat, Float64, ComplexF32, ComplexF64)
    perms = randperm(length(types)) # prevent compilation-time optimization
    n = length(types)
    for i in 1:n
        T1 = types[perms[i]]
        T2 = types[perms[mod(i,   n) + 1]]
        T3 = types[perms[mod(i+1, n) + 1]]
        T4 = types[perms[mod(i+2, n) + 1]]
        A1 = zeros(T1, 1)
        A2 = zeros(T2, 2)
        A3 = zeros(T3, 3)
        A4 = zeros(T4, 4)
        for λ in (1, π, 2 - 1im)
            # Check with type arguments.
            @test identical(promote_multiplier(λ, T1),
                            convert(isa(λ, Complex) ?
                                    Complex{real(T1)} :
                                    real(T1), λ))
            @test identical(promote_multiplier(λ,T1,T2),
                            convert(isa(λ, Complex) ?
                                    Complex{real(promote_type(T1,T2))} :
                                    real(promote_type(T1,T2)), λ))
            @test identical(promote_multiplier(λ,T1,T2,T3),
                            convert(isa(λ, Complex) ?
                                    Complex{real(promote_type(T1,T2,T3))} :
                                    real(promote_type(T1,T2,T3)), λ))
            @test identical(promote_multiplier(λ,T1,T2,T3,T4),
                            convert(isa(λ, Complex) ?
                                    Complex{real(promote_type(T1,T2,T3,T4))} :
                                    real(promote_type(T1,T2,T3,T4)), λ))
            # Check with array arguments.
            @test identical(promote_multiplier(λ, A1),
                            convert(isa(λ, Complex) ?
                                    Complex{real(T1)} :
                                    real(T1), λ))
            @test identical(promote_multiplier(λ, A1, A2),
                            convert(isa(λ, Complex) ?
                                    Complex{real(promote_type(T1,T2))} :
                                    real(promote_type(T1,T2)), λ))
            @test identical(promote_multiplier(λ, A1, A2, A3),
                            convert(isa(λ, Complex) ?
                                    Complex{real(promote_type(T1,T2,T3))} :
                                    real(promote_type(T1,T2,T3)), λ))
            @test identical(promote_multiplier(λ, A1, A2, A3, A4),
                            convert(isa(λ, Complex) ?
                                    Complex{real(promote_type(T1,T2,T3,T4))} :
                                    real(promote_type(T1,T2,T3,T4)), λ))
        end
    end
    for T in (AbstractFloat, Complex, Number)[randperm(3)]
        @test_throws ErrorException promote_multiplier(1, T)
    end
end
nothing
