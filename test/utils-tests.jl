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
    # Tests for `convert_multiplier`.
    #
    let convert_multiplier = LazyAlgebra.convert_multiplier,
        types = (Float32, Float16, BigFloat, Float64, ComplexF32, ComplexF64),
        perms = randperm(length(types)), # prevent compilation-time optimization
        n = length(types)
        # The ≡ (or ===) operator is too restrictive, let's define our own
        # method to compare multipliers: they must have the same value and the
        # same type to be considered as identical.
        identical(a::T, b::T) where {T} = (a == b)
        identical(a, b) = false
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
                @test identical(convert_multiplier(λ, T1),
                                convert(isa(λ, Complex) ?
                                        Complex{real(T1)} :
                                        real(T1), λ))
                @test identical(convert_multiplier(λ,T1,T2),
                                convert(isa(λ, Complex) ?
                                        Complex{real(promote_type(T1,T2))} :
                                        real(promote_type(T1,T2)), λ))
                @test identical(convert_multiplier(λ,T1,T2,T3),
                                convert(isa(λ, Complex) ?
                                        Complex{real(promote_type(T1,T2,T3))} :
                                        real(promote_type(T1,T2,T3)), λ))
                @test identical(convert_multiplier(λ,T1,T2,T3,T4),
                                convert(isa(λ, Complex) ?
                                        Complex{real(promote_type(T1,T2,T3,T4))} :
                                        real(promote_type(T1,T2,T3,T4)), λ))
                # Check with array arguments.
                @test identical(convert_multiplier(λ, A1),
                                convert(isa(λ, Complex) ?
                                        Complex{real(T1)} :
                                        real(T1), λ))
                @test identical(convert_multiplier(λ, A1, A2),
                                convert(isa(λ, Complex) ?
                                        Complex{real(promote_type(T1,T2))} :
                                        real(promote_type(T1,T2)), λ))
                @test identical(convert_multiplier(λ, A1, A2, A3),
                                convert(isa(λ, Complex) ?
                                        Complex{real(promote_type(T1,T2,T3))} :
                                        real(promote_type(T1,T2,T3)), λ))
                @test identical(convert_multiplier(λ, A1, A2, A3, A4),
                                convert(isa(λ, Complex) ?
                                        Complex{real(promote_type(T1,T2,T3,T4))} :
                                        real(promote_type(T1,T2,T3,T4)), λ))
            end
        end
        for T in (AbstractFloat, Real, Complex, Integer, Number, Unsigned)[randperm(6)]
            @test_throws ErrorException convert_multiplier(1, T)
        end
        @test LazyAlgebra.convert_multiplier(ZERO, Float16) === ZERO
        @test LazyAlgebra.convert_multiplier(ONE, Float32) === ONE
        @test LazyAlgebra.convert_multiplier(-ONE, Float64) === -ONE
    end
end # testset

@testset "Miscellaneous" begin
    #
    # Tests for `to_tuple`.
    #
    let to_tuple = LazyAlgebra.to_tuple, x = randn(5)
        @test to_tuple(x) === (x...,)
        @test to_tuple(to_tuple(x)) === (x...,)
    end
end # testset

@testset "Messages     " begin
    #
    # Tests of message, etc.
    #
    let bad_argument = LazyAlgebra.bad_argument,
        bad_size = LazyAlgebra.bad_size,
        arguments_have_incompatible_axes = LazyAlgebra.arguments_have_incompatible_axes,
        operands_have_incompatible_axes = LazyAlgebra.operands_have_incompatible_axes,
        message = LazyAlgebra.message,
        warn = LazyAlgebra.warn,
        siz = (3,4,5)
        @test_throws ArgumentError bad_argument("argument must be nonnegative")
        @test_throws ArgumentError bad_argument("invalid size ", siz)
        @test_throws DimensionMismatch arguments_have_incompatible_axes()
        @test_throws DimensionMismatch operands_have_incompatible_axes()
        @test_throws DimensionMismatch bad_size("invalid size")
        @test_throws DimensionMismatch bad_size("invalid size ", siz)
        message("Info:", "array size ", siz; color=:magenta)
        message(stdout, "Info:", "array size ", siz; color=:yellow)
        warn("array size ", siz)
        warn(stdout, "Info:", "array size ", siz)
    end
end # testset

nothing

end # module
