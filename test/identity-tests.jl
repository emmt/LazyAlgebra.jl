"""

# Tests for universal and shaped identities

Typical usage:

    include("test/identity-tests.jl").runtests();

"""
module LazyAlgebraIdentityTests

using LazyAlgebra
using LinearAlgebra
using Random
using Test
using TypeUtils

function runtests(; rng::AbstractRNG = MersenneTwister(314159),
                  sizes = ((), (3,), (2,3), (2,3,4)),
                  eltypes::Tuple{Vararg{Type}} = (Float64, Complex{Float32}),
                  kwds...)
    @testset "Identity" begin
        @testset "Universal identity" begin
            @test Id*Id === Id
            @test Id∘Id === Id
            @test Id\Id === Id
            @test Id/Id === Id
            @test Id + Id === 2Id
            @test 2Id - Id === 1Id
            @test 2Id - Id == Id
            @test isequal(2Id - Id, Id)

            A = SymbolicOperator(:A)
            @test Id/A === inv(A)
            @test A\Id === inv(A)
            @test Id/inv(A) === A
            @test inv(A)\Id === A

            for T in eltypes, dims in sizes
                x = rand(rng, T, dims)
                y = map(float, x) # not `float.(x)` because it collapses 0-dim array in a scalar
                LazyAlgebra.test_API(Id, x, y; name="universal identity", kwds...)
            end
        end

        @testset "Shaped identity" begin
            for T in eltypes, dims in sizes
                x = rand(rng, T, dims)
                y = map(float, x) # not `float.(x)` because it collapses 0-dim array in a scalar
                LazyAlgebra.test_API(Identity(dims...), x, y; name="shaped identity", kwds...)
            end
        end

        @testset "LinearAlgebra uniform scaling" begin
            # `I` is LinearAlgebra's uniform scaling with multiplier `true`.
            A = SymbolicOperator(:A)
            @test I + Id == 2Id
            @test I - Id == 0Id
            @test Id + I == 2Id
            @test Id - I == 0Id
            @test 2Id - 5I == -3Id

            @test A + I == A + Id
            @test A - I == A - Id
            @test I + A == Id + A
            @test I - A == Id - A
            @test A + 3I == A + 3Id
            @test A - 2I == A - 2Id

            @test I*Id == Id
            @test Id*I == Id
            @test I∘Id == Id
            @test Id∘I == Id
            @test (3I)*Id == 3Id
            @test Id*(-2I) == -2Id
            @test I/Id == Id
            @test Id/I == Id
            @test I\Id == Id
            @test Id\I == Id

            @test I*A == A
            @test I∘A == A
            @test A*I == A
            @test A∘I == A
            @test (3I)*A == 3A
            @test A*(-2I) == -2A
            @test I/A == inv(A)
            @test A\I == inv(A)
            @test I/inv(A) == A
            @test inv(A)\I == A
        end
    end
end

end # module
