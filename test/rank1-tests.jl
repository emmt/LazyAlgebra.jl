"""

# Tests for rank-1 operators

Typical usage:

    include("test/rank1-tests.jl").runtests();

"""
module LazyAlgebraRank1Tests

using LazyAlgebra
using LinearAlgebra
using Random
using Test
using TypeUtils

function runtests(; rng::AbstractRNG = MersenneTwister(314159),
                  sizes = ((3,) => (3,), (2,3) => (2,3), (2,3) => (4,), (5,) => (2,3,4)),
                  eltypes::Tuple{Vararg{Type}} = (Float64, Complex{Float32}),
                  kwds...)
    @testset "Rank-1 operators" begin
        for T in eltypes, pair in sizes
            u = rand(rng, T, first(pair))
            v = rand(rng, T, last(pair))
            A = @inferred RankOneOperator(u, v)
            x = rand(rng, T, size(v))
            # We must be sure that v'*x has the same numerical precision as u.
            vtx = LazyAlgebra.convert_multiplier(vdot(v, x), u)
            LazyAlgebra.test_API(A, x, vtx * u; name="rank-1 operator", kwds...)
            # Since (u*v')' === v*u' (as tested below), it should not be necessary to
            # fully test the adjoint of a rank-1 operator.
            B = @inferred RankOneOperator(v, u)
            @test A' === B
            @test B' === A
            y = rand(rng, T, size(u))
            uty = LazyAlgebra.convert_multiplier(vdot(u, y), v)
            LazyAlgebra.test_API(A', y, uty * v; name="adjoint of rank-1 operator", kwds...)
            if size(u) == size(v)
                A = SymmetricRankOneOperator(u)
                LazyAlgebra.test_API(A, x, vdot(u, x) * u; name="symmetric rank-1 operator", kwds...)
            end
        end
    end
end

end # module
