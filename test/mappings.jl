#
# mappings.jl -
#
# Tests for mappings.
#

const I = LazyAlgebra.Identity()

@testset "Mappings" begin
    dims = (3,4,5)
    n = prod(dims)

    @testset "Rank 1 operators ($T)" for T in (Float16, Float32, Float64)
        w = randn(T, dims)
        x = randn(T, dims)
        y = randn(T, dims)
        A = RankOneOperator(w, w)
        B = RankOneOperator(w, y)
        C = SymmetricRankOneOperator(w)
        @test approxsame(n, A*x , sum(w.*x)*w)
        @test approxsame(n, A'*x, sum(w.*x)*w)
        @test approxsame(n, B*x , sum(y.*x)*w)
        @test approxsame(n, B'*x, sum(w.*x)*y)
        @test approxsame(n, C*x , sum(w.*x)*w)
        @test approxsame(n, C'*x, sum(w.*x)*w)
    end

    @testset "Scaling operators ($T)" for T in (Float16, Float32, Float64)
        w = randn(T, dims)
        x = randn(T, dims)
        y = randn(T, dims)
        S = NonuniformScalingOperator(w)
        @test approxsame(2, S*x, w.*x)
        @test approxsame(2, S'*x, w.*x)

        alpha = sqrt(2)
        U = UniformScalingOperator(alpha)
        @test approxsame(2, U*x, alpha*x)
        @test approxsame(2, U'*x, alpha*x)
        @test approxsame(2, U\x, (1/alpha)*x)
        @test approxsame(2, U'\x, (1/alpha)*x)
    end

end
