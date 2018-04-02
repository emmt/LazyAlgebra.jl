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

    @testset "Uniform scaling ($T)" for T in (Float16, Float32, Float64)
        x = randn(T, dims)
        y = randn(T, dims)
        γ = sqrt(2)
        ϵ = eps(T)
        U = UniformScalingOperator(γ)
        @test approxsame(2, U*x, γ*x)
        @test approxsame(2, U'*x, γ*x)
        @test approxsame(2, U\x, (1/γ)*x)
        @test approxsame(2, U'\x, (1/γ)*x)
        for α in (0, 1, -1,  2.71, π),
            β in (0, 1, -1, -1.33, φ)
            for P in (Direct, Adjoint)
                @test 0 == maxrelabsdif(apply!(α, P, U, x, β, vcopy(y)),
                                        T(α*γ)*x + T(β)*y)
            end
            for P in (Inverse, InverseAdjoint)
                @test 0 == maxrelabsdif(apply!(α, P, U, x, β, vcopy(y)),
                                        T(α/γ)*x + T(β)*y)
            end
        end
    end

    @testset "Scaling operators ($T)" for T in (Float16, Float32, Float64)
        w = randn(T, dims)
        x = randn(T, dims)
        y = randn(T, dims)
        wx = w.*x
        qx = x./w
        z = vcreate(y)
        S = NonuniformScalingOperator(w)
        ϵ = eps(T)
        @test approxsame(2, S*x, wx)
        @test approxsame(2, S'*x, wx)
        for α in (0, 1, -1,  2.71, π),
            β in (0, 1, -1, -1.33, φ)
            for P in (Direct, Adjoint)
                @test 0 == maxrelabsdif(apply!(α, P, S, x, β, vcopy(y)),
                                        T(α)*w.*x + T(β)*y)
            end
            for P in (Inverse, InverseAdjoint)
                @test 0 == maxrelabsdif(apply!(α, P, S, x, β, vcopy(y)),
                                        T(α)*x./w + T(β)*y)
            end
        end
    end


end
