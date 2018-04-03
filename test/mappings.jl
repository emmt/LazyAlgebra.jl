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
        rtol = sqrt(eps(T))
        @test rtol ≥ maxrelabsdif(A*x , sum(w.*x)*w)
        @test rtol ≥ maxrelabsdif(A'*x, sum(w.*x)*w)
        @test rtol ≥ maxrelabsdif(B*x , sum(y.*x)*w)
        @test rtol ≥ maxrelabsdif(B'*x, sum(w.*x)*y)
        @test rtol ≥ maxrelabsdif(C*x , sum(w.*x)*w)
        @test rtol ≥ maxrelabsdif(C'*x, sum(w.*x)*w)
    end

    @testset "Uniform scaling ($T)" for T in (Float16, Float32, Float64)
        x = randn(T, dims)
        y = randn(T, dims)
        γ = sqrt(2)
        U = UniformScalingOperator(γ)
        rtol = sqrt(eps(T))
        @test rtol ≥ maxrelabsdif(U*x, γ*x)
        @test rtol ≥ maxrelabsdif(U'*x, γ*x)
        @test rtol ≥ maxrelabsdif(U\x, (1/γ)*x)
        @test rtol ≥ maxrelabsdif(U'\x, (1/γ)*x)
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

    @testset "Non-uniform scaling ($T)" for T in (Float16, Float32, Float64)
        w = randn(T, dims)
        x = randn(T, dims)
        y = randn(T, dims)
        wx = w.*x
        qx = x./w
        z = vcreate(y)
        S = NonuniformScalingOperator(w)
        rtol = sqrt(eps(T))
        @test rtol ≥ maxrelabsdif(S*x, wx)
        @test rtol ≥ maxrelabsdif(S'*x, wx)
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

    rows, cols = (2,3,4), (5,6)
    nrows, ncols = prod(rows), prod(cols)
    @testset "Generalized matrices ($T)" for T in (Float32, Float64)
        A = randn(T, rows..., cols...)
        x = randn(T, cols)
        y = randn(T, rows)
        G = GeneralMatrix(A)
        rtol = sqrt(eps(T))
        mA = reshape(A, nrows, ncols)
        vx = reshape(x, ncols)
        vy = reshape(y, nrows)
        Gx = G*x
        Gpy = G'*y
        @test rtol ≥ maxrelabsdif(Gx,  reshape(mA*vx, rows))
        @test rtol ≥ maxrelabsdif(Gpy, reshape(mA'*vy, cols))
        for α in (0, 1, -1,  2.71, π),
            β in (0, 1, -1, -1.33, φ)
            @test rtol ≥ maxrelabsdif(apply!(α, Direct, G, x, β, vcopy(y)),
                                      T(α)*Gx + T(β)*y)
            @test rtol ≥ maxrelabsdif(apply!(α, Adjoint, G, y, β, vcopy(x)),
                                      T(α)*Gpy + T(β)*x)
        end
    end

end
