#
# mappings.jl -
#
# Tests for basic mappings.
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
        atol, rtol = zero(T), sqrt(eps(T))
        @test A*x  ≈ sum(w.*x)*w atol=atol rtol=rtol norm=vnorm2
        @test A'*x ≈ sum(w.*x)*w atol=atol rtol=rtol norm=vnorm2
        @test B*x  ≈ sum(y.*x)*w atol=atol rtol=rtol norm=vnorm2
        @test B'*x ≈ sum(w.*x)*y atol=atol rtol=rtol norm=vnorm2
        @test C*x  ≈ sum(w.*x)*w atol=atol rtol=rtol norm=vnorm2
        @test C'*x ≈ sum(w.*x)*w atol=atol rtol=rtol norm=vnorm2
    end

    @testset "Uniform scaling ($T)" for T in (Float16, Float32, Float64)
        x = randn(T, dims)
        y = randn(T, dims)
        γ = sqrt(2)
        U = UniformScalingOperator(γ)
        atol, rtol = zero(T), sqrt(eps(T))
        @test U*x  ≈ γ*x     atol=atol rtol=rtol norm=vnorm2
        @test U'*x ≈ γ*x     atol=atol rtol=rtol norm=vnorm2
        @test U\x  ≈ (1/γ)*x atol=atol rtol=rtol norm=vnorm2
        @test U'\x ≈ (1/γ)*x atol=atol rtol=rtol norm=vnorm2
        for α in (0, 1, -1,  2.71, π),
            β in (0, 1, -1, -1.33, φ)
            for P in (Direct, Adjoint)
                @test apply!(α, P, U, x, β, vcopy(y)) ≈
                    T(α*γ)*x + T(β)*y atol=atol rtol=rtol norm=vnorm2
            end
            for P in (Inverse, InverseAdjoint)
                @test apply!(α, P, U, x, β, vcopy(y)) ≈
                    T(α/γ)*x + T(β)*y atol=atol rtol=rtol norm=vnorm2
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
        atol, rtol = zero(T), sqrt(eps(T))
        @test S*x  ≈ wx atol=atol rtol=rtol norm=vnorm2
        @test S'*x ≈ wx atol=atol rtol=rtol norm=vnorm2
        for α in (0, 1, -1,  2.71, π),
            β in (0, 1, -1, -1.33, φ)
            for P in (Direct, Adjoint)
                @test apply!(α, P, S, x, β, vcopy(y)) ≈
                    T(α)*w.*x + T(β)*y atol=atol rtol=rtol norm=vnorm2
            end
            for P in (Inverse, InverseAdjoint)
                @test apply!(α, P, S, x, β, vcopy(y)) ≈
                    T(α)*x./w + T(β)*y atol=atol rtol=rtol norm=vnorm2
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
        atol, rtol = zero(T), sqrt(eps(T))
        mA = reshape(A, nrows, ncols)
        vx = reshape(x, ncols)
        vy = reshape(y, nrows)
        Gx = G*x
        Gpy = G'*y
        @test Gx  ≈ reshape(mA*vx,  rows) atol=atol rtol=rtol norm=vnorm2
        @test Gpy ≈ reshape(mA'*vy, cols) atol=atol rtol=rtol norm=vnorm2
        for α in (0, 1, -1,  2.71, π),
            β in (0, 1, -1, -1.33, φ)
            @test apply!(α, Direct, G, x, β, vcopy(y)) ≈
                T(α)*Gx + T(β)*y atol=atol rtol=rtol norm=vnorm2
            @test apply!(α, Adjoint, G, y, β, vcopy(x)) ≈
                T(α)*Gpy + T(β)*x atol=atol rtol=rtol norm=vnorm2
        end
    end

end
