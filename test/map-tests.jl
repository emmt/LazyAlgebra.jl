#
# map-tests.jl -
#
# Tests for basic mappings.
#

using Test
using LazyAlgebra
using LinearAlgebra: ⋅, UniformScaling

@testset "Mappings" begin
    include("common.jl")
    Scaled = LazyAlgebra.Scaled
    Sum = LazyAlgebra.Sum
    Composition = LazyAlgebra.Composition
    Endomorphism = LazyAlgebra.Endomorphism
    MorphismType = LazyAlgebra.MorphismType
    is_same_mapping = LazyAlgebra.is_same_mapping
    ALPHAS = (0, 1, -1,  2.71, π)
    BETAS = (0, 1, -1, -1.33, Base.MathConstants.φ)

    @testset "UniformScaling" begin
        # Check + operator.
        @test Id + UniformScaling(1) === 2Id
        @test Id + UniformScaling(2) === 3Id
        @test UniformScaling(1) + Id === 2Id
        @test UniformScaling(2) + Id === 3Id
        # Check - operator.
        @test Id - UniformScaling(1) === 0Id
        @test Id - UniformScaling(2) === -Id
        @test UniformScaling(1) - Id === 0Id
        @test UniformScaling(2) - Id === Id
        # Check * operator.
        @test Id*UniformScaling(1) === Id
        @test Id*UniformScaling(2) === 2Id
        @test UniformScaling(1)*Id === Id
        @test UniformScaling(2)*Id === 2Id
        # Check \circ operator.
        @test Id∘UniformScaling(1) === Id
        @test Id∘UniformScaling(2) === 2Id
        @test UniformScaling(1)∘Id === Id
        @test UniformScaling(2)∘Id === 2Id
        # \cdot is specific.
        @test Id⋅UniformScaling(1) === Id
        @test Id⋅UniformScaling(2) === 2Id
        @test UniformScaling(1)⋅Id === Id
        @test UniformScaling(2)⋅Id === 2Id
        # Check / operator.
        @test Id/UniformScaling(1) === Id
        @test Id/UniformScaling(2) === (1/2)*Id
        @test UniformScaling(1)/Id === Id
        @test UniformScaling(2)/Id === 2Id
        # Check \ operator.
        @test Id\UniformScaling(1) === Id
        @test Id\UniformScaling(2) === 2Id
        @test UniformScaling(1)\Id === Id
        @test UniformScaling(2)\Id === (1/2)*Id
    end

    @testset "Uniform scaling ($T)" for T in (Float32, Float64)
        dims = (3,4,5)
        x = randn(T, dims)
        y = randn(T, dims)
        λ = sqrt(2)
        U = λ*Id
        atol, rtol = zero(T), sqrt(eps(T))
        @test U*x  ≈ λ*x       atol=atol rtol=rtol
        @test U'*x ≈ λ*x       atol=atol rtol=rtol
        @test U\x  ≈ (1/λ)*x   atol=atol rtol=rtol
        @test U'\x ≈ (1/λ)*x   atol=atol rtol=rtol
        for α in ALPHAS,
            β in BETAS
            for P in (Direct, Adjoint)
                @test apply!(α, P, U, x, β, vcopy(y)) ≈
                    T(α*λ)*x + T(β)*y atol=atol rtol=rtol norm=vnorm2
            end
            for P in (Inverse, InverseAdjoint)
                @test apply!(α, P, U, x, β, vcopy(y)) ≈
                    T(α/λ)*x + T(β)*y atol=atol rtol=rtol norm=vnorm2
            end
        end
    end

    @testset "Rank 1 operators ($T)" for T in (Float32, Float64)
        dims = (3,4,5)
        n = prod(dims)
        w = randn(T, dims)
        x = randn(T, dims)
        y = randn(T, dims)
        A = RankOneOperator(w, w)
        B = RankOneOperator(w, y)
        C = SymmetricRankOneOperator(w)
        atol, rtol = zero(T), sqrt(eps(T))
        @test LinearType(A) === Linear()
        @test LinearType(C) === Linear()
        @test MorphismType(C) === Endomorphism()
        @test A*Id === A
        @test Id*A === A
        @test A*x  ≈ sum(w.*x)*w atol=atol rtol=rtol norm=vnorm2
        @test A'*x ≈ sum(w.*x)*w atol=atol rtol=rtol norm=vnorm2
        @test B*x  ≈ sum(y.*x)*w atol=atol rtol=rtol norm=vnorm2
        @test B'*x ≈ sum(w.*x)*y atol=atol rtol=rtol norm=vnorm2
        @test C*x  ≈ sum(w.*x)*w atol=atol rtol=rtol norm=vnorm2
        @test C'*x ≈ sum(w.*x)*w atol=atol rtol=rtol norm=vnorm2
        for α in ALPHAS,
            β in BETAS
            for P in (Direct, Adjoint)
                @test apply!(α, P, C, x, β, vcopy(y)) ≈
                    T(α*vdot(w,x))*w + T(β)*y atol=atol rtol=rtol norm=vnorm2
            end
        end
    end


    @testset "Non-uniform scaling ($T)" for T in (Float32, Float64)
        dims = (3,4,5)
        n = prod(dims)
        w = randn(T, dims)
        for i in eachindex(w)
            while w[i] == 0
                w[i] = randn(T)
            end
        end
        x = randn(T, dims)
        y = randn(T, dims)
        z = vcreate(y)
        S = NonuniformScaling(w)
        @test diag(S) === w
        @test Diag(w) === S
        atol, rtol = zero(T), sqrt(eps(T))
        @test S*x  ≈ w.*x atol=atol rtol=rtol norm=vnorm2
        @test S'*x ≈ w.*x atol=atol rtol=rtol norm=vnorm2
        @test S\x ≈ x./w atol=atol rtol=rtol norm=vnorm2
        @test S'\x ≈ x./w atol=atol rtol=rtol norm=vnorm2
        for α in ALPHAS,
            β in BETAS
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

    @testset "Non-uniform scaling (Complex{$T})" for T in (Float32, Float64)
        dims = (3,4,5)
        n = prod(dims)
        w = complex.(randn(T, dims), randn(T, dims))
        for i in eachindex(w)
            while w[i] == 0
                w[i] = complex(randn(T), randn(T))
            end
        end
        x = complex.(randn(T, dims), randn(T, dims))
        y = complex.(randn(T, dims), randn(T, dims))
        wx = w.*x
        qx = x./w
        z = vcreate(y)
        S = NonuniformScaling(w)
        atol, rtol = zero(T), sqrt(eps(T))
        @test S*x  ≈ w.*x atol=atol rtol=rtol norm=vnorm2
        @test S'*x ≈ conj.(w).*x atol=atol rtol=rtol norm=vnorm2
        @test S\x ≈ x./w atol=atol rtol=rtol norm=vnorm2
        @test S'\x ≈ x./conj.(w) atol=atol rtol=rtol norm=vnorm2
        for α in ALPHAS,
            β in BETAS
            @test apply!(α, Direct, S, x, β, vcopy(y)) ≈
                T(α)*w.*x + T(β)*y atol=atol rtol=rtol norm=vnorm2
            @test apply!(α, Adjoint, S, x, β, vcopy(y)) ≈
                T(α)*conj.(w).*x + T(β)*y atol=atol rtol=rtol norm=vnorm2
            @test apply!(α, Inverse, S, x, β, vcopy(y)) ≈
                T(α)*x./w + T(β)*y atol=atol rtol=rtol norm=vnorm2
            @test apply!(α, InverseAdjoint, S, x, β, vcopy(y)) ≈
                T(α)*x./conj.(w) + T(β)*y atol=atol rtol=rtol norm=vnorm2
        end
    end

    @testset "Generalized matrices ($T)" for T in (Float32, Float64)
        rows, cols = (2,3,4), (5,6)
        nrows, ncols = prod(rows), prod(cols)
        A = randn(T, rows..., cols...)
        x = randn(T, cols)
        y = randn(T, rows)
        G = GeneralMatrix(A)
        atol, rtol = zero(T), sqrt(eps(T))
        mA = reshape(A, nrows, ncols)
        vx = reshape(x, ncols)
        vy = reshape(y, nrows)
        Gx = G*x
        Gty = G'*y
        @test Gx  ≈ reshape(mA*vx,  rows) atol=atol rtol=rtol norm=vnorm2
        @test Gty ≈ reshape(mA'*vy, cols) atol=atol rtol=rtol norm=vnorm2
        test_api(Direct, G, x, y)
        test_api(Adjoint, G, x, y)
    end
end
nothing
