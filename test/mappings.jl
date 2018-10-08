#
# mappings.jl -
#
# Tests for basic mappings.
#

#isdefined(:LazyAlgebra) || include("../src/LazyAlgebra.jl")

module LazyAlgebraMappingTests

using LazyAlgebra

# Deal with compatibility issues.
using Compat
using Compat.Test
@static if isdefined(Base, :MathConstants)
    import Base.MathConstants: φ
end
@static if VERSION ≥ v"0.7.0-DEV.1776"
    using FFTW
end

const I = Identity()

@testset "Mappings" begin
    dims = (3,4,5)
    n = prod(dims)
    alphas = (0, 1, -1,  2.71, π)
    betas = (0, 1, -1, -1.33, φ)
    operations = (Direct, Adjoint, Inverse, InverseAdjoint)
    floats = (Float32, Float64)
    complexes = (ComplexF32, ComplexF64)

    @testset "Identity" begin
        @test I === LazyAlgebra.I
        @test I' === I
        @test inv(I) === I
        @test I*I === I
        @test I\I === I
        @test I/I === I
        @test I+I === 2I
        @test I+2I === 3I
        @test I+I === 2I
        @test 2I+3I === 5I
        @test 3I + (I + 2I) === 6I
        @test inv(3I) === (1/3)*I
        @test I - I === 0I
        @test I + I - 2I === 0I
        @test 2I - (I + I) === 0I
        @test SelfAdjointType(I) <: SelfAdjoint
        @test MorphismType(I) <: Endomorphism
        @test DiagonalType(I) <: DiagonalMapping
        for P in operations
            @test InPlaceType(P, I) <: InPlace
        end
        for T in floats
            atol, rtol = zero(T), sqrt(eps(T))
            x = randn(T, dims)
            y = randn(T, dims)
            @test pointer(I*x) == pointer(x)
            for P in operations
                @test pointer(apply(P,I,x)) == pointer(I*x)
                z = vcreate(P, I, x)
                for α in alphas, β in betas
                    vcopy!(z, y)
                    @test apply!(α, P, I, x, β, z) ≈ α*x + β*y atol=atol rtol=rtol norm=vnorm2
                end
            end
        end
    end

    @testset "Rank 1 operators ($T)" for T in floats
        w = randn(T, dims)
        x = randn(T, dims)
        y = randn(T, dims)
        A = RankOneOperator(w, w)
        B = RankOneOperator(w, y)
        C = SymmetricRankOneOperator(w)
        atol, rtol = zero(T), sqrt(eps(T))
        @test LinearType(A) <: Linear
        @test LinearType(C) <: Linear
        @test MorphismType(C) <: Endomorphism
        for P in operations
            @test InPlaceType(P, C) <: InPlace
        end
        @test A*I === A
        @test I*A === A
        @test A*x  ≈ sum(w.*x)*w atol=atol rtol=rtol norm=vnorm2
        @test A'*x ≈ sum(w.*x)*w atol=atol rtol=rtol norm=vnorm2
        @test B*x  ≈ sum(y.*x)*w atol=atol rtol=rtol norm=vnorm2
        @test B'*x ≈ sum(w.*x)*y atol=atol rtol=rtol norm=vnorm2
        @test C*x  ≈ sum(w.*x)*w atol=atol rtol=rtol norm=vnorm2
        @test C'*x ≈ sum(w.*x)*w atol=atol rtol=rtol norm=vnorm2
        for α in alphas,
            β in betas
            for P in (Direct, Adjoint)
                @test apply!(α, P, C, x, β, vcopy(y)) ≈
                    T(α*vdot(w,x))*w + T(β)*y atol=atol rtol=rtol norm=vnorm2
            end
        end
    end

    @testset "Uniform scaling ($T)" for T in floats
        x = randn(T, dims)
        y = randn(T, dims)
        γ = sqrt(2)
        U = γ*I
        atol, rtol = zero(T), sqrt(eps(T))
        @test U*x  ≈ γ*x     atol=atol rtol=rtol norm=vnorm2
        @test U'*x ≈ γ*x     atol=atol rtol=rtol norm=vnorm2
        @test U\x  ≈ (1/γ)*x atol=atol rtol=rtol norm=vnorm2
        @test U'\x ≈ (1/γ)*x atol=atol rtol=rtol norm=vnorm2
        for α in alphas,
            β in betas
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

    @testset "Non-uniform scaling ($T)" for T in floats
        w = randn(T, dims)
        for i in eachindex(w)
            while w[i] == 0
                w[i] = randn(T)
            end
        end
        x = randn(T, dims)
        y = randn(T, dims)
        z = vcreate(y)
        S = NonuniformScalingOperator(w)
        atol, rtol = zero(T), sqrt(eps(T))
        @test S*x  ≈ w.*x atol=atol rtol=rtol norm=vnorm2
        @test S'*x ≈ w.*x atol=atol rtol=rtol norm=vnorm2
        @test S\x ≈ x./w atol=atol rtol=rtol norm=vnorm2
        @test S'\x ≈ x./w atol=atol rtol=rtol norm=vnorm2
        for α in alphas,
            β in betas
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

    @testset "Non-uniform scaling (Complex{$T})" for T in floats
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
        S = NonuniformScalingOperator(w)
        atol, rtol = zero(T), sqrt(eps(T))
        @test S*x  ≈ w.*x atol=atol rtol=rtol norm=vnorm2
        @test S'*x ≈ conj.(w).*x atol=atol rtol=rtol norm=vnorm2
        @test S\x ≈ x./w atol=atol rtol=rtol norm=vnorm2
        @test S'\x ≈ x./conj.(w) atol=atol rtol=rtol norm=vnorm2
        for α in alphas,
            β in betas
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

    rows, cols = (2,3,4), (5,6)
    nrows, ncols = prod(rows), prod(cols)
    @testset "Generalized matrices ($T)" for T in floats
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
        for α in alphas,
            β in betas
            @test apply!(α, Direct, G, x, β, vcopy(y)) ≈
                T(α)*Gx + T(β)*y atol=atol rtol=rtol norm=vnorm2
            @test apply!(α, Adjoint, G, y, β, vcopy(x)) ≈
                T(α)*Gty + T(β)*x atol=atol rtol=rtol norm=vnorm2
        end
    end

    rows, cols = (2,3,4), (5,6)
    nrows, ncols = prod(rows), prod(cols)
    @testset "Sparse matrices ($T)" for T in floats
        A = randn(T, rows..., cols...)
        A[rand(T, size(A)) .≤ 0.7] .= 0 # 70% of zeros
        x = randn(T, cols)
        y = randn(T, rows)
        G = GeneralMatrix(A)
        S = SparseOperator(A, length(rows))
        atol, rtol = zero(T), sqrt(eps(T))
        mA = reshape(A, nrows, ncols)
        vx = reshape(x, ncols)
        vy = reshape(y, nrows)
        Sx = S*x
        Sty = S'*y
        @test Sx  ≈ G*x atol=atol rtol=rtol norm=vnorm2
        @test Sty ≈ G'*y atol=atol rtol=rtol norm=vnorm2
        @test Sx  ≈ reshape(mA*vx,  rows) atol=atol rtol=rtol norm=vnorm2
        @test Sty ≈ reshape(mA'*vy, cols) atol=atol rtol=rtol norm=vnorm2
        for α in alphas,
            β in betas
            @test apply!(α, Direct, S, x, β, vcopy(y)) ≈
                T(α)*Sx + T(β)*y atol=atol rtol=rtol norm=vnorm2
            @test apply!(α, Adjoint, S, y, β, vcopy(x)) ≈
                T(α)*Sty + T(β)*x atol=atol rtol=rtol norm=vnorm2
        end
    end

    @testset "FFT ($T)" for T in floats
        for dims in ((45,), (20,), (33,12), (30,20), (4,5,6))
            for cmplx in (false, true)
                if cmplx
                    x = randn(T, dims) + 1im*randn(T, dims)
                else
                    x = randn(T, dims)
                end
                F = FFTOperator(x)
                if cmplx
                    y = randn(T, dims) + 1im*randn(T, dims)
                else
                    y = randn(T, output_size(F)) + 1im*randn(T, output_size(F))
                end
                ϵ = eps(T)
                atol, rtol = zero(T), eps(T)
                z = (cmplx ? fft(x) : rfft(x))
                w = (cmplx ? ifft(y) : irfft(y, dims[1]))
                @test F*x ≈ z atol=0 rtol=ϵ norm=vnorm2
                @test F\y ≈ w atol=0 rtol=ϵ norm=vnorm2
                for α in alphas,
                    β in betas
                    @test apply!(α, Direct, F, x, β, vcopy(y)) ≈
                        T(α)*z + T(β)*y atol=0 rtol=ϵ
                    @test apply!(α, Inverse, F, y, β, vcopy(x)) ≈
                        T(α)*w + T(β)*x atol=0 rtol=ϵ
                end
            end
        end
    end
end

end
