#
# diff-tests.jl -
#
# Tests for finite differences.
#

using LazyAlgebra
using Test

@testset "Sparse operators" begin
    types = (Float32, Float64)
    alphas = (0, 1, -1,  2.71, π)
    betas = (0, 1, -1, -1.33, Base.MathConstants.φ)
    rows = (2,3,4)
    cols = (5,6)
    nrows = prod(rows)
    ncols = prod(cols)
    for T in types
        A = randn(T, rows..., cols...)
        A[rand(T, size(A)) .≤ 0.7] .= 0 # 70% of zeros
        x = randn(T, cols)
        xsav = vcopy(x)
        y = randn(T, rows)
        ysav = vcopy(y)
        G = GeneralMatrix(A)
        S = SparseOperator(A, length(rows))
        @test is_endomorphism(S) == (rows == cols)
        @test (LazyAlgebra.EndomorphismType(S) ==
               LazyAlgebra.Endomorphism) == (rows == cols)
        @test output_size(S) == rows
        @test input_size(S) == cols
        @test_throws DimensionMismatch vcreate(Direct, S,
                                               randn(T, size(x) .+ 1))
        @test_throws DimensionMismatch vcreate(Adjoint, S,
                                               randn(T, size(y) .+ 1))
        @test A == Array(S)
        atol, rtol = zero(T), sqrt(eps(T))
        mA = reshape(A, nrows, ncols)
        vx = reshape(x, ncols)
        vy = reshape(y, nrows)
        Sx = S*x
        @test x == xsav
        Sty = S'*y
        @test y == ysav
        @test Sx  ≈ reshape(mA*vx,  rows)  atol=atol rtol=rtol
        @test Sty ≈ reshape(mA'*vy, cols)  atol=atol rtol=rtol
        @test Sx  ≈ G*x                    atol=atol rtol=rtol
        @test x == xsav
        @test Sty ≈ G'*y                   atol=atol rtol=rtol
        @test y == ysav
        ## Use another constructor with integer conversion.
        R = SparseOperator(Int32.(LazyAlgebra.rows(S)),
                           Int64.(LazyAlgebra.cols(S)),
                           LazyAlgebra.coefs(S),
                           Int32.(output_size(S)),
                           Int64.(input_size(S)))
        @test Sx  ≈ R*x  atol=atol rtol=rtol
        @test Sty ≈ R'*y atol=atol rtol=rtol
        for α in alphas,
            β in betas,
            scratch in (false, true)
            @test apply!(α, Direct, S, x, scratch, β, vcopy(y)) ≈
                T(α)*Sx + T(β)*y  atol=atol rtol=rtol
            if scratch
                vcopy!(x, xsav)
            else
                @test x == xsav
            end
            @test apply!(α, Adjoint, S, y, scratch, β, vcopy(x)) ≈
                T(α)*Sty + T(β)*x atol=atol rtol=rtol
            if scratch
                vcopy!(y, ysav)
            else
                @test y == ysav
            end
        end
    end
end
nothing
