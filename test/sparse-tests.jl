#
# diff-tests.jl -
#
# Tests for finite differences.
#

using SparseArrays
using LazyAlgebra
using Test

@testset "Sparse operators" begin
    rows = (2,3,4)
    cols = (5,6)
    nrows = prod(rows)
    ncols = prod(cols)
    for T in (Float32, Float64)
        ε = eps(T)
        A = randn(T, rows..., cols...)
        A[rand(T, size(A)) .≤ 0.7] .= 0 # 70% of zeros
        x = randn(T, cols)
        xsav = vcopy(x)
        y = randn(T, rows)
        ysav = vcopy(y)
        G = GeneralMatrix(A)
        S = SparseOperator(A, length(rows))
        @test eltype(S) === T
        @test ndims(S) == length(rows) + length(cols)
        @test is_endomorphism(S) == (rows == cols)
        @test (LazyAlgebra.EndomorphismType(S) ==
               LazyAlgebra.Endomorphism) == (rows == cols)
        @test output_size(S) == rows
        @test input_size(S) == cols
        @test SparseOperator(S) === S
        @test SparseOperator{T}(S) === S
        @test LazyAlgebra.are_same_mappings(SparseOperator(S), S)
        @test LazyAlgebra.are_same_mappings(SparseOperator{T}(S), S)

        # Extract coefficients as an array or as a matrix.
        A1 = Array(S)
        @test eltype(A1) === eltype(S)
        @test ndims(A1) == ndims(S)
        @test size(A1) == (rows..., cols...,)
        @test A1 == A
        A2 = Matrix(S)
        @test eltype(A2) === eltype(S)
        @test ndims(A2) == 2
        @test size(A2) == (prod(rows), prod(cols))
        @test A2 == reshape(A, size(A2))

        # Convert to another floating-point type.
        T1 = (T === Float32 ? Float64 : Float32)
        S1 = SparseOperator{T1}(S)
        @test eltype(S1) === T1
        @test ndims(S1) == ndims(S)
        @test LazyAlgebra.rows(S1) === LazyAlgebra.rows(S)
        @test LazyAlgebra.cols(S1) === LazyAlgebra.cols(S)
        @test LazyAlgebra.coefs(S1) ≈ LazyAlgebra.coefs(S) atol=0 rtol=2*eps(Float32)
        @test LazyAlgebra.are_same_mappings(S1, S) == false

        # Check reshaping.
        S2d = reshape(S, prod(output_size(S)), prod(input_size(S)))
        @test eltype(S2d) === eltype(S)
        @test ndims(S2d) == 2
        @test LazyAlgebra.rows(S2d) === LazyAlgebra.rows(S)
        @test LazyAlgebra.cols(S2d) === LazyAlgebra.cols(S)
        @test LazyAlgebra.coefs(S2d) === LazyAlgebra.coefs(S)
        @test LazyAlgebra.are_same_mappings(S2d, S) == false

        # Convert to a sparse matrix.
        S2 = sparse(S)
        @test eltype(S2) === eltype(S)
        S3 = SparseOperator(S2)
        @test eltype(S3) === eltype(S)
        x2 = randn(T, input_size(S3))
        y2 = randn(T, output_size(S3))
        @test S2*x2 ≈ S3*x2 atol=0 rtol=sqrt(ε)
        @test S2'*y2 ≈ S3'*y2 atol=0 rtol=sqrt(ε)

        # Check multiplication by a scalar.
        @test 1*S === S
        S0 = 0*S
        @test isa(S0, SparseOperator)
        @test length(LazyAlgebra.rows(S0)) == 0
        @test length(LazyAlgebra.cols(S0)) == 0
        @test length(LazyAlgebra.coefs(S0)) == 0
        @test eltype(S0) == eltype(S)
        @test input_size(S0) == input_size(S)
        @test output_size(S0) == output_size(S)
        α = T(π)
        αS = α*S
        @test isa(αS, SparseOperator)
        @test LazyAlgebra.rows(αS) === LazyAlgebra.rows(S)
        @test LazyAlgebra.cols(αS) === LazyAlgebra.cols(S)
        @test LazyAlgebra.coefs(αS) ≈ α*LazyAlgebra.coefs(S) atol=0 rtol=2ε
        @test eltype(αS) == eltype(S)
        @test input_size(αS) == input_size(S)
        @test output_size(αS) == output_size(S)

        # Check left and right multiplication by a non-uniform rescaling
        # operator.
        w1 = randn(T, output_size(S))
        W1 = NonuniformScalingOperator(w1)
        W1_S = W1*S
        c1 = (w1 .* A)[A .!= zero(T)]
        @test isa(W1_S, SparseOperator)
        @test eltype(W1_S) === T
        @test output_size(W1_S) == output_size(S)
        @test input_size(W1_S) == input_size(S)
        @test LazyAlgebra.cols(W1_S) === LazyAlgebra.cols(S)
        @test LazyAlgebra.rows(W1_S) === LazyAlgebra.rows(S)
        @test LazyAlgebra.coefs(W1_S) ≈ c1 atol=0 rtol=2ε
        w2 = randn(T, input_size(S))
        W2 = NonuniformScalingOperator(w2)
        S_W2 = S*W2
        c2 = (A .* reshape(w2, (ones(Int, length(output_size(S)))...,
                                input_size(S)...,)))[A .!= zero(T)]
        @test isa(S_W2, SparseOperator)
        @test eltype(S_W2) === T
        @test output_size(S_W2) == output_size(S)
        @test input_size(S_W2) == input_size(S)
        @test LazyAlgebra.cols(S_W2) === LazyAlgebra.cols(S)
        @test LazyAlgebra.rows(S_W2) === LazyAlgebra.rows(S)
        @test LazyAlgebra.coefs(S_W2) ≈ c2 atol=0 rtol=2ε

        # Check `apply!` and `vcreate`.
        @test_throws DimensionMismatch vcreate(Direct, S,
                                               randn(T, size(x) .+ 1))
        @test_throws DimensionMismatch vcreate(Adjoint, S,
                                               randn(T, size(y) .+ 1))
        atol, rtol = zero(T), sqrt(ε)
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

        # Use another constructor with integer conversion.
        R = SparseOperator(Int32.(LazyAlgebra.rows(S)),
                           Int64.(LazyAlgebra.cols(S)),
                           LazyAlgebra.coefs(S),
                           Int32.(output_size(S)),
                           Int64.(input_size(S)))
        @test Sx  ≈ R*x  atol=atol rtol=rtol
        @test Sty ≈ R'*y atol=atol rtol=rtol
        for α in (0, 1, -1,  2.71, π),
            β in (0, 1, -1, -1.33, Base.MathConstants.φ),
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
