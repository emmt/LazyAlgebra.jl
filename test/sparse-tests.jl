#
# diff-tests.jl -
#
# Tests for finite differences.
#
module TestingLazyAlgebraSparseArrays

using SparseArrays
using LazyAlgebra
using LazyAlgebra.SparseMethods
using Test

@testset "Sparse operators" begin
    rows = (2,3,4)
    cols = (5,6)
    M = length(rows)
    N = length(cols)
    for T in (Float32, Float64, Complex{Float64})
        R = real(T);
        ε = eps(R);
        A = rand(T, rows..., cols...);
        A[rand(Float64, size(A)) .≤ 0.7] .= 0; # 70% of zeros
        x = rand(T, cols);
        xsav = vcopy(x);
        y = rand(T, rows);
        ysav = vcopy(y);
        Scsc = SparseOperatorCSC{T,M,N}(A);
        Scsr = SparseOperatorCSR{T,M,N}(A);
        Scoo = SparseOperatorCOO{T,M,N}(A);

        # Check basic methods.
        for S in (Scsc, Scsr, Scoo)
            @test eltype(S) === T
            @test ndims(S) == length(rows) + length(cols)
            @test is_endomorphism(S) == (rows == cols)
            @test (LazyAlgebra.MorphismType(S) ===
                   LazyAlgebra.Endomorphism()) == (rows == cols)
            @test row_size(S) == rows
            @test col_size(S) == cols
            @test nrows(S) == prod(rows)
            @test ncols(S) == prod(cols)
            @test output_size(S) == rows
            @test input_size(S) == cols
            @test SparseOperator(S) === S
            @test SparseOperator{T}(S) === S
            @test SparseOperator{T,M}(S) === S
            @test SparseOperator{T,M,N}(S) === S
            @test LazyAlgebra.identical(SparseOperator(S), S)
            @test LazyAlgebra.identical(SparseOperator{T}(S), S)
            @test LazyAlgebra.identical(SparseOperator{T,M}(S), S)
            @test LazyAlgebra.identical(SparseOperator{T,M,N}(S), S)

            # Check `apply!` and `vcreate`.
            atol, rtol = zero(R), sqrt(ε);
            Sx  = S*x;  @test x == xsav;
            Sty = S'*y; @test y == ysav;
            @test vdot(y, Sx) ≈ vdot(Sty, x);
            for α in (0, 1, -1,  ),#2.71, π),
                β in (0, 1, -1, ),#-1.33, Base.MathConstants.φ),
                scratch in (false, true)
                # Test operator.
                @test apply!(α, Direct, S, x, scratch, β, vcopy(y)) ≈
                    R(α)*Sx + R(β)*y  atol=atol rtol=rtol
                if scratch
                    vcopy!(x, xsav)
                else
                    @test x == xsav
                end
                # Test  adjoint.
                @test apply!(α, Adjoint, S, y, scratch, β, vcopy(x)) ≈
                    R(α)*Sty + R(β)*x  atol=atol rtol=rtol
                if scratch
                    vcopy!(y, ysav)
                else
                    @test y == ysav
                end
            end

            # Compare to results with a general matrix.
            G = GeneralMatrix(A);
            Gx  = G*x;  @test x == xsav;
            Gty = G'*y; @test y == ysav;
            @test Sx  ≈ Gx   atol=atol rtol=rtol;
            @test Sty ≈ Gty  atol=atol rtol=rtol;

            # Compare to results with a 2D matrix and 1D vectors.
            Aflat = reshape(A, prod(rows), prod(cols));
            xflat = reshape(x, prod(cols));
            yflat = reshape(y, prod(rows));
            @test Sx  ≈ reshape(Aflat*xflat,  rows)  atol=atol rtol=rtol
            @test Sty ≈ reshape(Aflat'*yflat, cols)  atol=atol rtol=rtol

            # Extract coefficients as an array or as a matrix.
            A1 = Array(S);
            @test eltype(A1) === eltype(S)
            @test ndims(A1) == ndims(S)
            @test size(A1) == (rows..., cols...,)
            @test A1 == A
            # FIXME: A2 = Matrix(S)
            # FIXME: @test eltype(A2) === eltype(S)
            # FIXME: @test ndims(A2) == 2
            # FIXME: @test size(A2) == (prod(rows), prod(cols))
            # FIXME: @test A2 == reshape(A, size(A2))
            # FIXME: B = (A .!= 0) # make an array of booleans
            # FIXME: @test Array(SparseOperator(B, length(rows))) == B

            # Convert to another floating-point type.
            T1 = (T <: Complex ?
                  (real(T) === Float32 ? Complex{Float64} : Complex{Float32}) :
                  (T === Float32 ? Float64 : Float32))
            S1 = SparseOperator{T1}(S)
            @test eltype(S1) === T1
            @test ndims(S1) == ndims(S)
            if isa(S, AbstractSparseOperatorCSC) || isa(S, AbstractSparseOperatorCOO)
                @test get_rows(S1) === get_rows(S)
            else
                @test get_rows(S1) == get_rows(S)
            end
            if isa(S, AbstractSparseOperatorCSR) || isa(S, AbstractSparseOperatorCOO)
                @test get_cols(S1) === get_cols(S)
            else
                @test get_cols(S1) == get_cols(S)
            end
            @test coefficients(S1) ≈ coefficients(S) atol=0 rtol=2*eps(Float32)
            @test LazyAlgebra.identical(S1, S) == false

            # Check reshaping.
            S2d = reshape(S, prod(output_size(S)), prod(input_size(S)))
            @test eltype(S2d) === eltype(S)
            @test ndims(S2d) == 2
            if isa(S, AbstractSparseOperatorCSC) || isa(S, AbstractSparseOperatorCOO)
                @test get_rows(S2d) === get_rows(S)
            else
                @test get_rows(S2d) == get_rows(S)
            end
            if isa(S, AbstractSparseOperatorCSR) || isa(S, AbstractSparseOperatorCOO)
                @test get_cols(S2d) === get_cols(S)
            else
                @test get_cols(S2d) == get_cols(S)
            end
            @test coefficients(S2d) === coefficients(S)
            @test LazyAlgebra.identical(S2d, S) == false

            # FIXME: # Convert to a sparse matrix.
            # FIXME: S2 = sparse(S)
            # FIXME: @test eltype(S2) === eltype(S)
            # FIXME: S3 = SparseOperator(S2)
            # FIXME: @test eltype(S3) === eltype(S)
            # FIXME: x2 = randn(T, input_size(S3))
            # FIXME: y2 = randn(T, output_size(S3))
            # FIXME: @test S2*x2 ≈ S3*x2 atol=0 rtol=sqrt(ε)
            # FIXME: @test S2'*y2 ≈ S3'*y2 atol=0 rtol=sqrt(ε)
            # FIXME:
            # FIXME: # Check multiplication by a scalar.
            # FIXME: @test 1*S === S
            # FIXME: S0 = 0*S
            # FIXME: @test isa(S0, SparseOperator)
            # FIXME: @test length(getrows(S0)) == 0
            # FIXME: @test length(getcols(S0)) == 0
            # FIXME: @test length(coefficients(S0)) == 0
            # FIXME: @test eltype(S0) == eltype(S)
            # FIXME: @test input_size(S0) == input_size(S)
            # FIXME: @test output_size(S0) == output_size(S)
            # FIXME: α = R(π)
            # FIXME: αS = α*S
            # FIXME: @test isa(αS, SparseOperator)
            # FIXME: @test getrows(αS) === getrows(S)
            # FIXME: @test getcols(αS) === getcols(S)
            # FIXME: @test coefficients(αS) ≈ α*coefficients(S) atol=0 rtol=2ε
            # FIXME: @test eltype(αS) == eltype(S)
            # FIXME: @test input_size(αS) == input_size(S)
            # FIXME: @test output_size(αS) == output_size(S)
            # FIXME:
            # FIXME: # Check left and right multiplication by a non-uniform rescaling
            # FIXME: # operator.
            # FIXME: w1 = randn(T, output_size(S))
            # FIXME: W1 = NonuniformScaling(w1)
            # FIXME: W1_S = W1*S
            # FIXME: c1 = (w1 .* A)[A .!= zero(T)]
            # FIXME: @test isa(W1_S, SparseOperator)
            # FIXME: @test eltype(W1_S) === T
            # FIXME: @test output_size(W1_S) == output_size(S)
            # FIXME: @test input_size(W1_S) == input_size(S)
            # FIXME: @test getrows(W1_S) === getrows(S)
            # FIXME: @test getcols(W1_S) === getcols(S)
            # FIXME: @test coefficients(W1_S) ≈ c1 atol=0 rtol=2ε
            # FIXME: w2 = randn(T, input_size(S))
            # FIXME: W2 = NonuniformScaling(w2)
            # FIXME: S_W2 = S*W2
            # FIXME: c2 = (A .* reshape(w2, (ones(Int, length(output_size(S)))...,
            # FIXME:                         input_size(S)...,)))[A .!= zero(T)]
            # FIXME: @test isa(S_W2, SparseOperator)
            # FIXME: @test eltype(S_W2) === T
            # FIXME: @test output_size(S_W2) == output_size(S)
            # FIXME: @test input_size(S_W2) == input_size(S)
            # FIXME: @test getcols(S_W2) === getcols(S)
            # FIXME: @test getrows(S_W2) === getrows(S)
            # FIXME: @test coefficients(S_W2) ≈ c2 atol=0 rtol=2ε
            # FIXME:
            # FIXME: # Use another constructor with integer conversion.
            # FIXME: R = SparseOperator(Int32.(getrows(S)),
            # FIXME:                    Int64.(getcols(S)),
            # FIXME:                    coefficients(S),
            # FIXME:                    Int32.(output_size(S)),
            # FIXME:                    Int64.(input_size(S)))
            # FIXME: @test Sx  ≈ R*x   atol=atol rtol=rtol
            # FIXME: @test Sty ≈ R'*y  atol=atol rtol=rtol
        end
    end
end
nothing

end # module
