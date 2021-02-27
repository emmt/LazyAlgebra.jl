#
# sparse-tests.jl -
#
# Testing sparse operators.
#
module TestingLazyAlgebraSparseOperators

using SparseArrays
using StructuredArrays
using LazyAlgebra
using LazyAlgebra: identical
using LazyAlgebra.Foundations
using LazyAlgebra.SparseMethods
using LazyAlgebra.SparseOperators: check_structure, compute_offsets
using Test
using Random

is_csc(::Any) = false
is_csc(::CompressedSparseOperator{:CSC}) = true
is_csc(::Adjoint{<:CompressedSparseOperator{:CSR}}) = true

is_csr(::Any) = false
is_csr(::CompressedSparseOperator{:CSR}) = true
is_csr(::Adjoint{<:CompressedSparseOperator{:CSC}}) = true

is_coo(::Any) = false
is_coo(::CompressedSparseOperator{:COO}) = true
is_coo(::Adjoint{<:CompressedSparseOperator{:COO}}) = true

# Generate a possibly sparse array of random values.  Value are small signed
# integers so that all computations should be exact (except with non-integer
# multipliers).
genarr(T::Type, dims::Integer...; kwds...) = genarr(T, dims; kwds...)
function genarr(T::Type, dims::Tuple{Vararg{Integer}};
                sparsity::Real = 0,
                range::AbstractUnitRange{<:Integer} = -17:17)
    @assert 0 ≤ sparsity ≤ 1
    A = Array{T}(undef, dims)
    for i in eachindex(A)
        if sparsity > 0 && rand() ≤ sparsity
            A[i] = zero(T)
        elseif T <: Complex
            A[i] = T(rand(range), rand(range))
        else
            A[i] = T(rand(range))
        end
    end
    return A
end

# Unpack a sparse operator into a regular array using simplest iterator.  There
# may be duplicates.
function unpack_with_iterator!(dest::Array{T},
                               A::SparseOperator{E},
                               op = (E === Bool ? (|) : (+))) where {T,E}
    C = fill!(reshape(dest, (nrows(A), ncols(A))), zero(T))
    for (Aij, i, j) in A
        C[i,j] = op(C[i,j], Aij)
    end
    return dest
end

@testset "Low level sparse utilities" begin
    @test compute_offsets(2, Int[]) == [0,0,0]
    @test compute_offsets(5, [2,2,3,5]) == [0,0,2,3,3,4]
    @test compute_offsets(5, [1,3,3]) == [0,1,1,3,3,3]
    # Check for non-increasing order.
    @test_throws ErrorException compute_offsets(5, [1,3,2])
    # Check for out-of-bounds.
    @test_throws ErrorException compute_offsets(5, [0,3,3,7])
    @test_throws ErrorException compute_offsets(5, [1,3,7])
end

@testset "Compressed sparse formats " begin
    # Parameters.
    siz = (5, 6) # these tests only for A a 2-D array
    T = Float64;
    Tp = Float32; # for conversion

    # Make a banded matrix with random entries.
    A = genarr(T, siz) .* StructuredArray((i,j) -> -1 ≤ i - j ≤ 2, siz)
    spm = sparse(A);
    csr = convert(SparseOperatorCSR, A); # same as SparseOperatorCSR(A)
    csc = SparseOperatorCSC(A);
    coo = SparseOperatorCOO(A);
    x = genarr(T, siz[2]);
    y = genarr(T, siz[1]);

    # Make a COO version with randomly permuted entries.
    kp = randperm(nnz(coo));
    coo_perm = SparseOperatorCOO(get_vals(coo)[kp],
                                 get_rows(coo)[kp],
                                 get_cols(coo)[kp],
                                 row_size(coo),
                                 col_size(coo));

    # Make a COO version with randomly permuted entries and some duplicates.
    # Use fractions 1/3 and 3/4 for duplicating so that there is no loss of
    # precision.
    l = 7
    k = zeros(Int, length(kp) + l)
    w = ones(T, length(k))
    k[1:length(kp)] = kp
    for i in 1:l
        j1 = length(kp) - i + 1
        j2 = length(kp) + i
        w[j1] *= 1/4
        w[j2] *= 3/4
        k[j2] = k[j1]
    end
    coo_dups = SparseOperatorCOO(get_vals(coo)[k] .* w,
                                 get_rows(coo)[k],
                                 get_cols(coo)[k],
                                 row_size(coo),
                                 col_size(coo))

    # Check structures.
    @test check_structure(csr) === csr
    @test check_structure(csc) === csc
    @test check_structure(coo) === coo

    # Basic array-like methods
    @test eltype(csr) === eltype(A)
    @test eltype(csc) === eltype(A)
    @test eltype(coo) === eltype(A)

    @test length(csr) === length(A)
    @test length(csc) === length(A)
    @test length(coo) === length(A)

    @test ndims(csr) === ndims(A)
    @test ndims(csc) === ndims(A)
    @test ndims(coo) === ndims(A)

    @test size(csr) === size(A)
    @test size(csc) === size(A)
    @test size(coo) === size(A)

    @test nrows(csr) === size(A,1)
    @test nrows(csc) === size(A,1)
    @test nrows(coo) === size(A,1)
    @test nrows(spm) === size(A,1)

    @test ncols(csr) === size(A,2)
    @test ncols(csc) === size(A,2)
    @test ncols(coo) === size(A,2)
    @test ncols(spm) === size(A,2)

    # Number of structural non-zeros.
    nvals = count(x -> x != zero(x), A);
    @test nnz(csr) === nvals
    @test nnz(csc) === nvals
    @test nnz(coo) === nvals
    @test nnz(spm) === nvals
    @test length(get_vals(csr)) === nvals
    @test length(get_vals(csc)) === nvals
    @test length(get_vals(coo)) === nvals
    @test length(get_vals(spm)) === nvals

    # `nonzeros` and `get_vals` should yield the same object.
    @test get_vals(csr) === nonzeros(csr)
    @test get_vals(csc) === nonzeros(csc)
    @test get_vals(coo) === nonzeros(coo)
    @test get_vals(spm) === nonzeros(spm)

    # Julia arrays are column-major so values and row indices should be the
    # same in compressed sparse column (CSC) and compressed sparse coordinate
    # (COO) formats.
    @test get_vals(coo) == get_vals(csc)
    @test get_rows(coo) == get_rows(csc)
    @test get_cols(coo) == get_cols(csc)
    @test get_vals(coo) == get_vals(spm)
    @test get_rows(coo) == get_rows(spm)
    @test get_cols(coo) == get_cols(spm)

    # Check converting back to standard array.
    @test Array(csr) == A
    @test Array(csc) == A
    @test Array(coo) == A
    @test Array(spm) == A
    @test Array(coo_perm) == A
    @test Array(coo_dups) == A

    # Check matrix-vector multiplication (more serious tests in another
    # section).
    Ax = A*x
    Aty = A'*y
    @test csr*x == Ax
    @test csc*x == Ax
    @test coo*x == Ax
    @test csr'*y == Aty
    @test csc'*y == Aty
    @test coo'*y == Aty

    # Check iterators.
    B = Array{T}(undef, size(A))
    @test unpack_with_iterator!(B, csr) == A
    @test unpack_with_iterator!(B, csc) == A
    @test unpack_with_iterator!(B, coo) == A

    # Check conversions to COO, CSC and CSR formats.
    for F in (:COO, :CSC, :CSR)
        for src in (A, csc, csr, coo, coo_perm, coo_dups)
            for (t, cnv) in ((T, CompressedSparseOperator{F}(src)),
                             (T, CompressedSparseOperator{F,T}(src)),
                             (Tp, CompressedSparseOperator{F,Tp}(src)),)
                @test check_structure(cnv) === cnv
                @test eltype(cnv) === t
                if F === :COO
                    @test (cnv === coo) == (t === T && src === coo)
                    @test identical(cnv, coo) == (t === T && src === coo)
                    if is_csc(src) || is_csr(src)
                        if is_csc(src)
                            @test get_rows(cnv) === get_rows(src)
                        else
                            @test get_rows(cnv) == get_rows(src)
                        end
                        if is_csr(src)
                            @test get_cols(cnv) === get_cols(src)
                        else
                            @test get_cols(cnv) == get_cols(src)
                        end
                        if t === T
                            @test get_vals(cnv) === get_vals(src)
                        else
                            @test get_vals(cnv) == get_vals(src)
                        end
                    end
                elseif F === :CSC
                    @test (cnv === csc) == (t === T && src === csc)
                    @test identical(cnv, csc) == (t === T && src === csc)
                    if is_csc(src)
                        @test get_rows(cnv) === get_rows(csc)
                    else
                        @test get_rows(cnv) == get_rows(csc)
                    end
                    @test each_col(cnv) === each_col(csc)
                    @test get_cols(cnv) == get_cols(csc)
                    if is_csc(src) && t === T
                        @test get_vals(cnv) === get_vals(csc)
                    else
                        @test get_vals(cnv) == get_vals(csc)
                    end
                elseif F === :CSR
                    @test (cnv === csr) == (t === T && src === csr)
                    @test identical(cnv, csr) == (t === T && src === csr)
                    @test each_row(cnv) === each_row(csr)
                    @test get_rows(cnv) == get_rows(csr)
                    if is_csr(src)
                        @test get_cols(cnv) === get_cols(csr)
                    else
                        @test get_cols(cnv) == get_cols(csr)
                    end
                    if is_csr(src) && t === T
                        @test get_vals(cnv) === get_vals(csr)
                    else
                        @test get_vals(cnv) == get_vals(csr)
                    end
                end
            end
        end
    end

end # testset

@testset "Sparse operations         " begin
    rows = (2,3,4)
    cols = (5,6)
    M = length(rows)
    N = length(cols)
    for T in (Float32, Float64, Complex{Float64})
        R = real(T);
        ε = eps(R);
        A = genarr(T, rows..., cols...; sparsity=0.7); # 70% of zeros
        x = genarr(T, cols);
        xsav = vcopy(x);
        y = genarr(T, rows);
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

            # Check `apply!` and `vcreate` with integer valued multipliers so
            # that exact results are expected.
            Sx  = S*x;  @test x == xsav;
            Sty = S'*y; @test y == ysav;
            @test vdot(y, Sx) == vdot(Sty, x);
            for α in (0, 1, -1, 3),
                β in (0, 1, -1, 7),
                scratch in (false, true)
                # Test operator.
                @test apply!(α, Direct, S, x, scratch, β, vcopy(y)) ==
                    R(α)*Sx + R(β)*y
                if scratch
                    vcopy!(x, xsav)
                else
                    @test x == xsav
                end
                # Test  adjoint.
                @test apply!(α, Adjoint, S, y, scratch, β, vcopy(x)) ==
                    R(α)*Sty + R(β)*x
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
            @test Sx  == Gx
            @test Sty == Gty

            # Compare to results with a 2D matrix and 1D vectors.
            Aflat = reshape(A, prod(rows), prod(cols));
            xflat = reshape(x, prod(cols));
            yflat = reshape(y, prod(rows));
            @test Sx  == reshape(Aflat*xflat,  rows)
            @test Sty == reshape(Aflat'*yflat, cols)

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
            if is_csc(S) || is_coo(S)
                @test get_rows(S1) === get_rows(S)
            else
                @test get_rows(S1) == get_rows(S)
            end
            if is_csr(S) || is_coo(S)
                @test get_cols(S1) === get_cols(S)
            else
                @test get_cols(S1) == get_cols(S)
            end
            @test coefficients(S1) == coefficients(S)
            @test LazyAlgebra.identical(S1, S) == false

            # Check reshaping.
            S2d = reshape(S, prod(output_size(S)), prod(input_size(S)))
            @test eltype(S2d) === eltype(S)
            @test ndims(S2d) == 2
            if is_csc(S) || is_coo(S)
                @test get_rows(S2d) === get_rows(S)
            else
                @test get_rows(S2d) == get_rows(S)
            end
            if is_csr(S) || is_coo(S)
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
            # FIXME: x2 = genarr(T, input_size(S3))
            # FIXME: y2 = genarr(T, output_size(S3))
            # FIXME: @test S2*x2 == S3*x2
            # FIXME: @test S2'*y2 == S3'*y2
            # FIXME:
            # FIXME: # Check multiplication by a scalar.
            # FIXME: @test 1*S === S
            # FIXME: S0 = 0*S
            # FIXME: @test isa(S0, SparseOperator)
            # FIXME: @test length(get_rows(S0)) == 0
            # FIXME: @test length(get_cols(S0)) == 0
            # FIXME: @test length(coefficients(S0)) == 0
            # FIXME: @test eltype(S0) == eltype(S)
            # FIXME: @test input_size(S0) == input_size(S)
            # FIXME: @test output_size(S0) == output_size(S)
            # FIXME: α = R(π)
            # FIXME: αS = α*S
            # FIXME: @test isa(αS, SparseOperator)
            # FIXME: @test get_rows(αS) === get_rows(S)
            # FIXME: @test get_cols(αS) === get_cols(S)
            # FIXME: @test coefficients(αS) == α*coefficients(S)
            # FIXME: @test eltype(αS) == eltype(S)
            # FIXME: @test input_size(αS) == input_size(S)
            # FIXME: @test output_size(αS) == output_size(S)
            # FIXME:
            # FIXME: # Check left and right multiplication by a non-uniform rescaling
            # FIXME: # operator.
            # FIXME: w1 = genarr(T, output_size(S))
            # FIXME: W1 = NonuniformScaling(w1)
            # FIXME: W1_S = W1*S
            # FIXME: c1 = (w1 .* A)[A .!= zero(T)]
            # FIXME: @test isa(W1_S, SparseOperator)
            # FIXME: @test eltype(W1_S) === T
            # FIXME: @test output_size(W1_S) == output_size(S)
            # FIXME: @test input_size(W1_S) == input_size(S)
            # FIXME: @test get_rows(W1_S) === get_rows(S)
            # FIXME: @test get_cols(W1_S) === get_cols(S)
            # FIXME: @test coefficients(W1_S) == c1
            # FIXME: w2 = genarr(T, input_size(S))
            # FIXME: W2 = NonuniformScaling(w2)
            # FIXME: S_W2 = S*W2
            # FIXME: c2 = (A .* reshape(w2, (ones(Int, length(output_size(S)))...,
            # FIXME:                         input_size(S)...,)))[A .!= zero(T)]
            # FIXME: @test isa(S_W2, SparseOperator)
            # FIXME: @test eltype(S_W2) === T
            # FIXME: @test output_size(S_W2) == output_size(S)
            # FIXME: @test input_size(S_W2) == input_size(S)
            # FIXME: @test get_cols(S_W2) === get_cols(S)
            # FIXME: @test get_rows(S_W2) === get_rows(S)
            # FIXME: @test coefficients(S_W2) == c2
            # FIXME:
            # FIXME: # Use another constructor with integer conversion.
            # FIXME: R = SparseOperator(Int32.(get_rows(S)),
            # FIXME:                    Int64.(get_cols(S)),
            # FIXME:                    coefficients(S),
            # FIXME:                    Int32.(output_size(S)),
            # FIXME:                    Int64.(input_size(S)))
            # FIXME: @test Sx  == R*x
            # FIXME: @test Sty == R'*y
        end
    end
end
nothing

end # module
