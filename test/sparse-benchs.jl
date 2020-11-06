#
# sparse-benchs.jl -
#
# Benchmarking sparse operators.
#
module BenchmarkingLazyAlgebraSparseOperators

using LinearAlgebra
using SparseArrays
using StructuredArrays
using LazyAlgebra
using LazyAlgebra: identical, Adjoint, Direct
using LazyAlgebra.SparseMethods
using LazyAlgebra.SparseOperators: check_structure, compute_offsets
using BenchmarkTools
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

nickname(::AbstractMatrix) = "Matrix";
nickname(::AbstractSparseMatrix) = "SparseMatrix";
nickname(::GeneralMatrix) = "GeneralMatrix";
nickname(::CompressedSparseOperator{:COO}) = "SparseOperatorCOO";
nickname(::CompressedSparseOperator{:CSC}) = "SparseOperatorCSC";
nickname(::CompressedSparseOperator{:CSR}) = "SparseOperatorCSR";
nickname(n::Integer, x) = rpad(nickname(x), n);

function bench1(; m::Integer = 2_000, n::Integer = m, T::Type = Float64,
                sparsity::Real = 0.95)
    bench1(T, m, n, sparsity)
end

bench1(::Type{T}, m::Integer, n::Integer, sparsity::Real) where {T} =
    bench1(T, Int(m), Int(n), Float64(sparsity))

function bench1(::Type{T}, m::Int, n::Int, sparsity::Float64) where {T}
    A = genarr(T, (m, n); sparsity=sparsity);
    x = genarr(T, n);
    y = genarr(T, m);
    x1 = copy(x);
    y0 = copy(y);
    x1 = similar(x);
    y1 = similar(y);
    x2 = similar(x);
    y2 = similar(y);
    coo = SparseOperatorCOO(A);
    csc = SparseOperatorCSC(A);
    csr = SparseOperatorCSR(A);
    gen = GeneralMatrix(A);
    S = sparse(A);
    println("Tests are done for T=$T, (m,n)=($m,$n) and sparsity = ",
            round(sparsity*1e2, sigdigits=3), "% of entries.\n")
    nnz(S) == nnz(coo) || println("not same number on non-zeros (COO)");
    nnz(S) == nnz(csc) || println("not same number on non-zeros (CSS)");
    nnz(S) == nnz(csr) || println("not same number on non-zeros (CSR)");
    mul!(y1, A, x);
    mul!(x1, A', y);
    for B in (gen, S, coo, csc, csr)
        mul!(y2, B, x);
        mul!(x2, B', y);
        println("compare A*x  for a ", nickname(20, B), extrema(x1 - x2))
        println("compare A'*x for a ", nickname(20, B), extrema(y1 - y2))
    end
    println()
    for B in (A, gen, S, coo, csc, csr)
        print("benchmarking A*x for a ", nickname(20, B))
        @btime mul!($y1, $B, $x)
    end
    println()
    for B in (A, gen, S, coo, csc, csr)
        print("benchmarking A'*x for a ", nickname(20, B))
        @btime mul!($x1, $(B'), $y)
    end

end

end # module

BenchmarkingLazyAlgebraSparseOperators.bench1()
