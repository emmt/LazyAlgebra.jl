#
# conjgrad.jl -
#
# Tests for linear conjugate gradients methods.
#

#isdefined(:LazyAlgebra) || include("../src/LazyAlgebra.jl")

module LazyAlgebraConjGradTests

using LazyAlgebra
using Test

const DEBUG = true

@testset "ConjGrad" begin
    types = (Float32, Float64)
    rows = (7,8)
    cols = (2,3,4)
    @testset "least-square fit ($T)" for T in types
        H = GeneralMatrix(randn(T, rows..., cols...))
        x = randn(T, cols)
        y = H*x + 0.01*randn(T, rows)

        # Least-squares solution using Julia linear algebra.
        yj = reshape(y, length(y))
        Hj = reshape(contents(H), length(y), length(x))
        Aj = Hj'*Hj
        bj = Hj'*yj
        x0 = reshape(Aj\bj, cols)

        # Tolerances.
        rtol = 1e-4
        atol = rtol*vnorm2(x0)

        # LHS matrix and RHS vector of the normal equations.
        A = H'*H
        b = H'*y
        n = length(b)
        x1 = conjgrad(A, b; maxiter=10n, restart=n,
                      quiet=false, verb=true,
                      gtol=(0,0), ftol=0, xtol=rtol/10)
        DEBUG && println("x1: ", vnorm2(x1 - x0)/vnorm2(x0))
        @test vnorm2(x1 - x0) ≤ atol

        # Exercise random starting point.
        x2 = conjgrad(A, b, randn(T, cols); maxiter=2n, restart=n,
                      quiet=true, verb=false,
                      gtol=(0,0), ftol=0, xtol=rtol/10)
        DEBUG && println("x2: ", vnorm2(x2 - x0)/vnorm2(x0))
        @test vnorm2(x2 - x0) ≤ atol

        # Exercise other convergence tests.
        x3 = conjgrad(A, b; maxiter=n, restart=n,
                      quiet=false, verb=true,
                      gtol=(0,0), ftol=0, xtol=0)
        DEBUG && println("x3: ", vnorm2(x3 - x0)/vnorm2(x0))
        @test vnorm2(x3 - x0) ≤ atol
        x4 = conjgrad(A, b; maxiter=10n, restart=n,
                      quiet=false, verb=true,
                      gtol=(0,0), ftol=1e-9, xtol=0)
        DEBUG && println("x4: ", vnorm2(x4 - x0)/vnorm2(x0))
        @test vnorm2(x4 - x0) ≤ atol
        x5 = conjgrad(A, b; maxiter=10n, restart=n,
                      quiet=false, verb=true,
                      gtol=(0,1e-6), ftol=0, xtol=0)
        DEBUG && println("x5: ", vnorm2(x5 - x0)/vnorm2(x0))
        @test vnorm2(x5 - x0) ≤ atol

        # Directly use an array.
        x6 = conjgrad(reshape(Aj, cols..., cols...), b;
                      maxiter=2n, restart=n,
                      quiet=true, verb=false,
                      gtol=(0,0), ftol=0, xtol=rtol/10)
        DEBUG && println("x6: ", vnorm2(x6 - x0)/vnorm2(x0))
        @test vnorm2(x6 - x0) ≤ atol

        # Force no iterations (should yield exactly the initial solution).
        x7 = conjgrad(A, b, x0;
                      maxiter=0, restart=n,
                      quiet=true, verb=false,
                      gtol=(0,0), ftol=0, xtol=0)
        DEBUG && println("x7: ", vnorm2(x7 - x0)/vnorm2(x0))
        @test vnorm2(x7 - x0) ≤ 0

        # Force restarts.
        x8 = conjgrad(A, b;
                      maxiter=4n, restart=(n>>1),
                      quiet=true, verb=false,
                      gtol=(0,0), ftol=0, xtol=0)
        DEBUG && println("x8: ", vnorm2(x8 - x0)/vnorm2(x0))
        @test vnorm2(x8 - x0) ≤ atol

        # Non-positive definite.
        @test vnorm2(conjgrad(-A, b; strict=false)) ≤ 0
        @test_throws LazyAlgebra.NonPositiveDefinite conjgrad(-A, b; verb=true,
                                                              quiet=false)

    end
end

end # module
