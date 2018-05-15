#
# conjgrad.jl -
#
# Tests for linear conjugate gradients methods.
#

isdefined(:LazyAlgebra) || include("../src/LazyAlgebra.jl")

module LazyAlgebraConjGradTests

using LazyAlgebra

@static if VERSION < v"0.7.0-DEV.2005"
    using Base.Test
else
    using Test
end

const DEBUG = true

@testset "ConjGrad" begin
    types = (Float32, Float64)
    rows = (7,8)
    cols = (2,3,4)
    @testset "least-square fit ($T)" for T in types
        H = GeneralMatrix(randn(T, rows..., cols...))
        x = randn(T, cols)
        y = H*x + 0.01*randn(T, rows)

        # Lesat-squares solution using Julia linear algebra.
        yj = reshape(y, length(y))
        Hj = reshape(contents(H), length(y), length(x))
        x0 = reshape((Hj'*Hj)\(Hj'*yj), cols)

        # Tolerances.
        rtol = 1e-4
        atol = rtol*vnorm2(x0)

        # LHS matrix and RHS vector of the normal equations.
        A = H'*H
        b = H'*y
        n = length(b)
        x1 = conjgrad(A, b; maxiter=10n, restart=n,
                      quiet=true, verb=false,
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
                      quiet=false, verb=false,
                      gtol=(0,0), ftol=0, xtol=0)
        DEBUG && println("x3: ", vnorm2(x3 - x0)/vnorm2(x0))
        @test vnorm2(x3 - x0) ≤ atol
        x4 = conjgrad(A, b; maxiter=10n, restart=n,
                      quiet=true, verb=true,
                      gtol=(0,0), ftol=1e-9, xtol=0)
        DEBUG && println("x4: ", vnorm2(x4 - x0)/vnorm2(x0))
        @test vnorm2(x4 - x0) ≤ atol
        x5 = conjgrad(A, b; maxiter=10n, restart=n,
                      quiet=true, verb=false,
                      gtol=(0,1e-6), ftol=0, xtol=0)
        DEBUG && println("x5: ", vnorm2(x5 - x0)/vnorm2(x0))
        @test vnorm2(x5 - x0) ≤ atol

        # Exceptions.
        @test_throws LazyAlgebra.NonPositiveDefinite conjgrad(-A, b)

    end
end

end # module
