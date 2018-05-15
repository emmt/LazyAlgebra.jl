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

distance(a::Real, b::Real) = abs(a - b)

distance(a::NTuple{2,Real}, b::NTuple{2,Real}) =
    hypot(a[1] - b[1], a[2] - b[2])

distance(A::AbstractArray{Ta,N}, B::AbstractArray{Tb,N}) where {Ta,Tb,N} =
    maximum(abs.(A - B))

@testset "ConjGrad" begin
    types = (Float32, Float64)
    rows = (5,6)
    cols = (2,3,4)
    @testset "least-square fit ($T)" for T in types
        rtol = 1e-4
        H = GeneralMatrix(randn(T, rows..., cols...))
        x = randn(T, cols)
        y = H*x + 0.01*randn(T, rows)

        # LHS matrix and RHS vector of the normal equations.
        A = H'*H
        b = H'*y
        n = length(b)
        x1 = conjgrad(A, b; maxiter=2n, restart=n, quiet=true,
                      gtol=(0,0), ftol=0, xtol=rtol/10)

        # Lesat-squares solution using Julia linear algebra.
        yj = reshape(y, length(y))
        Hj = reshape(contents(H), length(y), length(x))
        x2 = reshape((Hj'*Hj)\(Hj'*yj), cols)

        @test x1 ≈ x2 rtol=rtol norm=vnorm2
    end
end

end # module
