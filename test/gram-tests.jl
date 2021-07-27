#
# gram-tests.jl -
#
# Test Gram operator.
#
module TestingLazyAlgebraRules
using LazyAlgebra
using LazyAlgebra.Foundations
using Test

@testset "Gram operators" begin
    @test gram(Id) === Id
    rows = (3,4,)
    cols = (2,5,)
    T = Float32
    a = rand(T, rows..., cols...)
    A = GeneralMatrix(a)
    AtA = gram(A)
    @test isa(AtA, Gram)
    @test isa(A'*A, Gram)
    @test A'*A === AtA
    @test AtA' === AtA
    x = rand(T, cols)
    y = A'*(A*x)
    z = AtA*x
    @test z ≈ y
end

end # module
