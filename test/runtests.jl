using LazyAlgebra
using Test
using LinearAlgebra

include("pseudo-tests.jl")

@testset "LazyAlgebra.jl" begin
    include("vectors.jl")
    include("rules-tests.jl")
    include("diag-tests.jl")

    TestingLazyAlgebraPseudoMatrices.runtests()
end
nothing
