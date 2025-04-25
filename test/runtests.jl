using LazyAlgebra
using Test
using LinearAlgebra

include("diag-tests.jl")
include("pseudo-tests.jl")

@testset "LazyAlgebra.jl" begin
    include("vectors.jl")
    include("rules-tests.jl")
    TestingLazyAlgebraDiag.runtests()
    TestingLazyAlgebraPseudoMatrices.runtests()
end
nothing
