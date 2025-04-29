using LazyAlgebra
using Test
using LinearAlgebra

include("crop-tests.jl")
include("diag-tests.jl")
include("pseudo-tests.jl")

@testset "LazyAlgebra.jl" begin
    include("vectors.jl")
    include("rules-tests.jl")
    LazyAlgebraCropTests.runtests()
    LazyAlgebraDiagTests.runtests()
    TestingLazyAlgebraPseudoMatrices.runtests()
end
nothing
