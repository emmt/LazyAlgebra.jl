using LazyAlgebra
using Test
using LinearAlgebra

include("identity-tests.jl")
include("rank1-tests.jl")
include("crop-tests.jl")
include("diag-tests.jl")
include("pseudo-tests.jl")

@testset "LazyAlgebra.jl" begin
    include("vect-tests.jl")
    include("rules-tests.jl")
    LazyAlgebraIdentityTests.runtests()
    LazyAlgebraRank1Tests.runtests()
    LazyAlgebraCropTests.runtests()
    LazyAlgebraDiagTests.runtests()
    TestingLazyAlgebraPseudoMatrices.runtests()
end
nothing
