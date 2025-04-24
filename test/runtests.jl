using LazyAlgebra
using Test
using LinearAlgebra

@testset "LazyAlgebra.jl" begin
    include("vectors.jl")
    include("rules-tests.jl")
    include("diag-tests.jl")
end
