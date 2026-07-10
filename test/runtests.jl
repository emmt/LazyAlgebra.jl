using LazyAlgebra
using Test

@testset "LazyAlgebra.jl" begin
    include("traits-tests.jl")
    include("utils-tests.jl")
    include("vect-tests.jl").runtests();
    include("rules-tests.jl")
    include("identity-tests.jl").runtests();
    include("diag-tests.jl").runtests();
    include("crop-tests.jl").runtests();
    include("rank1-tests.jl").runtests();
    include("pseudo-tests.jl").runtests();
    include("sparse-tests.jl").runtests();
    #include("simplify-tests.jl").runtests();
end
nothing
