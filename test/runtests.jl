#
# runtests.jl -
#
# Run all tests.
#
module TestingLazyAlgebra

using Test

@testset "Utilities                        " begin
    include("utils-tests.jl")
end
@testset "Rules                            " begin
    include("rules-tests.jl")
end
@testset "Generalized matrix multiplication" begin
    include("genmult-tests.jl")
end
@testset "Vectorized operations            " begin
    include("vect-tests.jl")
end
@testset "Mappings                         " begin
    include("map-tests.jl")
end
@testset "Finite differences               " begin
    include("diff-tests.jl")
end
@testset "Sparse operators                 " begin
    include("sparse-tests.jl")
end
@testset "Cropping and padding             " begin
    include("crop-tests.jl")
end
@testset "FFT methods                      " begin
    include("fft-tests.jl")
end
#@testset "Gram operators                   " begin
    include("gram-tests.jl")
#end
@testset "Conjugate gradient               " begin
    include("cg-tests.jl")
end

end # module
