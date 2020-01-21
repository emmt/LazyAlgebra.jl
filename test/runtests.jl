module LazyAlgebraTests

using LazyAlgebra
using Test

include("common.jl")
include("utils-tests.jl")
test_utilities()
include("genmult-tests.jl")
GenMultTests.test_all()
include("coder-tests.jl")
include("vect-tests.jl")
include("map-tests.jl")
LazyAlgebraMappingTests.test_all()
include("diff-tests.jl")
include("sparse-tests.jl")
include("crop-tests.jl")
include("fft-tests.jl")
include("cg-tests.jl")

end
