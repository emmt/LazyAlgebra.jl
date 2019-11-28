module LazyAlgebraTests

using LazyAlgebra
using Test

include("utils-test.jl")
test_utilities()
include("genmult-test.jl")
GenMultTests.test_all()
include("coder-test.jl")
include("utils.jl")
include("vectors.jl")
include("mappings.jl")
LazyAlgebraMappingTests.test_all()

include("conjgrad.jl")

end
