module LazyAlgebraTests

using LazyAlgebra

# Deal with compatibility issues.
using Compat
using Compat.Test

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
