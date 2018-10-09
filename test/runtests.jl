isdefined(Main, :LazyAlgebra) || include("../src/LazyAlgebra.jl")

module LazyAlgebraTests

using LazyAlgebra

# Deal with compatibility issues.
using Compat
using Compat.Test

include("coder-test.jl")
include("utils.jl")
include("vectors.jl")
include("mappings.jl")
include("conjgrad.jl")

end
