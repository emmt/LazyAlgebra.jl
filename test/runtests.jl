isdefined(:LazyAlgebra) || include("../src/LazyAlgebra.jl")

module LazyAlgebraTests

using Base.Test
using LazyAlgebra

include("utils.jl")
include("vectors.jl")
include("mappings.jl")

end
