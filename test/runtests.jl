isdefined(:LazyAlgebra) || include("../src/LazyAlgebra.jl")

module LazyAlgebraTests

using LazyAlgebra

@static if VERSION < v"0.7.0-DEV.2005"
    using Base.Test
else
    using Test
end

include("utils.jl")
include("vectors.jl")
include("mappings.jl")
include("conjgrad.jl")

end
