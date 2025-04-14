module LazyAlgebra

export
    AbstractDomain, ArrayDomain, OffsetArrayDomain,
    AbstractMapping, AbstractLinearMapping,
    input_domain, output_domain,
    Diag, Null, Identity

using Unitless

using LinearAlgebra

import Base: ndims, eltype, axes, size, length
import Base: adjoint, transpose, inv, parent, eltype
import Base: +, -, *, \, /, ∘
import Base: ∈, ∪, ∩, ⊆

include("types.jl")
include("utils.jl")
include("domains.jl")
include("mappings.jl")
include("rules.jl")

end # module
