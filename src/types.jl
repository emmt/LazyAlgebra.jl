# types.jl -
#
# Definitions of types in LazyAlgebra.
#

# Miscellaneous.

#const AcceptableArraySize{N} = NTuple{N,Integer}
const AcceptableArrayAxis = Union{Integer,AbstractUnitRange{<:Integer}}
const AcceptableArrayAxes{N} = NTuple{N,AcceptableArrayAxis}
const ArrayAxis = AbstractUnitRange{Int}
const ArrayAxes{N} = NTuple{N,ArrayAxis}

# Domains.

abstract type AbstractDomain end

abstract type AbstractArrayDomain{T,N} <: AbstractDomain end

struct EmptyDomain <: AbstractDomain end
const ∅ = EmptyDomain()

struct ArrayDomain{T,N} <: AbstractArrayDomain{T,N}
    length::Int
    size::Dims{N}
end

struct OffsetArrayDomain{T,N,I<:ArrayAxes{N}} <: AbstractArrayDomain{T,N}
    length::Int
    axes::I
end

# Mappings.

abstract type AbstractMapping{I<:AbstractDomain,O<:AbstractDomain} end
abstract type AbstractLinearMapping{I<:AbstractDomain,O<:AbstractDomain} <: AbstractMapping{I,O} end

struct Scaled{I<:AbstractDomain,O<:AbstractDomain,
              T<:Number,A<:AbstractMapping{I}} <: AbstractMapping{I,O}
    out::O
    multiplier::T
    mapping::A
    function Scaled(α::Number, B::AbstractMapping)
        out = (α*output_domain(B))::AbstractDomain
        return new{input_domain_type(B),typeof(out),typeof(α),typeof(B)}(out, α, B)
    end
end

struct Inverse{I<:AbstractDomain,O<:AbstractDomain,
               A<:AbstractMapping} <: AbstractMapping{I,O}
    parent::A
    function Inverse(A::AbstractMapping)
        # Swap input and output domains. FIXME invert?
        return new{output_domain_type(A),input_domain_type(A),typeof(A)}(A)
    end
end

struct Adjoint{I<:AbstractDomain,O<:AbstractDomain,
               A<:AbstractMapping{O,I}} <: AbstractMapping{I,O}
    parent::A
    Adjoint(A::Inverse) = error("use `inv(A)'` or `inv(A')`")
    function Adjoint(A::AbstractMapping)
        # Swap input and output domains. FIXME conjugacy?
        return new{output_domain_type(A),input_domain_type(A),typeof(A)}(A)
    end
end

"""
    LazyAlgebra.InverseAdjoint{I,O,A}

is an alias for the type of the adjoint-inverse `inv(A)'` or inverse-ajoint
`inv(A')` of a linear mapping `A`. Both are the same and are automatically
converted to `inv(A')`.

"""
const InverseAdjoint{I,O,A} = Inverse{I,O,<:Adjoint{O,I,A}}
#const AdjointInverse{I,O,A} = Adjoint{I,O,<:Inverse{O,I,A}}

struct Sum{I<:AbstractDomain,O<:AbstractDomain} <: AbstractMapping{I,O}
    inp::I # input domain
    out::O # output domain
    terms::Vector{AbstractMapping}
    function Sum(io::Pair{I,O},
                 terms::AbstractVector{<:AbstractMapping}) where {I<:AbstractDomain,
                                                                   O<:AbstractDomain}
        return new{I,O}(first(io), last(io), terms)
    end
end

struct Composition{I<:AbstractDomain,O<:AbstractDomain} <: AbstractMapping{I,O}
    terms::Vector{AbstractMapping}
end

struct Identity{D<:AbstractDomain} <: AbstractLinearMapping{D,D}
    io::D # input and output domains
    Identity(domain::D) where {D<:AbstractDomain} = new{D}(domain)
end

struct Null{I<:AbstractDomain,O<:AbstractDomain} <: AbstractLinearMapping{I,O}
    inp::I # input domain
    out::O # output domain
    Null(io::Pair{I,O}) where {I<:AbstractDomain,O<:AbstractDomain} =
        new{I,O}(first(io), last(io))
end

struct Diag{D<:AbstractDomain,A} <: AbstractLinearMapping{D,D}
    io::D # input and output domains
    diag::A
    function Diag(D::AbstractDomain, A)
        A ∈ D || throw(ArgumentError("diagonal must belong to i/o domain of mapping"))
        return new{typeof(D),typeof(A)}(D, A)
    end
end
