#
# symbolic.jl -
#
# Implement symbolic operators in LazyAlgebra. These operators have symbolic names and are
# mainly used for debugging or demonstration.
#
#-----------------------------------------------------------------------------------------

struct SymbolicOperator <: Operator
    name::Symbol
end

SymbolicOperator(name::AbstractString) = SymbolicOperator(Symbol(id))

Base.show(io::IO, A::SymbolicOperator) = print(io, A.name)

# FIXME identical(A::SymbolicOperator, B::SymbolicOperator) = A.name === B.name
