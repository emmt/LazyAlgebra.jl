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

# MIME"text/plain" is for the REPL.
Base.show(io::IO, ::MIME"text/plain", A::SymbolicOperator) = print(io, A.name)
Base.show(io::IO, A::SymbolicOperator) =
    print(io, "SymbolicOperator(:", A.name, ")")

# Testing for equality. Note that `isequal` amounts to calling `==` by default.
Base.:(==)(A::SymbolicOperator, B::SymbolicOperator) = A.name === B.name
