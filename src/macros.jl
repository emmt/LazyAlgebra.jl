"""
    LazyAlgebra.@callable T

makes instances of concrete type `T` callable as a regular `LazyAlgebra` operator, that is
`A(x)` behaves as `A*x` and calls [`vmul(A, x)`](@ref vmul) for any operator `A` of type
`T`.
g
!!! note
    Since Julia 1.3, methods can be added to an abstract type and it is not necessary to
    use this macro for user-defined operators unless retro-compatibility is needed.

"""
macro callable(T)
    quote
	(A::$(esc(T)))(x) = vmul(A, x)
    end
end
