#
# types.jl -
#
# Type definitions for linear algebra.
#
#-------------------------------------------------------------------------------
#
# This file is part of the MockAlgebra package released under the MIT "Expat"
# license.
#
# Copyright (c) 2017-2018 Éric Thiébaut.
#

struct SingularSystem <: Exception
    msg::String
end
Base.showerror(io::IO, err::SingularSystem) =
    print(io, "singular linear system ($(err.msg))")

struct NonPositiveDefinite <: Exception
    msg::String
end
Base.showerror(io::IO, err::NonPositiveDefinite) =
    print(io, "non-positive definite operator ($(err.msg))")

"""

Abstract type `LinearOperator` is introduced to extend the notion of
*matrices* and *vectors*.

Assuming the type of `A` inherits from `LinearOperator`, then:

* `A*x` yields the result of applying the operator `A` to `x`;

* `A'*x` yields the result of applying the adjoint of the operator `A` to `x`;

* `A\\x` yields the result of applying the inverse of operator `A` to `x`;

* `A'\\x` yields the result of applying the adjoint of the inverse of operator
  `A` to `x`.

The following methods should be implemented for a linear operator `A`:

```julia
newresult(Op, A, x) -> y
apply!(y, Op, A, x) -> y
```

for any supported operation `Op` among `Direct`, `Adjoint`, `Inverse` and
`InverseAdjoint`.  See the documentation of these methods for explanations.
Optionally, methods `Op(A)` may be extended, *e.g.* to throw exceptions if
operation`Op` is forbidden (or not implemented).  By default, all these
operations are assumed possible.

See also: [`apply`](@ref), [`apply!`](@ref), [`newresult`](@ref),
          [`is_applicable_in_place`](@ref), [`SelfAdjointOperator`](@ref),
          [`Direct`](@ref), [`Adjoint`](@ref), [`Inverse`](@ref),
          [`InverseAdjoint`](@ref).

"""
abstract type LinearOperator end

"""

Abstract type `SelfAdjointOperator` is to be inherited by linear operators
whose adjoint is equal to themself.  Such operators only need to implement
methods for the `Direct` and `Inverse` operations (if applicable).

See also: [`LinearOperator`](@ref).

"""
abstract type SelfAdjointOperator <: LinearOperator end

"""

Type `Direct` is a singleton type to indicate that a linear operator should
be directly applied.  This type is part of the union `Operations`.

See also: [`LinearOperator`](@ref), [`apply`](@ref), [`Operations`](@ref).

"""
struct Direct; end

"""

Types `Adjoint`, `Inverse` and `InverseAdjoint` are used to *decorate* the
conjugate transpose and/or inverse of any linear operator.  The `ctranspose`
method is extended, so that in the code, it is sufficient (and more readable)
to write `A'` instead of `Adjoint`.  `AdjointInverse` is just an alias for
`InverseAdjoint`.

See also: [`LinearOperator`](@ref), [`apply`](@ref), [`Operations`](@ref).

"""
struct Adjoint{T <: LinearOperator} <: LinearOperator
    op::T
end

struct Inverse{T <: LinearOperator} <: LinearOperator
    op::T
end
@doc @doc(Adjoint) Inverse

struct InverseAdjoint{T <: LinearOperator} <: LinearOperator
    op::T
end
@doc @doc(Adjoint) InverseAdjoint

const AdjointInverse = InverseAdjoint
@doc @doc(InverseAdjoint) AdjointInverse

"""

`Operations` is the union of the possible ways to apply a linear operator:
`Direct`, `Adjoint`, `Inverse` and `InverseAdjoint` (or its alias
`AdjointInverse).

See also: [`LinearOperator`](@ref), [`apply`](@ref), [`Direct`](@ref),
          [`Adjoint`](@ref).

"""
const Operations = Union{Direct,Adjoint,Inverse,InverseAdjoint}
