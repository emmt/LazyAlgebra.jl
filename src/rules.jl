#
# rules.jl -
#
# Implement rules for automatically simplifying expressions involving mappings.
#
#-------------------------------------------------------------------------------
#
# This file is part of LazyAlgebra (https://github.com/emmt/LazyAlgebra.jl)
# released under the MIT "Expat" license.
#
# Copyright (c) 2017-2022 Éric Thiébaut.
#

#------------------------------------------------------------------------------
# Identity and neutral elements for the addition and composition of mappings.

const Id = Identity()

# The neutral element ("zero") for the addition is unitless zero times a
# mapping of the proper type. NOTE: In the `apply` and `apply!` methods,
# applying such a scaled mapping should be optimized (although this depends on
# the specialization of the `apply!` method).
Base.zero(A::Mapping) = 0*A

Base.iszero(A::Scaled) = iszero(multiplier(A))
Base.iszero(::Mapping) = false

# The neutral element ("one") for the composition is the identity.
Base.one(A::Mapping) = one(typeof(A))
Base.one(::Type{<:Mapping}) = Id

Base.isone(::Identity) = true
Base.isone(::Mapping) = false

Base.oneunit(A::LinearMapping) = oneunit(typeof(A))
Base.oneunit(::Type{T}) where {T<:LinearMapping} = oneunit(eltype(T))*Id

#------------------------------------------------------------------------------
# Below are the only place where the `Scaled`, `Adjoint`, and `Inverse` inner
# constructors are directly called. To force automatic simplifications, exactly
# one expression amounts to directly calling a given constructor.
adjoint(A::LinearMapping) = Adjoint(A)
inv(A::Mapping) = Inverse(A)
*(α::Number, A::Mapping) = Scaled(α, A)
#------------------------------------------------------------------------------

# Automatic simplification of adjoint of adjoint and of inverse of inverse
# (also for constructors).
adjoint(A::Adjoint) = parent(A)
Adjoint(A::Adjoint) = parent(A)
inv(    A::Inverse) = parent(A)
Inverse(A::Inverse) = parent(A)

# Always build inverse-adjoint in that order (also for constructor).
adjoint(A::Inverse) = inv(adjoint(parent(A)))
Adjoint(A::Inverse) = inv(adjoint(parent(A)))

# Adjoint and inverse of a scaled mapping, the latter relying on further
# simplification of right-multiplication by a number.
adjoint(A::Scaled) = conj(multiplier(A))*adjoint(unscaled(A))
inv(    A::Scaled) = inv(unscaled(A))*inv(multiplier(A))

# Addition and composition of mappings.
+(A::Mapping, B::Mapping) = Sum((terms(+, A)..., terms(+, B)...,))
*(A::Mapping, B::Mapping) = Composition((terms(*, A)..., terms(*, B)...,))

# Distribute `adjoint` to the terms of sums and compositions.
adjoint(A::Sum) = Sum(map(adjoint, terms(A)))
adjoint(A::Composition) = Composition(reversemap(adjoint, terms(A)))

# Unary plus and minus.
+(A::Mapping) = A
-(A::Mapping) = (-1)*A
-(A::Scaled) = (-multiplier(A))*unscaled(A)

# Subtraction of mappings.
-(A::Mapping, B::Mapping) = A + (-B)

# Left and right division of mappings.
\(A::Mapping, B::Mapping) = inv(A)*B
/(A::Mapping, B::Mapping) = A*inv(B)

# Right-multiplication of a mapping by a number. For a linear mapping, the
# expression is rewritten as a left-multiplication of a mapping by a number (so
# that it is only necessary to specialize this expression for specific
# mappings). For a non-linear mapping, the expression is rewitten as
# right-multiplication of a mapping by the scaled identity.
*(A::LinearMapping, α::Number) = α*A
/(A::LinearMapping, α::Number) = α\A

*(A::NonLinearMapping, α::Number) = A*(α*Id)
/(A::NonLinearMapping, α::Number) = A*inv(α)

# Right- and left-division of a number by a mapping. The expression is
# rewritten as a multiplication of a mapping by a number which may be further
# simplified.
/(α::Number, A::Mapping) = α*inv(A)
\(A::Mapping, α::Number) = inv(A)*α

\(α::Number, A::Mapping) = inv(α)*A

# Special rules for scaled mappings.
*(α::Number, A::Scaled) = (α*multiplier(A))*unscaled(A)
/(α::Number, A::Scaled{true}) = (α/multiplier(A))*inv(unscaled(A))
/(A::Scaled{true}, α::Number) = (multiplier(A)/α)*unscaled(A)
\(α::Number, A::Scaled) = (multiplier(A)/α)*unscaled(A)
\(A::Scaled, α::Number) = inv(unscaled(A))*(α/multiplier(A))

# Dot operator (\cdot + tab) involving a mapping acts as the multiply or
# compose operator. FIXME: Restrict this to linear mappings?
⋅(A::Mapping, B::Mapping) = A*B
⋅(A::Mapping, B::Any    ) = A*B
⋅(A::Any,     B::Mapping) = A*B

# Compose operator (\circ + tab) beween mappings.
∘(A::Mapping, B::Mapping) = A*B

# Distribute multiplier to the terms of sums.
*(α::Number, A::Sum) = +(map(x -> α*x, terms(A))...)
\(α::Number, A::Sum) = +(map(x -> α\x, terms(A))...)

# Factorize multiplier to the left of compositions.
*(A::LinearMapping, B::Scaled) = multiplier(B)*(A*unscaled(B))
*(A::Scaled{true}, B::Scaled) = (multiplier(A)*multiplier(B))*(unscaled(A)*unscaled(B))
*(A::Scaled, B::Mapping) = multiplier(A)*(unscaled(A)*B)
/(A::Scaled{true}, B::Scaled) = (multiplier(A)/multiplier(B))*(unscaled(A)/unscaled(B))

# Rules for automatic type-stable simplifications involving the identity.
adjoint(A::Identity) = Id
inv(A::Identity) = Id
*(::Identity, ::Identity) = Id
for T in (:Scaled, :Composition, :Sum, :Mapping)
    @eval begin
        *(::Identity, A::$T) = A
        *(A::$T, ::Identity) = A
    end
end

"""
    ∇(A, x)

yields a result corresponding to the Jacobian (first partial derivatives) of
the mapping `A` for the variables `x`. If `A` is a linear mapping, `A` is
returned whatever `x`.

The call

    jacobian(A, x)

is an alias for `∇(A,x)`.

"""
∇(A::Mapping, x) = jacobian(A, x)
jacobian(A::NonLinearMapping, x) = Jacobian(A, x)
jacobian(A::LinearMapping, x) = A
jacobian(A::Scaled, x) = multiplier(A)*jacobian(unscaled(A), x)
@doc @doc(∇) jacobian
