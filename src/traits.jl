#
# traits.jl -
#
# Methods related to traits, that is properties which only depend on the types not on the
# values of objects.
#
#-----------------------------------------------------------------------------------------
#
# This file is part of LazyAlgebra (https://github.com/emmt/LazyAlgebra.jl)
# released under the MIT "Expat" license.
#
# Copyright (c) 2017-2025, Éric Thiébaut.
#

"""

```julia
SelfAdjointType(A)
```

yields the *self-adjoint* trait of mapping `A` indicating whether `A` is
certainly a self-adjoint linear map.  The returned value is one of the
singletons `SelfAdjoint()` for self-adjoint linear maps and `NonSelfAdjoint()`
for other mappings.

See also: [`Trait`](@ref), [`is_selfadjoint`](@ref).

"""
SelfAdjointType(::Operator) = NonSelfAdjoint()
SelfAdjointType(A::DecoratedOperator) = SelfAdjointType(unveil(A))
SelfAdjointType(A::Scaled) = SelfAdjointType(unscaled(A))
SelfAdjointType(A::Sum) =
    (allof(x -> SelfAdjointType(x) === SelfAdjoint(), terms(A)...) ?
     SelfAdjoint() : NonSelfAdjoint())
SelfAdjointType(A::Gram) = SelfAdjoint()

@doc @doc(SelfAdjointType) SelfAdjoint
@doc @doc(SelfAdjointType) NonSelfAdjoint

"""

```julia
MorphismType(A)
```

yields the *morphism* trait of mapping `A` indicating whether `A` is certainly
an endomorphism (its input and output spaces are the same).  The returned value
is one of the singletons `Endomorphism()` for mappings whose input and output
spaces are the same or `Morphism()` for other mappings.

See also: [`Trait`](@ref), [`is_endomorphism`](@ref).

"""
MorphismType(::Operator) = Morphism()
#=
MorphismType(A::DecoratedOperator) = MorphismType(unveil(A))
MorphismType(A::Gram) = Endomorphism()
MorphismType(A::Scaled) = MorphismType(unscaled(A))
MorphismType(A::Union{Sum,Composition}) =
    (allof(x -> MorphismType(x) === Endomorphism(), terms(A)...) ?
     Endomorphism() : Morphism())

@doc @doc(MorphismType) Morphism
@doc @doc(MorphismType) Endomorphism
=#

"""

```julia
DiagonalType(A)
```

yields the *diagonal* trait of mapping `A` indicating whether `A` is certainly
a diagonal linear mapping.  The returned value is one of the singletons
`DiagonalOperator()` for diagonal linear maps or `NonDiagonalOperator()` for other
mappings.

See also: [`Trait`](@ref), [`is_diagonal`](@ref).

"""
DiagonalType(::Operator) = NonDiagonalOperator()
#=
DiagonalType(A::DecoratedOperator) = DiagonalType(unveil(A))
DiagonalType(A::Scaled) = DiagonalType(unscaled(A))
DiagonalType(A::Union{Sum,Composition}) =
    (allof(x -> DiagonalType(x) === DiagonalOperator(), terms(A)...) ?
     DiagonalOperator() : NonDiagonalOperator())

@doc @doc(DiagonalType) NonDiagonalOperator
@doc @doc(DiagonalType) DiagonalOperator
=#

"""
```julia
is_selfadjoint(A)
```

yields whether mapping `A` is certainly a self-adjoint linear mapping.

!!! note
    This method is intended to perform certain automatic simplifications or
    optimizations.  It is guaranteed to return `true` when its argument is
    certainly a self-adjoint linear mapping but it may return `false` even
    though its argument behaves like a self-adjoint linear map because it is
    not always possible to figure out that a complex mapping construction has
    this property or because, for efficiency reasons, the coefficients of the
    mapping are not considered for this trait.

See also: [`SelfAdjointType`](@ref).

"""
is_selfadjoint(A::Operator) = _is_selfadjoint(SelfAdjointType(A))
_is_selfadjoint(::SelfAdjoint) = true
_is_selfadjoint(::NonSelfAdjoint) = false

"""
```julia
is_endomorphism(A)
```

yields whether mapping `A` is certainly an endomorphism.

!!! note
    This method is intended to perform certain automatic simplifications or
    optimizations.  It is guaranteed to return `true` when its argument is
    certainly an endomorphism but it may return `false` even though its
    argument behaves like an endomorphism because it is not always possible to
    figure out that a complex mapping assemblage has this property.

See also: [`MorphismType`](@ref).

"""
is_endomorphism(A::Operator) = _is_endomorphism(MorphismType(A))
_is_endomorphism(::Endomorphism) = true
_is_endomorphism(::Morphism) = false

"""
```julia
is_diagonal(A)
```

yields whether mapping `A` is certainly a diagonal linear map.

!!! note
    This method is intended to perform certain automatic simplifications or
    optimizations.  It is guaranteed to return `true` when its argument is
    certainly a diagonal linear map but it may return `false` even though its
    argument behaves like a diagonal linear map because it is not always
    possible to figure out that a complex mapping assemblage has this property.

See also: [`DiagonalType`](@ref).

"""
is_diagonal(A::Operator) = _is_diagonal(DiagonalType(A))
_is_diagonal(::DiagonalOperator) = true
_is_diagonal(::NonDiagonalOperator) = false
