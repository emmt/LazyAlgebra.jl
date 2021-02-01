#
# traits.jl -
#
# Methods related to mapping traits.
#
#-------------------------------------------------------------------------------
#
# This file is part of LazyAlgebra (https://github.com/emmt/LazyAlgebra.jl)
# released under the MIT "Expat" license.
#
# Copyright (c) 2017-2021 Éric Thiébaut.
#

"""

```julia
LinearType(A)
```

yields the *linear* trait of mapping `A` indicating whether `A` is certainly
linear.  The returned value is one of the singletons `Linear()` for linear maps
or `NonLinear()` for other mappings.

See also: [`Trait`](@ref), [`is_linear`](@ref).

"""
LinearType(::Mapping) = NonLinear() # any mapping assumed non-linear by default
LinearType(::LinearMapping) = Linear()
LinearType(::Inverse{<:LinearMapping}) = Linear()
LinearType(::Scaled{<:LinearMapping}) = Linear()
LinearType(A::Inverse) = LinearType(unveil(A))
LinearType(A::Scaled) = LinearType(unscaled(A))
LinearType(A::Union{Sum,Composition}) =
    (allof(x -> LinearType(x) === Linear(), terms(A)...) ?
     Linear() : NonLinear())
LinearType(A::Scaled{T,S}) where {T,S} =
    # If the multiplier λ of a scaled mapping A = (λ⋅M) is zero, then A behaves
    # linearly even though M is not a linear mapping.  FIXME: But acknowledging
    # this as a linear mapping may give rise to troubles later.
    (multiplier(A) == zero(S) ? Linear() : LinearType(unscaled(A)))

@doc @doc(LinearType) Linear
@doc @doc(LinearType) NonLinear

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
SelfAdjointType(::Mapping) = NonSelfAdjoint()
SelfAdjointType(A::DecoratedMapping) = SelfAdjointType(unveil(A))
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
MorphismType(::Mapping) = Morphism()
MorphismType(A::DecoratedMapping) = MorphismType(unveil(A))
MorphismType(A::Gram) = Endomorphism()
MorphismType(A::Scaled) = MorphismType(unscaled(A))
MorphismType(A::Union{Sum,Composition}) =
    (allof(x -> MorphismType(x) === Endomorphism(), terms(A)...) ?
     Endomorphism() : Morphism())

@doc @doc(MorphismType) Morphism
@doc @doc(MorphismType) Endomorphism

"""

```julia
DiagonalType(A)
```

yields the *diagonal* trait of mapping `A` indicating whether `A` is certainly
a diagonal linear mapping.  The returned value is one of the singletons
`DiagonalMapping()` for diagonal linear maps or `NonDiagonalMapping()` for other
mappings.

See also: [`Trait`](@ref), [`is_diagonal`](@ref).

"""
DiagonalType(::Mapping) = NonDiagonalMapping()
DiagonalType(A::DecoratedMapping) = DiagonalType(unveil(A))
DiagonalType(A::Scaled) = DiagonalType(unscaled(A))
DiagonalType(A::Union{Sum,Composition}) =
    (allof(x -> DiagonalType(x) === DiagonalMapping(), terms(A)...) ?
     DiagonalMapping() : NonDiagonalMapping())

@doc @doc(DiagonalType) NonDiagonalMapping
@doc @doc(DiagonalType) DiagonalMapping

"""
```julia
is_linear(A)
```

yields whether `A` is certainly a linear mapping.

!!! note
    This method is intended to perform certain automatic simplifications or
    optimizations.  It is guaranted to return `true` when its argument is
    certainly a linear mapping but it may return `false` even though its
    argument behaves linearly because it is not always possible to figure out
    that a complex mapping assemblage has this property.

See also: [`LinearType`](@ref).

"""
is_linear(A::LinearMapping) = true
is_linear(A::Mapping) = _is_linear(LinearType(A))
_is_linear(::Linear) = true
_is_linear(::NonLinear) = false

"""
```julia
is_selfadjoint(A)
```

yields whether mapping `A` is certainly a self-adjoint linear mapping.

!!! note
    This method is intended to perform certain automatic simplifications or
    optimizations.  It is guaranted to return `true` when its argument is
    certainly a self-adjoint linear mapping but it may return `false` even
    though its argument behaves like a self-adjoint linear map because it is
    not always possible to figure out that a complex mapping construction has
    this property or because, for efficiency reasons, the coefficients of the
    mapping are not considered for this trait.

See also: [`SelfAdjointType`](@ref).

"""
is_selfadjoint(A::Mapping) = _is_selfadjoint(SelfAdjointType(A))
_is_selfadjoint(::SelfAdjoint) = true
_is_selfadjoint(::NonSelfAdjoint) = false

"""
```julia
is_endomorphism(A)
```

yields whether mapping `A` is certainly an endomorphism.

!!! note
    This method is intended to perform certain automatic simplifications or
    optimizations.  It is guaranted to return `true` when its argument is
    certainly an endomorphism but it may return `false` even though its
    argument behaves like an endomorphism because it is not always possible to
    figure out that a complex mapping assemblage has this property.

See also: [`MorphismType`](@ref).

"""
is_endomorphism(A::Mapping) = _is_endomorphism(MorphismType(A))
_is_endomorphism(::Endomorphism) = true
_is_endomorphism(::Morphism) = false

"""
```julia
is_diagonal(A)
```

yields whether mapping `A` is certainly a diagonal linear map.

!!! note
    This method is intended to perform certain automatic simplifications or
    optimizations.  It is guaranted to return `true` when its argument is
    certainly a diagonal linear map but it may return `false` even though its
    argument behaves like a diagonal linear map because it is not always
    possible to figure out that a complex mapping assemblage has this property.

See also: [`DiagonalType`](@ref).

"""
is_diagonal(A::Mapping) = _is_diagonal(DiagonalType(A))
_is_diagonal(::DiagonalMapping) = true
_is_diagonal(::NonDiagonalMapping) = false
