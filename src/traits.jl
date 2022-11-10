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
# Copyright (c) 2017-2022 Éric Thiébaut.
#

"""
    LinearType(A)

yields the *linear* trait of mapping `A` indicating whether `A` is certainly
linear. The returned value is one of the singletons `Linear()` for linear
mappings or `NonLinear()` for other mappings.

See also: [`Trait`](@ref), [`is_linear`](@ref).

"""
LinearType(A::Mapping) = LinearType(typeof(A))
LinearType(::Type{<:LinearMapping}) = Linear()
LinearType(::Type{<:NonLinearMapping}) = NonLinear()

@doc @doc(LinearType) Linear
@doc @doc(LinearType) NonLinear

"""
    SelfAdjointType(A)

yields the *self-adjoint* trait of mapping `A` indicating whether `A` is
certainly a self-adjoint linear map. The returned value is one of the
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
    MorphismType(A)

yields the *morphism* trait of mapping `A` indicating whether `A` is certainly
an endomorphism (its input and output spaces are the same). The returned value
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
    DiagonalType(A)

yields the *diagonal* trait of mapping `A` indicating whether `A` is certainly
a diagonal linear mapping. The returned value is one of the singletons
`DiagonalMapping()` for diagonal linear maps or `NonDiagonalMapping()` for
other mappings.

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
    is_linear(A)

yields whether `A` is a linear mapping.

See also: [`LinearType`](@ref).

"""
is_linear(x) = (LinearType(x) === Linear())

"""
    is_selfadjoint(A)

yields whether mapping `A` is certainly a self-adjoint linear mapping.

!!! note
    This method is called to perform certain automatic simplifications or
    optimizations. It is guaranted to return `true` when its argument is
    certainly a self-adjoint linear mapping but it may return `false` even
    though its argument behaves like a self-adjoint linear map because it is
    not always possible to figure out that a complex mapping construction has
    this property or because, for efficiency reasons, the coefficients of the
    mapping are not considered for this trait.

See also: [`SelfAdjointType`](@ref).

"""
is_selfadjoint(x) = (SelfAdjointType(x) === SelfAdjoint())

"""
    is_endomorphism(A)

yields whether `A` is certainly an endomorphism.

!!! note
    This method is called to perform certain automatic simplifications or
    optimizations. It is guaranted to return `true` when its argument is
    certainly an endomorphism but it may return `false` even though its
    argument behaves like an endomorphism because it is not always possible to
    figure out that a complex mapping assemblage has this property.

See also: [`MorphismType`](@ref).

"""
is_endomorphism(x) = (MorphismType(x) === Endomorphism())

"""
    is_diagonal(A)

yields whether mapping `A` is certainly a diagonal linear map.

!!! note
    This method is called to perform certain automatic simplifications or
    optimizations. It is guaranted to return `true` when its argument is
    certainly a diagonal linear map but it may return `false` even though its
    argument behaves like a diagonal linear map because it is not always
    possible to figure out that a complex mapping assemblage has this property.

See also: [`DiagonalType`](@ref).

"""
is_diagonal(x) = (DiagonalType(x) === DiagonalMapping())
