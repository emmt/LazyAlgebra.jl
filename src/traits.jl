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

# Adjoint or inverse of a mapping type yields the corresponding mapping type
# without the hassle to propagate the linear trait or to apply automatic
# simplifications. These rules must appear very early as they may be used to
# define the types of function arguments. These rules must closely follow the
# logic implemented for taking the adjoint or inverse of mapping instances in
# `rules.jl`.
adjoint(::Type{M}) where {M<:LinearMapping} = Adjoint{M}
adjoint(::Type{Adjoint{M}}) where {M<:LinearMapping} = M
adjoint(::Type{Inverse{true,M}}) where {M<:LinearMapping} = inv(adjoint(M))
# FIXME: adjoint(::Type{InverseAdjoint{M}}) where {M<:LinearMapping} = Inverse{true,M}
inv(::Type{M}) where {L,M<:Mapping{L}} = Inverse{L,M}
inv(::Type{Adjoint{M}}) where {M<:LinearMapping} = InverseAdjoint{M}
inv(::Type{Inverse{L,M}}) where {L,M<:Mapping{L}} = M
# FIXME: inv(::Type{InverseAdjoint{M}}) where {M<:LinearMapping} = Adjoint{M}

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

yields the *self-adjoint* trait of mapping (instance or type) `A` indicating
whether `A` is certainly a self-adjoint linear map (instance or type). The
returned value is one of the singletons `SelfAdjoint()` for self-adjoint linear
maps and `NonSelfAdjoint()` for other mappings.

See also: [`Trait`](@ref), [`is_selfadjoint`](@ref).

"""
SelfAdjointType(A::Mapping) = SelfAdjointType(typeof(A))
SelfAdjointType(::Type{<:Mapping}) = NonSelfAdjoint()
SelfAdjointType(::Type{<:Adjoint{M}}) where {M} = SelfAdjointType(M)
SelfAdjointType(::Type{<:InverseAdjoint{M}}) where {M} = SelfAdjointType(M)
SelfAdjointType(::Type{<:Inverse{true,M}}) where {M} = SelfAdjointType(M)
SelfAdjointType(::Type{<:Scaled{true,M}}) where {M} = SelfAdjointType(M)
SelfAdjointType(::Type{<:Gram}) = SelfAdjoint()
@generated SelfAdjointType(::Type{M}) where {M<:Sum} =
    :($(all(is_selfadjoint, types_of_terms(M)) ? SelfAdjoint() : NonSelfAdjoint()))

@doc @doc(SelfAdjointType) SelfAdjoint
@doc @doc(SelfAdjointType) NonSelfAdjoint

"""
    MorphismType(A)

yields the *morphism* trait of mapping (instance or type) `A` indicating
whether `A` is certainly an endomorphism (instance or type), that is its input
and output spaces are the same. The returned value is one of the singletons
`Endomorphism()` for mappings whose input and output spaces are the same or
`Morphism()` for other mappings.

See also: [`Trait`](@ref), [`is_endomorphism`](@ref).

"""
MorphismType(A::Mapping) = MorphismType(typeof(A))
MorphismType(::Type{<:Mapping}) = Morphism()
MorphismType(::Type{<:Adjoint{M}}) where {M} = MorphismType(M)
MorphismType(::Type{<:InverseAdjoint{M}}) where {M} = MorphismType(M)
MorphismType(::Type{<:Inverse{<:Any,M}}) where {M} = MorphismType(M)
MorphismType(::Type{<:Scaled{<:Any,M}}) where {M} = MorphismType(M)
MorphismType(::Type{<:Gram}) = Endomorphism()
@generated MorphismType(::Type{M}) where {M<:Union{Sum,Composition}} =
    :($(all(is_endomorphism, types_of_terms(M)) ? Endomorphism() : Morphism()))

@doc @doc(MorphismType) Morphism
@doc @doc(MorphismType) Endomorphism

"""
    DiagonalType(A)

yields the *diagonal* trait of mapping (instance or type) `A` indicating
whether `A` is certainly a diagonal linear mapping (instance or type). The
returned value is one of the singletons `DiagonalMapping()` for diagonal linear
maps or `NonDiagonalMapping()` for other mappings.

See also: [`Trait`](@ref), [`is_diagonal`](@ref).

"""
DiagonalType(A::Mapping) = DiagonalType(typeof(A))
DiagonalType(::Type{<:Mapping}) = NonDiagonalMapping()
DiagonalType(::Type{<:Adjoint{M}}) where {M} = DiagonalType(M)
DiagonalType(::Type{<:InverseAdjoint{M}}) where {M} = DiagonalType(M)
DiagonalType(::Type{<:Inverse{true,M}}) where {M} = DiagonalType(M)
DiagonalType(::Type{<:Scaled{true,M}}) where {M} = DiagonalType(M)
DiagonalType(::Type{<:Gram{M}}) where {M} = DiagonalType(M)
@generated DiagonalType(::Type{M}) where {M<:Union{Sum,Composition}} =
    :($(all(is_diagonal, types_of_terms(M)) ? DiagonalMapping() : NonDiagonalMapping()))

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

"""
    lazyAlgebra.types_of_terms(A) -> itr

yields an iterable over the types of the terms that compose the mapping
instance or type `A`. This introspection method is mostly useful for sum or
composition of mappings; for other mappings, it just returns a 1-tuple of their
type.

"""
types_of_terms(A::Mapping) = types_of_terms(typeof(A))
types_of_terms(::Type{A}) where {A<:Mapping} = (A,)
types_of_terms(::Type{A}) where {A<:Union{Sum,Composition}} = terms(A).types
