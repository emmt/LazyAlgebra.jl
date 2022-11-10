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
SelfAdjointType(::Type{T}) where {T<:Sum} =
    (all_terms(is_selfadjoint, T) ? SelfAdjoint() : NonSelfAdjoint())

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
MorphismType(::Type{T}) where {T<:Union{Sum,Composition}} =
    (all_terms(is_endomorphism, T) ? Endomorphism() : Morphism())

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
DiagonalType(::Type{T}) where {T<:Union{Sum,Composition}} =
    (all_terms(is_diagonal, T) ? DiagonalMapping() : NonDiagonalMapping())

@doc @doc(DiagonalType) NonDiagonalMapping
@doc @doc(DiagonalType) DiagonalMapping

function all_terms(f::Function, ::Type{<:Union{Sum{L,N,T},Composition{L,N,T}}}) where {L,N,T}
    for x in T.types
        f(x) || return false
        end
    return true
end

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
