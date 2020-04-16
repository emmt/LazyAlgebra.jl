#
# utils.jl -
#
# General purpose methods.
#
#-------------------------------------------------------------------------------
#
# This file is part of LazyAlgebra (https://github.com/emmt/LazyAlgebra.jl)
# released under the MIT "Expat" license.
#
# Copyright (c) 2017-2020 Éric Thiébaut.
#

"""

```julia
promote_multiplier(λ, T)
```

yields multiplier `λ` converted to a suitable floating-point type for
multiplying values or expressions of type `T`.

Multiple arguments can be specified after the multiplier `λ`:

```julia
promote_multiplier(λ, args...)
```

to have `T` the promoted type of all types in `args...` or all element types of
arrays in `args...`.

This method is *type stable*.  The result has the same floating-point precision
as `T` and is a real if `λ` is real or a complex if `λ` is complex.

"""
promote_multiplier(λ::Real, ::Type{T}) where {T<:Floats} = begin
    # If λ ∈ ℝ, the returned multiplier is real with the same floating-point
    # precision as `T`.
    isconcretetype(T) || operand_type_not_concrete(T)
    convert(real(T), λ)
end

promote_multiplier(λ::Complex, ::Type{T}) where {T<:Floats} = begin
    # If λ ∈ ℂ, the returned multiplier is complex with the same floating-point
    # precision as `T`.
    isconcretetype(T) || operand_type_not_concrete(T)
    convert(Complex{real(T)}, λ)
end

@noinline promote_multiplier(λ::L, ::Type{T}) where {L<:Number,T} = begin
    # Other possible cases throw errors.
    isconcretetype(T) || operand_type_not_concrete(T)
    error(string("unsupported conversion of multiplier with type ", L,
                 " for operand with element type ",T))
end

promote_multiplier(λ::Number, ::AbstractArray{T}) where {T} =
    promote_multiplier(λ, T)
promote_multiplier(λ::Number, args::AbstractArray...) =
    promote_multiplier(λ, map(eltype, args)...)
promote_multiplier(λ::Number, args::Type...) =
    promote_multiplier(λ, promote_type(args...))

# Note: the only direct sub-types of `Number` are abstract types `Real` and
# `Complex`.
@noinline operand_type_not_concrete(::Type{T}) where {T} =
    error("operand type $T is not a concrete type")
