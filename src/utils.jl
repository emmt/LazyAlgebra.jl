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
# Copyright (c) 2017-2019 Éric Thiébaut.
#

# FIXME: this should be in julia/base/multidimensional.jl
Base.CartesianIndex(I::CartesianIndex) = I

"""

```julia
convert_multiplier(λ, T, S=T)
```

yields multiplier `λ` converted to a suitable type for multiplying array(s) whose
elements have type `T` and for storage in a destination array whose elements
have type `S`.

The following rules are applied:

1. The result has the same floating-point precision as `T`.

2. The result is a real if `λ` is a real or both `T` and `S` are real types (an
   error is thrown if `imag(λ)` is not zero); otherwise, the result is complex
   if both `λ` and `S` are complex.

""" convert_multiplier

# If λ ∈ ℝ, the returned multiplier is real with the same floating-point
# precision as `T`.
convert_multiplier(λ::Real, ::Type{T}) where {T<:Floats} =
    (isconcretetype(T) ? convert(real(T), λ) : operand_type_not_concrete(T))
convert_multiplier(λ::Real, ::Type{T}, ::Type{<:Number}) where {T<:Floats} =
    (isconcretetype(T) ? convert(real(T), λ) : operand_type_not_concrete(T))

# If λ ∈ ℂ, the returned multiplier can only be complex if `S` is complex;
# otherwise returned multiplier is real and the call to `convert` will clash if
# `imag(λ)` is non-zero (this is what we want).
convert_multiplier(λ::Complex, ::Type{T}) where {T<:Floats} =
    # `T` and `S` are the same and may be real or complex.  The multiplier is
    # converted to `T` which may be complex.
    (isconcretetype(T) ? convert(T, λ) : operand_type_not_concrete(T))
convert_multiplier(λ::Complex, ::Type{T}, ::Type{<:Real}) where {T<:Reals} =
    # `T` and `S` are reals.  The multiplier is converted to a real of same
    # numerical precision as `T`, that is `T`.
    (isconcretetype(T) ? convert(T, λ) : operand_type_not_concrete(T))
convert_multiplier(λ::Complex, ::Type{T}, ::Type{<:Complex}) where {T<:Floats} =
    # `S` and `λ` are complex, The multiplier is converted to a complex of same
    # numerical precision as `T`.
    (isconcretetype(T) ? convert(Complex{real(T)}, λ) : operand_type_not_concrete(T))

# Other possible cases throw errors.
convert_multiplier(λ::L, ::Type{T}) where {L<:Number,T} =
    (isconcretetype(T) ? unsupported_multiplier_conversion(L, T, T) :
     operand_type_not_concrete(T))
convert_multiplier(λ::L, ::Type{T}, ::Type{S}) where {L<:Number,T,S} =
    (isconcretetype(T) ? unsupported_multiplier_conversion(L, T, S) :
     operand_type_not_concrete(T))

@noinline unsupported_multiplier_conversion(::Type{L}, ::Type{O}, ::Type{S}) where {L<:Number,O,S} =
    error("unsupported conversion of multiplier with type $L for operand with element type $O and storage with element type $S")

# Note: the only direct sub-types of `Number` are abstract types `Real` and
# `Complex`.
@noinline operand_type_not_concrete(::Type{T}) where {T} =
    error("operand type $T is not a concrete type")
