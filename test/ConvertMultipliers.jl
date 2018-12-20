#
# ConvertMultipliers.jl -
#
# Benchmarking of `convert_multiplier` methods.  In particular, this code is
# useful to identify which simplifications occur at compile time.
#

module ConvertMultipliers

using BenchmarkTools

const Reals = AbstractFloat
const Complexes = Complex{<:Reals}
const Floats = Union{Reals,Complexes}

"""

```julia
convert_multiplier(λ, T [, S=T])
```

yields multiplier `λ` converted to a suitable type for multiplying array whose
elements have type `T` and for storage in a destination array whose elements
have type `S`.

The following rules are applied:

1. Convert `λ` to the same floating-point precision as `T`.

2. The result is a real if `λ` is a real or both `T` and `S` are real types;
   otherwise (that is if `λ` is complex and at least one of `T` or `S` is a
   complex type), the result is a complex.

Result can be a real if imaginary part of `λ` is zero but this would break the
rule of type-stability at compilation time.

"""
convert_multiplier(λ::Real, ::Type{T}) where {T<:Floats} =
    (isconcretetype(T) ? convert(real(T), λ) : operand_type_not_concrete(T))
convert_multiplier(λ::Complex, ::Type{T}) where {T<:Floats} =
    # Call to `convert` will clash if `T` is real and `imag(λ)` is non-zero
    # (this is what we want).
    (isconcretetype(T) ? convert(T, λ) : operand_type_not_concrete(T))

convert_multiplier(λ::Real, ::Type{T}, ::Type{<:Number}) where {T<:Floats} =
    (isconcretetype(T) ? convert(real(T), λ) : operand_type_not_concrete(T))
convert_multiplier(λ::Complex, ::Type{T}, ::Type{<:Real}) where {T<:Reals} =
    # Call to `convert` will clash if `imag(λ)` is non-zero (this is what we
    # want).
    (isconcretetype(T) ? convert(T, λ) : operand_type_not_concrete(T))
convert_multiplier(λ::Complex, ::Type{T}, ::Type{<:Complex}) where {T<:Floats} =
    (isconcretetype(T) ? convert(Complex{real(T)}, λ) : operand_type_not_concrete(T))

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


function print_left_justified(str::AbstractString, len::Integer, c::Char=' ')
    print(str)
    (n = len - length(str)) > 0 && print(c^n)
end

function print_right_justified(str::AbstractString, len::Integer, c::Char=' ')
    (n = len - length(str)) > 0 && print(c^n)
    print(str)
end

# What is costly is to convert a double/single precision constant or literal
# value in a single/double precision value.  Integer constants and literal
# values are converted by the compiler.  Complexes with integer valued parts are
# converted to Float32/Float64 at compilation time but their conversion to
# ComplexF32/ComplexF64 is slow.
function runtests()
    l1, c1 = 20, ' '
    l2, c2 = 25, '.'
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        λ in (1, 0, 1 + 0im, 1.0, 0f0)
        print_left_justified("λ = $(repr(λ)) ", l1, c1)
        print_left_justified(" T = $T ", l2, c2)
        @btime convert_multiplier($λ,$T);
    end
    for T in (ComplexF32, ComplexF64),
        λ in (1, 0, 1.0, 0f0, 1 + 0im, 0 + 1im, 1.0 + 2im)
        print_left_justified("λ = $(repr(λ)) ", l1, c1)
        print_left_justified(" T = $T ", l2, c2)
        @btime convert_multiplier($λ,$T);
    end
end

end
