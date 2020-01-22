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
convert_multiplier(λ, T, S=T)
```

yields multiplier `λ` converted to a suitable type for multiplying array(s)
whose elements have floating-point type `T` and for storage in a destination
array whose elements have type `S`.  This method is *type stable*.

The following rules are applied:

1. The result has the same floating-point precision as `T`.

2. The result is complex (of type `Complex{real(T)}`) if both `λ` and `S` are
   complex; otherwise the result is real (of type `real(T)`).  In the latter
   case, an `InexactError` exception may be thrown if `imag(λ)` is not zero.

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
    (isconcretetype(T) ? convert(Complex{real(T)}, λ) :
     operand_type_not_concrete(T))

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
