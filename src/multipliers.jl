"""
    LazyAlgebra.convert_multiplier(α::Number, T::Type)

yields the multiplier `α` converted to the same floating-point precision as `T`.
If `T` has no concrete floating-point type, `$default_precision` is assumed.

See also [`LazyAlgebra.multiplier_type`](@ref) and [`LazyAlgebra.output_eltype`](@ref).

"""
convert_multiplier(α::Number, ::Type{T}) where {T<:Number} =
    with_precision(get_precision(T), α)

# Leave "static multipliers" unchanged.
convert_multiplier(α::StaticMultiplier, ::Type{<:Number}) = α

"""
    LazyAlgebra.convert_multiplier(α::Number, [A::Operator,] x::AbstractArray)

yields the multiplier `α` converted so that the operation `α*x` (if `A` is not specified)
or `α*A*x` (if `A` is specified) has a floating-point precision driven by `x` or by `A*x`
respectively.

See also [`LazyAlgebra.inplace_multiplier`](@ref), [`LazyAlgebra.multiplier_type`](@ref)
and [`LazyAlgebra.output_eltype`](@ref).

"""
convert_multiplier(α::Number, x::AbstractArray) =
    convert_multiplier(α, eltype(x))

convert_multiplier(α::Number, A::Operator, x::AbstractArray) =
    convert_multiplier(α, output_eltype(A, x))

"""
    LazyAlgebra.multiplier_type(α::Number, x::AbstractArray) -> T
    LazyAlgebra.multiplier_type(typeof(α), typeof(x)) -> T
    LazyAlgebra.multiplier_type(typeof(α), eltype(x)) -> T
    LazyAlgebra.multiplier_type(α::Number, A::Operator, x::AbstractArray) -> T
    LazyAlgebra.multiplier_type(typeof(α), typeof(A), typeof(x)) -> T
    LazyAlgebra.multiplier_type(typeof(α), output_eltype(A, x)) -> T

yield the type of the multiplier `α` such that the operation `α*x` (if `A` is not
specified) or `α*A*x` (if `A` is specified) has a numerical precision respectively driven
by `x` or by `A*x` (not by `α`).

This method implements a *trait*: the result shall only depend on the types of the
arguments, and the method may also be directly called with the types of the arguments.

See also [`LazyAlgebra.convert_multiplier`](@ref) and [``LazyAlgebra.output_eltype`](@ref).

"""
multiplier_type(α::Number, x::AbstractArray) =
    multiplier_type(typeof(α), typeof(x))

multiplier_type(::Type{S}, ::Type{X}) where {S<:Number,X<:AbstractArray} =
    multiplier_type(S, eltype(X))

multiplier_type(α::Number, A::Operator, x::AbstractArray) =
    multiplier_type(typeof(α), typeof(A), typeof(x))

multiplier_type(::Type{S}, ::Type{A}, ::Type{X}) where {S<:Number,A<:Operator,X<:AbstractArray} =
    multiplier_type(S, output_eltype(A, X))

multiplier_type(::Type{S}, ::Type{T}) where {S<:Number,T<:Number} =
    convert_floating_point_type(T, S)

"""
    LazyAlgebra.inplace_multiplier(α, x) -> α′
    LazyAlgebra.inplace_multiplier(α, eltype(x)) -> α′

yield the dimensionless value of `α`, throwing an exception if `α` is not suitable for
scaling array `x` in-place. `α` must be dimensionless, `α` may be complex only if
`eltype(x)` is complex, and must be real otherwise. The result `α′` is a real or a complex
value.

See also [`LazyAlgebra.dimensionless`](@ref) and  [`LazyAlgebra.convert_multiplier`](@ref).

"""
inplace_multiplier(α::Number, x::AbstractArray) = inplace_multiplier(α, eltype(x))
inplace_multiplier(α::Real, ::Type{<:Number}) = α
inplace_multiplier(α::Complex, ::Type{<:Union{Complex,AbstractQuantity{<:Complex}}}) = α
inplace_multiplier(α::AbstractQuantity, ::Type{x}) where {x} =
    inplace_multiplier(dimensionless(α), x)
@noinline inplace_multiplier(α::Number, ::Type{x}) where {x<:Number} =
    throw(ArgumentError(
        "multiplier of type `$(typeof(α))` is not suitable to scale in-place an array with elements of type `$x`"))

convert_inplace_multiplier(α::Number, x::AbstractArray) =
    convert_inplace_multiplier(α, eltype(x))
convert_inplace_multiplier(α::Number, ::Type{x}) where {x} =
    convert_multiplier(inplace_multiplier(α, x), x)
