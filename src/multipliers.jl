"""
    LazyAlgebra.convert_multiplier(α::Number, T::Type)
    LazyAlgebra.convert_multiplier(T::Type, α::Number)

Return the multiplier `α` converted to the same floating-point precision as `T`. If `T`
has no concrete floating-point type, `TypeUtils.default_precision` is assumed.

See also [`LazyAlgebra.output_eltype`](@ref).

"""
convert_multiplier(α::Number, ::Type{T}) where {T<:Number} = convert_multiplier(T, α)
convert_multiplier(::Type{T}, α::Number) where {T<:Number} =
    adapt_precision(get_precision(T), α)

"""
    LazyAlgebra.convert_multiplier(α::Number, x::AbstractArray)
    LazyAlgebra.convert_multiplier(α::Number, A::Operator, x::AbstractArray)

Return the multiplier `α` converted so that the operation `α*x` (if `A` is not specified)
or `α*A*x` (if `A` is specified) has a precision driven by `x` or by `A*x` respectively,
not by the multiplier `α` itself.

See also [`LazyAlgebra.output_eltype`](@ref).

"""
convert_multiplier(α::Number, x::AbstractArray) = convert_multiplier(eltype(x), α)
convert_multiplier(α::Number, A::Operator, x::AbstractArray) =
    convert_multiplier(output_eltype(A, x), α)
