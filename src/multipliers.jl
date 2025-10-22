"""
    LazyAlgebra.convert_multiplier(α::Number, T::Type)
    LazyAlgebra.convert_multiplier(T::Type, α::Number)

Return the multiplier `α` converted to the same floating-point precision as `T`. If `T`
has no concrete floating-point type, `TypeUtils.default_precision` is assumed.

See also [`LazyAlgebra.multiplier_type`](@ref) and [`LazyAlgebra.output_eltype`](@ref).

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

See also [`LazyAlgebra.multiplier_type`](@ref) and [`LazyAlgebra.output_eltype`](@ref).

"""
convert_multiplier(α::Number, x::AbstractArray) = convert_multiplier(eltype(x), α)
convert_multiplier(α::Number, A::Operator, x::AbstractArray) =
    convert_multiplier(output_eltype(A, x), α)

"""
    LazyAlgebra.multiplier_type(α::Number, x::AbstractArray) -> T
    LazyAlgebra.multiplier_type(typeof(α), typeof(x)) -> T
    LazyAlgebra.multiplier_type(typeof(α), eltype(x)) -> T
    LazyAlgebra.multiplier_type(α::Number, A::Operator, x::AbstractArray) -> T
    LazyAlgebra.multiplier_type(typeof(α), typeof(A), typeof(x)) -> T
    LazyAlgebra.multiplier_type(typeof(α), output_eltype(A, x)) -> T

Return the type of the multiplier `α` such that the operation `α*x` (if `A` is not
specified) or `α*A*x` (if `A` is specified) has a numerical precision respectively driven
by `x` or by `A*x` (not by `α`).

This method implements a *trait*: the result shall only depend on the types of the
arguments, and the method may also be directly called with the types of the arguments.

See also [`LazyAlgebra.convert_multiplier`](@ref) and [``LazyAlgebra.output_eltype`](@ref).

"""
multiplier_type(α::Number, x::AbstractArray) =
    multiplier_type(typeof(α), typeof(x))

multiplier_type(::Type{α}, ::Type{x}) where {α<:Number,x<:AbstractArray} =
    multiplier_type(α, eltype(x))

multiplier_type(α::Number, A::Operator, x::AbstractArray) =
    multiplier_type(typeof(α), typeof(A), typeof(x))

multiplier_type(::Type{α}, ::Type{A}, ::Type{x}) where {α<:Number,A<:Operator,x<:AbstractArray} =
    multiplier_type(α, output_eltype(A, x))

multiplier_type(::Type{α}, ::Type{x}) where {α<:Number,x<:Number} =
    convert_floating_point_type(x, α)
