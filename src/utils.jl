# Yield the type of a product of two terms of respective types `S` and `T`.
prod_type(::Type{S}, ::Type{T}) where {S,T} = typeof(zero(S) * zero(T))

# Yield the type of a sum of two terms of respective types `S` and `T`. Same as
# `promote_type` except that the result is always concrete.
sum_type(::Type{S}, ::Type{T}) where {S,T} = typeof(zero(S) + zero(T))

# Yield the type of a sum of terms all of type `T`. The implemented logic is that `x[1] +
# x[2] + ...` shall have the same type as `n*x[i]` with `n` an `Int`.
sum_type(::Type{T}) where {T} = prod_type(Int, T)

# Yield the type of a sum of products of two terms, all of respective types `S` and `T`.
# The implemented logic is that `x[1]*y[1] + x[2]*y[2] + ...` shall have the same type as
# `n*x[i]*y[i]` with `n` an `Int`.
sumprod_type(::Type{S}, ::Type{T}) where {S,T} =
    typeof(zero(Int) * zero(S) * zero(T))

"""
    set_precision(T, x) -> y

yields an object `y` similar to `x` but with numerical precision specified by the
floating-point type `T`. If `x` has already the required precision or if setting its
precision is irrelevant or not implemented, `x` is returned unchanged. Setting the
precision shall not change the units if any. If `T` is `AbstractFloat`, the effect is as
if applying `float` recursively to the numeric values stored by `x`.

Example:

```julia
julia> set_precision(Float32, (1, 0x7, ("hello", 1.0, 1im)))
(1.0f0, 7.0f0, ("hello", 1.0f0, 0.0f0 + 1.0f0im))
```

!!! note
    Not all types of object implement `set_precision`.

"""
set_precision(::Type{T}, x::Any) where {T<:AbstractFloat} = x # pass-through is the default

# Error catcher.
@noinline set_precision(::Type{T}, x::Any) where {T} = throw(ArgumentError(
    "type `$T` is not a floating-point type"))

# Set precision of numbers.
set_precision(::Type{T}, α::Number) where {T<:AbstractFloat} = convert_real_type(T, α)
set_precision(::Type{AbstractFloat}, α::Number) = float(α)

# Set precision of numeric arrays.
set_precision(::Type{T}, A::AbstractArray{<:T}) where {T<:AbstractFloat} = A
set_precision(::Type{T}, A::AbstractArray) where {T<:AbstractFloat} =
    convert_eltype(set_precision(T, eltype(A)), A)

# Set precision of numerical types.
set_precision(::Type{T}, ::Type{S}) where {T<:AbstractFloat,S} = S
set_precision(::Type{T}, ::Type{S}) where {T<:AbstractFloat,S<:T} = S
set_precision(::Type{AbstractFloat}, ::Type{S}) where {S<:Number} = float(S)
set_precision(::Type{T}, ::Type{S}) where {T<:AbstractFloat,S<:Number} =
    convert_real_type(T, S)

## In other cases, map converter if object is an iterator and return the object otherwise.
#set_precision(::Type{T}, x::Any) where {T<:AbstractFloat} =
#    isiterable(x) ? maybe_unroll_map(set_precision(T), x) : x

# Set precision for tuples.
set_precision(::Type{T}, x::Tuple) where {T<:AbstractFloat} =
    maybe_unroll_map(set_precision(T), x)

maybe_unroll_map(f, x::Any) = map(f, x)
@inline maybe_unroll_map(f, x::Tuple) = length(x) ≤ 20 ? unroll_map(f, x) : map(f, x)

unroll_map(f, x::Tuple{}) = ()
unroll_map(f, x::Tuple{Any}) = (f(first(x)),)
@inline unroll_map(f, x::Tuple) = (f(first(x)), unroll_map(f, Base.tail(x))...)

# Set precision for Adjoint, Inverse, and Gram. Thanks to recursion, this also
# works for InverseAdjoint.
for W in (:Adjoint, :Inverse, :Gram)
    @eval begin
        set_precision(::Type{T}, A::$W) where {T<:AbstractFloat} =
            $W(set_precision(T, parent(A)))
    end
end

# Set precision for Sum.
set_precision(::Type{T}, (A,B)::Prod) where {T<:AbstractFloat} =
    set_precision(T, A) * set_precision(T, B)

# Set precision for Prod.
set_precision(::Type{T}, (A,B)::Sum) where {T<:AbstractFloat} =
    set_precision(T, A) + set_precision(T, B)

"""
    f = set_precision(T)

builds a callable object `f` such that `f(x)` is equivalent to `set_precision(T, x)`.

"""
set_precision(::Type{T}) where {T<:AbstractFloat} = TypeUtils.Converter(set_precision, T)
@noinline set_precision(::Type{T}) where {T} = throw(ArgumentError(
    "type `$T` is not a floating-point type"))

# Inlined functions called to perform `α*x + β*y` for specific values of the
# multipliers `α` and `β`.  Passing these (simple) functions to another method
# is to simplify the coding of vectorized methods and of the the `vmul!`
# method by operators.  NOTE: Forcing inlining may not be necessary but it does
# not hurt.
@inline axpby_yields_zero( α, x, β, y) = zero(typeof(y)) # α = 0, β = 0
@inline axpby_yields_y(    α, x, β, y) = y               # α = 0, β = 1
@inline axpby_yields_my(   α, x, β, y) = -y              # α = 0, β = -1
@inline axpby_yields_by(   α, x, β, y) = β*y             # α = 0, any β
@inline axpby_yields_x(    α, x, β, y) = x               # α = 1, β = 0
@inline axpby_yields_xpy(  α, x, β, y) = x + y           # α = 1, β = 1
@inline axpby_yields_xmy(  α, x, β, y) = x - y           # α = 1, β = -1
@inline axpby_yields_xpby( α, x, β, y) = x + β*y         # α = 1, any β
@inline axpby_yields_mx(   α, x, β, y) = -x              # α = -1, β = 0
@inline axpby_yields_ymx(  α, x, β, y) = y - x           # α = -1, β = 1
@inline axpby_yields_mxmy( α, x, β, y) = -x - y          # α = -1, β = -1
@inline axpby_yields_bymx( α, x, β, y) = β*y - x         # α = -1, any β
@inline axpby_yields_ax(   α, x, β, y) = α*x             # any α, β = 0
@inline axpby_yields_axpy( α, x, β, y) = α*x + y         # any α, β = 1
@inline axpby_yields_axmy( α, x, β, y) = α*x - y         # any α, β = -1
@inline axpby_yields_axpby(α, x, β, y) = α*x + β*y       # any α, any β
