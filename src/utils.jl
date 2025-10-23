"""
    LazyAlgebra.fast_max(x, y)

Return the greatest of `x` and `y`. If any of `x` and `y` is a NaN, the result is the NaN.
This latter behavior is not guaranteed if `@fastmath` is active.

"""
fast_max(x::Number, y::Number) = fast_max(promote(x, y)...)
fast_max(x::T, y::T) where {T<:Integer} = y < x ? x : y
fast_max(x::T, y::T) where {T<:Number} = ifelse(isnan(x)|(y < x), x, y)

if isdefined(Base.Core.Intrinsics, :max_float)
    const max_float = Base.Core.Intrinsics.max_float
    fast_max(x::T, y::T) where {T<:Base.IEEEFloat} = max_float(x, y)
    function fast_max(x::S, y::S) where {T<:Base.IEEEFloat,S<:AbstractQuantity{T}}
        u = unit(S)
        return max_float(ustrip(u, x), ustrip(u, y))*u
    end
end

"""
    LazyAlgebra.fast_min(x, y)

Return the least of `x` and `y`. If any of `x` and `y` is a NaN, the result is the NaN.
This latter behavior is not guaranteed if `@fastmath` is active.

"""
fast_min(x::Number, y::Number) = fast_min(promote(x, y)...)
fast_min(x::T, y::T) where {T<:Integer} = x < y ? x : y
fast_min(x::T, y::T) where {T<:Number} = ifelse(isnan(x)|(x < y), x, y)

if isdefined(Base.Core.Intrinsics, :min_float)
    const min_float = Base.Core.Intrinsics.min_float
    fast_min(x::T, y::T) where {T<:Base.IEEEFloat} = min_float(x, y)
    function fast_min(x::S, y::S) where {T<:Base.IEEEFloat,S<:AbstractQuantity{T}}
        u = unit(S)
        return min_float(ustrip(u, x), ustrip(u, y))*u
    end
end

#-----------------------------------------------------------------------------------------

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
    LazyAlgebra.isiterable(x) -> bool
    LazyAlgebra.isiterable(typeof(x)) -> bool

yield whether `x` is iterable, i.e. `iterate(x)` can be used to start iterating on `x`.

"""
isiterable(x) = isiterable(typeof(x))
isiterable(::Type{T}) where {T} = hasmethod(Base.iterate, (T,))

"""
    LazyAlgebra.ordinal_suffix(n) -> "st" or "nd" or "rd" or "th"

yields the ordinal suffix for integer `n`.

"""
function ordinal_suffix(n::Integer)
    if n > 0
        d = mod(n, 10)
        if d == 1
            return "st"
        elseif d == 2
            return "nd"
        elseif d == 3
            return "rd"
        end
    end
    return "th"
end

#-----------------------------------------------------------------------------------------

# Yield whether a number is a real with integer storage.
is_rationalizable(x::Number) = is_rationalizable(typeof(x))
is_rationalizable(::Type{T}) where {T<:Number} = bare_type(T) <: Union{Integer, Rational}

# Divide 2 multipliers.
divide(num::Number, den::Number) =
    is_rationalizable(num) && is_rationalizable(den) ? num//den : num/den

# Inverse a multiplier.
inverse(α::Number ) = is_rationalizable(α) ? one(α)//α : inv(α)
inverse(α::Neutral) = inv(α)

#-----------------------------------------------------------------------------------------

@noinline throw_bad_argument(msg::AbstractString) = throw(ArgumentError(msg))
@noinline throw_bad_argument(args...) = throw(ArgumentError(string(args...)))
