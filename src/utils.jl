#--------------------------------------------------------------------- Minimum and maximum -

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

Return the least of `x` and `y`. If any of `x` and `y` is a NaN, the result is the NaN. This
latter behavior is not guaranteed if `@fastmath` is active.

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

#-------------------------------------------------------------------------- Type inference -

"""
    LazyAlgebra.sample(x::Number) -> val
    LazyAlgebra.sample(T::Type{<:Number}) -> val

Return a predictable numerical value of type `T` (with `T = typeof(x)` in the first above
case). For singleton type `T`, the returned value is the only possible instance, otherwise
the value is `oneunit(T)`.

This function is intended to be used for type inference, e.g. by
[`LazyAlgebra.return_type`](@ref).

"""
sample(x::Number) = sample(typeof(x))
function sample(::Type{T}) where {T<:Number}
    isconcretetype(T) || throw_bad_argument("`$T` is not a concrete type")
    return isdefined(T, :instance) ? getfield(T, :instance) : oneunit(T)
end
# error catcher
sample(::Type{T}) where {T} = throw_bad_argument("`$T` is not a numeric type")

"""
    LazyAlgebra.return_type(f::Function, args::Type...) -> T::Type

Return the type of the result returned by calling `f` with arguments of concrete numeric
types `args...`.

!!! warning
    `f` shall be a pure function.

See also [`LazyAlgebra.sample`](@ref) and `Base.promote_op` for caveats about return type
inference.

"""
return_type(f::Function) = typeof(f())
@inline return_type(f::Function, args::Type...) = typeof(f(map(sample, args)...))

"""
    LazyAlgebra.prod_type(T₁::Type, T₂::Type) -> T::Type

Return the type of a product of two terms of respective types `T₁` and `T₂`.

See also [`LazyAlgebra.sum_type`](@ref) and [`LazyAlgebra.sum_prod_type`](@ref).

"""
prod_type(::Type{T₁}, ::Type{T₂}) where {T₁,T₂} = typeof(sample(T₁) * sample(T₂))

"""
    LazyAlgebra.sum_type(T₁::Type, T₂::Type) -> T::Type

Return the type of a sum of two terms of respective types `T₁` and `T₂`.

See also [`LazyAlgebra.prod_type`](@ref) and [`LazyAlgebra.sum_prod_type`](@ref).

"""
sum_type(::Type{T₁}, ::Type{T₂}) where {T₁,T₂} = typeof(sample(T₁) + sample(T₂))

"""
    LazyAlgebra.sum_type(T₁::Type, T₂::Type) -> T::Type

Return the type of a sum of terms, all of type `T`.

"""
function sum_type(::Type{T}) where {T}
    x = sample(T)
    return typeof(x + x)
end

"""
    LazyAlgebra.sum_type(T₁::Type, T₂::Type) -> T::Type

Return the type of a sum of products of two terms, all of respective types `T₁` and `T₂`.

See also [`LazyAlgebra.prod_type`](@ref) and [`LazyAlgebra.sum_type`](@ref).

"""
sum_prod_type(::Type{T₁}, ::Type{T₂}) where {T₁, T₂} = sum_type(prod_type(T₁, T₂))

"""
    LazyAlgebra.is_iterable(x) -> bool
    LazyAlgebra.is_iterable(typeof(x)) -> bool

yield whether `x` is iterable, i.e. `iterate(x)` can be used to start iterating on `x`.

"""
is_iterable(x) = is_iterable(typeof(x))
is_iterable(::Type{T}) where {T} = hasmethod(Base.iterate, (T,))

# Yield whether a number is a real with integer storage.
is_rationalizable(x::Number) = is_rationalizable(typeof(x))
is_rationalizable(::Type{T}) where {T<:Number} = bare_type(T) <: Union{Integer, Rational}

#-------------------------------------------------------------------------- Ordinal suffix -

"""
    LazyAlgebra.ordinal_suffix(n)

Return the ordinal suffix of integer `n`, one of: `"st"`, `"nd"`, `"rd"`, or `"th"`.

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

#----------------------------------------------------------------------------- Multipliers -

# Divide 2 multipliers.
divide(num::Number, den::Number) =
    is_rationalizable(num) && is_rationalizable(den) ? num//den : num/den

# Inverse a multiplier.
inverse(α::Number ) = is_rationalizable(α) ? one(α)//α : inv(α)
inverse(α::Neutral) = inv(α)
