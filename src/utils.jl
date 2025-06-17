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

#------------------------------------------------------------------------- DIMENSIONLESS -

"""
    LazyAlgebra.dimensionless(x)

yields the numerical value of `x` throwing an exception if `x` is dimensionful. If `x` is
a real or complex number, `x` is returned; if `x` is a dimensionless quantity, it is
converted to the equivalent real or complex number.

Examples:

```juliadoc
julia> Using LazyAlgebra, Unitful.DefaultSymbols

julia> LazyAlgebra.dimensionless(3.0)
3.0

julia> LazyAlgebra.dimensionless(8kg/g)
8000

```

"""
dimensionless(x::Real) = x
dimensionless(x::Complex) = x
dimensionless(x::AbstractQuantity{T,NoDims,U}) where {T,U} = uconvert(NoUnits, x)
@noinline dimensionless(x::AbstractQuantity) =
    throw(ArgumentError("expecting a dimensionless value, got dimensions `$(unit(x))`"))

#----------------------------------------------------------------------------- PRECISION -

"""
    get_precision(x) -> T<:AbstractFloat
    get_precision(typeof(x)) -> T<:AbstractFloat

yield the numerical precision of number/object `x`. If `x` is a floating-point value, its
floating-point type is returned; if `x` stores floating-point values, their promoted
floating-point type is returned; otherwise, `AbstractFloat` is returned.

See also [`adapt_precision`](@ref).

"""
get_precision(x::Any) = get_precision(typeof(x))
get_precision(::Type) = AbstractFloat # pass-through
get_precision(::Type{T}) where {T<:AbstractFloat} = T
get_precision(::Type{AbstractFloat}) = AbstractFloat
get_precision(::Type{<:AbstractIrrational}) = AbstractFloat
get_precision(::Type{<:Real}) = AbstractFloat
get_precision(::Type{<:Complex{T}}) where {T} = get_precision(T)
get_precision(::Type{<:AbstractArray{T}}) where {T} = get_precision(T)
get_precision(::Type{<:AbstractQuantity{T}}) where {T} = get_precision(T)

@generated function get_precision(::Type{T}) where {T<:Union{Tuple,NamedTuple}}
    # NOTE Using a `Ref` for `r` or `t` here is significantly slower.
    r = AbstractFloat
    for s in T.types
        t = get_precision(s)::AbstractFloat
        if isconcretetype(t)
            if r == AbstractFloat
                r = t
            else
                r = promote_type(r, t)
            end
        end
    end
    return r
end

"""
    adapt_precision(T::Type{<:AbstractFloat}, x) -> y

yields an object `y` similar to `x` but with numerical precision specified by the
floating-point type `T`. If `x` has already the required precision or if setting its
precision is irrelevant or not implemented, `x` is returned unchanged. Setting the
precision shall not change the dimensions of dimensionful numbers. If `T` is
`AbstractFloat`, the default floating-point type `$default_precision` is assumed.

Argument `x` may also be a type to infer the corresponding type with precision `T`.

Example:

```julia
julia> adapt_precision(Float32, (1, 0x07, ("hello", 1.0, 3.0 - 2.0im, π)))
(1, 0x07, ("hello", 1.0f0, 3.0f0 - 2.0f0im, 3.1415927f0))
```

As can be seen, only floating-point and irrational values are converted.


!!! note
    For objects of foreign type, say `ForeignType`, method `adapt_precision(::Type{T},
    x::ForeignType) where {T<:Precision}` shall be extended to make sure it is only called
    with a concrete floating-point type `T`.

See also [`get_precision`](@ref).

"""
adapt_precision(::Type{AbstractFloat}, x::Any) = adapt_precision(default_precision, x)
adapt_precision(::Type{T}, x::Any) where{T<:Precision} = x # pass-through by default
adapt_precision(::Type{T}, x::Any) where {T} = throw_not_precision(T)

@noinline throw_not_precision(::Type{T}) where {T} = throw(ArgumentError(
    "type `$T` is not a precision type"))

"""
    f = adapt_precision(T)

builds a callable object `f` such that `f(x)` is equivalent to `adapt_precision(T, x)`. If
`T` is `AbstractFloat`, the default floating-point type `$default_precision` is assumed.

"""
adapt_precision(::Type{AbstractFloat}) = adapt_precision(default_precision)
adapt_precision(::Type{T}) where {T<:Precision} = TypeUtils.Converter(adapt_precision, T)
adapt_precision(::Type{T}) where {T} = throw_not_precision(T)

# Set precision of floating-point or irrational numbers.
adapt_precision(::Type{T}, x::T) where {T<:Precision} = x
adapt_precision(::Type{T}, x::AbstractFloat) where {T<:Precision} = T(x)
adapt_precision(::Type{T}, x::Irrational) where {T<:Precision} = T(x)
adapt_precision(::Type{T}, x::Complex{T}) where {T<:Precision} = x
adapt_precision(::Type{T}, x::Complex{<:Union{AbstractFloat,Irrational}}) where {T<:Precision} =
    Complex{T}(real(x), imag(x))

# Set precision of numeric arrays.
adapt_precision(::Type{T}, A::AbstractArray{T}) where {T<:Precision} = A
adapt_precision(::Type{T}, A::AbstractArray{S}) where {T<:Precision,S} =
    convert_eltype(adapt_precision(T, S), A)

# Set precision of types.
adapt_precision(::Type{T}, ::Type{S}) where {T<:Precision,S} = S # pass-through by default
adapt_precision(::Type{T}, ::Type{<:AbstractFloat}) where {T<:Precision} = T
adapt_precision(::Type{T}, ::Type{<:Irrational}) where {T<:Precision} = T
adapt_precision(::Type{T}, ::Type{<:Complex}) where {T<:Precision} = Complex{T}
adapt_precision(::Type{T}, ::Type{Array{S,N}}) where {T<:Precision,S,N} =
    Array{adapt_precision{T, S}, N}

adapt_precision(::Type{T}, ::Type{Quantity{S,D,U}}) where {T<:Precision,S,D,U} =
    Quantity{adapt_precision(T, S), D, U}
adapt_precision(::Type{T}, x::Quantity{T,D,U}) where {T<:Precision,D,U} = x
adapt_precision(::Type{T}, x::Quantity{S,D,U}) where {T<:Precision,S,D,U} =
    Quantity{adapt_precision(T, S), D, U}(x)

## In other cases, map converter if object is an iterator and return the object otherwise.
#adapt_precision(::Type{T}, x::Any) where {T<:Precision} =
#    isiterable(x) ? maybe_unroll_map(adapt_precision(T), x) : x

# Set precision for tuples.
adapt_precision(::Type{T}, x::NamedTuple) where {T<:Precision} =
    map(adapt_precision(T), x)
adapt_precision(::Type{T}, x::Tuple) where {T<:Precision} =
    maybe_unroll_map(adapt_precision(T), x)

maybe_unroll_map(f, x::Any) = map(f, x)
@inline maybe_unroll_map(f, x::Tuple) = length(x) ≤ 20 ? unroll_map(f, x) : map(f, x)

unroll_map(f, x::Tuple{}) = ()
unroll_map(f, x::Tuple{Any}) = (f(first(x)),)
@inline unroll_map(f, x::Tuple) = (f(first(x)), unroll_map(f, Base.tail(x))...)

# Set precision for Adjoint, and Inverse. Thanks to recursion, this also works for
# InverseAdjoint.
for W in (:Adjoint, :Inverse)
    @eval begin
        adapt_precision(::Type{T}, A::$W) where {T<:Precision} =
            $W(_adapt_precision(T, parent(A)))
    end
end

# Set precision for Sum.
adapt_precision(::Type{T}, (A,B)::Prod) where {T<:Precision} =
    adapt_precision(T, A) * adapt_precision(T, B)

# Set precision for Prod.
adapt_precision(::Type{T}, (A,B)::Sum) where {T<:Precision} =
    adapt_precision(T, A) + adapt_precision(T, B)

# Like `adapt_precision` except that all numeric types may be converted.
force_precision(::Type{T}, ::Type{T}) where {T<:Precision} = T
force_precision(::Type{T}, ::Type{S}) where {T<:Precision,S<:Number} =
    convert_real_type(T, S)

force_precision(::Type{T}, x::T) where {T<:Precision} = x
force_precision(::Type{T}, x::Number) where {T<:Precision} = convert_real_type(T, x)

force_precision(::Type{T}, A::AbstractArray{T}) where {T<:Precision} = A
forcet_precision(::Type{T}, A::AbstractArray{S}) where {T<:Precision,S<:Number} =
    convert_eltype(convert_real_type(T, S), A)

#-----------------------------------------------------------------------------------------

# Yield whether a number is a real with integer storage.
is_rationalizable(x::Number) = is_rationalizable(typeof(x))
is_rationalizable(::Type{T}) where {T<:Number} = bare_type(T) <: Union{Integer, Rational}

divide(num::Number, den::Number) =
    is_rationalizable(num) && is_rationalizable(den) ? num//den : num/den

#-----------------------------------------------------------------------------------------

@noinline throw_bad_argument(msg::AbstractString) = throw(ArgumentError(msg))
@noinline throw_bad_argument(args...) = throw(ArgumentError(string(args...)))
