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

!!! note
    Not all types of object implement `get_precision`.

See also [`with_precision`](@ref).

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
    with_precision(T::Type{<:AbstractFloat}, x) -> y

yields an object `y` similar to `x` but with numerical precision specified by the
floating-point type `T`. If `x` has already the required precision or if setting its
precision is irrelevant or not implemented, `x` is returned unchanged. Setting the
precision shall not change the units if any. If `T` is `AbstractFloat`, the default
floating-point type `$default_precision` is assumed.

Argument `x` may also be a type to infer the type with precision `T`.

Example:

```julia
julia> with_precision(Float32, (1, 0x7, ("hello", 1.0, 1im)))
(1.0f0, 7.0f0, ("hello", 1.0f0, 0.0f0 + 1.0f0im))
```

!!! note
    For new object types, extend `_with_precision` (not directly `with_precision`). This
    auxiliary function shall only be called with a concrete floating-point type.

See also [`get_precision`](@ref).

"""
with_precision(::Type{AbstractFloat}, x::Any) = with_precision(default_precision, x)
with_precision(::Type{T}, x::Any) where{T<:AbstractFloat} = _with_precision(T, x)
with_precision(::Type{T}, x::Any) where {T} = throw_not_floating_point(T)

@noinline throw_not_floating_point(::Type{T}) where {T} = throw(ArgumentError(
    "type `$T` is not a floating-point type"))

"""
    f = with_precision(T)

builds a callable object `f` such that `f(x)` is equivalent to `with_precision(T, x)`. If
`T` is `AbstractFloat`, the default floating-point type `$default_precision` is assumed.

"""
with_precision(::Type{AbstractFloat}) = with_precision(default_precision)
with_precision(::Type{T}) where {T<:AbstractFloat} = _with_precision(T)
with_precision(::Type{T}) where {T} = throw_not_floating_point(T)

# NOTE Auxiliary function `_with_precision` is needed to avoid ambiguities. This auxiliary
#      function shall only be called with a concrete floating-point type. This auxiliary
#      function is the one to extend.
_with_precision(::Type{T}, x::Any) where {T<:AbstractFloat} = x # pass-through by default

# Converter.
_with_precision(::Type{T}) where {T<:AbstractFloat} = TypeUtils.Converter(_with_precision, T)

# Set precision of numbers.
_with_precision(::Type{T}, x::T) where {T<:AbstractFloat} = x
_with_precision(::Type{T}, x::Real) where {T<:AbstractFloat} = T(x)
_with_precision(::Type{T}, x::Complex{T}) where {T<:AbstractFloat} = x
_with_precision(::Type{T}, x::Complex) where {T<:AbstractFloat} = Complex{T}(real(x), imag(x))
_with_precision(::Type{T}, x::Number) where {T<:AbstractFloat} = convert_real_type(T, x)

# Set precision of numeric arrays.
_with_precision(::Type{T}, A::AbstractArray{T}) where {T<:AbstractFloat} = A
_with_precision(::Type{T}, A::AbstractArray{S}) where {T<:AbstractFloat,S} =
    convert_eltype(_with_precision(T, S), A)

# Set precision of types.
_with_precision(::Type{T}, ::Type{S}) where {T<:AbstractFloat,S} = S # pass-through by default
_with_precision(::Type{T}, ::Type{<:Real}) where {T<:AbstractFloat} = T
_with_precision(::Type{T}, ::Type{<:Complex}) where {T<:AbstractFloat} = Complex{T}
_with_precision(::Type{T}, ::Type{Array{S,N}}) where {T<:AbstractFloat,S,N} =
    Array{_with_precision{T, S}, N}
_with_precision(::Type{T}, ::Type{S}) where {T<:AbstractFloat,S<:Number} =
    convert_real_type(T, S)
_with_precision(::Type{T}, ::Type{Quantity{S,D,U}}) where {T<:AbstractFloat,S,D,U} =
    Quantity{_with_precision(T, S), D, U}
_with_precision(::Type{T}, x::Quantity{T,D,U}) where {T<:AbstractFloat,D,U} = x
_with_precision(::Type{T}, x::Quantity{S,D,U}) where {T<:AbstractFloat,S,D,U} =
    Quantity{_with_precision(T, S), D, U}(x)

## In other cases, map converter if object is an iterator and return the object otherwise.
#_with_precision(::Type{T}, x::Any) where {T<:AbstractFloat} =
#    isiterable(x) ? maybe_unroll_map(_with_precision(T), x) : x

# Set precision for tuples.
_with_precision(::Type{T}, x::NamedTuple) where {T<:AbstractFloat} =
    map(_with_precision(T), x)
_with_precision(::Type{T}, x::Tuple) where {T<:AbstractFloat} =
    maybe_unroll_map(_with_precision(T), x)

maybe_unroll_map(f, x::Any) = map(f, x)
@inline maybe_unroll_map(f, x::Tuple) = length(x) ≤ 20 ? unroll_map(f, x) : map(f, x)

unroll_map(f, x::Tuple{}) = ()
unroll_map(f, x::Tuple{Any}) = (f(first(x)),)
@inline unroll_map(f, x::Tuple) = (f(first(x)), unroll_map(f, Base.tail(x))...)

# Set precision for Adjoint, Inverse, and Gram. Thanks to recursion, this also
# works for InverseAdjoint.
for W in (:Adjoint, :Inverse, :Gram)
    @eval begin
        _with_precision(::Type{T}, A::$W) where {T<:AbstractFloat} =
            $W(_with_precision(T, parent(A)))
    end
end

# Set precision for Sum.
_with_precision(::Type{T}, (A,B)::Prod) where {T<:AbstractFloat} =
    _with_precision(T, A) * _with_precision(T, B)

# Set precision for Prod.
_with_precision(::Type{T}, (A,B)::Sum) where {T<:AbstractFloat} =
    _with_precision(T, A) + _with_precision(T, B)

#-----------------------------------------------------------------------------------------

# Yield whether a number has integer storage.
is_rationalizable(x::Number) = is_rationalizable(typeof(x))
is_rationalizable(::Type{T}) where {T<:Number} = real_type(T) <: Union{Integer, Rational}

divide(num::Number, den::Number) =
    is_rationalizable(num) && is_rationalizable(den) ? num//den : num/den

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
