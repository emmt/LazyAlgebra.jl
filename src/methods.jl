#
# methods.jl -
#
# Implement non-specific methods for operators.
#
#-----------------------------------------------------------------------------------------
#
# This file is part of LazyAlgebra (https://github.com/emmt/LazyAlgebra.jl) released under
# the MIT "Expat" license.
#
# Copyright (c) 2017-2025, Éric Thiébaut.
#

"""
    LazyAlgebra.output_eltype([α::Number,] A::Operator, x::AbstractArray) -> T

yields the element type `T` of the result of `A*x` or of `α*A*x` if the multiplier `α` is
specified.

As a simplification, it is assumed that the element type of `A*x` is a *trait* that only
depends on the type of the operator `A` and on the element type of the input array `x`.
Following this assumption, this method infers its result from that of:

    LazyAlgebra.output_eltype(typeof(A), eltype(x))

and it is thus expected that a method with this signature exists for the operator `A` and
that it returns the element type of `A*x`. If such a method does not exists, a fallback method
is provided which amounts to calling:

    Base.eltype(typeof(A))

to infer the type of the coefficients of `A` and which assumes that the element type of
`A*x` is that of the floating-point conversion of the multiplication of two values of
respective types `eltype(A)` and `eltype(x)`.

See also [`LazyAlgebra.output_axes`](@ref), [`LazyAlgebra.create_output`](@ref), and
[`LazyAlgebra.multiplier_type`](@ref).

"""
function output_eltype(α::Number, A::Operator, x::AbstractArray)
    T = output_eltype(A, x) # element type of A*x
    return prod_type(multiplier_type(typeof(α), T), T)
end

output_eltype(A::Operator, x::AbstractArray) = output_eltype(typeof(A), eltype(x))

# Fallback method, assumes that `eltype(A)` is extended.
output_eltype(::Type{A}, ::Type{X}) where {A<:Operator,X} =
    float(prod_type(eltype(A), X))

# Extend `Base.eltype` for operators and their variants. NOTE This is not necessary for
# `Sum` and `Prod` as they implement `output_eltype` properly.
Base.eltype(A::Operator) = eltype(typeof(A))
Base.eltype(::Type{Adjoint{A}}) where {A} = eltype(A)
Base.eltype(::Type{Inverse{A}}) where {A} = float(eltype(A))
Base.eltype(::Type{InverseAdjoint{A}}) where {A} = float(eltype(A))

# Output element type for products and sums assuming right-associativity.
output_eltype(::Type{Prod{L,R}}, ::Type{X}) where {L,R,X} =
    output_eltype(L, output_eltype(R, X))

output_eltype(::Type{Sum{L,R}}, ::Type{X}) where {L,R,X} =
    sum_type(output_eltype(L, X), output_eltype(R, X))

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
    LazyAlgebra.output_axes(A::Operator, x::AbstractArray)

yields the axes of the result of `A*x`.

As a simplification, it is assumed that the axes of `A*x` only depend on the operator `A`
and on the axes of the input array `x`. Following this assumption, this method returns the
result of:

    LazyAlgebra.output_axes(A, axes(x))

and it is thus expected that a method with this signature exists for the operator `A`.

See also [`LazyAlgebra.output_eltype`](@ref) and [`LazyAlgebra.create_output`](@ref).

"""
output_axes(A::Operator, x::AbstractArray) = output_axes(A, axes(x))

# Output axes for products assuming right-associativity.
output_axes(A::Prod{<:Number}, J::ArrayAxes) = output_axes(A[2], J)
output_axes(A::Prod, J::ArrayAxes) = output_axes(A[1], output_axes(A[2], J))

# Output axes for sums assuming right-associativity.
output_axes(A::Sum, J::ArrayAxes) =
    output_axes_in_sum(output_axes(A[1], J), A[2], J)

output_axes_in_sum(I::ArrayAxes, A::Sum, J::ArrayAxes) =
    output_axes(A[1], J) == I ? output_axes_in_sum(I, A[2], J) :
    throw_incompatible_output_axes_in_sum()

output_axes_in_sum(I::ArrayAxes, A::Operator, J::ArrayAxes) =
    output_axes(A, J) == I ? I : throw_incompatible_output_axes_in_sum()

@noinline  throw_incompatible_output_axes_in_sum() =
    throw(DimensionMismatch("incompatible output axes in sum"))

"""
    y = LazyAlgebra.create_output([α::Number,] A::Operator, x::AbstractArray)

creates an array `y` to store the result of `A*x` or of `α*A*x` if the multiplier `α` is
specified. In this latter case, it shall be assumed that `α` has been already converted
by [`LazyAlgebra.convert_multiplier`](@ref).

The method may be specialized in the operator type. The default implementations are:

```julia
create_output(A::Operator, x::AbstractArray) =
    new_array(output_eltype(A, x), output_axes(A, x))

create_output(α::Number, A::Operator, x::AbstractArray) =
    new_array(output_eltype(α, A, x), output_axes(A, x))
```

where the `new_array` method is taken from the `TypeUtils` package. Hence, by default, if
`LazyAlgebra.output_axes(A, x)` yields a tuple consisting of `Base.OneTo` instances, an
array of type `Array` with 1-based indices is returned; otherwise, an `OffsetArray` is
returned.

!!! warning
    This method is called by [`vmul`](@ref) to create its output before calling
    [`LazyAlgebra.unsafe_vmul!`](@ref) assuming that `x` and `y` have correct indices to
    compute `A*x` and to store the result in `y`. Hence, it is important that any
    specialization of `LazyAlgebra.create_output` throws an exception if the axes of `x`
    are not valid.

See also [`LazyAlgebra.output_axes`](@ref) and [`LazyAlgebra.output_eltype`](@ref).

"""
create_output(A::Operator, x::AbstractArray) =
    new_array(output_eltype(A, x), output_axes(A, x))

create_output(α::Number, A::Operator, x) =
    new_array(output_eltype(α, A, x), output_axes(A, x))

"""
    LazyAlgebra.convert_multiplier(α::Number, T::Type)

yields the multiplier `α` converted to the same floating-point precision as `T`.

See also [`LazyAlgebra.multiplier_type`](@ref) and [`LazyAlgebra.output_eltype`](@ref).

"""
convert_multiplier(α::Number, ::Type{T}) where {T<:Number} =
    convert_floating_point_type(T, α)

"""
    LazyAlgebra.convert_multiplier(α::Number, [A::Operator,] x::AbstractArray)

yields the multiplier `α` converted so that the operation `α*x` (if `A` is not specified)
or `α*A*x` (if `A` is specified) has a floating-point precision driven by `x` or by `A*x`
respectively.

See also [`LazyAlgebra.multiplier_type`](@ref) and [`LazyAlgebra.output_eltype`](@ref).

"""
convert_multiplier(α::Number, x::AbstractArray) =
    convert_multiplier(α, eltype(x))

convert_multiplier(α::Number, A::Operator, x::AbstractArray) =
    convert_multiplier(α, output_type(A, x))

"""
    LazyAlgebra.multiplier_type(α::Number, x::AbstractArray) -> T
    LazyAlgebra.multiplier_type(α::Number, A::Operator, x::AbstractArray) -> T

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
    convert_floating_point_type(eltype(X), S)

multiplier_type(α::Number, A::Operator, x::AbstractArray) =
    multiplier_type(typeof(α), typeof(A), typeof(x))

multiplier_type(::Type{S}, ::Type{A}, ::Type{X}) where {S<:Number,A<:Operator,X<:AbstractArray} =
    convert_floating_point_type(output_eltype(A, X), S)
@noinline function unimplemented(::Type{P},
                                 ::Type{T}) where {P<:Operations, T<:Operator}
    throw(UnimplementedOperation("unimplemented operation `$P` for operator $T"))
end

@noinline function unimplemented(func::Union{AbstractString,Symbol},
                                 ::Type{T}) where {T<:Operator}
    throw(UnimplementedMethod("unimplemented method `$func` for operator $T"))
end

"""
    LazyAlgebra.@callable T

makes instances of concrete type `T` callable as a regular `LazyAlgebra` operator, that is
`A(x)` yields [`vmul(A,x)`](@ref vmul) for any operator `A` of
type `T`.

"""
macro callable(T)
    quote
	(A::$(esc(T)))(x) = vmul(A, x)
    end
end
@callable Adjoint
@callable Inverse
@callable Gram
@callable Sum
@callable Prod

Base.show(io::IO, ::MIME"text/plain", A::Operator) = show(io, A)

Base.show(io::IO, A::Identity) = write(io, "Id")

function Base.show(io::IO, A::Adjoint)
    B = parent(A)
    show_paren(io, B, B isa Union{Sum,Prod,Adjoint})
    write(io, '\'')
end

function Base.show(io::IO, A::Inverse)
    write(io, "inv(")
    show(io, parent(A))
    write(io, ')')
end

function show(io::IO, A::Prod)
    protect = A[2] isa Sum
    if A[1] isa Number
        λ = A[1]
        if λ == -1
            write(io, '-')
        elseif λ == 1
            protect = false
        else
            show_multiplier(io, λ)
            write(io, '*')
        end
    else
        show_in_prod(io, A[1])
        write(io, '*')
    end
    show_paren(io, A[2], protect)
end

function Base.show(io::IO, A::Sum)
    show(io, A[1])
    show_next_in_sum(io, A[2])
end

# Show a multiplier.
show_multiplier(io::IO, λ::Number) =  show_paren(io, λ, is_complex(λ))
is_complex(λ::Number) = false
is_complex(λ::Complex) = true

# Show a term in a product.
show_in_prod(io::IO, A::Operator) = show_paren(io, A, A isa Sum)

# Show a term optionally enclosed by parentheses.
function show_paren(io::IO, x, paren::Bool)
    paren && print(io, '(')
    show(io, x)
    paren && print(io, ')')
end

# `show_next_in_sum` shows a term in a sum (not the first one).
function show_next_in_sum(io::IO, A::Operator)
    write(io, " + ")
    show(io, A)
end

function show_next_in_sum(io::IO, A::Sum)
    show_next_in_sum(io, A[1])
    show_next_in_sum(io, A[2])
end

function show_next_in_sum(io::IO, A::Prod)
    if A[1] isa Number
        λ = A[1]
        if λ < zero(λ)
            λ = -λ
            write(io, " - ")
        else
            write(io, " + ")
        end
        if λ != one(λ)
            show_multiplier(io, λ)
            write(io, '*')
        end
    else
        write(io, " + ")
        show_in_prod(io, A[1])
        write(io, '*')
    end
    show_in_prod(io, A[2])
end

"""
    unveil(A)

unveils the operator embedded in operator `A` if it is a *decorated* operator (see
[`LazyAlgebra.DecoratedOperator`](@ref)); otherwise, just returns `A` if it is
not a *decorated* operator.

As a special case, `A` may be an instance of `LinearAlgebra.UniformScaling` and
the result is the LazyAlgebra operator corresponding to `A`.

"""
unveil(A::Union{Adjoint,Inverse,Gram}) = parent(A)
unveil(A::Union{Adjoint{<:Inverse},Inverse{<:Adjoint}}) = parent(parent(A))
unveil(A::Operator) = A
unveil(A::UniformScaling) = Operator(A)

"""
    unscaled(A)

yields the operator `M` of the scaled operator `A = λ*M` (see [`Scaled`](@ref));
otherwise yields `A`. This method also works for intances of
`LinearAlgebra.UniformScaling`. Call [`multiplier`](@ref) to get the multiplier
`λ`.

"""
unscaled(A::Operator) = A
unscaled(A::Prod{<:Number}) = A[1]
unscaled(A::UniformScaling) = Id

"""
    multiplier(A)

yields the multiplier `λ` of the scaled operator `A = λ*M` (see
[`Scaled`](@ref)); otherwise yields `1`. Note that this method also works for
intances of `LinearAlgebra.UniformScaling`. Call [`unscaled`](@ref) to get the
operator `M`. `λ`.

"""
multiplier(A::Prod{<:Number}) = A[2]
multiplier(A::Operator) = 1
multiplier(A::UniformScaling) = getfield(A, :λ)

"""
    identifier(A)

yields a hash value identifying almost uniquely the unscaled operator `A`. This
identifier is used for sorting terms in a sum of operators.

!!! warning
    For now, the identifier is computed as `objectid(unscaled(A))` and is
    unique with a very high probability.

"""
identifier(A::Operator) = objectid(unscaled(A))

Base.isless(A::Operator, B::Operator) = isless(identifier(A), identifier(B))

"""
    nrows(A)

yields the *equivalent* number of rows of the linear operator `A`. Not all
operators extend this method.

In the implemented generalization of linear operators, the equivalent number of
rows is the number of element of the result of applying the operator be it
single- or multi-dimensional.

"""
nrows(A::Operator) = prod(row_size(A))

"""
    ncols(A)

yields the *equivalent* number of columns of the linear operator `A`. Not all
operators extend this method.

In the implemented generalization of linear operators, the equivalent number of
columns is the number of element of an argument of the operator be it single-
or multi-dimensional.

"""
ncols(A::Operator) = prod(col_size(A))

"""
    row_size(A)

yields the dimensions of the result of applying the linear operator `A`, this
is equivalent to `output_size(A)`. Not all operators extend this method.

"""
row_size(A::Operator) = output_size(A)

"""
    col_size(A)

yields the dimensions of the argument of the linear operator `A`, this is
equivalent to `input_size(A)`. Not all operators extend this method.

"""
col_size(A::Operator) = input_size(A)

"""
    coefficients(A)

yields the object backing the storage of the coefficients of the linear operator
`A`. Not all linear operators extend this method.

""" coefficients

"""
    check(A) -> A

checks integrity of operator `A` and returns it.

"""
check(A::Operator) = A

"""
    LazyAlgebra.check_vmul(y, A, x) -> (v1, v2, v1 - v2)

yields `v1 = vdot(y, A*x)`, `v2 = vdot(A'*y, x)` and their difference for `A` a linear
operator, `y` a *vector* of the output space of `A` and `x` a *vector* of the input space
of `A`. In principle, the two inner products should be equal whatever `x` and `y`;
otherwise the implementation of the operator has a bug.

Simple linear operators operating on Julia arrays can be tested on random
*vectors* with:

    check_vmul([T=Float64,] outdims, A, inpdims) -> (v1, v2, v1 - v2)

with `outdims` and `outdims` the dimensions of the output and input *vectors*
for `A`. Optional argument `T` is the element type.

If `A` operates on Julia arrays and methods `input_eltype`, `input_size`,
`output_eltype` and `output_size` have been specialized for `A`, then:

    check_vmul(A) -> (v1, v2, v1 - v2)

is sufficient to check `A` against automatically generated random arrays.

See also: [`vdot`](@ref), [`vcreate`](@ref), [`vmul!`](@ref),
[`input_type`](@ref).

"""
function check_vmul(y::Ty, A::Operator, x::Tx) where {Tx, Ty}
    v1 = vdot(y, A*x)
    v2 = vdot(A'*y, x)
    (v1, v2, v1 - v2)
end

function check_vmul(::Type{T},
                    outdims::Tuple{Vararg{Int}},
                    A::Operator,
                    inpdims::Tuple{Vararg{Int}}) where {T<:AbstractFloat}
    check_vmul(randn(T, outdims), A, randn(T, inpdims))
end

function check_vmul(outdims::Tuple{Vararg{Int}},
                    A::Operator,
                    inpdims::Tuple{Vararg{Int}})
    check_vmul(Float64, outdims, A, inpdims)
end

check_vmul(A::Operator) =
    check_vmul(randn(output_eltype(A), output_size(A)), A,
               randn(input_eltype(A), input_size(A)))

"""
    identical(A, B)

yields whether `A` is the same operator as `B` in the sense that their effects
will always be the same. This method is used to perform some simplifications
and optimizations and may have to be specialized for specific operator types.
The default implementation is to return `A === B`.

!!! note
    The returned result may be true although `A` and `B` are not necessarily
    the same objects. For instance, if `A` and `B` are two sparse matrices
    whose coefficients and indices are stored in the same arrays (as can be
    tested with the `===` or `≡` operators, `identical(A,B)` should return
    `true` because the two operators will always behave identically (any
    changes in the coefficients or indices of `A` will be reflected in `B`). If
    any of the arrays storing the coefficients or the indices are not the same
    objects, then `identical(A,B)` must return `false` even though the stored
    values may be the same because it is possible, later, to change one
    operator without affecting identically the other.

"""
@inline identical(::Operator, ::Operator) = false # false if not same types
@inline identical(A::T, B::T) where {T<:Operator} = (A === B)

"""
    gram(A) -> A'*A

yields the Gram operator built out of the linear operator `A`. The result is
equivalent to `A'*A` but its type depends on simplifications that may occur.

See also [`Gram`](@ref).

"""
gram(A::Operator) = A'*A

@noinline throw_forbidden_Gram_of_non_linear_operator() =
    bad_argument("making a Gram operator out of a non-linear operator is not allowed")

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

"""
    y = A*x
    y = vmul(A, x)

or:

    y = (α*A)*x
    y = vmul(α, A, x)

yield the result of applying the linear operator `A` or the scaled linear operator `α*A`
to the argument `x`.

!!! warning
    Do not extend this method for specific operator types, but rather the
    [`LazyAlgebra.unsafe_vmul!`](@ref) method.

See also [`vmul](@ref), [`LazyAlgebra.Operator`](@ref), and
[`LazyAlgebra.unsafe_vmul!`](@ref).

"""
function vmul(A::Operator, x::AbstractArray)
    y = create_output(A, x)
    T = floating_point_type(eltype(y))
    unsafe_vmul!(one(T), A, x, zero(T), y)
    return y
end

function vmul(α::Number, A::Operator, x::AbstractArray)
    α = convert_multiplier(α, A, x)
    y = create_output(α, A, x)
    T = floating_point_type(eltype(y))
    unsafe_vmul!(α, A, x, zero(T), y)
    return y
end

Base.:(*)(A::Operator, x::AbstractArray) = vmul(A, x)
Base.:(\)(A::Operator, x::AbstractArray) = vmul(inv(A), x)

"""
    vmul!(α::Number, A::Operator, x::AbstractArray, β::Number, y::AbstractArray) -> y

overwrites `y` with `α*A⋅x + β*y`. The convention is that the prior contents of `y` is not
used at all if `iszero(β)` holds so `y` can be directly used to store the result even
though it is not initialized.

Another supported syntax is:

    vmul!(y::AbstractArray, [α::Number=1], A::Operator, x::AbstractArray) -> y

which overwrites `y` with `α*A*x` and returns `y` and thus amounts to calling:

    vmul!(α, A, x, 0, y)

The `vmul!` method can be seen as a generalization of the `LinearAlgebra.mul!` method.

!!! warning
    Do not extend this method for specific operator types, but rather the
    [`LazyAlgebra.unsafe_vmul!`](@ref) method.

See also [`vmul`](@ref), [`LazyAlgebra.Operator`](@ref), and
[`LazyAlgebra.unsafe_vmul!`](@ref).

"""
vmul!(y::AbstractArray, A::Operator, x::AbstractArray) =
    vmul!(1, A, x, 0, y)

vmul!(y::AbstractArray, α::Number, A::Operator, x::AbstractArray) =
    vmul!(α, A, x, 0, y)

function vmul!(α::Number, A::Operator, x::AbstractArray, β::Number, y::AbstractArray)
    axes_Ax = output_axes(A, x)
    axes_y = axes(y)
    axes_y == axes_Ax || throw_incompatible_axes("`y`", axes_y, axes_Ax)
    α = convert_multiplier(α, A, x)
    β = convert_multiplier(β, y)
    if !iszero(α)
        unsafe_vmul!(α, A, x, β, y)
    elseif !iszero(β)
        unsafe_vscale!(y, β)
    else
        vzero!(y)
    end
    return y
end

@noinline throw_incompatible_axes(arg_name, arg_axes, ref_axes) =
    throw(DimensionMismatch("axes of $(arg_name) should be `$(axes_to_string(ref_axes))`, got `$(axes_to_string(arg_axes))`"))

function axes_to_string(rngs::Tuple{Vararg{AbstractUnitRange{<:Integer}}})
    io = IOBuffer()
    print_axes(io, rngs)
    return String(take!(io))
end

print_axis(io::IO, rng::AbstractUnitRange{<:Integer}) =
    print(io, first(rng), ':', last(rng))

function print_axes(io::IO, rngs::Tuple{Vararg{AbstractUnitRange{<:Integer}}})
    write(io, '(')
    for (i, rng) in enumerate(rngs)
        i > 1 && write(io, ", ")
        print_axis(rng)
    end
    length(rngs) == 1 && write(io, ',')
    write(io, ')')
    nothing
end

"""
    LazyAlgebra.unsafe_vmul!(α::Number, A::Operator, x::AbstractArray,
                             β::Number, y::AbstractArray)

overwrites `y` with `α*A⋅x + β*y`. This method (not [`vmul`](@ref) nor [`vmul!`](@ref))
is supposed to be specialized for any supported operator type.

This method is called by [`vmul`](@ref) and [`vmul!`](@ref) after checking that arguments
`x` and `y` have correct axes (so that `@inbounds` may be assumed to compute the result
stored in `y`), with multipliers `α` and `β` converted to suitable floating-point types,
and only if `iszero(α)` does not hold. The convention is that the prior contents of `y` is
not used at all if `iszero(β)` holds so that `y` can be directly used to store the result
even though it is not initialized. `LazyAlgebra.unsafe_vmul!` shall return `nothing` (any
returned value is ignored by [`vmul`](@ref) and [`vmul!`](@ref).

After checking the axes of `x` and of `y` and converting the multipliers `α` and `β`,
[`vmul`](@ref) and [`vmul!`](@ref) do something like:

```julia
if !iszero(α)
    LazyAlgebra.unsafe_vmul!(α, A, x, β, y)
elseif !iszero(β)
    LazyAlgebra.unsafe_vscale!(y, β)
else
    LazyAlgebra.vzero!(y)
end
```

See also [`vmul`](@ref), [`vmul!`](@ref), [`LazyAlgebra.Operator`](@ref),
[`LazyAlgebra.unsafe_vscale!`](@ref), [`vzero!`](@ref),
[`LazyAlgebra.output_eltype`](@ref) [`LazyAlgebra.output_axes`](@ref), and
[`LazyAlgebra.create_output`](@ref).

"""
function unsafe_vmul! end

# Specialize `unsafe_vmul!` for a sum of operators knowing that a sum of more than 2
# operators is stored according to right-associativity.
function unsafe_vmul!(α::Number, A::Sum, x::AbstractArray, β::Number, y::AbstractArray)
    unsafe_vmul!(α, A[1], x, β,      y)
    unsafe_vmul!(α, A[2], x, one(β), y)
end

# Specialize `unsafe_vmul!` for a product whose leading operand is a scalar.
function unsafe_vmul!(α::Number, A::Prod{<:Number}, x::AbstractArray,
                      β::Number, y::AbstractArray)
    # Compute product of multipliers with the precision of `α`. This is necessary because
    # there is no constraints on the precision of the multiplier `λ` of a scaled operator.
    αλ = convert_floating_point_type(typeof(α), α*A[1])
    unsafe_vmul!(αλ, A[2], x, β, y)
end

# Specialize `unsafe_vmul!` for a product whose leading operand is a linear operator. This
# requires allocating temporaries. Thanks to the right-associativity imposed by the
# constructors, passing unused arguments `α`, `β`, and `y`, is avoided.
unsafe_vmul!(α::Number, A::Prod{<:Operator}, x::AbstractArray, β::Number, y::AbstractArray) =
    unsafe_vmul!(α, A[1], A[2]*x, β, y)

@noinline function unsafe_vmul!(α::Number, A::Union{Inverse{<:Sum},InverseAdjoint{<:Sum}},
                                 x::AbstractArray, β::Number, y::AbstractArray)
    error("automatic dispatching of the inverse of a sum of operators is not supported")
end

# Default rules to apply a Gram operator. Gram matrices are Hermitian by construction
# which left only 2 cases to deal with.
function unsafe_vmul!(α::Number, G::Gram, x::AbstractArray,
                      β::Number, y::AbstractArray)
    A = G[] # yields A such that G = A'*A
    unsafe_vmul!(α, A', A*x, β, y)
end
#
function unsafe_vmul!(α::Number, G::Inverse{<:Gram}, x::AbstractArray,
                      β::Number, y::AbstractArray)
    A = G[][] # yields A such that G = inv(A'*A) = inv(A)*inv(A')
    unsafe_vmul!(α, inv(A), inv(A')*x, β, y)
end

# Extend `LinearAlgebra.ldiv!(y, A, b)` to overwrite `y` with `A\b`.
LinearAlgebra.ldiv!(y::AbstractArray, A::Operator, b::AbstractArray) =
    vmul!(y, inv(A), b)

# Extend `LinearAlgebra.mul!(c, A, b, α, β)` to overwrite `c` with `α*A*b + β*c`.
LinearAlgebra.mul!(c::AbstractArray, A::Operator, b::AbstractArray, α::Number, β::Number) =
    vmul!(α, A, b, β, c)

# Extend `LinearAlgebra.mul!(y, A, b)` to overwrite `y` with `A*b`.
LinearAlgebra.mul!(y::AbstractArray, A::Operator, b::AbstractArray) =
    vmul!(y, A, b)
