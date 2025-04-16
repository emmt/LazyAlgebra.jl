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
    LazyAlgebra.output_eltype([alpha::Number,] A::Operator, x::AbstractArray) -> T

yields the element type `T` of the result of `A*x` or of `alpha*A*x` if the multiplier
`alpha` is specified.

As a simplification, it is assumed that the element type of `A*x` is a *trait* that only
depends on the type of the operator `A` and on the element type of the input array `x`.
Following this assumption, this method infers its result from that of:

    LazyAlgebra.output_eltype(typeof(A), eltype(x))

and it is thus expected that a method with this signature exists for the operator `A` and
that it returns the element type of `A*x`. If such a method does not exists, a fallback method
is provided which calls:

    Base.eltype(typeof(A))

to infer the type of the elements of `A` and which assumes that the element type of `A*x`
is that of the floating-point conversion of the product of two values of respective types
`eltype(A)` and `eltype(x)` converted to floating-point.

This machinery is needed to support quantities with units.

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

# Output element type for products and sums assuming right-associativity.
output_eltype(::Type{Prod{L,R}}, ::Type{X}) where {L,R,X} =
    output_eltype(L, output_eltype(R, X))
output_eltype(::Type{Sum{L,R}}, ::Type{X}) where {L,R,X} =
    sum_type(output_eltype(L, X), output_eltype(R, X))
#
#output_eltype(::Type{S}, ::Type{X}) where {S<:Number,X} =
#    prod_type(multiplier_type(S, X), X)

"""
    LazyAlgebra.output_axes(A::Operator, x::AbstractArray)

yields the axes of the result of `A*x`.

As a simplification, it is assumed that the axes of the output only depend on the operator
`A` and on the axes of the input array `x`. Following this assumption, this method returns
the result of:

    LazyAlgebra.output_axes(A, axes(x))

and it is thus expected that a method with this signature exists for the operator `A`.

See also [`LazyAlgebra.output_eltype`](@ref) and [`LazyAlgebra.create_output`](@ref).

"""
output_axes(A::Operator, x::AbstractArray) = output_axes(A, axes(x))

# Output axes for products assuming right-associativity.
output_axes(A::Prod{<:Number}, J::ArrayAxes) = output_axes(last(A), J)
output_axes(A::Prod, J::ArrayAxes) = output_axes(first(A), output_axes(last(A), J))

# Output axes for sums assuming right-associativity.
output_axes(A::Sum, J::ArrayAxes) =
    output_axes_in_sum(output_axes(first(A), J), last(A), J)
output_axes_in_sum(I::ArrayAxes, A::Sum, J::ArrayAxes) =
    output_axes(first(A), J) == I ? output_axes_in_sum(I, last(A), J) :
    throw_incompatible_axes_in_sum()
output_axes_in_sum(I::ArrayAxes, A::Operator, J::ArrayAxes) =
    output_axes(A, J) == I ? I : throw_incompatible_axes_in_sum()

@noinline  throw_incompatible_axes_in_sum() =
    throw(DimensionMismatch("incompatible axes in sum"))

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
    This method is called by [`LazyAlgebra.apply`](@ref) to create its output before
    calling [`LazyAlgebra.unsafe_apply!`](@ref) assuming that `x` and `y` have correct
    indices to compute `A*x` and store the result in `y`. Hence, it is important that any
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


@noinline function unimplemented(::Type{P},
                                 ::Type{T}) where {P<:Operations, T<:Operator}
    throw(UnimplementedOperation("unimplemented operation `$P` for mapping $T"))
end

@noinline function unimplemented(func::Union{AbstractString,Symbol},
                                 ::Type{T}) where {T<:Operator}
    throw(UnimplementedMethod("unimplemented method `$func` for mapping $T"))
end

"""
    @callable T

makes instances of concrete type `T` callable as a regular `LazyAlgebra`
mapping, that is `A(x)` yields `apply(A,x)` for any `A` of type `T`.

"""
macro callable(T)
    quote
	(A::$(esc(T)))(x) = apply(A, x)
    end
end
@callable Adjoint
@callable Inverse
@callable Gram
@callable Sum
@callable Prod

Base.show(io::IO, ::MIME"text/plain", A::Operator) = show(io, A)
Base.show(io::IO, A::Operator) = _show(io, A)

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
    protect = A[2] isa Union{Sum,Prod} # FIXME: only Sum?
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

unveils the mapping embedded in mapping `A` if it is a *decorated* mapping (see
[`LazyAlgebra.DecoratedOperator`](@ref)); otherwise, just returns `A` if it is
not a *decorated* mapping.

As a special case, `A` may be an instance of `LinearAlgebra.UniformScaling` and
the result is the LazyAlgebra mapping corresponding to `A`.

"""
unveil(A::Union{Adjoint,Inverse,Gram}) = parent(A)
unveil(A::Union{Adjoint{<:Inverse},Inverse{<:Adjoint}}) = parent(parent(A))
unveil(A::Operator) = A
unveil(A::UniformScaling) = Operator(A)

"""
    unscaled(A)

yields the mapping `M` of the scaled mapping `A = λ*M` (see [`Scaled`](@ref));
otherwise yields `A`. This method also works for intances of
`LinearAlgebra.UniformScaling`. Call [`multiplier`](@ref) to get the multiplier
`λ`.

"""
unscaled(A::Operator) = A
unscaled(A::Prod{<:Number}) = first(A)
unscaled(A::UniformScaling) = Id

"""
    multiplier(A)

yields the multiplier `λ` of the scaled mapping `A = λ*M` (see
[`Scaled`](@ref)); otherwise yields `1`. Note that this method also works for
intances of `LinearAlgebra.UniformScaling`. Call [`unscaled`](@ref) to get the
mapping `M`. `λ`.

"""
multiplier(A::Prod{<:Number}) = last(A)
multiplier(A::Operator) = 1
multiplier(A::UniformScaling) = getfield(A, :λ)

"""
    identifier(A)

yields a hash value identifying almost uniquely the unscaled mapping `A`. This
identifier is used for sorting terms in a sum of mappings.

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

yields the object backing the storage of the coefficients of the linear mapping
`A`. Not all linear mappings extend this method.

""" coefficients

"""
    check(A) -> A

checks integrity of mapping `A` and returns it.

"""
check(A::Operator) = A

"""
    checkmapping(y, A, x) -> (v1, v2, v1 - v2)

yields `v1 = vdot(y, A*x)`, `v2 = vdot(A'*y, x)` and their difference for `A` a
linear mapping, `y` a *vector* of the output space of `A` and `x` a *vector* of
the input space of `A`. In principle, the two inner products should be equal
whatever `x` and `y`; otherwise the mapping has a bug.

Simple linear mappings operating on Julia arrays can be tested on random
*vectors* with:

    checkmapping([T=Float64,] outdims, A, inpdims) -> (v1, v2, v1 - v2)

with `outdims` and `outdims` the dimensions of the output and input *vectors*
for `A`. Optional argument `T` is the element type.

If `A` operates on Julia arrays and methods `input_eltype`, `input_size`,
`output_eltype` and `output_size` have been specialized for `A`, then:

    checkmapping(A) -> (v1, v2, v1 - v2)

is sufficient to check `A` against automatically generated random arrays.

See also: [`vdot`](@ref), [`vcreate`](@ref), [`apply!`](@ref),
[`input_type`](@ref).

"""
function checkmapping(y::Ty, A::Operator, x::Tx) where {Tx, Ty}
    v1 = vdot(y, A*x)
    v2 = vdot(A'*y, x)
    (v1, v2, v1 - v2)
end

function checkmapping(::Type{T},
                      outdims::Tuple{Vararg{Int}},
                      A::Operator,
                      inpdims::Tuple{Vararg{Int}}) where {T<:AbstractFloat}
    checkmapping(randn(T, outdims), A, randn(T, inpdims))
end

function checkmapping(outdims::Tuple{Vararg{Int}},
                      A::Operator,
                      inpdims::Tuple{Vararg{Int}})
    checkmapping(Float64, outdims, A, inpdims)
end

checkmapping(A::Operator) =
    checkmapping(randn(output_eltype(A), output_size(A)), A,
                 randn(input_eltype(A), input_size(A)))

"""
    identical(A, B)

yields whether `A` is the same mapping as `B` in the sense that their effects
will always be the same. This method is used to perform some simplifications
and optimizations and may have to be specialized for specific mapping types.
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

yields the Gram operator built out of the linear mapping `A`. The result is
equivalent to `A'*A` but its type depends on simplifications that may occur.

See also [`Gram`](@ref).

"""
gram(A::Operator) = A'*A

@noinline throw_forbidden_Gram_of_non_linear_mapping() =
    bad_argument("making a Gram operator out of a non-linear mapping is not allowed")

# Inlined functions called to perform `α*x + β*y` for specific values of the
# multipliers `α` and `β`.  Passing these (simple) functions to another method
# is to simplify the coding of vectorized methods and of the the `apply!`
# method by mappings.  NOTE: Forcing inlining may not be necessary but it does
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

#------------------------------------------------------------------------------
# VCREATE, APPLY AND APPLY!

"""
    vmul(A, x) -> y

yields `y = A*x`. The default behavior is to call `apply(A,x,false)`.
Method [`vmul!`](@ref) is the in-place version.

"""
vmul(A, x) = apply(A, x, false)

"""
    vmul!(y, A, x) -> y

overwrites `y` with the result of `A*x` and returns `y`. The default behavior
is to call `apply!(1,A,x,false,0,y)`.

!!! note
    This method is intended to be used by algorithms such as the conjugate
    gradient to apply operators. It may be specialized by the caller for its
    needs which is much easier than specializing [`apply!`](@ref) which
    requires to consider the specific values of the multipliers `α` and `β`.

"""
vmul!(y, A, x) = apply!(1, A, x, false, 0, y)

"""
    apply(A, x, scratch=false) -> y

yields the result `y` of applying mapping `A` to the argument `x`. Optional
parameter `P` can be used to specify how `A` is to be applied:

* `Adjoint` to apply the adjoint of `A` and yield `y = A'⋅x`;
* `Inverse` to apply the inverse of `A` and yield `y = A\\x`;
* `InverseAdjoint` or `AdjointInverse` to apply the inverse of `A'` and
  yield `y = A'\\x`.

Not all operations may be implemented by the different types of mappings and
`Adjoint` and `InverseAdjoint` may only be applicable for linear mappings.

Optional argument `scratch` indicates whether input argument `x` can be
overwritten by the operation. This may be exploited to avoid allocating
temporary workspace(s). The caller should set `scratch = true` if `x` is not
needed after calling `apply`. If `scratch = true`, then it is possible that `y`
be the same object as `x`; otherwise, `y` is a new object unless applying the
operation yields the same contents as `y` for the result `x` (this is always
true for the identity for instance). Thus, in general, it should not be assumed
that the result of applying a mapping is different from the input.

Julia methods are provided so that `apply(A', x)` automatically calls
`apply(Adjoint, A, x)` so the shorter syntax may be used without impacting
performances.

See also: [`Operator`](@ref), [`apply!`](@ref), [`vcreate`](@ref).

"""
apply(A::Operator, x, scratch::Bool=false) =
    apply!(1, P, A, x, scratch, 0, vcreate(A, x, scratch))

*(A::Operator, x::AbstractArray) = apply(A, x)
\(A::Operator, x::AbstractArray) = apply(inv(A), x)

"""
    apply!([α=1,] A::Operator, x, [scratch=false,] [β=0,] y) -> y

overwrites `y` with `α*A⋅x + β*y`. The convention is that the prior contents of `y` is not
used at all if `β = 0` so `y` can be directly used to store the result even though it is
not initialized. The `scratch` optional argument indicates whether the input `x` is no
longer needed by the caller and can thus be used as a scratch array. Having `scratch =
true` or `β = 0` may be exploited by the specific implementation of the `apply!` method
for the mapping type to avoid allocating temporary workspace(s).

The `apply!` method can be seen as a generalization of the `LinearAlgebra.mul!` method.

The order of arguments can be changed and the same result as above is obtained with:

    apply!([β=0,] y, [α=1,] A::Operator, x, scratch=false) -> y

The result `y` may have been allocated by:

    y = vcreate(A, x, scratch=false)

Operator sub-types only need to extend `vcreate` and `apply!` with the specific
signatures:

    vcreate(A::M, x, scratch::Bool=false) -> y
    apply!(α::Number, ::Type{P}, A::M, x, scratch::Bool, β::Number, y) -> y

for any supported operation `P` and where `M` is the type of the mapping. Of
course, the types of arguments `x` and `y` may be specified as well.

Optionally, the method with signature:

    apply(::Type{P}, A::M, x, scratch::Bool=false) -> y

may also be extended to improve the default implementation which is:

    apply(P::Type{<:Operations}, A::Operator, x, scratch::Bool=false) =
        apply!(1, P, A, x, scratch, 0, vcreate(P, A, x, scratch))

See also: [`Operator`](@ref), [`apply`](@ref), [`vcreate`](@ref).

"""
apply!(A::Operator, x::AbstractArray, y::AbstractArray) =
    apply!(1, A, x, false, 0, y)
apply!(α::Number, A::Operator, x::AbstractArray, y::AbstractArray) =
    apply!(α, A, x, false, 0, y)
apply!(A::Operator, x::AbstractArray, β::Number, y::AbstractArray) =
    apply!(1, A, x, false, β, y)
apply!(α::Number, A::Operator, x::AbstractArray, β::Number, y::AbstractArray) =
    apply!(α, A, x, false, β, y)

apply!(A::Operator, x::AbstractArray, scratch::Bool, y::AbstractArray) =
    apply!(1, A, x, scratch, 0, y)
apply!(α::Number, A::Operator, x::AbstractArray, scratch::Bool, y::AbstractArray) =
    apply!(α, A, x, scratch, 0, y)
apply!(A::Operator, x::AbstractArray, scratch::Bool, β::Number, y::AbstractArray) =
    apply!(1, A, x, scratch, β, y)

apply!(y::AbstractArray, A::Operator, x::AbstractArray, scratch::Bool=false) =
    apply!(1, A, x, scratch, 0, y)
apply!(y::AbstractArray, α::Number, A::Operator, x::AbstractArray, scratch::Bool=false) =
    apply!(α, A, x, scratch, 0, y)
apply!(β::Number, y::AbstractArray, A::Operator, x::AbstractArray, scratch::Bool=false) =
    apply!(1, A, x, scratch, β, y)
apply!(β::Number, y::AbstractArray, α::Number, A::Operator, x::AbstractArray, scratch::Bool=false) =
    apply!(α, A, x, scratch, β, y)

# Extend `LinearAlgebra.ldiv!(y, A, b)` to overwrite `y` with `A\b`.
LinearAlgebra.ldiv!(y::AbstractArray, A::Operator, b::AbstractArray) =
    apply!(y, inv(A), b)

# Extend `LinearAlgebra.mul!(c, A, b, α, β)` to overwrite `c` with `α*A*b + β*c`.
LinearAlgebra.mul!(y::AbstractArray, A::Operator, x::AbstractArray) =
    apply!(1, A, x, false, 0, y)

# Extend `LinearAlgebra.mul!(c, A, b, α, β)` to overwrite `c` with `α*A*b + β*c`.
LinearAlgebra.mul!(c::AbstractArray, A::Operator, b::AbstractArray, α::Number, β::Number) =
    apply!(α, A, x, false, β, y)

# Implemention of the `apply!(α,P,A,x,scratch,β,y)` and
# `vcreate(P,A,x,scratch)` methods for a scaled mapping.
for (P, expr) in ((:Adjoint, :(α*conj(multiplier(A)))),
                  (:Inverse, :(α/multiplier(A))),
                  (:InverseAdjoint, :(α/conj(multiplier(A)))))
    @eval begin # FIXME:

        apply!(α::Number, A::$P{<:Scaled}, x, scratch::Bool, β::Number, y) =
            apply!($expr, unscaled(unveil(A)), x, scratch, β, y)

    end
end

"""
    overwritable(scratch, x, y) -> bool

yields whether the result `y` of applying a mapping to `x` with scratch flag
`scratch` can overwritten. Arguments `x` and `y` can be reversed.

"""
overwritable(scratch::Bool, x, y) = (scratch || x !== y)

# Implement `apply` for scaled operators to avoid the needs of explicitly
# calling `vcreate` as done by the default implementation of `apply`.  This is
# needed for scaled compositions among others.
function apply(A::Scaled, x, scratch::Bool)
    y = apply(Operator, unscaled(A), x, scratch)
    vscale!((overwritable(scratch, x, y) ? y : vcopy(y)), multiplier(A))
end

function apply(::Type{Adjoint}, A::Scaled, x, scratch::Bool)
    y = apply(Operator, unscaled(A), x, scratch)
    vscale!((overwritable(scratch, x, y) ? y : vcopy(y)), conj(multiplier(A)))
end

function apply(::Type{Inverse}, A::Scaled, x, scratch::Bool)
    y = apply(Operator, unscaled(A), x, scratch)
    vscale!((overwritable(scratch, x, y) ? y : vcopy(y)), 1/multiplier(A))
end

function apply(::Type{InverseAdjoint}, A::Scaled, x, scratch::Bool)
    y = apply(Operator, unscaled(A), x, scratch)
    vscale!((overwritable(scratch, x, y) ? y : vcopy(y)), 1/conj(multiplier(A)))
end

vcreate(P::Type{<:Operations}, A::Scaled, x, scratch::Bool) =
    vcreate(P, unscaled(A), x, scratch)

# Implemention of the `vcreate(P,A,x,scratch)` and
# `apply!(α,P,A,x,scratch,β,y)` methods for the various decorations of a
# mapping so as to automatically unveil the embedded mapping.
for (T1, T2, T3) in ((:Inverse,        :Adjoint,        :InverseAdjoint),
                     (:InverseAdjoint, :Adjoint,        :Inverse),
                     (:Operator,       :Inverse,        :Inverse),
                     (:Adjoint,        :Inverse,        :InverseAdjoint),
                     (:InverseAdjoint, :Inverse,        :Adjoint),
                     (:Adjoint,        :InverseAdjoint, :Inverse),
                     (:Inverse,        :InverseAdjoint, :Adjoint))
    @eval begin

        vcreate(::Type{$T1}, A::$T2, x, scratch::Bool) =
            vcreate($T3, unveil(A), x, scratch)

        apply!(α::Number, ::Type{$T1}, A::$T2, x, scratch::Bool, β::Number, y) =
            apply!(α, $T3, unveil(A), x, scratch, β, y)

    end
end

function apply!(α::Number, P::Type{<:Union{Operator,Adjoint}}, A::Sum{N},
                x, scratch::Bool, β::Number, y) where {N}
    if α == 0
        # Just scale the destination.
        vscale!(y, β)
    else
        # Apply first mapping with β and then other with β=1.  Scratch flag is
        # always false until last mapping because we must preserve x as there
        # is more than one term.
        apply!(α, P, A[1], x, false, β, y)
        for i in 2:N
            apply!(α, P, A[i], x, (scratch && i == N), 1, y)
        end
    end
    return y
end

vcreate(::Type{<:Union{Inverse,InverseAdjoint}}, A::Sum, x, scratch::Bool) =
    throw_unsupported_inverse_of_sum()

apply(::Type{<:Union{Inverse,InverseAdjoint}}, A::Sum, x, scratch::Bool) =
    throw_unsupported_inverse_of_sum()

function apply!(α::Number, ::Type{<:Union{Inverse,InverseAdjoint}}, A::Sum,
                x, scratch::Bool, β::Number, y)
    throw_unsupported_inverse_of_sum()
end

throw_unsupported_inverse_of_sum() =
    error("automatic dispatching of the inverse of a sum of mappings is not supported")

# Implementation of the `apply!(α,P,A,x,scratch,β,y)` method for a composition
# of mappings.  There is no possible `vcreate(P,A,x,scratch)` method for a
# composition so we directly extend the `apply(P,A,x,scratch)` method.  Note
# that `Composition` instances are warranted to have at least 2 components.
#
# The unrolled code (taking care of allowing as few temporaries as possible and
# for the Direct or InverseAdjoint operation) writes:
#
#     w1 = apply(P, A[N], x, scratch)
#     scratch = overwritable(scratch, x, w1)
#     w2 = apply!(1, P, A[N-1], w1, scratch)
#     scratch = overwritable(scratch, w1, w2)
#     w3 = apply!(1, P, A[N-2], w2, scratch)
#     scratch = overwritable(scratch, w2, w3)
#     ...
#     return apply!(α, P, A[1], wNm1, scratch, β, y)
#
# To break the type barrier, this is done by a recursion.  The recursion is
# just done in the other direction for the Adjoint or Inverse operation.

function vcreate(::Type{<:Operations},
                 A::Composition{N}, x, scratch::Bool) where {N}
    error("it is not possible to create the output of a composition of mappings")
end

function apply!(α::Number, ::Type{P}, A::Composition{N}, x, scratch::Bool,
                β::Number, y) where {N,P<:Union{Operator,InverseAdjoint}}
    if α == 0
        # Just scale the destination.
        vscale!(y, β)
    else
        ops = terms(A)
        w = apply(P, *, ops[2:N], x, scratch)
        scratch = overwritable(scratch, w, x)
        apply!(α, P, ops[1], w, scratch, β, y)
    end
    return y
end

function apply(::Type{P}, A::Composition{N}, x,
               scratch::Bool) where {N,P<:Union{Operator,InverseAdjoint}}
    apply(P, *, terms(A), x, scratch)
end

function apply(::Type{P}, ::typeof(*), ops::NTuple{N,Operator}, x,
               scratch::Bool) where {N,P<:Union{Operator,InverseAdjoint}}
    w = apply(P, ops[N], x, scratch)
    N == 1 && return w
    scratch = overwritable(scratch, w, x)
    apply(P, *, ops[1:N-1], w, scratch)
end

function apply!(α::Number, ::Type{P}, A::Composition{N}, x, scratch::Bool,
                β::Number, y) where {N,P<:Union{Adjoint,Inverse}}
    if α == 0
        # Just scale the destination.
        vscale!(y, β)
    else
        ops = terms(A)
        w = apply(P, *, ops[1:N-1], x, scratch)
        scratch = overwritable(scratch, w, x)
        apply!(α, P, ops[N], w, scratch, β, y)
    end
    return y
end

function apply(::Type{P}, A::Composition{N}, x,
               scratch::Bool) where {N,P<:Union{Adjoint,Inverse}}
    apply(P, *, terms(A), x, scratch)
end

function apply(::Type{P}, ::typeof(*), ops::NTuple{N,Operator}, x,
               scratch::Bool) where {N,P<:Union{Adjoint,Inverse}}
    w = apply(P, ops[1], x, scratch)
    N == 1 && return w
    scratch = overwritable(scratch, w, x)
    apply(P, *, ops[2:N], w, scratch)
end

# Default rules to apply a Gram operator.  Gram matrices are Hermitian by
# construction which left only 2 cases to deal with.

apply!(α::Number, ::Type{Adjoint}, A::Gram, x, scratch::Bool, β::Number, y) =
    apply!(α, Operator, A, x, scratch, β, y)

apply!(α::Number, ::Type{InverseAdjoint}, A::Gram, x, scratch::Bool, β::Number, y) =
    apply!(α, Inverse, A, x, scratch, β, y)

function apply!(α::Number, ::Type{Operator}, A::Gram, x, scratch::Bool, β::Number, y)
    if α == 0
        vscale!(y, β)
    else
        B = unveil(A) # A ≡ B'*B
        z = apply(Operator, B, x, scratch) # z <- B⋅x
        apply!(α, Adjoint, B, z, (z !== x), β, y) # y <- α⋅B'⋅z + β⋅y
    end
    return y
end

function apply!(α::Number, ::Type{Inverse}, A::Gram, x, scratch::Bool, β::Number, y)
    if α == 0
        vscale!(y, β)
    else
        B = unveil(A) # A ≡ B'⋅B
        # Compute α⋅inv(A)⋅x + β⋅y = α⋅inv(B'⋅B)⋅x + β⋅y
        #                          = α⋅inv(B)⋅inv(B')⋅x + β⋅y
        z = apply(InverseAdjoint, B, x, scratch) # z <- inv(B')⋅x
        apply!(α, Inverse, B, z, (z !== x), β, y) # y <- α⋅inv(B)⋅z + β⋅y
    end
    return y
end

# A Gram operator is self-adjoint by construction and yields result of same
# kind as input.
vcreate(::Type{<:Operations}, ::Gram, x, scratch::Bool) = vcreate(x)
