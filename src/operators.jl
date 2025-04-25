#
# operators.jl -
#
# Implement non-specific methods for operators.
#
#-----------------------------------------------------------------------------------------

# Convert constructor.
Operator(A::Operator) = A
Operator(A::LinearAlgebra.UniformScaling) = multiplier(A) * Id

Base.convert(::Type{Operator}, A::Operator) = A
Base.convert(::Type{Operator}, A) = Operator(A)

# Some traits need to be transposed.
Base.transpose(trait::InputShapeUnknown) = OutputShapeUnknown()
Base.transpose(trait::OutputShapeUnknown) = InputShapeUnknown()
Base.transpose(trait::HasInputShape{N}) where {N} = HasOutputShape{N}()
Base.transpose(trait::HasOutputShape{N}) where {N} = HasInputShape{N}()
Base.transpose(trait::InputEltypeUnknown) = OutputEltypeUnknown()
Base.transpose(trait::OutputEltypeUnknown) = InputEltypeUnknown()
Base.transpose(trait::HasInputEltype) = HasOutputEltype()
Base.transpose(trait::HasOutputEltype) = HasInputEltype()

"""
    LazyAlgebra.InputShape(typeof(A))

given the type of an operator `A`, yields one of:

* `LazyAlgebra.InputShapeUnknown()` if the shape of the input of `A` cannot be determined
  in advance. This is the assumed default.

* `LazyAlgebra.HasInputShape{N}()` if the input of `A` has a known `N`-dimensional shape
  whose axes and size are respectively given by `LazyAlgebra.input_axes(A)` and
  `LazyAlgebra.input_size(A)`.

See also [`LazyAlgebra.InputEltype](@ref) and [`LazyAlgebra.OutputShape](@ref).

"""
InputShape(A) = InputShape(typeof(A))
InputShape(::Type) = InputShapeUnknown()
InputShape(::Type{A}) where {A<:Union{Adjoint,Inverse}} =
    transpose(OutputShape(parent(A)))

"""
    LazyAlgebra.OutputShape(typeof(A))

given the type of an operator `A`, yields one of:

* `LazyAlgebra.OutputShapeUnknown()` if the shape of the output of `A` cannot be
  determined in advance.

* `LazyAlgebra.HasOutputShape{N}()` if the output of `A` has a known `N`-dimensional shape
  whose axes and size are respectively given by `LazyAlgebra.output_axes(A)` and
  `LazyAlgebra.output_size(A)`.

!!! note
    In any case, the output shape of `A*x` can be determined by
    `LazyAlgebra.output_axes(A,x)`.

See also [`LazyAlgebra.OutputEltype](@ref), [`LazyAlgebra.InputShape](@ref), and
[`LazyAlgebra.output_axes](@ref).

"""
OutputShape(A) = OutputShape(typeof(A))
OutputShape(::Type) = OutputShapeUnknown()
OutputShape(::Type{A}) where {A<:Union{Adjoint,Inverse}} =
    transpose(InputShape(parent(A)))

"""
    LazyAlgebra.InputEltype(typeof(A))

given the type of an operator `A`, yields one of:

* `LazyAlgebra.InputEltypeUnknown()` if the element type of the input of `A` cannot be
  determined in advance. This is the assumed default.

* `LazyAlgebra.HasInputEltype()` if the element type of the input of `A` is known and
  given by `LazyAlgebra.input_eltype(typeof(A))`.

See also [`LazyAlgebra.InputShape](@ref), [`LazyAlgebra.OutputEltype](@ref), and
[`LazyAlgebra.output_eltype](@ref).

"""
InputEltype(A) = InputEltype(typeof(A))
InputEltype(::Type) = InputEltypeUnknown()
InputEltype(::Type{A}) where {A<:Union{Adjoint,Inverse}} =
    transpose(OutputEltype(parent(A)))

"""
    LazyAlgebra.OutputEltype(typeof(A))

given the type of an operator `A`, yields one of:

* `LazyAlgebra.OutputEltypeUnknown()` if the element type of the output of `A` cannot be
  determined in advance. This is the assumed default.

* `LazyAlgebra.HasOutputEltype()` if the element type of the output of `A` is known and
  given by `LazyAlgebra.output_eltype(typeof(A))`.

!!! note
    In any case, the output element type of `A*x` and `α*A*xcan be determined by
    `LazyAlgebra.output_eltype(A, x)` and `LazyAlgebra.output_eltype(α, A, x)`.

See also [`LazyAlgebra.OutputShape](@ref), [`LazyAlgebra.InputEltype](@ref), and
[`LazyAlgebra.output_eltype](@ref).

"""
OutputEltype(A) = OutputEltype(typeof(A))
OutputEltype(::Type) = OutputEltypeUnknown()
OutputEltype(::Type{A}) where {A<:InverseAdjoint} =
    transpose(InputEltype(parent(A)))

"""
    LazyAlgebra.input_eltype(A) -> T
    LazyAlgebra.input_eltype(typeof(A)) -> T

yields the element type `T` of `x` for computing `A*x` with operator `A`. Not all operators
implement this trait.

To implement this trait for an operator, the following two methods shall be specialized:

```julia
LazyAlgebra.InputEltype(typeof(A)) = LazyAlgebra.HasInputEltype()
LazyAlgebra.input_eltype(typeof(A)) = ...
```

If this trait is implemented, argument `x` with a different element type is automatically
converted by [`vmul`](@ref) and [`vmul!`](@ref). As consequence, consider carefully
whether this is advisable or not. In general, this is only needed if the operator is
implemented by an external library which imposes the element type.

!!! warning
    This function shall only be called if `LazyAlgebra.InputEltype(typeof(A))`
    yields `LazyAlgebra.HasInputEltype()`.

See also [`vmul`](@ref), [`vmul!`](@ref), [`LazyAlgebra.InputEltype`](@ref) and
[`LazyAlgebra.output_eltype`](@ref).

"""
input_eltype(A) = input_eltype(typeof(A))
input_eltype(::Type{A}) where {A<:Union{Adjoint,Inverse}} = output_eltype(parent(A))
@noinline input_eltype(::Type{T}) where {T} =
    error("`LazyAlgebra.input_eltype(T)` not defined for objects of type `T=$T`")

"""
    LazyAlgebra.output_eltype(A) -> T
    LazyAlgebra.output_eltype(typeof(A)) -> T

yields the element type `T` of the result of `A*x` for operator `A` and for any acceptable
`x`. Not all operators implement this trait.

To implement this trait for an operator, the following two methods shall be specialized:

```julia
LazyAlgebra.OutputEltype(typeof(A)) = LazyAlgebra.HasOutputEltype()
LazyAlgebra.output_eltype(typeof(A)) = ...
```

!!! warning
    Do not confuse this `LazyAlgebra.output_eltype` method which takes a single argument
    with the one that takes 2 or 3 arguments. The single argument method shall only be
    called if `LazyAlgebra.OutputEltype(typeof(A))` yields
    `LazyAlgebra.HasOutputEltype()`.

See also [`vmul`](@ref), [`vmul!`](@ref), [`LazyAlgebra.OutputEltype`](@ref) and
[`LazyAlgebra.output_eltype`](@ref).

"""
output_eltype(A) = output_eltype(typeof(A))
output_eltype(::Type{A}) where {A<:Union{Adjoint,Inverse}} = input_eltype(parent(A))
@noinline output_eltype(::Type{T}) where {T} =
    error("`LazyAlgebra.output_eltype(T)` not defined for objects of type `T=$T`")

"""
    LazyAlgebra.output_eltype([α::Number,] A::Operator, x::AbstractArray) -> T
    LazyAlgebra.output_eltype([typeof(α),] typeof(A), typeof(x)) -> T

yield the element type `T` of the result of `A*x` or of `α*A*x` if the multiplier `α` is
specified.

As a simplification, it is assumed that the element type of `A*x` is a *trait* that only
depends on the type of the operator `A` and on the type of the input array `x`. Following
this assumption, this method infers its result from that of:

    LazyAlgebra.output_eltype(typeof(A), typeof(x))

and it is thus expected that a method with this signature exists for the operator `A` and
that it returns the element type of `A*x`. If such a method does not exists but
[`LazyAlgebra.OutputEltype(typeof(A))`](@ref LazyAlgebra.OutputEltype) yields
`LazyAlgebra.HasOutputEltype()`, then `T` is given by:

    T = float(LazyAlgebra.output_eltype(typeof(A)))

otherwise

    Base.eltype(typeof(A))

is called to infer the type of the coefficients of `A` and which assumes that the element
type of `A*x` is that of the floating-point conversion of the multiplication of two values
of respective types `eltype(typeof(A))` and `eltype(x)`.

See also [`LazyAlgebra.output_axes`](@ref), [`LazyAlgebra.create_output`](@ref), and
[`LazyAlgebra.multiplier_type`](@ref).

"""
output_eltype(α::Number, A::Operator, x::AbstractArray) =
    output_eltype(typeof(α), typeof(A), typeof(x))

output_eltype(A::Operator, x::AbstractArray) =
    output_eltype(typeof(A), typeof(x))

function output_eltype(::Type{α}, ::Type{A}, ::Type{x}) where {α<:Number, A<:Operator,
                                                               x<:AbstractArray}
    T = output_eltype(A, x) # element type of A*x
    return prod_type(multiplier_type(α, T), T)
end

# Fallback method, assumes that one of `output_eltype(A)` or `eltype(A)` is applicable.
output_eltype(::Type{A}, ::Type{x}) where {A<:Operator, x<:AbstractArray} =
    OutputEltype(A) === HasOutputEltype() ? float(output_eltype(A)) :
    float(prod_type(eltype(A), eltype(x)))

# Extend `Base.eltype` for operators and their variants. NOTE This is not necessary for
# `Sum` and `Prod` as they implement `output_eltype` properly.
Base.eltype(A::Operator) = eltype(typeof(A))
Base.eltype(::Type{<:Adjoint{A}}) where {A} = eltype(A)
Base.eltype(::Type{<:Inverse{A}}) where {A} = float(eltype(A))
Base.eltype(::Type{<:InverseAdjoint{A}}) where {A} = float(eltype(A))

# Output element type for products and sums assuming right-associativity.
output_eltype(::Type{Prod{L,R}}, ::Type{x}) where {L,R,x<:AbstractArray} =
    output_eltype(L, output_eltype(R, x))

output_eltype(::Type{Sum{L,R}}, ::Type{x}) where {L,R,x<:AbstractArray} =
    sum_type(output_eltype(L, x), output_eltype(R, x))

"""
    LazyAlgebra.output_ndims(A)

yields the number dimensions of the result of `A*x` based on the type of `A`.

!!! note
    If the number `N` of dimensions of `A*x` is known in advance, do not extend this
    method but rather extend `LazyAlgebra.OutputShape(typeof(A))` to yield
    `LazyAlgebra.HasOutputShape{N}()`.

See also [`LazyAlgebra.input_ndims`](@ref), [`LazyAlgebra.output_axes`](@ref),
[`LazyAlgebra.output_eltype`](@ref), and [`LazyAlgebra.OutputShape`](@ref).

"""
output_ndims(A) = _output_ndims(OutputShape(A))
_output_ndims(::HasOutputShape{N}) where {N} = N
@noinline _output_ndims(::OutputShapeUnknown) =
    error("`LazyAlgebra.output_ndims` not defined for objects of this type")

"""
    LazyAlgebra.input_ndims(A)

yields the number dimensions of the input `x` for `A*x` based on the type of `A`.

!!! note
    If the number `M` of dimensions of `A*x` is known in advance, do not extend this
    method but rather extend `LazyAlgebra.OutputShape(typeof(A))` to yield
    `LazyAlgebra.HasOutputShape{M}()`.

See also [`LazyAlgebra.output_ndims`](@ref), [`LazyAlgebra.input_axes`](@ref),
[`LazyAlgebra.input_eltype`](@ref), and [`LazyAlgebra.InputShape`](@ref).

"""
input_ndims(A) = _input_ndims(InputShape(A))
_input_ndims(::HasInputShape{N}) where {N} = N
@noinline _input_ndims(::InputShapeUnknown) =
    error("`LazyAlgebra.input_ndims` not defined for objects of this type")

"""
    LazyAlgebra.output_axes(A::Operator, x::AbstractArray)

yields the axes of the result of `A*x`.

As a simplification, it is assumed that the axes of `A*x` only depend on the operator `A`
and on the axes of the input array `x`. Following this assumption, this method returns the
result of:

    LazyAlgebra.output_axes(A, axes(x))

and it is thus expected that a method with this signature exists for the operator `A`.

If the axes of the input and output of `A` do not depend on the input `x`, an alternative
is to implement:

    LazyAlgebra.output_axes(A)
    LazyAlgebra.input_axes(A)

to respectively yield the the axes of the output and input of `A` when these axes do not
depend on `x`. In that case, the following traits shall be implemented:

```julia
LazyAlgebra.OutputShape(typeof(A)) = LazyAlgebra.HasOutputShape{M}()
LazyAlgebra.InputShape(typeof(A)) = LazyAlgebra.HasInputShape{N}()
```

with `M` and `N` the number of dimensions of the output and input of `A`.

See also [`LazyAlgebra.output_eltype`](@ref), [`LazyAlgebra.create_output`](@ref),
[`LazyAlgebra.input_axes`](@ref), [`LazyAlgebra.input_eltype`](@ref),
[`LazyAlgebra.OutputShape`](@ref), and [`LazyAlgebra.InputShape`](@ref).

"""
output_axes(A::Operator, x::AbstractArray) = output_axes(A, axes(x))
output_axes(A::InverseAdjoint, x::AbstractArray) = output_axes(parent(parent(A)), axes(x))

# Fallback version of `output_axes(A, x)` assuming `input_axes(A)` and `input_axes(A)` are
# defined for `A`.
function output_axes(A::Operator, x_axes::ArrayAxes)
    InputShape(A) isa HasInputShape || throw_input_shape_not_implemented(typeof(A))
    check_input_axes(x_axes, input_axes(A))
    OutputShape(A) isa HasOutputShape || throw_output_shape_not_implemented(typeof(A))
    return output_axes(A)
end

@noinline output_axes(A::Operator) = throw_output_shape_not_implemented(typeof(A))

@noinline throw_input_shape_not_implemented(::Type{T}) where {T} =
    error(string("checking the dimension of the input of an operator of type `", T,
                 "` is not correctly implemented, see doc. of `LazyAlgebra.output_axes`"))

@noinline throw_output_shape_not_implemented(::Type{T}) where {T} =
    error(string("inferring the dimension of the output of an operator of type `", T,
                 "` is not correctly implemented, see doc. of `LazyAlgebra.output_axes`"))

"""
    LazyAlgebra.input_axes(A::Operator)

yields the axes that `x` must have to compute `A*x`. Not all operators `A` implement this.

To implement this method for an operator, the following two methods shall be specialized:

```julia
LazyAlgebra.InputShape(typeof(A)) = LazyAlgebra.HasInputShape{N}()
LazyAlgebra.input_axes(A) = ...
```

with `N` the number of dimensions of input `x` to compute `A*x`.

See also [`LazyAlgebra.output_axes`](@ref) and [`LazyAlgebra.InputShape`](@ref).

"""
@noinline input_axes(A::Operator) =
    error("`LazyAlgebra.input_axes(A)` not defined for operator `A` of type `$(typeof(A))`")

output_axes(A::Union{Adjoint,Inverse}) =  input_axes(A[])
input_axes( A::Union{Adjoint,Inverse}) = output_axes(A[])

output_axes(A::InverseAdjoint, J::ArrayAxes) = output_axes(A[][], J)
output_axes(A::InverseAdjoint) = output_axes(A[][])
input_axes( A::InverseAdjoint) = input_axes(A[][])

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

@noinline throw_incompatible_output_axes_in_sum() =
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

Base.show(io::IO, ::MIME"text/plain", A::Operator) = show(io, A)

function Base.show(io::IO, A::Operator)
    show(io, typeof(A))
end

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

function Base.show(io::IO, A::Prod)
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

# Show a multiplier (surrounded by parentheses if not a real, i.e. if a complex).
show_multiplier(io::IO, λ::Number) = show_paren(io, λ, !isreal(λ))

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
        if isreal(λ) && λ < zero(λ)
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
    LazyAlgebra.unscaled(A)

yields the operator `B` of the *scaled operator* `A = λ*B`; otherwise yields `A`. This
method is also applicable to instances of `LinearAlgebra.UniformScaling`. Call
[`LazyAlgebra.multiplier`](@ref) to get the multiplier `λ`.

"""
unscaled(A::Operator) = A
unscaled(A::Prod{<:Number}) = A[2]
unscaled(A::UniformScaling) = Id

"""
    LazyAlgebra.multiplier(A)

yields the multiplier `λ` of the *scaled operator* `A = λ*B`; otherwise yields `1`. This
method is also applicable to instances of `LinearAlgebra.UniformScaling`. Call
[`LazyAlgebra.unscaled`](@ref) to get the operator `B`.

"""
multiplier(A::Prod{<:Number}) = A[2]
multiplier(A::Operator) = 1
multiplier(A::UniformScaling) = getfield(A, :λ)

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

See also [`vmul!`](@ref), [`LazyAlgebra.Operator`](@ref), and
[`LazyAlgebra.unsafe_vmul!`](@ref).

"""
function vmul(A::Operator, x::AbstractArray)
    y = create_output(A, x)
    T = floating_point_type(eltype(y))
    dispatch_vmul!(one(T), A, x, zero(T), y)
end

function vmul(α::Number, A::Operator, x::AbstractArray)
    α = convert_multiplier(α, A, x)
    y = create_output(α, A, x)
    T = floating_point_type(eltype(y))
    dispatch_vmul!(α, A, x, zero(T), y)
end

Base.:(*)(A::Operator, x::AbstractArray) = vmul(A, x)
Base.:(\)(A::Operator, x::AbstractArray) = vmul(inv(A), x)

"""
    vmul!(α::Number, A::Operator, x::AbstractArray, β::Number, y::AbstractArray) -> y

overwrites `y` with `α*A⋅x + β*y` and returns `y`. The convention is that the prior
contents of `y` is not used at all if `iszero(β)` holds so `y` can be directly used to
store the result even though it is not initialized.

Another supported syntax is:

    vmul!(y::AbstractArray, [α::Number=1], A::Operator, x::AbstractArray) -> y

which overwrites `y` with `α*A*x` and returns `y`, this is a shortcut for:

    vmul!(α, A, x, 0, y)

The `vmul!` method can be seen as a generalization of the `LinearAlgebra.mul!` method.

!!! warning
    Do not extend this method for specific operator types, but rather the
    [`LazyAlgebra.unsafe_vmul!`](@ref) method.

See also [`vmul`](@ref), [`LazyAlgebra.Operator`](@ref),
[`LazyAlgebra.dispatch_vmul!`](@ref), and [`LazyAlgebra.unsafe_vmul!`](@ref).

"""
vmul!(y::AbstractArray, A::Operator, x::AbstractArray) =
    vmul!(1, A, x, 0, y)

vmul!(y::AbstractArray, α::Number, A::Operator, x::AbstractArray) =
    vmul!(α, A, x, 0, y)

function vmul!(α::Number, A::Operator, x::AbstractArray, β::Number, y::AbstractArray)
    check_output_axes(y, output_axes(A, x))
    dispatch_vmul!(convert_multiplier(α, A, x), A, x,
                   convert_multiplier(β, y), y)
end

"""
    LazyAlgebra.dispatch_vmul!(α, A, x, β, y) -> y

overwrites `y` with `α*A⋅x + β*y` and returns `y`.

If `iszero(α)` does not hold, this method calls [`LazyAlgebra.unsafe_vmul!(α, A, x, β,
y)`](@ref LazyAlgebra.unsafe_vmul!); otherwise, if `iszero(β)` does not hold, this method
calls [`LazyAlgebra.unsafe_vscale!(β, y)`](@ref LazyAlgebra.unsafe_vscale!)this method
calls [`vzero!(y)`](@ref LazyAlgebra.vzero!).

!!! warning
    This method assumes that the axes of `x` and `y` have been checked to be correct as
    the respective input and output for `A` and that the multipliers `α` and `β` have been
    converted to suitable floating-point type.

See also [`vmul`](@ref), [`vmul!`](@ref), and [`LazyAlgebra.unsafe_vmul!`](@ref).

"""
function dispatch_vmul!(α::Number, A::Operator, x::AbstractArray, β::Number, y::AbstractArray)
    if !iszero(α)
        unsafe_vmul!(α, A, x, β, y)
    elseif !iszero(β)
        unsafe_vscale!(y, β)
    else
        vzero!(y)
    end
    return y
end

"""
    LazyAlgebra.check_input_axes(x, inp_axes) -> nothing
    LazyAlgebra.check_input_axes(axes(x), inp_axes) -> nothing

throw a `DimensionMismatch` exception if the axes of the input array `x` are not equal to
the given `inp_axes`.

See also [`vmul`](@ref), [`vmul!`](@ref), and [`LazyAlgebra.check_output_axes`](@ref).

"""
check_input_axes(x::AbstractArray, inp_axes::ArrayAxes) = check_input_axes(axes(x), inp_axes)
check_input_axes(x_axes::ArrayAxes, inp_axes::ArrayAxes) =
    x_axes == inp_axes ? nothing : throw_incompatible_input_axes(x_axes, inp_axes)

@noinline throw_incompatible_input_axes(x_axes::ArrayAxes, inp_axes::ArrayAxes) =
    throw(DimensionMismatch(incompatible_axes("input array", x_axes, inp_axes)))

"""
    LazyAlgebra.check_output_axes(y, out_axes) -> nothing
    LazyAlgebra.check_output_axes(axes(y), out_axes) -> nothing

throw a `DimensionMismatch` exception if the axes of the output array `y` are not equal to
the given `out_axes`.

See also [`vmul`](@ref), [`vmul!`](@ref), and [`LazyAlgebra.check_input_axes`](@ref).

"""
check_output_axes(y::AbstractArray, out_axes::ArrayAxes) = check_output_axes(axes(y), out_axes)
check_output_axes(y_axes::ArrayAxes, out_axes::ArrayAxes) =
    y_axes == out_axes ? nothing : throw_incompatible_axes("output array", y_axes, out_axes)

@noinline throw_incompatible_output_axes(y_axes::ArrayAxes, out_axes::ArrayAxes) =
    throw(DimensionMismatch(incompatible_axes("output array", y_axes, out_axes)))

function incompatible_axes(arg_name::AbstractString, arg_axes::ArrayAxes, ref_axes::ArrayAxes)
    io = IOBuffer()
    write(io, "axes of ")
    print(io, arg_name)
    write(io, " should be ")
    print_axes(io, arg_axes)
    write(io, ", got ")
    print_axes(io, ref_axes)
    return String(take!(io))
end

function axes_to_string(rngs::Tuple{Vararg{AbstractUnitRange{<:Integer}}})
    io = IOBuffer()
    print_axes(io, rngs)
    return String(take!(io))
end

print_axis(io::IO, dim::Integer) =
    print(io, "1:", max(0, Int(dim)))

print_axis(io::IO, rng::AbstractUnitRange{<:Integer}) =
    print(io, first(rng), ':', last(rng))

function print_axes(io::IO, rngs::Tuple{Vararg{AbstractUnitRange{<:Integer}}})
    write(io, '(')
    for (i, rng) in enumerate(rngs)
        i > 1 && write(io, ", ")
        print_axis(io, rng)
    end
    length(rngs) == 1 && write(io, ',')
    write(io, ')')
    nothing
end

function print_shape(io::IO, shape::ArrayShape)
    if shape isa Tuple{Vararg{Union{Integer,Base.OneTo}}}
        show(io, as_array_size(shape))
    else
        print_axes(io, shape)
    end
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
