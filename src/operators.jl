#
# operators.jl -
#
# Implement non-specific methods for operators.
#
#-----------------------------------------------------------------------------------------

# Conversion constructors.
Operator(A::Operator) = A
Operator(A::LinearAlgebra.UniformScaling) = multiplier(A) * Id
Operator(A::AbstractMatrix) = PseudoMatrix(A, Val(1))

Base.convert(::Type{Operator}, A::Operator) = A
Base.convert(::Type{Operator}, A) = Operator(A)

# Rules to automatically convert `LinearAlgebra.UniformScaling` into `λ*Id` and abstract
# matrix into `PseudoMatrix` when combined with any `LazyAlgebra` operator or when
# specific constructors are applied.
let NonMatrix = LinearAlgebra.UniformScaling, Other = Union{NonMatrix,AbstractMatrix}
    for op in (:(*), :(∘), :(/), Symbol("\\"))
        # For compositions, the left-hand operand must not be an array otherwise this
        # contradicts the rule that `A*x` calls `vmul`.
        @eval begin
            Base.$op(A::$Other, B::Operator) = $op(Operator(A), B)
            Base.$op(A::Operator, B::$NonMatrix) = $op(A, Operator(B))
        end
    end
    for op in (:(+), :(-))
        @eval begin
            Base.$op(A::$Other, B::Operator) = $op(Operator(A), B)
            Base.$op(A::Operator, B::$Other) = $op(A, Operator(B))
        end
    end
    @eval begin
        Sum(A::$Other,   B::$Other  ) = Sum(Operator(A), Operator(B))
        Sum(A::$Other,   B::Operator) = Sum(Operator(A), B)
        Sum(A::Operator, B::$Other  ) = Sum(A, Operator(B))

        Prod(A::$Other,  B::$Other ) = Prod(Operator(A), Operator(B))
        Prod(A::$Other,  B::Operand) = Prod(Operator(A), B)
        Prod(A::Operand, B::$Other ) = Prod(A, Operator(B))
    end
    for constructor in (:Adjoint, :Inverse, :Gram)
        @eval $constructor(A::$Other) = $constructor(Operator(A))
    end
end

# Some traits need to be transposed.
Base.transpose(trait::InputShapeUnknown) = OutputShapeUnknown()
Base.transpose(trait::OutputShapeUnknown) = InputShapeUnknown()
Base.transpose(trait::HasInputShape{N}) where {N} = HasOutputShape{N}()
Base.transpose(trait::HasOutputShape{N}) where {N} = HasInputShape{N}()
Base.transpose(trait::InputEltypeUnknown) = OutputEltypeUnknown()
Base.transpose(trait::OutputEltypeUnknown) = InputEltypeUnknown()
Base.transpose(trait::HasInputEltype) = HasOutputEltype()
Base.transpose(trait::HasOutputEltype) = HasInputEltype()

Base.ndims(x::OutputShape) = ndims(typeof(x))
Base.ndims(::HasOutputShape{N}) where {N} = N
@noinline Base.ndims(::OutputShapeUnknown) =
    throw_argument_error("unknown number of output dimensions")

Base.ndims(x::InputShape) = ndims(typeof(x))
Base.ndims(::HasInputShape{N}) where {N} = N
@noinline Base.ndims(::InputShapeUnknown) =
    throw_argument_error("unknown number of input dimensions")

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
InputShape(::Type{A}) where {A<:AbstractMatrix} = HasInputShape{1}()
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
OutputShape(::Type{A}) where {A<:AbstractMatrix} = HasOutputShape{1}()
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
    In any case, the output element type of `A*x` and `α*A*x` can be determined by
    `LazyAlgebra.output_eltype(A, x)` and `LazyAlgebra.output_eltype(α, A, x)`.

See also [`LazyAlgebra.OutputShape](@ref), [`LazyAlgebra.InputEltype](@ref), and
[`LazyAlgebra.output_eltype](@ref).

"""
OutputEltype(A) = OutputEltype(typeof(A))
OutputEltype(::Type) = OutputEltypeUnknown()
OutputEltype(::Type{A}) where {A<:Union{Adjoint,Inverse}} =
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

output_eltype(::Type{α}, ::Type{A}, ::Type{x}) where {α<:Number, A<:Operator, x<:AbstractArray} =
    output_eltype(α, AbstractArray{output_eltype(A, x)})

# Output element type for scaling a vector. NOTE This should be the same for `vscale`.
output_eltype(α::Number, x::AbstractArray) = output_eltype(typeof(α), typeof(x))
output_eltype(::Type{α}, ::Type{x}) where {α<:Number, x<:AbstractArray} =
    prod_type(multiplier_type(α, x), eltype(x))

# Fallback method, assumes that one of `output_eltype(A)` or `eltype(A)` is applicable.
output_eltype(::Type{A}, ::Type{x}) where {A<:Operator, x<:AbstractArray} =
    OutputEltype(A) === HasOutputEltype() ? float(output_eltype(A)) :
    float(sumprod_type(eltype(A), eltype(x)))

# Extend `Base.eltype` for operators and their variants. NOTE This is not necessary for
# `Sum` and `Prod` as they implement `output_eltype` properly.
Base.eltype(A::Operator) = eltype(typeof(A))
Base.eltype(::Type{<:Adjoint{A}}) where {A} = eltype(A)
Base.eltype(::Type{<:Inverse{A}}) where {A} = float(eltype(A))
Base.eltype(::Type{<:InverseAdjoint{A}}) where {A} = float(eltype(A))
Base.eltype(::Type{<:Prod{A,B}}) where {A,B} = prod_type(eltype(A), eltype(B))
Base.eltype(::Type{<:Sum{A,B}}) where {A,B} = sum_type(eltype(A), eltype(B))
@noinline Base.eltype(::Type{T}) where {T<:Operator} =
    error("`eltype` trait not implemented for operators of type `$T`")

# Output element type for sums and products assuming right-associativity.
output_eltype(::Type{Sum{L,R}}, ::Type{x}) where {L,R,x<:AbstractArray} =
    sum_type(output_eltype(L, x), output_eltype(R, x))

output_eltype(::Type{Prod{L,R}}, ::Type{x}) where {L,R,x<:AbstractArray} =
    output_eltype(L, AbstractArray{output_eltype(R, x)})

# Output element type for scaled operators.
output_eltype(::Type{Prod{L,R}}, ::Type{x}) where {L<:Number,R,x<:AbstractArray} =
    output_eltype(L, R, x)

"""
    LazyAlgebra.output_ndims(A)
    LazyAlgebra.output_ndims(typeof(A))

yield the number dimensions of the result of `A*x` based on the type of `A`.

!!! note
    If the number `M` of dimensions of `A*x` is known in advance, do not extend this
    method but rather extend `LazyAlgebra.OutputShape(typeof(A))`,
    `LazyAlgebra.output_axes(A)`, and optionally `LazyAlgebra.output_size(A)` to
    respectively yield `LazyAlgebra.HasOutputShape{M}()`, the axes and the size of `A*x`.

See also [`LazyAlgebra.input_ndims`](@ref), [`LazyAlgebra.output_axes`](@ref),
[`LazyAlgebra.output_eltype`](@ref), [`LazyAlgebra.OutputShape`](@ref), and.
[`LazyAlgebra.row_ndims`](@ref).

"""
output_ndims(A) = ndims(OutputShape(A))

"""
    LazyAlgebra.input_ndims(A)
    LazyAlgebra.input_ndims(typeof(A))

yield the number dimensions of the input `x` for `A*x` based on the type of `A`.

!!! note
    If the number `N` of dimensions of `x` to compute `A*x` is known in advance, do not
    extend this method but rather extend `LazyAlgebra.OutputShape(typeof(A))`,
    `LazyAlgebra.input_axes(A)`, and optionally `LazyAlgebra.input_size(A)` to
    respectively yield `LazyAlgebra.HasInputShape{N}()`, the axes and the size of `A*x`.

See also [`LazyAlgebra.output_ndims`](@ref), [`LazyAlgebra.input_axes`](@ref),
[`LazyAlgebra.input_eltype`](@ref), and [`LazyAlgebra.InputShape`](@ref), and.
[`LazyAlgebra.col_ndims`](@ref).

"""
input_ndims(A) = ndims(InputShape(A))

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

# Output and input axes and size are known in advance for a regular matrix.
output_axes(A::AbstractMatrix) = (axes(A, 1),)
output_size(A::AbstractMatrix) = (size(A, 1),)
input_axes( A::AbstractMatrix) = (axes(A, 2),)
input_size( A::AbstractMatrix) = (size(A, 2),)

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

yields the operator `B` of the *scaled operator* `A = λ*B` where `λ` is a number;
otherwise yields `A`. This method is also applicable to instances of
`LinearAlgebra.UniformScaling`. Call [`LazyAlgebra.multiplier`](@ref) to get the
multiplier `λ`.

"""
unscaled(A::Operator) = A
unscaled(A::Prod{<:Number}) = A[2]
unscaled(A::UniformScaling) = Id

"""
    LazyAlgebra.multiplier(A)

yields the multiplier `λ` of the *scaled operator* `A = λ*B` where `λ` is a number and `B`
an operator; otherwise yields `1`. This method is also applicable to instances of
`LinearAlgebra.UniformScaling`. Call [`LazyAlgebra.unscaled`](@ref) to get the operator
`B`.

"""
multiplier(A::Prod{<:Number}) = A[1]
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
function vmul end

Base.:(*)(A::Operator, x::AbstractArray) = vmul(A, x)
Base.:(\)(A::Operator, x::AbstractArray) = vmul(inv(A), x)

# First, factorize out multipliers so that only `vmul(α,A,x)` with `A` a non-scaled
# operator shall be implemented after this stage.
vmul(A::Scaled, x::AbstractArray) = vmul(A[1], A[2], x)
vmul(α::Number, A::Scaled, x::AbstractArray) = vmul(α*A[1], A[2], x)
vmul(A::Operator, x::AbstractArray) = vmul(𝟙, A, x)

# Second, deal with products of operators.
vmul(α::Number, A::Prod, x::AbstractArray) = vmul(α, A[1], vmul(A[2], x))

# Finally, consider `vmul(α,A,x)` for non-scaled, non-product operator `A`.
function vmul(α::Number, A::Operator, x::AbstractArray)
    # Create output array and call `vmul!` at stage 1 to skip checking of axes.
    y = create_output(α, A, x) # FIXME convert multiplier before?
    return vmul!(Stage(1), α, A, x, 𝟘, y)
end

"""
    vmul!(α::Number, A::Operator, x::AbstractArray, β::Number, y::AbstractArray) -> y

overwrites `y` with `α*A⋅x + β*y` and returns `y`. The convention is that the prior
contents of `y` is not used at all if `iszero(β)` holds so `y` can be directly used to
store the result even though it is not initialized.

Multiplier `β` must be dimensionless; it can be complex if `y` also has complex element
type, and must be real otherwise.

Another supported syntax is:

    vmul!(y::AbstractArray, [α::Number=𝟙], A::Operator, x::AbstractArray) -> y

to overwrite `y` with `α*A*x`, this is a shortcut for:

    vmul!(α, A, x, 0, y)

The `vmul!` method can be seen as a generalization of the `LinearAlgebra.mul!` method.

!!! warning
    Do not extend this method for specific operator types, but rather the
    [`LazyAlgebra.unsafe_vmul!`](@ref) method.

See also [`vmul`](@ref), [`LazyAlgebra.Operator`](@ref), and
[`LazyAlgebra.unsafe_vmul!`](@ref).

"""
function vmul! end

# Rewrite `vmul!(y,A,x)` and `vmul!(y,α,A,x)` into equivalent call to `vmul!(α,A,x,β,y)`.
# factorizing out the multipliers in the process.
vmul!(y::AbstractArray, A::Scaled, x::AbstractArray) = vmul!(A[1], A[2], x, 𝟘, y)
vmul!(y::AbstractArray, A::Operator, x::AbstractArray) = vmul!(𝟙, A, x, 𝟘, y)

vmul!(y::AbstractArray, α::Number, A::Scaled, x::AbstractArray) = vmul!(α*A[1], A[2], x, 𝟘, y)
vmul!(y::AbstractArray, α::Number, A::Operator, x::AbstractArray) = vmul!(α, A, x, 𝟘, y)

# Factorize out multipliers so that only `vmul!(α,A,x,β,y)` with `A` a non-scaled operator
# shall be implemented next.
vmul!(α::Number, A::Scaled, x::AbstractArray, β::Number, y::AbstractArray) =
    vmul!(α*A[1], A[2], x, β, y)

# Deal with products of operators.
vmul!(α::Number, A::Prod, x::AbstractArray, β::Number, y::AbstractArray) =
    vmul!(α, A[1], vmul(A[2], x), β, y)

# Now, consider `vmul!(α,A,x,β,y)` with `A` a non-scaled and non-product operator.
#
# Stages:
#   0. check axes;
#   1. convert `α`;
#   2. dispatch on `α`;
#   3. if `iszero(α)`, call `vscale!(y, β)` and stop; otherwise, convert `β` and
#      proceed with next stage;
#   4. dispatch on `β` to call `unsafe_vnul!`.
#
# Notes:
#  - The conversion and dispatching steps are separated in case conversion is not
#    type-stable which should not be the case.
#  - If axes are known to be ok, step 0 can be skipped.

function vmul!(α::Number, A::Operator, x::AbstractArray, β::Number, y::AbstractArray)
    @assert !(A isa Prod)
    check_output_axes(y, output_axes(A, x))
    return vmul!(Stage(1), α, A, x, β, y)
end

function vmul!(::Stage{1},
               α::Number, A::Operator, x::AbstractArray, β::Number, y::AbstractArray)
    α′ = convert_multiplier(α, output_eltype(A, x))
    return vmul!(Stage(2), α′, A, x, β, y)
end

function vmul!(::Stage{2},
               α::Number, A::Operator, x::AbstractArray, β::Number, y::AbstractArray)
    @dispatch_on_multiplier α vmul!(Stage(3), α, A, x, β, y)
    return y
end

function vmul!(::Stage{3},
               α::Number, A::Operator, x::AbstractArray, β::Number, y::AbstractArray)
    if α isa StaticMultiplier{0}
        vscale!(y, β)
    else
        β′ = convert_inplace_multiplier(β, eltype(y))
        vmul!(Stage(4), α, A, x, β′, y)
    end
    return y
end

function vmul!(::Stage{4},
               α::Number, A::Operator, x::AbstractArray, β::Number, y::AbstractArray)
    @dispatch_on_multiplier β unsafe_vmul!(α, A, x, β, y)
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
    y_axes == out_axes ? nothing : throw_incompatible_output_axes(y_axes, out_axes)

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
    LazyAlgebra.vzeros!(y)
end
```

See also [`vmul`](@ref), [`vmul!`](@ref), [`LazyAlgebra.Operator`](@ref),
[`LazyAlgebra.unsafe_vscale!`](@ref), [`vzeros!`](@ref),
[`LazyAlgebra.output_eltype`](@ref) [`LazyAlgebra.output_axes`](@ref), and
[`LazyAlgebra.create_output`](@ref).

"""
function unsafe_vmul! end

# Specialize `unsafe_vmul!` for a sum of operators knowing that a sum of more than 2
# operators is stored according to right-associativity. Compared to specializing `vmul!`
# instead, this saves re-checking axes, re-conversion of multipliers, and re-dispatching
# on multipliers.
function unsafe_vmul!(α::Number, A::Sum, x::AbstractArray, β::Number, y::AbstractArray)
    # There should be no needs to dispatch on the multipliers because, inputs `α` and `β`
    # have already been processed. Thus `unsafe_vmul!` can be directly called. In
    # principle, the second call should be with `𝟙*unit(β)`, but, being an in-place
    # multiplier, `β` is dimensionless and `𝟙*unit(β)` and `𝟙` are the same thing.
    unsafe_vmul!(α, A[1], x, β, y)
    unsafe_vmul!(α, A[2], x, 𝟙, y)
    return y
end

@noinline function unsafe_vmul!(α::Number, A::Union{Inverse{<:Sum},InverseAdjoint{<:Sum}},
                                 x::AbstractArray, β::Number, y::AbstractArray)
    error("automatic dispatching of the inverse of a sum of operators is not supported")
end

# Default rules to apply a Gram operator. Gram matrices are Hermitian by construction
# which left only 2 cases to deal with.
function vmul!(α::Number, G::Gram, x::AbstractArray, β::Number, y::AbstractArray)
    A = G[] # A is such that G = A'*A
    return vmul!(α, A', A*x, β, y)
end
#
function vmul!(α::Number, G::Inverse{<:Gram}, x::AbstractArray, β::Number, y::AbstractArray)
    A = G[][] # A is such that G = inv(A'*A) = inv(A)*inv(A')
    return vmul!(α, inv(A), inv(A')*x, β, y)
end

"""
    LazyAlgebra.test_API(A::Operator, x, y)

tests that operator API is correctly implemented for `A`. `x` is a chosen input for `A`
and `y` is the expected output. The shapes and element types of `x` and `y` must be
correct. The returned value is that of a `@testset`.

Keywords `alphas` and `betas` are tuples of values for `α` and `β` to test
`LazyAgebra.vmul(α, A, x)`, `LazyAgebra.vmul!(dst, α, A, x)`, and `LazyAgebra.vmul!(α, A,
x, β, y)`.

Keywords `atol` and `rtol` are the absolute and relative tolerances for comparing `A*x`
and `y`.

The `norm` keyword defaults to `LinearAlgebra.norm` for arrays.

"""
function test_API(A::Operator, x::AbstractArray, y::AbstractArray;
                  alphas::Tuple{Vararg{Number}} = (-1, 0, 1, 3, -2 + 1im),
                  betas::Tuple{Vararg{Number}} = (-1, 0, 1, 2, π),
                  rtol::Real = 4e-7, atol=0, norm::Function=norm,
                  name::AbstractString = repr(typeof(A)))

    @testset "Operator API for $name, T=$(eltype(x)), and dims=$(size(x))" begin
        # Output element type.
        o = @inferred OutputEltype(A)
        @test o === HasOutputEltype() || o === OutputEltypeUnknown()
        if o === HasOutputEltype()
            # `output_eltype(typeof(A))` must be implemented.
            @test hasmethod(output_eltype, Tuple{typeof(A)})
            @test @inferred(output_eltype(typeof(A))) === eltype(y)
        elseif hasmethod(output_eltype, Tuple{typeof(A),typeof(x)})
            # `output_eltype(typeof(A), typeof(x))` is implemented.
            @test @inferred(output_eltype(typeof(A),typeof(x))) === eltype(y)
        else
            # `Base.eltype(typeof(A))` must not be the default implementation.
            @test eltype(A) !== Any
            @test float(sumprod_type(eltype(A), eltype(x))) === eltype(y)
        end

        # Input element type.
        i = @inferred InputEltype(A)
        @test i === HasInputEltype() || i === InputEltypeUnknown()
        if i === HasInputEltype()
            @test @inferred(input_eltype(A)) === eltype(x)
        end

        # Output axes.
        o = @inferred OutputShape(A)
        @test o isa HasOutputShape || o === OutputShapeUnknown()
        if o isa HasOutputShape
            @test ndims(o) == ndims(y)
            @test @inferred(output_axes(A)) isa ArrayAxes{ndims(o)}
            @test @inferred(output_axes(A)) == axes(y)
        else
            @test hasmethod(output_axes, Tuple{typeof(A), typeof(axes(x))})
            @test output_axes(A, axes(x)) === axes(y)
        end

        # Input axes.
        i = @inferred InputShape(A)
        @test i isa HasInputShape || i === InputShapeUnknown()
        if i isa HasInputShape
            @test ndims(i) == ndims(x)
            @test @inferred(input_axes(A)) isa ArrayAxes{ndims(i)}
            @test @inferred(input_axes(A)) == axes(x)
        end

        xsav = copy(x)
        Ax = A*x
        @test x == xsav # x must be left unchanged
        @test eltype(Ax) == eltype(y)
        @test axes(Ax) == axes(y)
        @test Ax ≈ y atol=atol rtol=rtol norm=norm

        @testset "α*A*x with α=$α" for α in alphas
            α′ = convert_multiplier(α, Ax)
            αAx = @inferred(vmul(α, A, x))
            @test x == xsav
            @test eltype(αAx) == typeof(zero(α′)*zero(eltype(Ax)))
            @test αAx ≈ α′*y atol=atol rtol=rtol norm=norm
        end

        @testset "α*A*x + β*y with α=$α and β=$β" for α in alphas, β in betas
            α′ = convert_multiplier(α, Ax)
            β′ = convert_multiplier(β, y)
            T = typeof(zero(α′)*zero(eltype(Ax)) + zero(β′)*zero(eltype(y)))
            z = similar(y, T)
            iszero(β′) ? vnans!(z) : vcopy!(z, y) # fill with NaNs if values not to be used
            @test @inferred(vmul!(α, A, x, β, z)) === z
            @test x == xsav
            if atol == 0 && α == -β
                # Result should be ≈ 0 which is a delicate case for tolerances.
                @test z ≈ α′*Ax + β′*y atol=rtol*max(norm(α′*Ax), norm(β′*y)) rtol=0 norm=norm
            else
                @test z ≈ α′*Ax + β′*y atol=atol rtol=rtol norm=norm
            end
        end
    end
end
