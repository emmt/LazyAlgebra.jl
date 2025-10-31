#
# operators.jl -
#
# Implement non-specific methods for operators.
#
#-----------------------------------------------------------------------------------------

# Conversion constructors.
Operator(A::Operator) = A
Operator(A::LinearAlgebra.UniformScaling) = multiplier(A) * Id
Operator(A::AbstractMatrix) = PseudoMatrix(A, Dims{1})

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
    for constructor in (:Adjoint, :Transpose, :Inverse)
        @eval $constructor(A::$Other) = $constructor(Operator(A))
    end
end

#----------------------------------------------------------------- Input Shape, Size, etc. -

"""
    LazyAlgebra.input_shape(A)

Return the shape that `x` must have to compute `A*x` with the matrix or linear operator
`A`. The shape is a tuple of array dimensions or index unit ranges.

This method has no default implementation. To provide this method for an operator `A`, the
following two methods shall be specialized:

```julia
LazyAlgebra.InputShape(typeof(A)) = LazyAlgebra.HasInputShape{N}()
LazyAlgebra.input_shape(A) = ...
```

with `N` the number of dimensions of suitable input `x` to compute `A*x`.

See also [`LazyAlgebra.output_shape`](@ref), [`LazyAlgebra.InputShape`](@ref),
[`LazyAlgebra.input_ndims`](@ref), [`LazyAlgebra.input_axes`](@ref), and
[`LazyAlgebra.input_size`](@ref).

"""
input_shape(A::Operator) = throw_input_shape_not_implemented(typeof(A))
input_shape(A::AbstractMatrix) = input_axes(A)

@noinline throw_input_shape_not_implemented(::Type{T}) where {T} =
    error(string("checking the dimension of the input of an operator of type `", T,
                 "` is not correctly implemented, see doc. of `LazyAlgebra.output_axes`"))

"""
    LazyAlgebra.input_size(A)

Return the dimensions of the input argument `x` to compute `A*x` with the matrix or linear
operator `A`.

This method is only applicable to operators whose input shape is fixed (see
[`LazyAlgebra.input_shape`](@ref)).

"""
input_size(A::Operator) = as_array_size(input_shape(A))
input_size(A::AbstractMatrix) = (size(A, 2),)

"""
    LazyAlgebra.input_axes(A)

Return the dimensions of the input argument `x` to compute `A*x` with the matrix or linear
operator `A`.

This method is only applicable to operators whose input shape is fixed (see
[`LazyAlgebra.input_shape`](@ref)).

"""
input_axes(A::Operator) = as_array_axes(input_shape(A))
input_axes(A::AbstractMatrix) = (axes(A, 2),)
input_axes(A::Conjugate) = input_axes(parent(A))
input_axes(A::Union{Adjoint,Transpose,Inverse}) = output_axes(parent(A))

"""
    LazyAlgebra.ncols(A)

yields the *equivalent* number of columns of the matrix or linear operator `A` that is the
number of elements of any valid input `x` for `A*x` whatever the number of dimensions of
`x`.

This method is only applicable to operators whose input shape is fixed (see
[`LazyAlgebra.input_shape`](@ref)).

"""
ncols(A::Operator) = prod(input_size(A))
ncols(A::AbstractMatrix) = size(A, 2)

#---------------------------------------------------------------- Output Shape, Size, etc. -

"""
    LazyAlgebra.output_shape(A::Operator)

Return the shape of `A*x` when it is known in advance for the matrix or linear operator
`A`. The shape is a tuple of array dimensions or index unit ranges.

This method has no default implementation. To implement this method for an operator `A`,
the following two methods shall be specialized:

```julia
LazyAlgebra.OutputShape(typeof(A)) = LazyAlgebra.HasOutputShape{M}()
LazyAlgebra.output_shape(A) = ...
```

with `M` the number of dimensions of `A*x`.

See also [`LazyAlgebra.input_shape`](@ref), [`LazyAlgebra.OutputShape`](@ref),
[`LazyAlgebra.output_ndims`](@ref), [`LazyAlgebra.output_axes`](@ref), and
[`LazyAlgebra.output_size`](@ref).

"""
output_shape(A::Operator) = throw_output_shape_not_implemented(typeof(A))
output_shape(A::AbstractMatrix) = output_axes(A)

@noinline throw_output_shape_not_implemented(::Type{T}) where {T} =
    error(string("inferring the dimension of the output of an operator of type `", T,
                 "` is not correctly implemented, see doc. of `LazyAlgebra.output_axes`"))

"""
    LazyAlgebra.output_size(A)

Return the dimensions of the result of left-multiplying a *vector* (of suitable size) by
the matrix or linear operator `A`.

This method is only applicable to operators whose output shape is fixed (see
[`LazyAlgebra.output_shape`](@ref)).

"""
output_size(A::Operator) = as_array_size(output_shape(A))
output_size(A::AbstractMatrix) = (size(A, 1),)

"""
    LazyAlgebra.output_axes(A)

Return the axes of the result of left-multiplying a *vector* (of suitable size) by the
matrix or linear operator `A`.

This method is only applicable to operators whose output shape is fixed (see
[`LazyAlgebra.output_shape`](@ref)).

"""
output_axes(A::Operator) = as_array_axes(output_shape(A))
output_axes(A::AbstractMatrix) = (axes(A, 1),)
output_axes(A::Conjugate) = output_axes(parent(A))
output_axes(A::Union{Adjoint,Transpose,Inverse}) = input_axes(parent(A))

"""
    LazyAlgebra.nrows(A)

yields the *equivalent* number of rows of the matrix or linear operator `A` that is the
number of elements of the result of `A*x` whatever its number of dimensions.

This method is only applicable to operators whose output shape is fixed (see
[`LazyAlgebra.output_shape`](@ref)).

"""
nrows(A::Operator) = prod(output_size(A))
nrows(A::AbstractMatrix) = size(A, 1)

#------------------------------------------------------------------------- Shape of Result -

"""
    LazyAlgebra.output_axes(A::Operator, x::AbstractArray)

Return the axes of the result of `A*x`.

To have this method applicable to a given linear operator type, there are several
possibilities:

1. The method `LazyAlgebra.output_axes(A, x)` may be directly implemented for the type of
   `A` an perhaps `x`.

2. If the axes of `A*x` only depend on the operator `A` and on the axes of the input array
   `x`, then it is sufficient to provide:

   ```julia
   LazyAlgebra.output_axes(A, axes(x))
   ```

3. If the shapes of the input and output of `A` are known in advance, then it is simpler
   to provide:

   ```julia
   LazyAlgebra.input_shape(A)
   LazyAlgebra.output_shape(A)
   ```

   to respectively yield the shapes of the input and output of `A` as tuples of
   dimension lengths and/or index unit ranges (the two may be mixed). For `LazyAlgebra`
   to be aware of this, the following traits must also be implemented:

   ```julia
   LazyAlgebra.InputShape(typeof(A)) = LazyAlgebra.HasInputShape{N}()
   LazyAlgebra.OutputShape(typeof(A)) = LazyAlgebra.HasOutputShape{M}()
   ```

   with `N` and `M` the number of dimensions of the input and output of `A`.

See also [`LazyAlgebra.create_output`](@ref), [`LazyAlgebra.InputShape`](@ref),
[`LazyAlgebra.input_shape`](@ref), [`LazyAlgebra.OutputShape`](@ref), and
[`LazyAlgebra.output_shape`](@ref).

"""
output_axes(A::Operator, x::AbstractArray) = output_axes(A, axes(x))

# Fallback version of `output_axes(A, x)` assuming `input_axes(A)` and `input_axes(A)` are
# defined for `A`.
function output_axes(A::Operator, x_axes::ArrayAxes)
    InputShape(A) isa HasInputShape || throw_input_shape_not_implemented(typeof(A))
    OutputShape(A) isa HasOutputShape || throw_output_shape_not_implemented(typeof(A))
    check_input_axes(x_axes, input_axes(A))
    return output_axes(A)
end

output_axes(A::InverseAdjoint, x_axes::ArrayAxes) = output_axes(parent(parent(A)), x_axes)
output_axes(A::InverseTranspose, x_axes::ArrayAxes) = output_axes(parent(parent(A)), x_axes)

# Output axes for products assuming right-associativity.
output_axes((α,A)::Scaled, x_axes::ArrayAxes) = output_axes(A, x_axes)
output_axes((A,B)::Prod, x_axes::ArrayAxes) = output_axes(A, output_axes(B, x_axes))

# Output axes for sums assuming right-associativity.
output_axes((A,B)::Sum, x_axes::ArrayAxes) =
    output_axes_in_sum(output_axes(A, x_axes), B, x_axes)

output_axes_in_sum(y_axes::ArrayAxes, (A,B)::Sum, x_axes::ArrayAxes) =
    output_axes(A, x_axes) == y_axes ? output_axes_in_sum(y_axes, B, x_axes) :
    throw_incompatible_output_axes_in_sum()

output_axes_in_sum(y_axes::ArrayAxes, A::Operator, x_axes::ArrayAxes) =
    output_axes(A, x_axes) == y_axes ? y_axes : throw_incompatible_output_axes_in_sum()

@noinline throw_incompatible_output_axes_in_sum() =
    throw(DimensionMismatch("incompatible output axes in sum"))

"""
    y = LazyAlgebra.create_output([α::Number,] A::Operator, x::AbstractArray)

Create a new array `y` to store the result of `A*x` or of `α*A*x` if the multiplier `α` is
specified. In this latter case, the returned type does not depend on the numerical
precision of `α`, only on its units, if any, and on whether it is a real or a complex
number.

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
    LazyAlgebra.unscaled(A)

If `A = λ*B` is a *scaled operator* with `B` an operator and `λ` a number, returns the
operator `B`; otherwise returns operator `A`. This method is also applicable to instances
of `LinearAlgebra.UniformScaling`. Call [`LazyAlgebra.multiplier`](@ref) to get the
multiplier `λ`.

"""
unscaled(A::Operator) = A
unscaled(A::Scaled) = A[2]
unscaled(A::UniformScaling) = Id

"""
    LazyAlgebra.multiplier(A)

If `A = λ*B` is a *scaled operator* with `B` an operator and `λ` a number, returns the
multiplier `λ`; otherwise returns `𝟙`. This method is also applicable to instances of
`LinearAlgebra.UniformScaling`. Call [`LazyAlgebra.unscaled`](@ref) to get the operator
`B`.

"""
multiplier(A::Scaled) = A[1]
multiplier(A::Operator) = 𝟙
multiplier(A::UniformScaling) = getfield(A, :λ)

"""
    LazyAlgebra.check_vmul(y, A, x) -> (v1, v2, v1 - v2)

Return `v1 = vdot(y, A*x)`, `v2 = vdot(A'*y, x)` and their difference for `A` a linear
operator, `y` a *vector* of the output space of `A` and `x` a *vector* of the input space
of `A`. In principle, the two inner products should be equal whatever `x` and `y`;
otherwise the implementation of the operator has a bug.

Simple linear operators operating on Julia arrays can be tested on random *vectors* with:

    check_vmul([T=Float64,] outdims, A, inpdims) -> (v1, v2, v1 - v2)

with `outdims` and `outdims` the dimensions of the output and input *vectors* for `A`.
Optional argument `T` is the element type.

If `A` operates on Julia arrays and methods `input_eltype`, `input_size`, `output_eltype`
and `output_size` have been specialized for `A`, then:

    check_vmul(A) -> (v1, v2, v1 - v2)

is sufficient to check `A` against automatically generated random arrays.

See also: [`vdot`](@ref), [`vcreate`](@ref), [`vmul!`](@ref), [`input_type`](@ref).

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
    A*x
    vmul(A, x)
    (α*A)*x
    vmul(α, A, x)

Return the result of applying the linear operator `A` or the scaled linear operator `α*A`
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
    # Convert multiplier before creating output and dispatching.
    α = convert_multiplier(α, output_eltype(A, x))
    # Create output and apply operator unless `α` is zero.
    y = create_output(α, A, x)
    if iszero(α)
        # Zero-fill output.
        vzeros!(y)
    else
        # Dispatch on `α` to call the unsafe method.
        @dispatch_on_multiplier α unsafe_vmul!(α, A, x, 𝟘, y)
    end
    return y
end

"""
    vmul!(α::Number, A::Operator, x::AbstractArray, β::Number, y::AbstractArray) -> y

Overwrite `y` with `α*A⋅x + β*y` and return `y`.

Multiplier `β` must be dimensionless; it can be complex if `y` also has complex element
type, and must be real otherwise. The convention is that the prior content of `y` is not
used at all if `iszero(β)` holds so `y` can be directly used to store the result even
though it is not initialized.

Other supported methods are:

    vmul!(y::AbstractArray, [α::Number=𝟙], A::Operator, x::AbstractArray) -> y
    vmul!(z::AbstractArray, α::Number, A::Operator, x::AbstractArray, β::Number, y::AbstractArray) -> z

to respectively overwrite `y` with `α*A*x` and `z` with `α*A⋅x + β*y`, these are shortcuts
for:

    vmul!(α, A, x, 𝟘, y)
    vmul!(α, A, x, 𝟙, vscale!(z, β, y))

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

# Now, implement `vmul!(α,A,x,β,y)` with `A` a non-scaled and non-product operator.
function vmul!(α::Number, A::Operator, x::AbstractArray, β::Number, y::AbstractArray)
    # Check arguments indices.
    check_output_axes(y, output_axes(A, x))
    # Check the compatibility of types and units.
    _ = convert(eltype(y), sample(α)*sample(output_eltype(A, x))
                + sample(β)*sample(eltype(y)))::eltype(y)
    # Deal with multipliers.
    unsafe_vmul!(Val(:alpha_beta), α, A, x, β, y)
    return y
end

function unsafe_vmul!(::Val{:alpha_beta},
                      α::Number, A::Operator, x::AbstractArray,
                      β::Number, y::AbstractArray)
    # Deal with `β` than `α`.
    β = convert_multiplier(β, eltype(y))
    @dispatch_on_multiplier β unsafe_vmul!(Val(:alpha), α, A, x, β, y)
    return nothing
end

function unsafe_vmul!(::Val{:alpha},
                      α::Number, A::Operator, x::AbstractArray,
                      β::Number, y::AbstractArray)
    α = convert_multiplier(α, output_eltype(A, x))
    if iszero(α)
        # Skip computing `α*A*x`.
        unsafe_vscale!(y, β)
    else
        @dispatch_on_multiplier α unsafe_vmul!(α, A, x, β, y)
    end
    return nothing
end

function unsafe_vmul!(::Val{:beta},
                      α::Number, A::Operator, x::AbstractArray,
                      β::Number, y::AbstractArray)
    β = convert_multiplier(β, eltype(y))
    if iszero(α)
        # Skip computing `α*A*x`.
        @dispatch_on_multiplier β unsafe_vscale!(y, β)
    else
        @dispatch_on_multiplier β unsafe_vmul!(α, A, x, β, y)
    end
    return nothing
end

function vmul!(z::AbstractArray, α::Number, A::Operator, x::AbstractArray,
               β::Number, y::AbstractArray)
    # Check compatibility of arguments `β`, `y`m and `z`.
    @assert_same_axes y z
    _ = convert(eltype(z), sample(β)*sample(eltype(y)))::eltype(z)

    β = convert_multiplier(β, eltype(y))
    if iszero(β)
        vmul!(α, A, x, 𝟘, z)
    else
        @dispatch_on_multiplier β unsafe_vscale!(z, β, y)
        vmul!(α, A, x, 𝟙, z)
    end
    return z
end

# Extend `vmul!` for regular matrices.
vmul!(y::AbstractVector, A::AbstractMatrix, x::AbstractVector) =
    LinearAlgebra.mul!(y, A, x)
function vmul!(α::Number, A::AbstractMatrix, x::AbstractVector,
               β::Number, y::AbstractVector)
    α = convert_multiplier(α, output_eltype(typeof(A), typeof(x)))
    β = convert_multiplier(β, output_eltype(typeof(A), typeof(x)))
    LinearAlgebra.mul!(y, A, x, α, β)
    return y
end

# Extend `unsafe_vmul!` for regular matrices.
unsafe_vmul!(α::Number, A::AbstractMatrix, x::AbstractVector, β::Number, y::AbstractVector) =
    LinearAlgebra.mul!(y, A, x, α, β)

# Extend `LinearAlgebra.ldiv!(y, A, b)` to overwrite `y` with `A\b`.
LinearAlgebra.ldiv!(y::AbstractArray, A::Operator, b::AbstractArray) =
    vmul!(y, inv(A), b)

# Extend `LinearAlgebra.mul!(c, A, b, α, β)` to overwrite `c` with `α*A*b + β*c`.
LinearAlgebra.mul!(c::AbstractArray, A::Operator, b::AbstractArray, α::Number, β::Number) =
    vmul!(α, A, b, β, c)

# Extend `LinearAlgebra.mul!(y, A, b)` to overwrite `y` with `A*b`.
LinearAlgebra.mul!(y::AbstractArray, A::Operator, b::AbstractArray) =
    vmul!(y, A, b)

"""
    LazyAlgebra.check_input_axes(x, inp_axes) -> nothing
    LazyAlgebra.check_input_axes(axes(x), inp_axes) -> nothing

Throw a `DimensionMismatch` exception if the axes of the input array `x` are not equal to
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

Throw a `DimensionMismatch` exception if the axes of the output array `y` are not equal to
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
    print(io, "axes of ", arg_name, " should be ")
    print_axes(io, arg_axes)
    print(io, ", got ")
    print_axes(io, ref_axes)
    return String(take!(io))
end

function axes_to_string(rngs::Tuple{Vararg{AbstractUnitRange{<:Integer}}})
    io = IOBuffer()
    print_axes(io, rngs)
    return String(take!(io))
end

"""
    LazyAlgebra.unsafe_vmul!(α::Number, A::Operator, x::AbstractArray,
                             β::Number, y::AbstractArray)

Overwrite `y` with `α*A⋅x + β*y`. This method (not [`vmul`](@ref) nor [`vmul!`](@ref))
is supposed to be specialized for any supported operator type.

This method is called by [`vmul`](@ref) and [`vmul!`](@ref) after checking that arguments
`x` and `y` have correct axes (so that `@inbounds` may be assumed to compute the result
stored in `y`), with multipliers `α` and `β` converted to suitable numbers, and only if
`iszero(α)` does not hold. The convention is that the prior contents of `y` is not used at
all if `iszero(β)` holds so that `y` can be directly used to store the result even though
it is not initialized. `LazyAlgebra.unsafe_vmul!` shall return `nothing` (any returned
value is ignored by [`vmul`](@ref) and [`vmul!`](@ref).

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
    return nothing
end

@noinline function unsafe_vmul!(α::Number, A::Union{Inverse{<:Sum},InverseAdjoint{<:Sum}},
                                x::AbstractArray, β::Number, y::AbstractArray)
    error("applying the inverse of a general sum of operators is not supported")
end

# Deal with scaled operator.
unsafe_vmul!(α::Number, (λ,A)::Scaled, x::AbstractArray, β::Number, y::AbstractArray) =
    unsafe_vmul!(Val(:alpha), α*λ, A, x, β, y)

# Deal with products of operators. FIXME In principle, there are no needs to recheck
# indices, convert multipliers, and dispatch on their values.
unsafe_vmul!(α::Number, (A,B)::Prod, x::AbstractArray, β::Number, y::AbstractArray) =
    unsafe_vmul!(α, A, vmul(B, x), β, y)

"""
    LazyAlgebra.test_API(A::Operator, x, y)

Test that operator API is correctly implemented for `A`. `x` is a chosen input for `A` and
`y` is the expected output. The shapes and element types of `x` and `y` must be correct.
The returned value is that of a `@testset`.

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
            @test float(sum_prod_type(eltype(A), eltype(x))) === eltype(y)
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
