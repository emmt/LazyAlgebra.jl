#--------------------------------------------------------------------------- Storage Order -

"""
    LazyAlgebra.StorageOrder(x)
    LazyAlgebra.StorageOrder(typeof(x))

Return the singleton object representing the storage order of `x`, one of:

- `LazyAlgebra.ColumMajor()` if the entries of `x` are stored in column-major order;

- `LazyAlgebra.RowMajor()` if the entries of `x` are stored in row-major order;

- `LazyAlgebra.StorageOrderUnknown()` otherwise (this is the default).

See also [`LazyAlgebra.is_column_major`](@ref) and [`LazyAlgebra.is_row_major`](@ref) .

"""
StorageOrder(x::Any) = StorageOrder(typeof(x))
StorageOrder(x::StorageOrder) = x

StorageOrder(::Type{<:Any}) = StorageOrderUnknown()
StorageOrder(::Type{<:ColumnMajor}) = ColumnMajor()
StorageOrder(::Type{<:RowMajor}) = RowMajor()
StorageOrder(::Type{<:Array}) = ColumnMajor()
StorageOrder(::Type{<:SparseMatrixCSC}) = ColumnMajor()
StorageOrder(::Type{<:SparseOperatorCSC}) = ColumnMajor()
StorageOrder(::Type{<:SparseOperatorCSR}) = RowMajor()
StorageOrder(::Type{<:Swapped{A}}) where {A} = transpose(StorageOrder(A))
StorageOrder(::Type{<:LinearAlgebra.Adjoint{<:Any,A}}) where {A} = transpose(StorageOrder(A))
StorageOrder(::Type{<:LinearAlgebra.Transpose{<:Any,A}}) where {A} = transpose(StorageOrder(A))

"""
    LazyAlgebra.is_row_major(x)
    LazyAlgebra.is_row_major(typeof(x))
    LazyAlgebra.is_row_major(ASAP.Format(A)) # FIXME

Return whether `x` has row-major storage order.

See also [`LazyAlgebra.is_column_major`](@ref) and [`LazyAlgebra.StorageOrder`](@ref) .

"""
is_row_major(::RowMajor) = true
is_row_major(::Type{<:RowMajor}) = true
is_row_major(::StorageOrder) = false
is_row_major(::Type{<:StorageOrder}) = false
is_row_major(x) = is_row_major(typeof(x))
is_row_major(::Type{T}) where {T<:Any} = is_row_major(StorageOrder(T))

"""
    LazyAlgebra.is_column_major(x)
    LazyAlgebra.is_column_major(typeof(x))
    LazyAlgebra.is_column_major(ASAP.Format(A)) # FIXME

Return whether `x` has column-major storage order.

See also [`LazyAlgebra.is_row_major`](@ref) and [`LazyAlgebra.StorageOrder`](@ref) .

yield whether `A` has column-major storage order. This trait is the opposite of
[`LazyAlgebra.is_row_major`](@ref).

"""
is_column_major(::ColumnMajor) = true
is_column_major(::Type{<:ColumnMajor}) = true
is_column_major(::StorageOrder) = false
is_column_major(::Type{<:StorageOrder}) = false
is_column_major(x) = is_column_major(typeof(x))
is_column_major(::Type{T}) where {T<:Any} = is_column_major(StorageOrder(T))

#----------------------------------------------------------------- Input and Output Shapes -

"""
    LazyAlgebra.InputShape(A)
    LazyAlgebra.InputShape(typeof(A))

Depending on the type of operator `A`, return one of:

* `LazyAlgebra.InputShapeUnknown()` if the shape of the input of `A` cannot be determined
  in advance. This is the assumed default.

* `LazyAlgebra.HasInputShape{N}()` if the input of `A` has a known `N`-dimensional shape
  whose axes and size are respectively given by `LazyAlgebra.input_axes(A)` and
  `LazyAlgebra.input_size(A)`.

See also [`LazyAlgebra.InputEltype](@ref) and [`LazyAlgebra.OutputShape](@ref).

"""
InputShape(A) = InputShape(typeof(A))
InputShape(::Type{T}) where {T<:Any} = InputShapeUnknown()
InputShape(::Type{T}) where {T<:Union{Swapped,Inverse}} = transpose(OutputShape(parent(T)))

"""
    LazyAlgebra.OutputShape(A)
    LazyAlgebra.OutputShape(typeof(A))

Depending on the type of operator `A`, return one of:

* `LazyAlgebra.OutputShapeUnknown()` if the shape of the output of `A` cannot be
  determined in advance.

* `LazyAlgebra.HasOutputShape{N}()` if the output of `A` has a known `N`-dimensional shape
  whose axes and size are respectively given by `LazyAlgebra.output_axes(A)` and
  `LazyAlgebra.output_size(A)`.

!!! note
    In any case, the output shape of `A*x` can be determined by
    `LazyAlgebra.output_axes(A, x)`.

See also [`LazyAlgebra.OutputEltype](@ref), [`LazyAlgebra.InputShape](@ref), and
[`LazyAlgebra.output_axes](@ref).

"""
OutputShape(A) = OutputShape(typeof(A))
OutputShape(::Type{T}) where {T<:Any} = OutputShapeUnknown()
OutputShape(::Type{T}) where {T<:Union{Swapped,Inverse}} = transpose(InputShape(parent(T)))

Base.ndims(x::OutputShape) = ndims(typeof(x))
Base.ndims(::HasOutputShape{N}) where {N} = N
@noinline Base.ndims(::OutputShapeUnknown) =
    throw_argument_error("unknown number of output dimensions")

Base.ndims(x::InputShape) = ndims(typeof(x))
Base.ndims(::HasInputShape{N}) where {N} = N
@noinline Base.ndims(::InputShapeUnknown) =
    throw_argument_error("unknown number of input dimensions")

"""
    LazyAlgebra.output_ndims(A)
    LazyAlgebra.output_ndims(typeof(A))

Return the number dimensions of the result of `A*x` based on the type of `A`.

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

Return the number dimensions of the input `x` for `A*x` based on the type of `A`.

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

#---------------------------------------------------------- Input and Output Element Types -

"""
    LazyAlgebra.InputEltype(A)
    LazyAlgebra.InputEltype(typeof(A))

Depending on the type of operator `A`, return one of:

* `LazyAlgebra.InputEltypeUnknown()` if the element type of the input of `A` cannot be
  determined in advance. This is the assumed default.

* `LazyAlgebra.HasInputEltype()` if the element type of the input of `A` is known and
  given by `LazyAlgebra.input_eltype(typeof(A))`.

See also [`LazyAlgebra.InputShape](@ref), [`LazyAlgebra.OutputEltype](@ref), and
[`LazyAlgebra.output_eltype](@ref).

"""
InputEltype(A) = InputEltype(typeof(A))
InputEltype(::Type{T}) where {T<:Any} = InputEltypeUnknown()
InputEltype(::Type{T}) where {T<:Union{Swapped,Inverse}} =
    transpose(OutputEltype(parent(T)))

"""
    LazyAlgebra.input_eltype(A) -> T
    LazyAlgebra.input_eltype(typeof(A)) -> T

Return the element type `T` of `x` for computing `A*x` with operator `A`. Not all
operators implement this trait.

To implement this trait for an operator, the following two methods shall be specialized:

```julia
LazyAlgebra.InputEltype(typeof(A)) = LazyAlgebra.HasInputEltype()
LazyAlgebra.input_eltype(typeof(A)) = T
```

If this trait is implemented, argument `x` with a different element type is automatically
converted by [`vmul`](@ref) and [`vmul!`](@ref). As a consequence, consider carefully
whether this is advisable or not. In general, this is only needed if the operator is
implemented by an external library which imposes the element type.

!!! warning
    This function shall only be called if `LazyAlgebra.InputEltype(typeof(A))`
    yields `LazyAlgebra.HasInputEltype()`.

See also [`vmul`](@ref), [`vmul!`](@ref), [`LazyAlgebra.InputEltype`](@ref) and
[`LazyAlgebra.output_eltype`](@ref).

"""
input_eltype(A) = input_eltype(typeof(A))
input_eltype(::Type{T}) where {T<:Union{Swapped,Inverse}} = output_eltype(parent(T))
@noinline input_eltype(::Type{T}) where {T<:Operator} =
    error("`LazyAlgebra.input_eltype(T)` not defined for operators of type `T=$T`")

"""
    LazyAlgebra.OutputEltype(A)
    LazyAlgebra.OutputEltype(typeof(A))

Depending on the type of operator `A`, return one of:

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
OutputEltype(::Type{T}) where {T<:Any} = OutputEltypeUnknown()
OutputEltype(::Type{T}) where {T<:Union{Swapped,Inverse}} =
    transpose(InputEltype(parent(T)))

"""
    LazyAlgebra.output_eltype(A) -> T
    LazyAlgebra.output_eltype(typeof(A)) -> T

Return the element type `T` of the result of `A*x` for operator `A` and any acceptable
`x`. Not all operators implement this trait.

To implement this trait for an operator, the following two methods shall be specialized:

```julia
LazyAlgebra.OutputEltype(typeof(A)) = LazyAlgebra.HasOutputEltype()
LazyAlgebra.output_eltype(typeof(A)) = T
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
output_eltype(::Type{T}) where {T<:Union{Swapped,Inverse}} = input_eltype(parent(T))
@noinline output_eltype(::Type{T}) where {T<:Operator} =
    error("`LazyAlgebra.output_eltype(T)` not defined for operators of type `T = $T`")

"""
    LazyAlgebra.output_eltype([α::Number,] A::Operator, x::AbstractArray) -> T
    LazyAlgebra.output_eltype([typeof(α),] typeof(A), typeof(x)) -> T

Return the element type `T` of the result of `A*x` or of `α*A*x` if the multiplier `α` is
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

See also [`LazyAlgebra.output_axes`](@ref) and [`LazyAlgebra.create_output`](@ref).

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
    prod_type(convert_floating_point_type(eltype(x), α), eltype(x))

# Fallback method, assumes that one of `output_eltype(A)` or `eltype(A)` is applicable.
output_eltype(::Type{A}, ::Type{x}) where {A<:Operator, x<:AbstractArray} =
    OutputEltype(A) === HasOutputEltype() ? float(output_eltype(A)) :
    float(sum_prod_type(eltype(A), eltype(x)))

# Output element type for sums and products assuming right-associativity.
output_eltype(::Type{Sum{L,R}}, ::Type{x}) where {L,R,x<:AbstractArray} =
    sum_type(output_eltype(L, x), output_eltype(R, x))

output_eltype(::Type{Prod{L,R}}, ::Type{x}) where {L,R,x<:AbstractArray} =
    output_eltype(L, AbstractArray{output_eltype(R, x)})

# Output element type for scaled operators.
output_eltype(::Type{Prod{L,R}}, ::Type{x}) where {L<:Number,R,x<:AbstractArray} =
    output_eltype(L, R, x)

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

#-------------------------------------------------------------------------------------------

# Some traits need to be transposed.

Base.transpose(trait::StorageOrderUnknown) = StorageOrderUnknown()
Base.transpose(trait::RowMajor) = ColumnMajor()
Base.transpose(trait::ColumnMajor) = RowMajor()

Base.transpose(trait::InputShapeUnknown) = OutputShapeUnknown()
Base.transpose(trait::OutputShapeUnknown) = InputShapeUnknown()
Base.transpose(trait::HasInputShape{N}) where {N} = HasOutputShape{N}()
Base.transpose(trait::HasOutputShape{N}) where {N} = HasInputShape{N}()

Base.transpose(trait::InputEltypeUnknown) = OutputEltypeUnknown()
Base.transpose(trait::OutputEltypeUnknown) = InputEltypeUnknown()
Base.transpose(trait::HasInputEltype) = HasOutputEltype()
Base.transpose(trait::HasOutputEltype) = HasInputEltype()

is_complex(x) = is_complex(typeof(x))
is_complex(::Type{T}) where {T<:Number} = is_complex(bare_type(T))
is_complex(::Type{T}) where {T<:Real} = false
is_complex(::Type{T}) where {T<:Complex} = true
is_complex(::Type{T}) where {T<:Any} = false
