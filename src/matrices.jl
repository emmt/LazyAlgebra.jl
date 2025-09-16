# Extend LazyAlgebra to regular matrices and vectors and provide methods to generalize
# the API for matrices and vectors.

# Extend `vmul!` for regular matrices.
vmul!(y::AbstractVector, A::AbstractMatrix, x::AbstractVector) =
    LinearAlgebra.mul!(y, A, x)

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
    LazyAlgebra.output_size(A)

yields the dimensions of the result of applying the linear operator `A` or of multiplying
by the matrix `A`. This method relies on [`LazyAlgebra.output_axes(A)`](@ref
LazyAlgebra.output_axes) which may not be implemented for all operators.

"""
output_size(A::Operator) = map(length, output_axes(A))

"""
    LazyAlgebra.input_size(A)

yields the dimensions of the input argument `x` to compute `A*x` with the matrix or linear
operator `A`. This method relies on [`LazyAlgebra.input_axes(A)`](@ref
LazyAlgebra.input_axes) which may not be implemented for all operators.

"""
input_size(A::Operator) = map(length, input_axes(A))

"""
    LazyAlgebra.row_ndims(A)
    LazyAlgebra.row_ndims(typeof(A))

yield the number of dimensions of the result of applying the linear operator `A` or
multiplying by the matrix `A`. This function is an alias to
[`LazyAlgebra.output_ndims`](@ref).

"""
const row_ndims = output_ndims

"""
    LazyAlgebra.col_ndims(A)
    LazyAlgebra.col_ndims(typeof(A))

yield the number of dimensions of the input argument `x` to compute `A*x` with the matrix
or linear operator `A`. This function is an alias to [`LazyAlgebra.input_ndims`](@ref).

"""
const col_ndims = input_ndims

"""
    LazyAlgebra.row_axes(A)

yields the axes of the result of applying the linear operator `A` or of multiplying by the
matrix `A`. This function is an alias to [`LazyAlgebra.output_axes`](@ref).

"""
const row_axes = output_axes

"""
    LazyAlgebra.col_axes(A)

yields the axes of the input argument `x` to compute `A*x` with the matrix or linear
operator `A`. This function is an alias to [`LazyAlgebra.input_axes`](@ref).

"""
const col_axes = input_axes

"""
    LazyAlgebra.row_size(A)

yields the dimensions of the result of applying the linear operator `A` or of multiplying
by the matrix `A`. This function is an alias to [`LazyAlgebra.output_size`](@ref).

"""
const row_size = output_size

"""
    LazyAlgebra.col_size(A)

yields the dimensions of the input argument `x` to compute `A*x` with the matrix or linear
operator `A`. This function is an alias to [`LazyAlgebra.input_size`](@ref).

"""
const col_size = input_size

"""
    LazyAlgebra.nrows(A)

yields the *equivalent* number of rows of the matrix or linear operator `A` that is the
number of elements of the result of `A*x` whatever its number of dimensions.

This method is only applicable to operators whose output size is fixed.

"""
nrows(A::Operator) = prod(row_size(A))
nrows(A::AbstractMatrix) = size(A, 1)

"""
    LazyAlgebra.ncols(A)

yields the *equivalent* number of columns of the linear operator `A` that is the number of
elements of any valid input `x` for `A*x` whatever the number of dimensions of `x`.

This method is only applicable to operators whose input size is fixed.

"""
ncols(A::Operator) = prod(col_size(A))
ncols(A::AbstractMatrix) = size(A, 2)
