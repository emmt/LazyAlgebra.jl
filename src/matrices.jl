# Extend LazyAlgebra to regular matrices and vectors and provide methods that generalize
# the usual API for matrices and vectors.

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

output_axes(A::AbstractMatrix) = (axes(A, 1),)
input_axes( A::AbstractMatrix) = (axes(A, 2),)

"""
    LazyAlgebra.output_size(A)

yields the dimensions of the result of applying the linear operator `A` or multiplying by
the matrix `A`. This method relies on [`LazyAlgebra.output_axes(A)`](@ref
LazyAlgebra.output_axes) which may not be implemented for all operators.

"""
output_size(A::Operator) = map(length, output_axes(A))
output_size(A::AbstractMatrix) = (size(A, 1),)

"""
    LazyAlgebra.input_size(A)

yields the dimensions of the input argument `x` to compute `A*x` with the matrix or linear
operator `A`. This method relies on [`LazyAlgebra.input_axes(A)`](@ref
LazyAlgebra.input_axes) which may not be implemented for all operators.

"""
input_size(A::Operator) = map(length, input_axes(A))
input_size(A::AbstractMatrix) = (size(A, 2),)

"""
    LazyAlgebra.row_axes(A)

yields the axes of the result of applying the linear operator `A` or multiplying by the
matrix `A`. This is equivalent to [`LazyAlgebra.output_axes(A)`](@ref
LazyAlgebra.output_axes).

"""
row_axes(A::Union{Operator,AbstractMatrix}) = output_axes(A)

"""
    LazyAlgebra.col_axes(A)

yields the axes of the input argument `x` to compute `A*x` with the matrix or linear
operator `A`. This is equivalent to [`LazyAlgebra.input_axes(A)`](@ref
LazyAlgebra.input_axes).

"""
col_axes(A::Union{Operator,AbstractMatrix}) = input_axes(A)

"""
    LazyAlgebra.row_size(A)

yields the dimensions of the result of applying the linear operator `A` or multiplying by
the matrix `A`. This is equivalent to [`LazyAlgebra.output_size(A)`](@ref
LazyAlgebra.output_size).

"""
row_size(A::Union{Operator,AbstractMatrix}) = output_size(A)

"""
    LazyAlgebra.col_size(A)

yields the dimensions of the input argument `x` to compute `A*x` with the matrix or linear
operator `A`. This is equivalent to [`LazyAlgebra.input_size(A)`](@ref
LazyAlgebra.input_size).

"""
col_size(A::Union{Operator,AbstractMatrix}) = input_size(A)

"""
    LazyAlgebra.nrows(A)

yields the *equivalent* number of rows of the matrix or linear operator `A`. Not all
operators extend this method.

In the implemented generalization of linear operators, the equivalent number of rows is
the number of element of the result of applying the operator be it single- or
multi-dimensional.

"""
nrows(A::Operator) = prod(row_size(A))
nrows(A::AbstractMatrix) = size(A, 1)

"""
    LazyAlgebra.ncols(A)

yields the *equivalent* number of columns of the linear operator `A`. Not all operators
extend this method.

In the implemented generalization of linear operators, the equivalent number of columns is
the number of element of an argument of the operator be it single- or multi-dimensional.

"""
ncols(A::Operator) = prod(col_size(A))
ncols(A::AbstractMatrix) = size(A, 2)
