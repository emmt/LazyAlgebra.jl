#
# blas.jl -
#
# Code based on BLAS (Basic Linear Algebra Subroutines).
#
#-------------------------------------------------------------------------------
#
# This file is part of LazyAlgebra (https://github.com/emmt/LazyAlgebra.jl)
# released under the MIT "Expat" license.
#
# Copyright (c) 2017-2019 Éric Thiébaut.
#

# The idea is to generalize the dot product as follows:
#
#   `vdot(x,y)` yields the sum of `conj(x[i])*y[i]` for each `i` in
#               `eachindex(x,y)` providing `x` and `y` have the same dimensions
#               (i.e., same `indices`).
#
#   `A*x` yields the matrix-vector product providing that the trailing
#                dimensions of `A` match the dimensions of `x`.  The result has
#                the same dimensions as the leading dimensions of `A`.
#
# We may want to use fast BLAS routines.
#
# According to the following timings (for n = 96 and 4 threads), the fastest
# method is the BLAS version of `apply!(,Adjoint,,)`.  When looking at the
# loops, this is understandable as `apply!(,Adjoint,,)` is easier to
# parallelize than `apply!(,Direct,,)`.  Note that Julia implementations are
# with SIMD and no bounds checking.
#
#             A⋅x       A'.x      x'⋅y
#     ---------------------------------
#     BLAS   3.4 µs     2.0 µs   65 ns
#     Julia  4.5 µs    24.2 μs   65 ns

#import LinearAlgebra.BLAS
#import Compat.LinearAlgebra.BLAS: libblas, BlasInt, BlasReal, BlasFloat, BlasComplex, @blasfunc

const BlasVec{T} = Union{DenseVector{T},StridedVector{T}}
const BlasArr{T,N} = DenseArray{T,N}

for T in (Float32, Float64)

    @eval begin

        vdot(::Type{$T}, x::BlasVec{$T}, y::BlasVec{$T}) =
            __call_blas_dot(BLAS.dot, x, y)

        vdot(::Type{$T}, x::BlasArr{$T,N}, y::BlasArr{$T,N}) where {N} =
            __call_blas_dot(BLAS.dot, x, y)

        vdot(::Type{Complex{$T}}, x::BlasVec{Complex{$T}}, y::BlasVec{Complex{$T}}) =
            __call_blas_dot(BLAS.dotc, x, y)

        vdot(::Type{Complex{$T}}, x::BlasArr{Complex{$T},N}, y::BlasArr{Complex{$T},N}) where {N} =
            __call_blas_dot(BLAS.dotc, x, y)

    end

end

@inline function __call_blas_dot(f, x, y)
    size(x) == size(y) ||
        __baddims("`x` and `y` must have the same dimensions")
    return f(length(x), pointer(x), stride(x, 1), pointer(y), stride(y, 1))
end

function vupdate!(y::BlasVec{T}, alpha::Number,
                  x::BlasVec{T}) where {T<:BlasFloat}
    size(x) == size(y) ||
        __baddims("`x` and `y` must have the same dimensions")
    BLAS.axpy!(length(x), convert(T, alpha),
               pointer(x), stride(x, 1),
               pointer(y), stride(y, 1))
    return y
end
