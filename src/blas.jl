#
# blas.jl -
#
# Code based on BLAS (Basic Linear Algebra Subroutines).
#
#-------------------------------------------------------------------------------
#
# This file is part of the LazyAlgebra package released under the MIT "Expat"
# license.
#
# Copyright (c) 2017-2018 Éric Thiébaut.
#

# The idea is to generalize the dot product as follows:
#
#   `vdot(x,y)` yields the sum of `conj(x[i])*y[i]` for each `i` in
#               `eachindex(x,y)` provided `x` and `y` have the same dimensions
#               (i.e., same `indices`).
#
#   `A*x` yields the matrix-vector product provided that the trailing
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

import Base.BLAS
import Base.BLAS: libblas, BlasInt, BlasReal, BlasFloat, BlasComplex, @blasfunc

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

function apply!(α::Number,
                ::Type{Direct},
                A::DenseArray{T},
                x::DenseArray{T},
                β::Number,
                y::DenseArray{T}) where {T<:BlasFloat}
    (size(y)..., size(x)...) == size(A) ||
        __baddims("the dimensions of `y` and `x` must match those of `A`")
    m, n = length(y), length(x)
    return __gemv!('N', m, n, convert(T, α), A, m, x, 1, convert(T, β), y, 1)
end

function apply!(α::Number,
                ::Type{Adjoint},
                A::DenseArray{T},
                x::DenseArray{T},
                β::Number,
                y::DenseArray{T}) where {T<:BlasFloat}
    (size(x)..., size(y)...) == size(A) ||
        __baddims("the dimensions of `x` and `y` must match those of `A`")
    m, n = length(x), length(y)
    return __gemv!('C', m, n, convert(T, α), A, m, x, 1, convert(T, β), y, 1)
end

# Wrappers for BLAS level 2 GEMV routine, assuming arguments have been checked
# by the caller.  This is to allow for a wider interpretation of the matrix
# vector product.
for (fname, elty) in ((:dgemv_,:Float64),
                      (:sgemv_,:Float32),
                      (:zgemv_,:Complex128),
                      (:cgemv_,:Complex64))
    @eval begin
        #SUBROUTINE xGEMV(TRANS,M,N,ALPHA,A,LDA,X,INCX,BETA,Y,INCY)
        #*     .. Scalar Arguments ..
        #      DOUBLE PRECISION ALPHA,BETA
        #      INTEGER INCX,INCY,LDA,M,N
        #      CHARACTER TRANS
        #*     .. Array Arguments ..
        #      DOUBLE PRECISION A(LDA,*),X(*),Y(*)
        function __gemv!(trans::Char, m::Int, n::Int, alpha::($elty),
                         A::DenseArray{$elty}, lda::Int,
                         x::DenseArray{$elty}, incx::Int, beta::($elty),
                         y::DenseArray{$elty}, incy::Int)
            ccall((@blasfunc($fname), libblas), Void,
                  (Ptr{UInt8}, Ptr{BlasInt}, Ptr{BlasInt}, Ptr{$elty},
                   Ptr{$elty}, Ptr{BlasInt}, Ptr{$elty}, Ptr{BlasInt},
                   Ptr{$elty}, Ptr{$elty}, Ptr{BlasInt}),
                  &trans, &m, &n, &alpha, A, &max(1,lda), x, &incx,
                  &beta, y, &incy)
            return y
        end
    end
end
