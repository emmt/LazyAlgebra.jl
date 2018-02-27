#
# blas.jl -
#
# Code based on BLAS (Basic Linear Algebra Subroutines).
#
#-------------------------------------------------------------------------------
#
# This file is part of the MockAlgebra package released under the MIT "Expat"
# license.
#
# Copyright (c) 2017-2018 Éric Thiébaut.
#

# The idea is to generalize the dot product as follows:
#
#   `vdot(x,y)` yields the sum of `x[i]*y[i]` for each `i` in `eachindex(x,y)`
#               provided `x` and `y` have the same dimensions (i.e., same
#               `indices`).
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
import Base.BLAS: libblas, BlasInt, BlasReal, @blasfunc

for (fname, elty) in ((:ddot_,:Float64),
                      (:sdot_,:Float32))
    @eval begin
                #       DOUBLE PRECISION FUNCTION DDOT(N,DX,INCX,DY,INCY)
                # *     .. Scalar Arguments ..
                #       INTEGER INCX,INCY,N
                # *     ..
                # *     .. Array Arguments ..
                #       DOUBLE PRECISION DX(*),DY(*)
        function _dot(n::Integer,
                      x::Union{Ptr{$elty},DenseArray{$elty}}, incx::Integer,
                      y::Union{Ptr{$elty},DenseArray{$elty}}, incy::Integer)
            ccall((@blasfunc($fname), libblas), $elty,
                  (Ptr{BlasInt}, Ptr{$elty}, Ptr{BlasInt},
                   Ptr{$elty}, Ptr{BlasInt}), &n, x, &incx, y, &incy)
        end
    end
end

function blas_vdot(x::Union{DenseArray{T},StridedVector{T}},
                   y::Union{DenseArray{T},StridedVector{T}}
                   ) where {T <: BlasReal}
    if (n = length(x)) != length(y)
        throw(DimensionMismatch("`x` and `y` must have the same length"))
    end
    return _dot(n, pointer(x), stride(x, 1), pointer(y), stride(y, 1))
end

if USE_BLAS_DOT
    # Use BLAS whenever possible.
    function vdot(x::Union{DenseArray{T},StridedVector{T}},
                  y::Union{DenseArray{T},StridedVector{T}}
                  ) where {T <: BlasReal}
        return blas_vdot(x, y)
    end
    function vdot(::Type{T},
                  x::Union{DenseArray{T},StridedVector{T}},
                  y::Union{DenseArray{T},StridedVector{T}}
                  ) where {T <: BlasReal}
        return blas_vdot(x, y)
    end
end

function blas_vupdate!(y::Union{DenseArray{T},StridedVector{T}},
                       alpha::Number,
                       x::Union{DenseArray{T},StridedVector{T}}
                       ) where {T <: BlasReal}
    return BLAS.axpy!(alpha, x, y)
end

if USE_BLAS_AXPY
    # Use BLAS whenever possible.
    function vupdate!(y::Union{DenseArray{T},StridedVector{T}},
                      alpha::Number,
                      x::Union{DenseArray{T},StridedVector{T}}
                      ) where {T <: BlasReal}
        return blas_vupdate!(y, alpha, x)
    end
end

function blas_apply!(y::DenseArray{T},
                     ::Type{Direct},
                     A::DenseArray{T},
                     x::DenseArray{T}
                     ) where {T <: BlasReal}
    @assert size(A) == (size(y)..., size(x)...)
    m, n = length(y), length(x)
    return _gemv!('N', m, n, one(T), A, m, x, 1, zero(T), y, 1)
end

function blas_apply!(y::DenseArray{T},
                     ::Type{Adjoint},
                     A::DenseArray{T},
                     x::DenseArray{T}) where {T <: BlasReal}
    @assert size(A) == (size(x)..., size(y)...)
    m, n = length(x), length(y)
    return _gemv!('T', m, n, one(T), A, m, x, 1, zero(T), y, 1)
end

# Wrappers for BLAS level 2 GEMV routine, assuming arguments have been checked
# by the caller.  This is to allow for a wider interpretation of the matrix
# vector product.
for (fname, elty) in ((:dgemv_,:Float64),
                      (:sgemv_,:Float32))
    @eval begin
        #SUBROUTINE DGEMV(TRANS,M,N,ALPHA,A,LDA,X,INCX,BETA,Y,INCY)
        #*     .. Scalar Arguments ..
        #      DOUBLE PRECISION ALPHA,BETA
        #      INTEGER INCX,INCY,LDA,M,N
        #      CHARACTER TRANS
        #*     .. Array Arguments ..
        #      DOUBLE PRECISION A(LDA,*),X(*),Y(*)
        function _gemv!(trans::Char, m::Int, n::Int, alpha::($elty),
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

if USE_BLAS_GEMV
    # Use BLAS whenever possible.
    function apply!(y::DenseArray{R},
                    ::Type{Op},
                    A::DenseArray{R},
                    x::DenseArray{R}
                    ) where {R<:BlasReal, Op<:Union{Direct,Adjoint}}
        return blas_apply!(y, Op, A, x)
    end
end
