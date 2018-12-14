#
# lgemv.jl -
#
# Loosely Generalized Matrix Vector Mutiplication.
#
#-------------------------------------------------------------------------------
#
# This file is part of LazyAlgebra (https://github.com/emmt/LazyAlgebra.jl)
# released under the MIT "Expat" license.
#
# Copyright (c) 2017-2018 Éric Thiébaut.
#

module LGEMV

export
    lgemv!,
    lgemv

using ..LazyAlgebra # for `allindices`, `is_flat_array`, etc.
using ..LazyAlgebra: LinearAlgebra, axes

using Compat

const BLAS = LinearAlgebra.BLAS
using .BLAS: libblas, @blasfunc, BlasInt, BlasReal, BlasFloat, BlasComplex

const Reals = AbstractFloat
const Complexes = Complex{<:AbstractFloat}
const Floats = Union{Reals,Complexes}

@static if !isdefined(Base, :AbstractChar)
    const AbstractChar = Char
end

"""
# Loosely Generalized Matrix Vector Mutiplication

```julia
lgemv([tr='N',] [α=1,] A, x) -> y
```

yields `y = α*tr(A)*x` where `tr(A)` is `A` if `tr` is `'N'`, `transpose(A)` if
`tr` is `'T'` and `adjoint(A)` if `tr` is `'C'`.  The expression `tr(A)*x` is a
matrix-vector multiplication but loosely interpreted by grouping consecutive
dimensions of `A` and `x` as follows:

- if `tr` is `'N'`, the trailing dimensions of `x` must match those of `A` and
  the leading dimensions of `A` are those of the result `y`;

- if `tr` is `'T'` or `'C'`, the leading dimensions of `x` must match those of
  `A` and the trailing dimensions of `A` are those of the result `y`.

The in-place version is called by:

```julia
lgemv!([tr='N',] [α=1,] A, x, [β=0,] y) -> y
```

and overwrites the contents of `y` with `α*tr(A)*x + β*y`.  Note that `x` and
`y` must not be aliased.

The multipliers `α` and `β` must be both specified or omitted, they can be any
scalar numbers but are respectively converted to
`promote_type(eltype(A),eltype(x))` and `eltype(y)` which may throw an
`InexactError` exception.

See also: [`LinearAlgebra.BLAS.gemv`](@ref), [`LinearAlgebra.BLAS.gemv!`](@ref).

"""
function lgemv(trans::AbstractChar,
               α::Number,
               A::AbstractArray{Ta,Na},
               x::AbstractArray{Tx,Nx}) where {Ta<:Floats,Na,
                                               Tx<:Floats,Nx}
    # Check that indices are compatible and detremine the shape of the output.
    # Then call the low-level method.
    1 ≤ Nx < Na || incompatible_dimensions()
    Ny = Na - Nx
    if trans == 'N'
        # Non-transposed matrix.  Trailing dimensions of X must match those of
        # A, leading dimensions of A are those of the result.
        @inbounds for d in 1:Nx
            axes(x, d) == axes(A, Ny + d) || incompatible_dimensions()
        end
        shape = ntuple(d -> axes(A, d), Val(Ny))
    elseif trans == 'T' || trans == 'C'
        # Transposed matrix.  Leading dimensions of X must match those of A,
        # trailing dimensions of A are those of the result.
        @inbounds for d in 1:Nx
            axes(x, d) == axes(A, d) || incompatible_dimensions()
        end
        shape = ntuple(d -> axes(A, Nx + d), Val(Ny))
    else
        invalid_transpose_character()
    end
    T = promote_type(Ta, Tx)
    _lgemv!(trans, α, A, x, zero(T), similar(A, T, shape))
end

function lgemv!(trans::AbstractChar,
                α::Number,
                A::AbstractArray{<:Floats,Na},
                x::AbstractArray{<:Floats,Nx},
                β::Number,
                y::AbstractArray{<:Floats,Ny}) where {Na,Nx,Ny}
    # Check that indices are compatible.  Then call the low-level method.
    Na == Nx + Ny || incompatible_dimensions()
    if trans == 'N'
        @inbounds for d in 1:Ny
            axes(y, d) == axes(A, d) || incompatible_dimensions()
        end
        @inbounds for d in 1:Nx
            axes(x, d) == axes(A, Ny + d) || incompatible_dimensions()
        end
    elseif trans == 'T' || trans == 'C'
        @inbounds for d in 1:Nx
            axes(x, d) == axes(A, d) || incompatible_dimensions()
        end
        @inbounds for d in 1:Ny
            axes(y, d) == axes(A, Nx + d) || incompatible_dimensions()
        end
    else
        invalid_transpose_character()
    end
    _lgemv!(trans, α, A, x, β, y)
end

@doc @doc(lgemv) lgemv!

lgemv(A::AbstractArray, x::AbstractArray) =
    lgemv('N', 1, A, x)

lgemv!(A::AbstractArray, x::AbstractArray, y::AbstractArray) =
    lgemv!('N', 1, A, x, 0, y)

lgemv(trans::AbstractChar, A::AbstractArray, x::AbstractArray) =
    lgemv(trans, 1, A, x)

lgemv!(trans::AbstractChar, A::AbstractArray, x::AbstractArray, y::AbstractArray) =
    lgemv!(trans, 1, A, x, 0, y)

lgemv(α::Number, A::AbstractArray, x::AbstractArray) =
    lgemv('N', α, A, x)

lgemv!(α::Number, A::AbstractArray, x::AbstractArray, β::Number, y::AbstractArray) =
    lgemv!('N', α, A, x, β, y)

@static if isdefined(LinearAlgebra, :Transpose)
    lgemv(A::LinearAlgebra.Transpose, x::AbstractArray) =
        lgemv('T', 1, A.parent, x)
    lgemv!(A::LinearAlgebra.Transpose, x::AbstractArray, y::AbstractArray) =
        lgemv!('T', 1, A.parent, x, 0, y)
    lgemv(α::Number, A::LinearAlgebra.Transpose, x::AbstractArray) =
        lgemv('T', α, A.parent, x)
    lgemv!(α::Number, A::LinearAlgebra.Transpose, x::AbstractArray, β::Number, y::AbstractArray) =
        lgemv!('T', α, A.parent, x, β, y)
end

@static if isdefined(LinearAlgebra, :Adjoint)
    lgemv(A::LinearAlgebra.Adjoint, x::AbstractArray) =
        lgemv('C', 1, A.parent, x)
    lgemv!(A::LinearAlgebra.Adjoint, x::AbstractArray, y::AbstractArray) =
        lgemv!('C', 1, A.parent, x, 0, y)
    lgemv(α::Number, A::LinearAlgebra.Adjoint, x::AbstractArray) =
        lgemv('C', α, A.parent, x)
    lgemv!(α::Number, A::LinearAlgebra.Adjoint, x::AbstractArray, β::Number, y::AbstractArray) =
        lgemv!('C', α, A.parent, x, β, y)
end

incompatible_dimensions() =
    throw(DimensionMismatch("incompatible dimensions for generalized matrix-vector multiplication"))

invalid_transpose_character() =
    throw(ArgumentError("invalid transpose character"))

function _lgemv!(trans::AbstractChar,
                 α::Number,
                 A::AbstractArray{Ta},
                 x::AbstractArray{Tx},
                 β::Number,
                 y::AbstractArray{Ty}) where {Ta<:Floats,
                                              Tx<:Floats,
                                              Ty<:Floats}
    T = promote_type(Ta, Tx)
    flat = is_flat_array(A, x, y)
    _lgemv!(Val(flat), trans, convert(T, α), A, x, convert(Ty, β), y)
end

# Reference (non-BLAS) version for non-flat arrays.  First argument can be any
# Val but Val(true) to force using this version even though other versions may
# be faster.  Multiplier α is assumed to have proper type, that is
# promote_type(Ta, Tx).  Loops are ordered assuming the coefficients of A have
# column-major storage order.
function _lgemv!(::Val,
                 trans::AbstractChar,
                 α::Floats,
                 A::AbstractArray{Ta},
                 x::AbstractArray{Tx},
                 β::Ty,
                 y::AbstractArray{Ty}) where {Ta<:Floats,
                                              Tx<:Floats,
                                              Ty<:Floats}
    T = promote_type(Ta, Tx)
    typeof(α) == T || error("bad type for multiplier α")
    if α == zero(α)
        _scale!(y, β)
    elseif trans == 'N'
        if β != one(β)
            _scale!(y, β)
        end
        rows, cols = allindices(y), allindices(x)
        @inbounds for j in cols
            axj = α*x[j]
            if axj != zero(axj)
                @simd for i in rows
                    y[i] += A[i,j]*axj
                end
            end
        end
    else
        rows, cols = allindices(x), allindices(y)
        if β == zero(β)
            if Ta <: Reals || trans == 'T'
                @inbounds for j in cols
                    s = zero(T)
                    @simd for i in rows
                        s += A[i,j]*x[i]
                    end
                    y[j] = α*s
                end
            else
                @inbounds for j in cols
                    s = zero(T)
                    @simd for i in rows
                        s += conj(A[i,j])*x[i]
                    end
                    y[j] = α*s
                end
            end
        else
            if Ta <: Reals || trans == 'T'
                @inbounds for j in cols
                    s = zero(T)
                    @simd for i in rows
                        s += A[i,j]*x[i]
                    end
                    y[j] = α*s + β*y[j]
                end
            else
                @inbounds for j in cols
                    s = zero(T)
                    @simd for i in rows
                        s += conj(A[i,j])*x[i]
                    end
                    y[j] = α*s + β*y[j]
                end
            end
        end
    end
    return y
end

# Reference (non-BLAS) version for flat arrays.  Multiplier α is assumed to
# have proper type, that is promote_type(Ta, Tx).  Loops are ordered assuming
# the coefficients of A have column-major storage order.
function _lgemv!(::Val{true},
                 trans::AbstractChar,
                 α::Floats,
                 A::AbstractArray{Ta},
                 x::AbstractArray{Tx},
                 β::Ty,
                 y::AbstractArray{Ty}) where {Ta<:Floats,
                                              Tx<:Floats,
                                              Ty<:Floats}
    T = promote_type(Ta, Tx)
    typeof(α) == T || error("bad type for multiplier α")
    if α == zero(α)
        _scale!(y, β)
    elseif trans == 'N'
        if β != one(β)
            _scale!(y, β)
        end
        nrows, ncols = length(y), length(x)
        @inbounds for j in 1:ncols
            axj = α*x[j]
            if axj != zero(axj)
                off = (j - 1)*nrows
                @simd for i in 1:nrows
                    y[i] += A[off + i]*axj
                end
            end
        end
    else
        nrows, ncols = length(x), length(y)
        if β == zero(β)
            if Ta <: Reals || trans == 'T'
                @inbounds for j in 1:ncols
                    off = (j - 1)*nrows
                    s = zero(T)
                    @simd for i in 1:nrows
                        s += A[off + i]*x[i]
                    end
                    y[j] = α*s
                end
            else
                @inbounds for j in 1:ncols
                    off = (j - 1)*nrows
                    s = zero(T)
                    @simd for i in 1:nrows
                        s += conj(A[off + i])*x[i]
                    end
                    y[j] = α*s
                end
            end
        else
            if Ta <: Reals || trans == 'T'
                @inbounds for j in 1:ncols
                    off = (j - 1)*nrows
                    s = zero(T)
                    @simd for i in 1:nrows
                        s += A[off + i]*x[i]
                    end
                    y[j] = α*s + β*y[j]
                end
            else
                @inbounds for j in 1:ncols
                    off = (j - 1)*nrows
                    s = zero(T)
                    @simd for i in 1:nrows
                        s += conj(A[off + i])*x[i]
                    end
                    y[j] = α*s + β*y[j]
                end
            end
        end
    end
    return y
end

# BLAS version.  The differences with LinearAlgebra.BLAS.gemv! are that inputs
# are assumed to be flat arrays (see is_flat_array) and that multipliers are
# automatically converted.
for (fname, elty) in ((:dgemv_,:Float64),
                      (:sgemv_,:Float32),
                      (:zgemv_,:ComplexF64),
                      (:cgemv_,:ComplexF32))
    @eval begin
        #SUBROUTINE DGEMV(TRANS,M,N,ALPHA,A,LDA,X,INCX,BETA,Y,INCY)
        #*     .. Scalar Arguments ..
        #      DOUBLE PRECISION ALPHA,BETA
        #      INTEGER INCX,INCY,LDA,M,N
        #      CHARACTER TRANS
        #*     .. Array Arguments ..
        #      DOUBLE PRECISION A(LDA,*),X(*),Y(*)
        function _lgemv!(::Val{true},
                         trans::AbstractChar,
                         α::($elty), # FIXME: can just be Number
                         A::AbstractArray{$elty},
                         x::AbstractArray{$elty},
                         β::($elty), # FIXME: can just be Number
                         y::AbstractArray{$elty})
            if trans == 'N'
                nrows, ncols = length(y), length(x)
            else
                nrows, ncols = length(x), length(y)
            end
            length(A) == nrows*ncols ||
                throw(DimensionMismatch("incompatible sizes")) # FIXME: this should never occurs
            ccall((@blasfunc($fname), libblas), Cvoid,
                  (Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt}, Ref{$elty},
                   Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                   Ref{$elty}, Ptr{$elty}, Ref{BlasInt}),
                  trans, nrows, ncols, α, A, nrows, x, 1, β, y, 1)
            return y
        end
    end
end

# FIXME: call vscale!(y, β)
function _scale!(A::AbstractArray{T}, α::T) where {T<:Floats}
    if α == zero(T)
        fill!(A, zero(T))
    elseif α != one(T)
        @inbounds @simd for i in eachindex(A)
            A[i] *= α
        end
    end
end

end # module
