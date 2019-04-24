#
# lgemv.jl -
#
# Lazily Generalized Matrix-Vector mutiplication.
#
#-------------------------------------------------------------------------------
#
# This file is part of LazyAlgebra (https://github.com/emmt/LazyAlgebra.jl)
# released under the MIT "Expat" license.
#
# Copyright (c) 2017-2019 Éric Thiébaut.
#

"""
# Lazily Generalized Matrix-Vector mutiplication

```julia
lgemv([α=1,] [tr='N',] A, x) -> y
```

yields `y = α*op(A)*x` where `op(A)` is `A` if `tr` is `'N'`, `transpose(A)` if
`tr` is `'T'` and `adjoint(A)` if `tr` is `'C'`.  The expression `op(A)*x` is a
matrix-vector multiplication but interpreted by grouping consecutive dimensions
of `A` and `x` as follows:

- if `tr` is `'N'`, the trailing dimensions of `x` must match those of `A` and
  the leading dimensions of `A` are those of the result `y`;

- if `tr` is `'T'` or `'C'`, the leading dimensions of `x` must match those of
  `A` and the trailing dimensions of `A` are those of the result `y`.

The in-place version is called by:

```julia
lgemv!([α=1,] [tr='N',] A, x, [β=0,] y) -> y
```

and overwrites the contents of `y` with `α*op(A)*x + β*y`.  Note that `x` and
`y` must not be aliased.

The multipliers `α` and `β` must be both specified or omitted, they can be any
scalar numbers but are respectively converted to
`promote_type(eltype(A),eltype(x))` and `eltype(y)` which may throw an
`InexactError` exception.

See also: [`lgemm`](@ref), [`LinearAlgebra.BLAS.gemv`](@ref),
[`LinearAlgebra.BLAS.gemv!`](@ref).

"""
function lgemv(α::Number,
               trans::Char,
               A::AbstractArray{<:Floats},
               x::AbstractArray{<:Floats})
    return _lgemv(Implementation(Val(:lgemv), α, trans, A, x),
                  α, trans, A, x)
end

function lgemv!(α::Number,
                trans::Char,
                A::AbstractArray{<:Floats},
                x::AbstractArray{<:Floats},
                β::Number,
                y::AbstractArray{<:Floats})
    return _lgemv!(Implementation(Val(:lgemv), α, trans, A, x, β, y),
                   α, trans, A, x, β, y)
end

@doc @doc(lgemv) lgemv!

lgemv(A::AbstractArray, x::AbstractArray) = lgemv(1, 'N', A, x)

lgemv!(A::AbstractArray, x::AbstractArray, y::AbstractArray) =
    lgemv!(1, 'N', A, x, 0, y)

lgemv(trans::Char, A::AbstractArray, x::AbstractArray) =
    lgemv(1, trans, A, x)

lgemv!(trans::Char, A::AbstractArray, x::AbstractArray,
       y::AbstractArray) = lgemv!(1, trans, A, x, 0, y)

lgemv(α::Number, A::AbstractArray, x::AbstractArray) = lgemv(α, 'N', A, x)

lgemv!(α::Number, A::AbstractArray, x::AbstractArray, β::Number,
       y::AbstractArray) = lgemv!(α, 'N', A, x, β, y)

@static if isdefined(LinearAlgebra, :Transpose)
    lgemv(A::LinearAlgebra.Transpose, x::AbstractArray) =
        lgemv(1, 'T', A.parent, x)
    lgemv!(A::LinearAlgebra.Transpose, x::AbstractArray, y::AbstractArray) =
        lgemv!(1, 'T', A.parent, x, 0, y)
    lgemv(α::Number, A::LinearAlgebra.Transpose, x::AbstractArray) =
        lgemv(α, 'T', A.parent, x)
    lgemv!(α::Number, A::LinearAlgebra.Transpose, x::AbstractArray,
           β::Number, y::AbstractArray) = lgemv!(α, 'T', A.parent, x, β, y)
end

@static if isdefined(LinearAlgebra, :Adjoint)
    lgemv(A::LinearAlgebra.Adjoint, x::AbstractArray) =
        lgemv(1, 'C', A.parent, x)
    lgemv!(A::LinearAlgebra.Adjoint, x::AbstractArray, y::AbstractArray) =
        lgemv!(1, 'C', A.parent, x, 0, y)
    lgemv(α::Number, A::LinearAlgebra.Adjoint, x::AbstractArray) =
        lgemv(α, 'C', A.parent, x)
    lgemv!(α::Number, A::LinearAlgebra.Adjoint, x::AbstractArray, β::Number,
           y::AbstractArray) = lgemv!(α, 'C', A.parent, x, β, y)
end

# Best implementations for lgemv and lgemv!

for (atyp, eltyp) in ((Real,   BlasReal),
                      (Number, BlasComplex))
    @eval begin
        function Implementation(::Val{:lgemv},
                                α::$atyp,
                                trans::Char,
                                A::Matrix{T},
                                x::Vector{T}) where {T<:$eltyp}
            return Blas()
        end

        function Implementation(::Val{:lgemv},
                                α::$atyp,
                                trans::Char,
                                A::Matrix{T},
                                x::Vector{T},
                                β::$atyp,
                                y::Vector{T}) where {T<:$eltyp}
            return Blas()
        end

        function Implementation(::Val{:lgemv},
                                α::$atyp,
                                trans::Char,
                                A::AbstractMatrix{T},
                                x::AbstractVector{T}) where {T<:$eltyp}
            return (isflatarray(A, x) ? Blas() : Basic())
        end

        function Implementation(::Val{:lgemv},
                                α::$atyp,
                                trans::Char,
                                A::AbstractMatrix{T},
                                x::AbstractVector{T},
                                β::$atyp,
                                y::AbstractVector{T}) where {T<:$eltyp}
            return (isflatarray(A, x, y) ? Blas() : Basic())
        end

        function Implementation(::Val{:lgemv},
                                α::$atyp,
                                trans::Char,
                                A::Array{T},
                                x::Array{T}) where {T<:$eltyp}
            return Blas()
        end

        function Implementation(::Val{:lgemv},
                                α::$atyp,
                                trans::Char,
                                A::Array{T},
                                x::Array{T},
                                β::$atyp,
                                y::Array{T}) where {T<:$eltyp}
            return Blas()
        end

        function Implementation(::Val{:lgemv},
                                α::$atyp,
                                trans::Char,
                                A::AbstractArray{T},
                                x::AbstractArray{T}) where {T<:$eltyp}
            return (isflatarray(A, x) ? Blas() : Basic())
        end

        function Implementation(::Val{:lgemv},
                                α::$atyp,
                                trans::Char,
                                A::AbstractArray{T},
                                x::AbstractArray{T},
                                β::$atyp,
                                y::AbstractArray{T}) where {T<:$eltyp}
            return (isflatarray(A, x, y) ? Blas() : Basic())
        end
    end
end

function Implementation(::Val{:lgemv},
                        α::Number,
                        trans::Char,
                        A::AbstractMatrix,
                        x::AbstractVector)
    return Basic()
end

function Implementation(::Val{:lgemv},
                        α::Number,
                        trans::Char,
                        A::AbstractMatrix,
                        x::AbstractVector,
                        β::Number,
                        y::AbstractVector)
    return Basic()
end

function Implementation(::Val{:lgemv},
                        α::Number,
                        trans::Char,
                        A::AbstractArray,
                        x::AbstractArray)
    return (isflatarray(A, x) ? Linear() : Generic())
end

function Implementation(::Val{:lgemv},
                        α::Number,
                        trans::Char,
                        A::AbstractArray,
                        x::AbstractArray,
                        β::Number,
                        y::AbstractArray)
    return (isflatarray(A, x, y) ? Linear() : Generic())
end

# BLAS implementations for (generalized) matrices and vectors.
# Linear indexing is assumed (this must have been checked before).

function _lgemv(::Blas,
                α::Number,
                trans::Char,
                A::AbstractArray{T},
                x::AbstractArray{T}) where {T<:BlasFloat}
    nrows, ncols, shape = _lgemv_dims(trans, A, x)
    return _blas_lgemv!(nrows, ncols, convert(T, α), trans, A, x,
                        zero(T), Array{T}(undef, shape))
end

function _lgemv!(::Blas,
                 α::Number,
                 trans::Char,
                 A::AbstractArray{T},
                 x::AbstractArray{T},
                 β::Number,
                 y::AbstractArray{T}) where {T<:BlasFloat}
    nrows, ncols = _lgemv_dims(trans, A, x, y)
    return _blas_lgemv!(nrows, ncols, convert(T, α), trans, A, x,
                        convert(T, β), y)
end

# Julia implementations for (generalized) matrices and vectors.
# Linear indexing is assumed (this must have been checked before).

function _lgemv(::Linear,
                α::Number,
                trans::Char,
                A::AbstractArray{Ta},
                x::AbstractArray{Tx}) where {Ta<:Floats,
                                             Tx<:Floats}
    nrows, ncols, shape = _lgemv_dims(trans, A, x)
    Tax, Ty = _lgemv_types(α, Ta, Tx)
    return _linear_lgemv!(nrows, ncols,
                          convert_multiplier(α, Tax, Ty), trans, A, x,
                          convert_multiplier(0, Ty), Array{Ty}(undef, shape))
end

function _lgemv!(::Linear,
                 α::Number,
                 trans::Char,
                 A::AbstractArray{Ta},
                 x::AbstractArray{Tx},
                 β::Number,
                 y::AbstractArray{Ty}) where {Ta<:Floats,
                                               Tx<:Floats,
                                               Ty<:Floats}
    nrows, ncols = _lgemv_dims(trans, A, x, y)
    Tax = promote_type(Ta, Tx)
    return _linear_lgemv!(nrows, ncols,
                          convert_multiplier(α, Tax, Ty), trans, A, x,
                          convert_multiplier(β, Ty), y)
end

# Basic Julia implementations when vectors and matrices are, respectively, 1D
# and 2D arrays.

function _lgemv(::Basic,
                α::Number,
                trans::Char,
                A::AbstractMatrix{Ta},
                x::AbstractVector{Tx}) where {Ta<:Floats,Tx<:Floats}
    rows, cols = _lgemv_indices(trans, A, x)
    Tax, Ty = _lgemv_types(α, Ta, Tx)
    return _generic_lgemv!(rows, cols,
                           convert_multiplier(α, Tax, Ty), trans, A, x,
                           convert_multiplier(0, Ty),
                           similar(Array{Ty}, trans == 'N' ? rows : cols))
end

function _lgemv!(::Basic,
                 α::Number,
                 trans::Char,
                 A::AbstractMatrix{Ta},
                 x::AbstractVector{Tx},
                 β::Number,
                 y::AbstractVector{Ty}) where {Ta<:Floats,Tx<:Floats,Ty<:Floats}
    rows, cols = _lgemv_indices(trans, A, x, y)
    Tax = promote_type(Ta, Tx)
    return _generic_lgemv!(rows, cols,
                           convert_multiplier(α, Tax, Ty), trans, A, x,
                           convert_multiplier(β, Ty), y)
end

# Generic implementations for any other cases.

function _lgemv(::Generic,
                α::Number,
                trans::Char,
                A::AbstractArray{Ta,Na},
                x::AbstractArray{Tx,Nx}) where {Ta<:Floats,Na,
                                                Tx<:Floats,Nx}
    rows, cols = _lgemv_indices(trans, A, x)
    Tax, Ty = _lgemv_types(α, Ta, Tx)
    return _generic_lgemv!(allindices(rows), allindices(cols),
                           convert_multiplier(α, Tax, Ty), trans, A, x,
                           convert_multiplier(0, Ty),
                           similar(Array{Ty}, trans == 'N' ? rows : cols))
end

function _lgemv!(::Generic,
                 α::Number,
                 trans::Char,
                 A::AbstractArray{Ta,Na},
                 x::AbstractArray{Tx,Nx},
                 β::Number,
                 y::AbstractArray{Ty,Ny}) where {Ta<:Floats,Na,
                                                 Tx<:Floats,Nx,
                                                 Ty<:Floats,Ny}
    rows, cols = _lgemv_indices(trans, A, x, y)
    Tax = promote_type(Ta, Tx)
    return _generic_lgemv!(allindices(rows), allindices(cols),
                           convert_multiplier(α, Tax, Ty), trans, A, x,
                           convert_multiplier(β, Ty), y)
end

#
# Call low-level BLAS version.  The differences with LinearAlgebra.BLAS.gemv!
# are that inputs are assumed to be flat arrays (see isflatarray) and that
# multipliers are automatically converted.
#
for (f, T) in ((:dgemv_, Float64),
               (:sgemv_, Float32),
               (:zgemv_, ComplexF64),
               (:cgemv_, ComplexF32))
    @eval begin
        #
        # FORTRAN prototype:
        #     SUBROUTINE ${pfx}GEMV(TRANS,M,N,ALPHA,A,LDA,X,INCX,BETA,Y,INCY)
        #         ${T} ALPHA,BETA
        #         INTEGER M,N,LDA,INCX,INCY
        #         CHARACTER TRANS
        #         ${T} A(LDA,*),X(*),Y(*)
        #
        # Scalar arguments, α and β, can just be `Number` and integer arguments
        # can just be `Integer` but we want to keep the signature strict
        # because it is a low-level private method.
        #
        function _blas_lgemv!(nrows::Int,
                              ncols::Int,
                              α::($T),
                              trans::Char,
                              A::AbstractArray{$T},
                              x::AbstractArray{$T},
                              β::($T),
                              y::AbstractArray{$T})
            #@static if DEBUG
            #    (length(x) == (trans == 'N' ? ncols : nrows) &&
            #     length(y) == (trans == 'N' ? nrows : ncols) &&
            #     length(A) == nrows*ncols) ||
            #     throw(DimensionMismatch("incompatible sizes"))
            #end
            ccall((@blasfunc($f), libblas), Cvoid,
                  (Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt}, Ref{$T},
                   Ptr{$T}, Ref{BlasInt}, Ptr{$T}, Ref{BlasInt},
                   Ref{$T}, Ptr{$T}, Ref{BlasInt}),
                  trans, nrows, ncols, α, A, nrows, x, 1, β, y, 1)
            return y
        end
    end
end
#
# Reference (non-BLAS) version for *flat* arrays which can be linearly indexed.
# Loops are ordered assuming the coefficients of A have column-major storage
# order.
#
function _linear_lgemv!(nrows::Int, ncols::Int,
                        α::Floats,
                        trans::Char,
                        A::AbstractArray{Ta},
                        x::AbstractArray{Tx},
                        β::Floats,
                        y::AbstractArray{Ty}) where {Ta<:Floats,
                                                     Tx<:Floats,
                                                     Ty<:Floats}
    if α == 0 || trans == 'N'
        # Form: y := β⋅y
        if β == 0
            @inbounds @simd for k in eachindex(y)
                y[k] = zero(Ty)
            end
        elseif β != 1
            @inbounds @simd for k in eachindex(y)
                y[k] *= β
            end
        end
    end
    if α != 0
        if trans == 'N'
            @inbounds for j in 1:ncols
                temp = α*x[j]
                if temp != zero(temp)
                    off = (j - 1)*nrows
                    @simd for i in 1:nrows
                        y[i] += A[off + i]*temp
                    end
                end
            end
        else
            T = promote_type(Ta, Tx)
            if Ta <: Reals || trans == 'T'
                @inbounds for j in 1:ncols
                    off = (j - 1)*nrows
                    temp = zero(T)
                    @simd for i in 1:nrows
                        temp += A[off + i]*x[i]
                    end
                    y[j] = (β == 0 ? α*temp : α*temp + β*y[j])
                end
            else
                @inbounds for j in 1:ncols
                    off = (j - 1)*nrows
                    temp = zero(T)
                    @simd for i in 1:nrows
                        temp += conj(A[off + i])*x[i]
                    end
                    y[j] = (β == 0 ? α*temp : α*temp + β*y[j])
                end
            end
        end
    end
    return y
end
#
# At the lowest level, the same code can serve for the very general case
# (multi-dimensional Cartesian indices) and the basic case (A is a 2D array
# while x and y are both vectors).
#
# The elements of A are accessed sequentially with one pass through A assuming
# they are stored in colum-major order.
#
function _generic_lgemv!(I, J,
                         α::Floats,
                         trans::Char,
                         A::AbstractArray{Ta},
                         x::AbstractArray{Tx},
                         β::Floats,
                         y::AbstractArray{Ty}) where {Ta<:Floats,
                                                      Tx<:Floats,
                                                      Ty<:Floats}
    if α == 0 || trans == 'N'
        # Form: y := β⋅y
        if β == 0
            @inbounds @simd for k in eachindex(y)
                y[k] = zero(Ty)
            end
        elseif β != 1
            @inbounds @simd for k in eachindex(y)
                y[k] *= β
            end
        end
    end
    if α != 0
        if trans == 'N'
            #
            # Form  y := α*A*x + y.
            #
            @inbounds for j in J
                temp = α*x[j]
                if temp != zero(temp)
                    @simd for i in I
                        y[i] += A[i,j]*temp
                    end
                end
            end
        else
            T = promote_type(Ta,Tx)
            if Ta <: Real || trans == 'T'
                #
                # Form  y := α*A^T*x + y
                #
                @inbounds for j in J
                    temp = zero(T)
                    @simd for i in I
                        temp += A[i,j]*x[i]
                    end
                    y[j] = (β == 0 ? α*temp : α*temp + β*y[j])
                end
            else
                #
                # Form  y := α*A^H*x + y.
                #
                @inbounds for j in J
                    temp = zero(T)
                    @simd for i in I
                        temp += conj(A[i,j])*x[i]
                    end
                    y[j] = (β == 0 ? α*temp : α*temp + β*y[j])
                end
            end
        end
    end
    return y
end
#
# This method yields promote_type(Ta, Tx) and Ty, the type of the elements for
# the result of lgemv.
#
@inline function _lgemv_types(α::Real, ::Type{Ta},
                              ::Type{Tx}) where {Ta<:Floats,Tx<:Floats}
    Tax = promote_type(Ta, Tx)
    return Tax, Tax
end
#
@inline function _lgemv_types(α::Complex, ::Type{Ta},
                              ::Type{Tx}) where {Ta<:Floats,Tx<:Floats}
    Tax = promote_type(Ta, Tx)
    return Tax, complex(Tax)
end
#
# This method yields the number of rows and columns for lgemv assuming linear
# indexing and check arguments.
#
@inline function _lgemv_dims(trans::Char,
                             A::AbstractMatrix,
                             x::AbstractVector)
    nrows, ncols = size(A, 1), size(A, 2)
    if trans == 'N'
        length(x) == ncols || incompatible_dimensions()
    elseif trans == 'T' || trans == 'C'
        length(x) == nrows || incompatible_dimensions()
    else
        invalid_transpose_character()
    end
    return nrows, ncols, (trans == 'N' ? nrows : ncols)
end
#
# Idem for general "flat" arrays.
#
function _lgemv_dims(trans::Char,
                     A::AbstractArray{<:Any,Na},
                     x::AbstractArray{<:Any,Nx}) where {Na,Nx}
    1 ≤ Nx < Na || incompatible_dimensions()
    @inbounds begin
        Ny = Na - Nx
        if trans == 'N'
            ncols = 1
            for d in 1:Nx
                dim = size(x, d)
                size(A, Ny + d) == dim || incompatible_dimensions()
                ncols *= dim
            end
            shape = ntuple(d -> size(A, d), Ny)
            nrows = prod(shape)
        elseif trans == 'T' || trans == 'C'
            nrows = 1
            for d in 1:Nx
                dim = size(x, d)
                size(A, d) == dim || incompatible_dimensions()
                nrows *= dim
            end
            shape = ntuple(d -> size(A, Nx + d), Ny)
            ncols = prod(shape)
        else
            invalid_transpose_character()
        end
        return nrows, ncols, shape
    end
end
#
# This method yields the number of rows and columns for lgemv! assuming linear
# indexing and check arguments.
#
@inline function _lgemv_dims(trans::Char,
                             A::AbstractMatrix,
                             x::AbstractVector,
                             y::AbstractVector)
    nrows, ncols = size(A, 1), size(A, 2)
    if trans == 'N'
        (length(x) == ncols && length(y) == nrows) || incompatible_dimensions()
    elseif trans == 'T' || trans == 'C'
        (length(x) == nrows && length(y) == ncols) || incompatible_dimensions()
    else
        invalid_transpose_character()
    end
    return nrows, ncols
end
#
# Idem for general arrays.
#
function _lgemv_dims(trans::Char,
                     A::AbstractArray{<:Any,Na},
                     x::AbstractArray{<:Any,Nx},
                     y::AbstractArray{<:Any,Ny}) where {Na,Nx,Ny}
    (Na == Nx + Ny && Nx ≥ 1 && Ny ≥ 1) || incompatible_dimensions()
    nrows = ncols = 1
    @inbounds begin
        if trans == 'N'
            for d in 1:Ny
                dim = size(y, d)
                size(A, d) == dim || incompatible_dimensions()
                nrows *= dim
            end
            for d in 1:Nx
                dim = size(x, d)
                size(A, d + Ny) == dim || incompatible_dimensions()
                ncols *= dim
            end
        elseif trans == 'T' || trans == 'C'
            for d in 1:Nx
                dim = size(x, d)
                size(A, d) == dim || incompatible_dimensions()
                nrows *= dim
            end
            for d in 1:Ny
                dim = size(y, d)
                size(A, d + Nx) == dim || incompatible_dimensions()
                ncols *= dim
            end
        else
            invalid_transpose_character()
        end
    end
    return (nrows, ncols)
end
#
# Build tuples rows and cols of index intervals to access A[i,j] in lgemv and
# check arguments.
#
@inline function _lgemv_indices(trans::Char,
                                A::AbstractMatrix,
                                x::AbstractVector)
    (trans == 'N' || trans == 'T' || trans == 'C') ||
        invalid_transpose_character()
    rows, cols = axes(A, 1), axes(A, 2)
    axes(x, 1) == (trans == 'N' ? cols : rows) || incompatible_dimensions()
    return rows, cols
end
#
# Idem for general arrays.
#
@inline function _lgemv_indices(trans::Char,
                                A::AbstractArray{<:Any,Na},
                                x::AbstractArray{<:Any,Nx}) where {Na,Nx}
    1 ≤ Nx < Na || incompatible_dimensions()
    @inbounds begin
        Ny = Na - Nx
        if trans == 'N'
            rows = ntuple(d -> axes(A, d), Ny)
            cols = ntuple(d -> axes(A, d + Ny), Nx)
            for d in 1:Nx
                axes(x, d) == cols[d] || incompatible_dimensions()
            end
        elseif trans == 'T' || trans == 'C'
            rows = ntuple(d -> axes(A, d), Nx)
            cols = ntuple(d -> axes(A, d + Nx), Ny)
            for d in 1:Nx
                axes(x, d) == rows[d] || incompatible_dimensions()
            end
        else
            invalid_transpose_character()
        end
        return rows, cols
    end
end
#
# Build tuples rows and cols of index intervals to access A[i,j] in lgemv! and
# check arguments.
#
@inline function _lgemv_indices(trans::Char,
                                A::AbstractMatrix,
                                x::AbstractVector,
                                y::AbstractVector)
    (trans == 'N' || trans == 'T' || trans == 'C') ||
        invalid_transpose_character()
    rows, cols = axes(A, 1), axes(A, 2)
    (axes(x, 1) == (trans == 'N' ? cols : rows) &&
     axes(y, 1) == (trans == 'N' ? rows : cols)) || incompatible_dimensions()
    return rows, cols
end
#
# Idem for general arrays.
#
@inline function _lgemv_indices(trans::Char,
                                A::AbstractArray{<:Any,Na},
                                x::AbstractArray{<:Any,Nx},
                                y::AbstractArray{<:Any,Ny}) where {Na,Nx,Ny}
    (Na == Nx + Ny && Nx ≥ 1 && Ny ≥ 1) || incompatible_dimensions()
    @inbounds begin
        if trans == 'N'
            rows = ntuple(d -> axes(A, d), Ny)
            cols = ntuple(d -> axes(A, d + Ny), Nx)
            for d in 1:Nx
                axes(x, d) == cols[d] || incompatible_dimensions()
            end
            for d in 1:Ny
                axes(y, d) == rows[d] || incompatible_dimensions()
            end
        elseif trans == 'T' || trans == 'C'
            rows = ntuple(d -> axes(A, d), Nx)
            cols = ntuple(d -> axes(A, d + Nx), Ny)
            for d in 1:Nx
                axes(x, d) == rows[d] || incompatible_dimensions()
            end
            for d in 1:Ny
                axes(y, d) == cols[d] || incompatible_dimensions()
            end
        else
            invalid_transpose_character()
        end
        return rows, cols
    end
end
