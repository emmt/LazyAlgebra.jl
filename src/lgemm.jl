#
# lgemm.jl -
#
# Lazily Generalized Matrix-Matrix mutiplication.
#
#-------------------------------------------------------------------------------
#
# This file is part of LazyAlgebra (https://github.com/emmt/LazyAlgebra.jl)
# released under the MIT "Expat" license.
#
# Copyright (c) 2017-2018 Éric Thiébaut.
#

"""
# Lazily Generalized Matrix-Matrix mutiplication

```julia
lgemm([α=1,] [transA='N',] A, [transB='N',] B, Nc=2) -> C
```

yields `C = α*op(A)*op(B)` where `op(A)` is `A` if `transA` is `'N'`,
`transpose(A)` if `transA` is `'T'` and `adjoint(A)` if `transA` is `'C'` and
simlarly for `op(B)` and `transB`.  The expression `op(A)*op(B)` is a
matrix-matrix multiplication but interpreted by grouping consecutive dimensions
of `op(A)` and `op(B)` (see [`lgemv`](@ref) for explanations).  Argument `Nc`
specifies the number of dimensions of the result.

The in-place version is called by:

```julia
lgemm!([α=1,] [transA='N',] A,  [transB='N',] B, [β=0,] C) -> C
```

and overwrites the contents of `C` with `α*op(A)*op(B)`.  Note that `C` must
not be aliased with `A` or `B`.

The multipliers `α` and `β` must be both specified or omitted, they can be any
scalar numbers but are respectively converted to
`promote_type(eltype(A),eltype(B))` and `eltype(C)` which may throw an
`InexactError` exception.

See also: [`lgemv`](@ref), [`LinearAlgebra.BLAS.gemm`](@ref),
[`LinearAlgebra.BLAS.gemm!`](@ref).

"""
function lgemm(α::Number,
               transA::Char,
               A::AbstractArray{<:Floats},
               transB::Char,
               B::AbstractArray{<:Floats},
               Nc::Integer=2)
    return _lgemm(Implementation(Val(:lgemm), α, transA, A, transB, B, Int(Nc)),
                  α, transA, A, transB, B, Int(Nc))
end

function lgemm!(α::Number,
                transA::Char,
                A::AbstractArray{<:Floats},
                transB::Char,
                B::AbstractArray{<:Floats},
                β::Number,
                C::AbstractArray{<:Floats})
    return _lgemm!(Implementation(Val(:lgemm), α, transA, A, transB, B, β, C),
                   α, transA, A, transB, B, β, C)
end

@doc @doc(lgemm) lgemm!

# Best implementations for lgemm and lgemm!

for (atyp, eltyp) in ((Real,   BlasReal),
                      (Number, BlasComplex))
    @eval begin
        function Implementation(::Val{:lgemm},
                                α::$atyp,
                                transA::Char,
                                A::DenseMatrix{T},
                                transB::Char,
                                B::DenseMatrix{T},
                                Nc::Int) where {T<:$eltyp}
            return Blas()
        end

        function Implementation(::Val{:lgemm},
                                α::$atyp,
                                transA::Char,
                                A::DenseMatrix{T},
                                transB::Char,
                                B::DenseMatrix{T},
                                β::$atyp,
                                C::DenseMatrix{T}) where {T<:$eltyp}
            return Blas()
        end

        function Implementation(::Val{:lgemm},
                                α::$atyp,
                                transA::Char,
                                A::AbstractMatrix{T},
                                transB::Char,
                                B::AbstractMatrix{T},
                                Nc::Int) where {T<:$eltyp}
            return (is_flat_array(A, B) ? Blas() : Basic())
        end

        function Implementation(::Val{:lgemm},
                                α::$atyp,
                                transA::Char,
                                A::AbstractMatrix{T},
                                transB::Char,
                                B::AbstractMatrix{T},
                                β::$atyp,
                                C::AbstractMatrix{T}) where {T<:$eltyp}
            return (is_flat_array(A, B, C) ? Blas() : Basic())
        end

        function Implementation(::Val{:lgemm},
                                α::$atyp,
                                transA::Char,
                                A::DenseArray{T},
                                transB::Char,
                                B::DenseArray{T},
                                Nc::Int) where {T<:$eltyp}
            return Blas()
        end

        function Implementation(::Val{:lgemm},
                                α::$atyp,
                                transA::Char,
                                A::DenseArray{T},
                                transB::Char,
                                B::DenseArray{T},
                                β::$atyp,
                                C::DenseArray{T}) where {T<:$eltyp}
            return Blas()
        end

        function Implementation(::Val{:lgemm},
                                α::$atyp,
                                transA::Char,
                                A::AbstractArray{T},
                                transB::Char,
                                B::AbstractArray{T},
                                Nc::Int) where {T<:$eltyp}
            return (is_flat_array(A, B) ? Blas() : Basic())
        end

        function Implementation(::Val{:lgemm},
                                α::$atyp,
                                transA::Char,
                                A::AbstractArray{T},
                                transB::Char,
                                B::AbstractArray{T},
                                β::$atyp,
                                C::AbstractArray{T}) where {T<:$eltyp}
            return (is_flat_array(A, B, C) ? Blas() : Basic())
        end
    end
end

function Implementation(::Val{:lgemm},
                        α::Number,
                        transA::Char,
                        A::AbstractMatrix,
                        transB::Char,
                        B::AbstractMatrix,
                        Nc::Int)
    return Basic()
end

function Implementation(::Val{:lgemm},
                        α::Number,
                        transA::Char,
                        A::AbstractMatrix,
                        transB::Char,
                        B::AbstractMatrix,
                        β::Number,
                        C::AbstractMatrix)
    return Basic()
end

function Implementation(::Val{:lgemm},
                        α::Number,
                        transA::Char,
                        A::AbstractArray,
                        transB::Char,
                        B::AbstractArray,
                        Nc::Int)
    return (is_flat_array(A, B) ? Linear() : Generic())
end

function Implementation(::Val{:lgemm},
                        α::Number,
                        transA::Char,
                        A::AbstractArray,
                        transB::Char,
                        B::AbstractArray,
                        β::Number,
                        C::AbstractArray)
    return (is_flat_array(A, B, C) ? Linear() : Generic())
end

"""

```julia
_lgemm_ndims(Na, Nb, Nc) -> Ni, Nj, Nk
```

yields the number of consecutive indices involved in meta-indices `i`, `j` and
`k` for `lgemm` and `lgemm!` methods which compute:

```
C[i,j] = α⋅sum_k op(A)[i,k]*op(B)[k,j] + β⋅C[i,j]
```

Here `Na`, `Nb` and `Nc` are the respective number of dimensions of `A`, `B`
and `C`.  Calling this method also check the compatibility of the number of
dimensions.

"""
function _lgemm_ndims(Na::Int, Nb::Int, Nc::Int)
    #
    # The relations are:
    #
    #     Na = Ni + Nk         Ni = (Na + Nc - Nb)/2
    #     Nb = Nk + Nj   <==>  Nj = (Nc + Nb - Na)/2
    #     Nc = Ni + Nj         Nk = (Nb + Na - Nc)/2
    #
    # Note: Ni ≥ 1 and Nj ≥ 1 and Nk ≥ 1 implies that Na ≥ 2 and Nb ≥ 2
    # and Nc ≥ 2.
    #
    Li = Na + Nc - Nb
    Lj = Nc + Nb - Na
    Lk = Nb + Na - Nc
    if Li > 0 && iseven(Li) && Lj > 0 && iseven(Lj) && Lk > 0 && iseven(Lk)
        return Li >> 1, Lj >> 1, Lk >> 1
    end
    incompatible_dimensions()
end

# BLAS implementations for (generalized) matrices.
# Linear indexing is assumed (this must have been checked before).

function _lgemm(::Blas,
                α::Number,
                transA::Char,
                A::AbstractArray{T},
                transB::Char,
                B::AbstractArray{T},
                Nc::Int) where {T<:BlasFloat}
    m, n, p, shape = _lgemm_dims(transA, A, transB, B, Nc)
    return _blas_lgemm!(m, n, p, convert(T, α), transA, A,
                        transB, B, zero(T), Array{T}(undef, shape))
end

function _lgemm!(::Blas,
                 α::Number,
                 transA::Char,
                 A::AbstractArray{T},
                 transB::Char,
                 B::AbstractArray{T},
                 β::Number,
                 C::AbstractArray{T}) where {T<:BlasFloat}
    m, n, p = _lgemm_dims(transA, A, transB, B, C)
    return _blas_lgemm!(m, n, p, convert(T, α), transA, A,
                        transB, B, convert(T, β), C)
end

# Julia implementations for (generalized) matrices.
# Linear indexing is assumed (this must have been checked before).

function _lgemm(::Linear,
                α::Number,
                transA::Char,
                A::AbstractArray{Ta},
                transB::Char,
                B::AbstractArray{Tb},
                Nc::Int) where {Ta<:Floats,Tb<:Floats}
    m, n, p, shape = _lgemm_dims(transA, A, transB, B, Nc)
    Tab, Tc = _lgemm_types(α, Ta, Tb)
    return _linear_lgemm!(m, n, p,
                          convert_multiplier(α, Tab, Tc), transA, A,
                          transB, B,
                          convert_multiplier(0, Tc), Array{Tc}(undef, shape))
end

function _lgemm!(::Linear,
                 α::Number,
                 transA::Char,
                 A::AbstractArray{Ta},
                 transB::Char,
                 B::AbstractArray{Tb},
                 β::Number,
                 C::AbstractArray{Tc}) where {Ta<:Floats,Tb<:Floats,Tc<:Floats}
    m, n, p = _lgemm_dims(transA, A, transB, B, C)
    Tab = promote_type(Ta, Tb)
    return _linear_lgemm!(m, n, p,
                          convert_multiplier(α, Tab, Tc), transA, A,
                          transB, B,
                          convert_multiplier(β, Tc), C)
end

# Julia implementations for any kind of abstract matrices.

function _lgemm(::Basic,
                α::Number,
                transA::Char,
                A::AbstractMatrix{Ta},
                transB::Char,
                B::AbstractMatrix{Tb},
                Nc::Int) where {Ta<:Floats,Tb<:Floats}
    I, J, K = _lgemm_indices(transA, A, transB, B, Nc)
    Tab, Tc = _lgemm_types(α, Ta, Tb)
    return _generic_lgemm!(I, J, K,
                           convert_multiplier(α, Tab, Tc), transA, A,
                           transB, B,
                           convert_multiplier(0, Tc),
                           similar(Array{Tc}, (I, J)))
end

function _lgemm!(::Basic,
                 α::Number,
                 transA::Char,
                 A::AbstractMatrix{Ta},
                 transB::Char,
                 B::AbstractMatrix{Tb},
                 β::Number,
                 C::AbstractMatrix{Tc}) where {Ta<:Floats,Tb<:Floats,Tc<:Floats}
    I, J, K = _lgemm_indices(transA, A, transB, B, C)
    Tab = promote_type(Ta, Tb)
    return _generic_lgemm!(I, J, K,
                           convert_multiplier(α, Tab, Tc), transA, A,
                           transB, B,
                           convert_multiplier(β, Tc), C)
end

# Generic Julia implementation.

function _lgemm(::Generic,
                α::Number,
                transA::Char,
                A::AbstractMatrix{Ta},
                transB::Char,
                B::AbstractMatrix{Tb},
                Nc::Int) where {Ta<:Floats,Tb<:Floats}
    I, J, K = _lgemm_indices(transA, A, transB, B, Nc)
    Tab, Tc = _lgemm_types(α, Ta, Tb)
    return _generic_lgemm!(I, J, K,
                           convert_multiplier(α, Tab, Tc), transA, A,
                           transB, B,
                           convert_multiplier(0, Tc),
                           similar(Array{Tc}, (I, J)))
end

function _lgemm!(::Generic,
                 α::Number,
                 transA::Char,
                 A::AbstractMatrix{Ta},
                 transB::Char,
                 B::AbstractMatrix{Tb},
                 β::Number,
                 C::AbstractMatrix{Tc}) where {Ta<:Floats,Tb<:Floats,Tc<:Floats}
    I, J, K = _lgemm_indices(transA, A, transB, B, C)
    Tab = promote_type(Ta, Tb)
    return _generic_lgemm!(I, J, K,
                           convert_multiplier(α, Tab, Tc), transA, A,
                           transB, B,
                           convert_multiplier(β, Tc), C)
end

function _lgemm(::Generic,
                α::Number,
                transA::Char,
                A::AbstractArray{Ta},
                transB::Char,
                B::AbstractArray{Tb},
                Nc::Int) where {Ta<:Floats,Tb<:Floats}
    I, J, K = _lgemm_indices(transA, A, transB, B, Nc)
    Tab, Tc = _lgemm_types(α, Ta, Tb)
    return _generic_lgemm!(allindices(I), allindices(J), allindices(K),
                           convert_multiplier(α, Tab, Tc), transA, A,
                           transB, B,
                           convert_multiplier(0, Tc),
                           similar(Array{Tc}, (I..., J...)))
end

function _lgemm!(::Generic,
                 α::Number,
                 transA::Char,
                 A::AbstractArray{Ta},
                 transB::Char,
                 B::AbstractArray{Tb},
                 β::Number,
                 C::AbstractArray{Tc}) where {Ta<:Floats,Tb<:Floats,Tc<:Floats}
    I, J, K = _lgemm_indices(transA, A, transB, B, C)
    Tab = promote_type(Ta, Tb)
    return _generic_lgemm!(allindices(I), allindices(J), allindices(K),
                           convert_multiplier(α, Tab, Tc), transA, A,
                           transB, B,
                           convert_multiplier(β, Tc), C)
end

#
# Call low-level BLAS version.  The differences with LinearAlgebra.BLAS.gemm!
# are that inputs are assumed to be flat arrays (see is_flat_array) and that
# multipliers are automatically converted.
#
for (f, T) in ((:dgemm_, Float64),
                      (:sgemm_, Float32),
                      (:zgemm_, ComplexF64),
                      (:cgemm_, ComplexF32))
    @eval begin
        #
        # FORTRAN prototype:
        #
        #     SUBROUTINE ${pfx}GEMM(TRANSA,TRANSB,M,N,K,ALPHA,A,LDA,B,LDB,BETA,
        #                           C,LDC)
        #         ${T} ALPHA,BETA
        #         INTEGER M,N,K,LDA,LDB,LDC
        #         CHARACTER TRANSA,TRANSB
        #         ${T} A(LDA,*),B(LDB,*),C(LDC,*)
        #
        # Scalar arguments, α and β, can just be `Number` and integer arguments
        # can just be `Integer` but we want to keep the signature strict
        # because it is a low-level private method.
        #
        function _blas_lgemm!(m::Int, n::Int, p::Int, α::($T),
                              transA::Char, A::AbstractArray{$T},
                              transB::Char, B::AbstractArray{$T},
                              β::($T), C::AbstractArray{$T})
            lda = (transA == 'N' ? m : p)
            ldb = (transB == 'N' ? p : n)
            ldc = m
            ccall((@blasfunc($f), libblas), Cvoid,
                  (Ref{UInt8}, Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt},
                   Ref{BlasInt}, Ref{$T}, Ptr{$T}, Ref{BlasInt},
                   Ptr{$T}, Ref{BlasInt}, Ref{$T}, Ptr{$T},
                   Ref{BlasInt}),
                  transA, transB, m, n, p, α, A, lda, B, ldb, β, C, ldc)
            return C
        end
    end
end
#
# This method is based on reference BLAS level 2 routine ZGEMM of LAPACK
# (version 3.7.0).  It is assumed that arrays A, B and C are flat arrays which
# can be linearly indexed.  Arguments are assumed to be correct, no checking is
# done.
#
function _linear_lgemm!(m::Int,
                        n::Int,
                        p::Int,
                        α::Number,
                        transA::Char,
                        A::AbstractArray{Ta,Na},
                        transB::Char,
                        B::AbstractArray{Tb,Nb},
                        β::Number,
                        C::AbstractArray{Tc,Nc}) where {Ta<:Floats,Na,
                                                        Tb<:Floats,Nb,
                                                        Tc<:Floats,Nc}
    #
    # Quick return if possible.
    #
    if m > 0 && n > 0 && (β != 1 || (p > 0 && α != 0))
        T = promote_type(Ta, Tb)
        if α == 0
            #
            # Quick computations when α = 0.
            #
            if β == 0
                @inbounds @simd for k in eachindex(C)
                    C[k] = zero(Tc)
                end
            elseif β != 1
                @inbounds @simd for k in eachindex(C)
                    C[k] *= β
                end
            end
        elseif transB == 'N'
            if transA == 'N'
                #
                # Form  C := α*A*B + β*C.
                #
                @inbounds for j in 1:n
                    if β == 0
                        @simd for i in 1:m
                            C[m*(j - 1) + i] = zero(Tc)
                        end
                    elseif β != 1
                        @simd for i in 1:m
                            C[m*(j - 1) + i] *= β
                        end
                    end
                    for k in 1:p
                        temp = α*B[p*(j - 1) + k]
                        if temp != zero(temp)
                            @simd for i in 1:m
                                C[m*(j - 1) + i] += temp*A[m*(k - 1) + i]
                            end
                        end
                    end
                end
            elseif Ta <: Real || transA == 'T'
                #
                # Form  C := α*A^T*B + β*C
                #
                @inbounds for j in 1:n
                    for i in 1:m
                        temp = zero(T)
                        @simd for k in 1:p
                            temp += A[p*(i - 1) + k]*B[p*(j - 1) + k]
                        end
                        C[m*(j - 1) + i] = (β == 0 ? α*temp : α*temp + β*C[m*(j - 1) + i])
                    end
                end
            else
                #
                # Form  C := α*A^H*B + β*C.
                #
                @inbounds for j in 1:n
                    for i in 1:m
                        temp = zero(T)
                        @simd for k in 1:p
                            temp += conj(A[p*(i - 1) + k])*B[p*(j - 1) + k]
                        end
                        C[m*(j - 1) + i] = (β == 0 ? α*temp : α*temp + β*C[m*(j - 1) + i])
                    end
                end
            end
        elseif transA == 'N'
            if Tb <: Real || transB == 'T'
                #
                # Form  C := α*A*B^T + β*C
                #
                @inbounds for j in 1:n
                    if β == 0
                        @simd for i in 1:m
                            C[m*(j - 1) + i] = zero(Tc)
                        end
                    elseif β != one
                        @simd for i in 1:m
                            C[m*(j - 1) + i] *= β
                        end
                    end
                    for k in 1:p
                        temp = α*B[n*(k - 1) + j]
                        if temp != zero(temp)
                            @simd for i in 1:m
                                C[m*(j - 1) + i] += temp*A[m*(k - 1) + i]
                            end
                        end
                    end
                end
            else
                #
                # Form  C := α*A*B^H + β*C.
                #
                @inbounds for j in 1:n
                    if β == 0
                        @simd for i in 1:m
                            C[m*(j - 1) + i] = zero(Tc)
                        end
                    elseif β != one
                        @simd for i in 1:m
                            C[m*(j - 1) + i] *= β
                        end
                    end
                    for k in 1:p
                        temp = α*conj(B[n*(k - 1) + j])
                        if temp != zero(temp)
                            @simd for i in 1:m
                                C[m*(j - 1) + i] += temp*A[m*(k - 1) + i]
                            end
                        end
                    end
                end
            end
        elseif Ta <: Real || transA == 'T'
            if Tb <: Real || transB == 'T'
                #
                # Form  C := α*A^T*B^T + β*C
                #
                @inbounds for j in 1:n
                    for i in 1:m
                        temp = zero(T)
                        @simd for k in 1:p
                            temp += A[p*(i - 1) + k]*B[n*(k - 1) + j]
                        end
                        C[m*(j - 1) + i] = (β == 0 ? α*temp : α*temp + β*C[m*(j - 1) + i])
                    end
                end
            else
                #
                # Form  C := α*A^T*B^H + β*C
                #
                @inbounds for j in 1:n
                    for i in 1:m
                        temp = zero(T)
                        @simd for k in 1:p
                            temp += A[p*(i - 1) + k]*conj(B[n*(k - 1) + j])
                        end
                        C[m*(j - 1) + i] = (β == 0 ? α*temp : α*temp + β*C[m*(j - 1) + i])
                    end
                end
            end
        else
            if Tb <: Real || transB == 'T'
                #
                # Form  C := α*A^H*B^T + β*C
                #
                @inbounds for j in 1:n
                    for i in 1:m
                        temp = zero(T)
                        @simd for k in 1:p
                            temp += conj(A[p*(i - 1) + k])*B[n*(k - 1) + j]
                        end
                        C[m*(j - 1) + i] = (β == 0 ? α*temp : α*temp + β*C[m*(j - 1) + i])
                    end
                end
            else
                #
                # Form  C := α*A^H*B^H + β*C.
                #
                @inbounds for j in 1:n
                    for i in 1:m
                        temp = zero(T)
                        @simd for k in 1:p
                            temp += conj(A[p*(i - 1) + k])*conj(B[n*(k - 1) + j])
                        end
                        C[m*(j - 1) + i] = (β == 0 ? α*temp : α*temp + β*C[m*(j - 1) + i])
                    end
                end
            end
        end
    end
    return C
end
#
# This method is based on reference BLAS level 2 routine ZGEMM of LAPACK
# (version 3.7.0).  Arguments A, B and C are any kind of arrays indexed by the
# Cartesian indices in I, J, and K.  Arguments are assumed to be correct, no
# checking is done.
#
function _generic_lgemm!(I, J, K,
                         α::Number,
                         transA::Char,
                         A::AbstractArray{Ta,Na},
                         transB::Char,
                         B::AbstractArray{Tb,Nb},
                         β::Number,
                         C::AbstractArray{Tc,Nc}) where {Ta<:Floats,Na,
                                                         Tb<:Floats,Nb,
                                                         Tc<:Floats,Nc}
    #
    # Quick return if possible.
    #
    if length(I) > 0 && length(J) > 0 && (β != 1 || (length(K) > 0 && α != 0))
        T = promote_type(Ta, Tb)
        if α == 0
            #
            # Quick computations when  α = 0.
            #
            if β == 0
                @inbounds @simd for k in eachindex(C)
                    C[k] = zero(Tc)
                end
            elseif β != 1
                @inbounds @simd for k in eachindex(C)
                    C[k] *= β
                end
            end
        elseif transB == 'N'
            if transA == 'N'
                #
                # Form  C := α*A*B + β*C.
                #
                @inbounds for j in J
                    if β == 0
                        @simd for i in I
                            C[i,j] = zero(Tc)
                        end
                    elseif β != 1
                        @simd for i in I
                            C[i,j] *= β
                        end
                    end
                    for k in K
                        temp = α*B[k,j]
                        if temp != zero(temp)
                            @simd for i in I
                                C[i,j] += temp*A[i,k]
                            end
                        end
                    end
                end
            elseif Ta <: Real || transA == 'T'
                #
                # Form  C := α*A^T*B + β*C
                #
                @inbounds for j in J
                    for i in I
                        temp = zero(T)
                        @simd for k in K
                            temp += A[k,i]*B[k,j]
                        end
                        C[i,j] = (β == 0 ? α*temp : α*temp + β*C[i,j])
                    end
                end
            else
                #
                # Form  C := α*A^H*B + β*C.
                #
                @inbounds for j in J
                    for i in I
                        temp = zero(T)
                        @simd for k in K
                            temp += conj(A[k,i])*B[k,j]
                        end
                        C[i,j] = (β == 0 ? α*temp : α*temp + β*C[i,j])
                    end
                end
            end
        elseif transA == 'N'
            if Tb <: Real || transB == 'T'
                #
                # Form  C := α*A*B^T + β*C
                #
                @inbounds for j in J
                    if β == 0
                        @simd for i in I
                            C[i,j] = zero(Tc)
                        end
                    elseif β != one
                        @simd for i in I
                            C[i,j] *= β
                        end
                    end
                    for k in K
                        temp = α*B[j,k]
                        if temp != zero(temp)
                            @simd for i in I
                                C[i,j] += temp*A[i,k]
                            end
                        end
                    end
                end
            else
                #
                # Form  C := α*A*B^H + β*C.
                #
                @inbounds for j in J
                    if β == 0
                        @simd for i in I
                            C[i,j] = zero(Tc)
                        end
                    elseif β != one
                        @simd for i in I
                            C[i,j] *= β
                        end
                    end
                    for k in K
                        temp = α*conj(B[j,k])
                        if temp != zero(temp)
                            @simd for i in I
                                C[i,j] += temp*A[i,k]
                            end
                        end
                    end
                end
            end
        elseif Ta <: Real || transA == 'T'
            if Tb <: Real || transB == 'T'
                #
                # Form  C := α*A^T*B^T + β*C
                #
                @inbounds for j in J
                    for i in I
                        temp = zero(T)
                        @simd for k in K
                            temp += A[k,i]*B[j,k]
                        end
                        C[i,j] = (β == 0 ? α*temp : α*temp + β*C[i,j])
                    end
                end
            else
                #
                # Form  C := α*A^T*B^H + β*C
                #
                @inbounds for j in J
                    for i in I
                        temp = zero(T)
                        @simd for k in K
                            temp += A[k,i]*conj(B[j,k])
                        end
                        C[i,j] = (β == 0 ? α*temp : α*temp + β*C[i,j])
                    end
                end
            end
        else
            if Tb <: Real || transB == 'T'
                #
                # Form  C := α*A^H*B^T + β*C
                #
                @inbounds for j in J
                    for i in I
                        temp = zero(T)
                        @simd for k in K
                            temp += conj(A[k,i])*B[j,k]
                        end
                        C[i,j] = (β == 0 ? α*temp : α*temp + β*C[i,j])
                    end
                end
            else
                #
                # Form  C := α*A^H*B^H + β*C.
                #
                @inbounds for j in J
                    for i in I
                        temp = zero(T)
                        @simd for k in K
                            temp += conj(A[k,i])*conj(B[j,k])
                        end
                        C[i,j] = (β == 0 ? α*temp : α*temp + β*C[i,j])
                    end
                end
            end
        end
    end
    return C
end
#
# This method yields promote_type(Ta, Tb) and Tc, the type of the elements for
# the result of lgemm.
#
@inline function _lgemm_types(α::Real, ::Type{Ta},
                              ::Type{Tb}) where {Ta<:Floats,Tb<:Floats}
    Tab = promote_type(Ta, Tb)
    return Tab, Tab
end
#
@inline function _lgemm_types(α::Complex, ::Type{Ta},
                              ::Type{Tb}) where {Ta<:Floats,Tb<:Floats}
    Tab = promote_type(Ta, Tb)
    return Tab, complex(Tab)
end
#
# This method yields the lengths of the meta-dimensions for lgemm assuming
# linear indexing and check arguments.
#
@inline function _lgemm_dims(transA::Char,
                             A::AbstractMatrix,
                             transB::Char,
                             B::AbstractMatrix,
                             Nc::Int)
    Nc == 2 || incompatible_dimensions()
    lda = size(A, 1)
    ldb = size(B, 1)
    if transA == 'N'
        m, p = lda, size(A, 2)
    elseif transA == 'T' || transA == 'C'
        p, m = lda, size(A, 2)
    else
        invalid_transpose_character()
    end
    if transB == 'N'
        ldb == p || incompatible_dimensions()
        n = size(B, 2)
    elseif transB == 'T' || transB == 'C'
        size(B, 2) == p || incompatible_dimensions()
        n = ldb
    else
        invalid_transpose_character()
    end
    return m, n, p, (m, n)
end
#
# Idem for general arrays.
#
function _lgemm_dims(transA::Char,
                     A::AbstractArray{<:Any,Na},
                     transB::Char,
                     B::AbstractArray{<:Any,Nb},
                     Nc::Int) where {Na,Nb}
    nota = (transA == 'N')
    notb = (transB == 'N')
    if ((!nota && transA != 'T' && transA != 'C') ||
        (!notb && transB != 'T' && transB != 'C'))
        invalid_transpose_character()
    end
    Ni, Nj, Nk = _lgemm_ndims(Na, Nb, Nc)
    @inbounds begin
        p = 1
        if nota
            if notb
                # C[i,j] = sum_k A[i,k]*B[k,j]
                for d in 1:Nk
                    dim = size(A, d + Ni)
                    size(B, d) == dim || incompatible_dimensions()
                    p *= dim
                end
                shape = ntuple(d -> d ≤ Ni ? size(A, d) : size(B, d + (Nk - Ni)), Nc)
            else
                # C[i,j] = sum_k A[i,k]*B[j,k]
                for d in 1:Nk
                    dim = size(A, d + Ni)
                    size(B, d + Nj) == dim || incompatible_dimensions()
                    p *= dim
                end
                shape = ntuple(d -> d ≤ Ni ? size(A, d) : size(B, d - Ni), Nc)
            end
        else
            if notb
                # C[i,j] = sum_k A[k,i]*B[k,j]
                for d in 1:Nk
                    dim = size(A, d)
                    size(B, d) == dim || incompatible_dimensions()
                    p *= dim
                end
                shape = ntuple(d -> d ≤ Ni ? size(A, d + Nk) : size(B, d + (Nk - Ni)), Nc)
            else
                # C[i,j] = sum_k A[k,i]*B[j,k]
                for d in 1:Nk
                    dim = size(A, d)
                    size(B, d + Nj) == dim || incompatible_dimensions()
                    p *= dim
                end
                shape = ntuple(d -> d ≤ Ni ? size(A, d + Nk) : size(B, d - Ni), Nc)
            end
        end
        m = 1
        for d in 1:Ni
            m *= shape[d]
        end
        n = 1
        for d in Ni+1:Ni+Nj
            n *= shape[d]
        end
       return m, n, p, shape
    end
end
#
# This method yields the lenghts of the meta-dimensions for lgemm! assuming
# linear indexing and check arguments.
#
@inline function _lgemm_dims(transA::Char,
                             A::AbstractMatrix,
                             transB::Char,
                             B::AbstractMatrix,
                             C::AbstractMatrix)
    m, n = size(C, 1), size(C, 2)
    adim1, adim2 = size(A, 1), size(A, 2)
    bdim1, bdim2 = size(B, 1), size(B, 2)
    if transA == 'N'
        adim1 == m || incompatible_dimensions()
        p = adim2
    elseif transA == 'T' || transA == 'C'
        adim2 == m || incompatible_dimensions()
        p = adim1
    else
        invalid_transpose_character()
    end
    if transB == 'N'
        (bdim1 == p && bdim2 == n) || incompatible_dimensions()
    elseif transB == 'T' || transB == 'C'
        (bdim1 == n && bdim2 == p) || incompatible_dimensions()
    else
        invalid_transpose_character()
    end
    return m, n, p
end
#
# Idem for general arrays.
#
function _lgemm_dims(transA::Char,
                     A::AbstractArray{<:Any,Na},
                     transB::Char,
                     B::AbstractArray{<:Any,Nb},
                     C::AbstractArray{<:Any,Nc}) where {Na,Nb,Nc}
    nota = (transA == 'N')
    notb = (transB == 'N')
    if ((!nota && transA != 'T' && transA != 'C') ||
        (!notb && transB != 'T' && transB != 'C'))
        invalid_transpose_character()
    end
    Ni, Nj, Nk = _lgemm_ndims(Na, Nb, Nc)
    @inbounds begin
        m = p = 1
        if nota
            for d in 1:Ni
                dim = size(C, d)
                size(A, d) == dim || incompatible_dimensions()
                m *= dim
            end
            if notb
                # C[i,j] = sum_k A[i,k]*B[k,j]
                for d in 1:Nk
                    dim = size(A, d + Ni)
                    size(B, d) == dim || incompatible_dimensions()
                    p *= dim
                end
            else
                # C[i,j] = sum_k A[i,k]*B[j,k]
                for d in 1:Nk
                    dim = size(A, d + Ni)
                    size(B, d + Nj) == dim || incompatible_dimensions()
                    p *= dim
                end
            end
        else
            for d in 1:Ni
                dim = size(C, d)
                size(A, d + Nk) == dim || incompatible_dimensions()
                m *= dim
            end
            if notb
                # C[i,j] = sum_k A[k,i]*B[k,j]
                for d in 1:Nk
                    dim = size(A, d)
                    size(B, d) == dim || incompatible_dimensions()
                    p *= dim
                end
            else
                # C[i,j] = sum_k A[k,i]*B[j,k]
                for d in 1:Nk
                    dim = size(A, d)
                    size(B, d + Nj) == dim || incompatible_dimensions()
                    p *= dim
                end
            end
        end
        n = 1
        if notb
            for d in 1:Nj
                dim = size(C, d + Ni)
                size(B, d + Nk) == dim || incompatible_dimensions()
                n *= dim
            end
        else
            for d in 1:Nj
                dim = size(C, d + Ni)
                size(B, d) == dim || incompatible_dimensions()
                n *= dim
            end
        end
        return m, n, p
    end
end
#
# This method yields the indices of the meta-dimensions for lgemm and check
# arguments.
#
@inline function _lgemm_indices(transA::Char,
                                A::AbstractMatrix,
                                transB::Char,
                                B::AbstractMatrix,
                                Nc::Int)
    Nc == 2 || incompatible_dimensions()
    if transA == 'N'
        I, K = axes(A, 1), axes(A, 2)
    elseif transA == 'T' || transA == 'C'
        K, I = axes(A, 1), axes(A, 2)
    else
        invalid_transpose_character()
    end
    if transB == 'N'
        axes(B, 1) == K || incompatible_dimensions()
        J = axes(B, 2)
    elseif transB == 'T' || transB == 'C'
        axes(B, 2) == K || incompatible_dimensions()
        J = axes(B, 1)
    else
        invalid_transpose_character()
    end
    return I, J, K
end
#
# Idem for general arrays.
#
function _lgemm_indices(transA::Char,
                        A::AbstractArray{<:Any,Na},
                        transB::Char,
                        B::AbstractArray{<:Any,Nb},
                        Nc::Int) where {Na,Nb}
    nota = (transA == 'N')
    notb = (transB == 'N')
    if ((!nota && transA != 'T' && transA != 'C') ||
        (!notb && transB != 'T' && transB != 'C'))
        invalid_transpose_character()
    end
    Ni, Nj, Nk = _lgemm_ndims(Na, Nb, Nc)
    @inbounds begin
        if nota
            I = ntuple(d -> axes(A, d), Ni)
            K = ntuple(d -> axes(A, d + Ni), Nk)
        else
            I = ntuple(d -> axes(A, d + Nk), Ni)
            K = ntuple(d -> axes(A, d), Nk)
        end
        if notb
            for d in 1:Nk
                axes(B, d) == K[d] || incompatible_dimensions()
            end
            J = ntuple(d -> axes(B, d + Nk), Nj)
        else
            for d in 1:Nk
                axes(B, d + Nj) == K[d] || incompatible_dimensions()
            end
            J = ntuple(d -> axes(B, d), Nj)
        end
        return I, J, K
    end
end
#
# This method yields the indices of the meta-dimensions for lgemm! and check
# arguments.
#
@inline function _lgemm_indices(transA::Char,
                                A::AbstractMatrix,
                                transB::Char,
                                B::AbstractMatrix,
                                C::AbstractMatrix)
    I, J = axes(C, 1), axes(C, 2)
    if transA == 'N'
        axes(A, 1) == I || incompatible_dimensions()
        K = axes(A, 2)
    elseif transA == 'T' || transA == 'C'
        axes(A, 2) == I || incompatible_dimensions()
        K = axes(A, 1)
    else
        invalid_transpose_character()
    end
    if transB == 'N'
        (axes(B, 1) == K && axes(B, 2) == J) || incompatible_dimensions()
    elseif transB == 'T' || transB == 'C'
        (axes(B, 1) == J && axes(B, 2) == K) || incompatible_dimensions()
    else
        invalid_transpose_character()
    end
    return I, J, K
end
#
# Idem for general arrays.
#
function _lgemm_indices(transA::Char,
                        A::AbstractArray{<:Any,Na},
                        transB::Char,
                        B::AbstractArray{<:Any,Nb},
                        C::AbstractArray{<:Any,Nc}) where {Na,Nb,Nc}
    nota = (transA == 'N')
    notb = (transB == 'N')
    if ((!nota && transA != 'T' && transA != 'C') ||
        (!notb && transB != 'T' && transB != 'C'))
        invalid_transpose_character()
    end
    Ni, Nj, Nk = _lgemm_ndims(Na, Nb, Nc)
    @inbounds begin
        I = ntuple(d -> axes(C, d), Ni)
        J = ntuple(d -> axes(C, d + Ni), Nj)
        if nota
            for d in 1:Ni
                axes(A, d) == I[d] || incompatible_dimensions()
            end
            K = ntuple(d -> axes(A, d + Ni), Nk)
        else
            for d in 1:Ni
                axes(A, d + Nk) == I[d] || incompatible_dimensions()
            end
            K = ntuple(d -> axes(A, d), Nk)
        end
        if notb
            for d in 1:Nk
                axes(B, d) == K[d] || incompatible_dimensions()
            end
            for d in 1:Nj
                axes(B, d + Nk) == J[d] || incompatible_dimensions()
            end
        else
            for d in 1:Nk
                axes(B, d + Nj) == K[d] || incompatible_dimensions()
            end
            for d in 1:Nj
                axes(B, d) == J[d] || incompatible_dimensions()
            end
        end
        return I, J, K
    end
end
