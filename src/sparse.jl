#
# sparse.jl -
#
# Implement sparse linear mappings.
#
#-------------------------------------------------------------------------------
#
# This file is part of LazyAlgebra (https://github.com/emmt/LazyAlgebra.jl)
# released under the MIT "Expat" license.
#
# Copyright (c) 2017-2018 Éric Thiébaut.
#

"""
```julia
SparseOperator(outdims, inpdims, A, I, J)
```

yields a sparse linear map whose *rows* and *columns* have respective
dimensions `outdims` and `inpdims` and whose non-zero coefficients are given by
`A` with corresponding row and column linear indices respectively given by `I`
and `J`.

```julia
SparseOperator(A, n=1)
```

yields an instance of `SparseOperator` whose coefficients are the non-zero
coefficients of array `A` and which implements generalized matrix
multiplication by `A` in such a way that the result of applying the operator
are arrays whose dimensions are the `n` leading dimensions of `A` for input
arrays whose dimensions are the remaining trailing dimensions of `A`.

!!! note
    For efficiency reasons, sparse operators are currrently limited to dense
    Julia arrays because they can be indexed linearly with no loss of
    performances.  If `A`, `I` and/or `J` are not dense arrays, they will be
    automatically converted to regular arrays.

"""
struct SparseOperator{T,M,N,
                      Ta<:DenseVector{T},
                      Ti<:DenseVector{Int},
                      Tj<:DenseVector{Int}} <: LinearMapping
    outdims::NTuple{M,Int}
    inpdims::NTuple{N,Int}
    A::Ta
    I::Ti
    J::Tj
    samedims::Bool
    function SparseOperator{T,M,N,Ta,Ti,Tj}(
        outdims::NTuple{M,Int},
        inpdims::NTuple{N,Int},
        A::Ta,
        I::Ti,
        J::Tj
    ) where {
        Ta<:DenseVector{T},
        Ti<:DenseVector{Int},
        Tj<:DenseVector{Int}
    } where {
        T,M,N
    }
        samedims = (M == N && outdims == inpdims)
        return new{T,M,N,Ta,Ti,Tj}(outdims, inpdims, A, I, J, samedims)
    end
end

@callable SparseOperator

function SparseOperator(outdims::NTuple{M,Int},
                        inpdims::NTuple{N,Int},
                        A::Ta,
                        I::Ti,
                        J::Tj) where {Ta<:DenseVector{T},
                                      Ti<:DenseVector{Int},
                                      Tj<:DenseVector{Int}} where {T,M,N}
    return SparseOperator{T,M,N,Ta,Ti,Tj}(outdims, inpdims, A, I, J)
end

function SparseOperator(outdims::NTuple{M,Integer},
                        inpdims::NTuple{N,Integer},
                        A::AbstractVector{T},
                        I::AbstractVector{<:Integer},
                        J::AbstractVector{<:Integer}) where {T,M,N}
    return SparseOperator(map(Int, outdims),
                          map(Int, inpdims),
                          densevector(T,   A),
                          densevector(Int, I),
                          densevector(Int, J))
end

function SparseOperator(A::DenseArray{T,N},
                        n::Integer = 1) where {T,N}
    1 ≤ n < N || throw(ArgumentError("bad number of of leading dimensions"))
    dims = size(A)
    outdims = dims[1:n]
    inpdims = dims[n+1:end]
    nrows = prod(outdims)
    ncols = prod(inpdims)
    nz = 0
    @inbounds for k in 1:length(A)
        if A[k] != zero(T)
            nz += 1
        end
    end
    C = Array{T}(undef, nz)
    I = Array{Int}(undef, nz)
    J = Array{Int}(undef, nz)
    k = 0
    l = 0
    @inbounds for j in 1:ncols, i in 1:nrows
        k += 1
        if (a = A[k]) != zero(T)
            l += 1
            C[l] = a
            I[l] = i
            J[l] = j
        end
    end
    @assert l == nz
    return SparseOperator(outdims, inpdims, C, I, J)
end

input_size(S::SparseOperator) = S.inpdims
output_size(S::SparseOperator) = S.outdims

EndomorphismType(S::SparseOperator) = (S.samedims ? Endomorphism : Morphism)

function vcreate(::Type{Direct}, S::SparseOperator{Ts,M,N},
                 x::DenseArray{Tx,N}) where {Ts<:Real,Tx<:Real,M,N}
    size(x) == input_size(S) ||
        throw(DimensionMismatch("bad dimensions for input"))
    return Array{promote_type(Ts,Tx)}(undef, output_size(S))
end

function vcreate(::Type{Adjoint}, S::SparseOperator{Ts,M,N},
                 x::DenseArray{Tx,M}) where {Ts<:Real,Tx<:Real,M,N}
    size(x) == output_size(S) ||
        throw(DimensionMismatch("bad dimensions for input"))
    return Array{promote_type(Ts,Tx)}(undef, input_size(S))
end

function apply!(alpha::Real,
                ::Type{Direct},
                S::SparseOperator{Ts,M,N},
                x::DenseArray{Tx,N},
                beta::Real,
                y::DenseArray{Ty,M}) where {Ts<:Real,Tx<:Real,Ty<:Real,M,N}
    size(x) == input_size(S) ||
        throw(DimensionMismatch("bad dimensions for input"))
    size(y) == output_size(S) ||
        throw(DimensionMismatch("bad dimensions for output"))
    beta == 1 || vscale!(y, beta)
    if alpha != 0
        A, I, J = S.A, S.I, S.J
        length(I) == length(J) == length(A) ||
            error("corrupted sparse operator structure")
        for k in 1:length(A)
            @inbounds a, i, j = A[k], I[k], J[k]
            y[i] += alpha*a*x[j]
        end
    end
    return y
end

function apply!(alpha::Real,
                ::Type{Adjoint},
                S::SparseOperator{Ts,M,N},
                x::DenseArray{Tx,M},
                beta::Real,
                y::DenseArray{Ty,N}) where {Ts<:Real,Tx<:Real,Ty<:Real,M,N}
    size(x) == output_size(S) ||
        throw(DimensionMismatch("bad dimensions for input"))
    size(y) == input_size(S) ||
        throw(DimensionMismatch("bad dimensions for output"))
    beta == 1 || vscale!(y, beta)
    if alpha != 0
        A, I, J = S.A, S.I, S.J
        length(I) == length(J) == length(A) ||
            error("corrupted sparse operator structure")
        for k in 1:length(A)
            @inbounds a, i, j = A[k], I[k], J[k]
            y[j] += alpha*a*x[i]
        end
    end
    return y
end
