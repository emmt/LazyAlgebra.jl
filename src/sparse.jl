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
SparseOperator(outdims, inpdims, C, I, J)
```

yields a sparse linear map whose *rows* and *columns* have respective
dimensions `outdims` and `inpdims` and whose non-zero coefficients are given by
`C` with corresponding row and column linear indices respectively given by `I`
and `J`.

```julia
SparseOperator(C, n=1)
```

yields an instance of `SparseOperator` whose coefficients are the non-zero
coefficients of array `C` and which implements generalized matrix
multiplication by `C` in such a way that the result of applying the operator
are arrays whose dimensions are the `n` leading dimensions of `C` for input
arrays whose dimensions are the remaining trailing dimensions of `C`.

!!! note
    For efficiency reasons, sparse operators are currently limited to dense
    Julia arrays because they can be indexed linearly with no loss of
    performances.  If `C`, `I` and/or `J` are not dense arrays, they will be
    automatically converted to regular arrays.

"""
struct SparseOperator{T,M,N,
                      Tc<:DenseVector{T},
                      Ti<:DenseVector{Int},
                      Tj<:DenseVector{Int}} <: LinearMapping
    outdims::NTuple{M,Int}
    inpdims::NTuple{N,Int}
    C::Tc
    I::Ti
    J::Tj
    samedims::Bool
    function SparseOperator{T,M,N,Tc,Ti,Tj}(
        outdims::NTuple{M,Int},
        inpdims::NTuple{N,Int},
        C::Tc,
        I::Ti,
        J::Tj
    ) where {
        Tc<:DenseVector{T},
        Ti<:DenseVector{Int},
        Tj<:DenseVector{Int}
    } where {
        T,M,N
    }
        samedims = (M == N && outdims == inpdims)
        return new{T,M,N,Tc,Ti,Tj}(outdims, inpdims, C, I, J, samedims)
    end
end

@callable SparseOperator

function SparseOperator(outdims::NTuple{M,Int},
                        inpdims::NTuple{N,Int},
                        C::Tc,
                        I::Ti,
                        J::Tj) where {Tc<:DenseVector{T},
                                      Ti<:DenseVector{Int},
                                      Tj<:DenseVector{Int}} where {T,M,N}
    return SparseOperator{T,M,N,Tc,Ti,Tj}(outdims, inpdims, C, I, J)
end

function SparseOperator(outdims::NTuple{M,Integer},
                        inpdims::NTuple{N,Integer},
                        C::AbstractVector{T},
                        I::AbstractVector{<:Integer},
                        J::AbstractVector{<:Integer}) where {T,M,N}
    return SparseOperator(map(Int, outdims),
                          map(Int, inpdims),
                          densevector(T,   C),
                          densevector(Int, I),
                          densevector(Int, J))
end

function SparseOperator(A::DenseArray{T,N}, n::Integer = 1) where {T,N}
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

coefs(S::SparseOperator) = S.C
rows(S::SparseOperator) = S.I
cols(S::SparseOperator) = S.J

input_size(S::SparseOperator) = S.inpdims
output_size(S::SparseOperator) = S.outdims

EndomorphismType(S::SparseOperator) = (S.samedims ? Endomorphism : Morphism)

function vcreate(::Type{Direct},
                 S::SparseOperator{Ts,M,N},
                 x::DenseArray{Tx,N},
                 scratch::Bool=false) where {Ts<:Real,Tx<:Real,M,N}
    # In-place operation is not possible so we simply ignore the scratch flag.
    size(x) == input_size(S) ||
        throw(DimensionMismatch("bad dimensions for input"))
    return Array{promote_type(Ts,Tx)}(undef, output_size(S))
end

function vcreate(::Type{Adjoint},
                 S::SparseOperator{Ts,M,N},
                 x::DenseArray{Tx,M},
                 scratch::Bool=false) where {Ts<:Real,Tx<:Real,M,N}
    # In-place operation is not possible so we simply ignore the scratch flag.
    size(x) == output_size(S) ||
        throw(DimensionMismatch("bad dimensions for input"))
    return Array{promote_type(Ts,Tx)}(undef, input_size(S))
end

function apply!(α::Real,
                ::Type{Direct},
                S::SparseOperator{Ts,M,N},
                x::DenseArray{Tx,N},
                scratch::Bool,
                β::Real,
                y::DenseArray{<:Number,M}) where {Ts<:Number,Tx<:Number,M,N}
    size(x) == input_size(S) ||
        throw(DimensionMismatch("bad dimensions for input"))
    size(y) == output_size(S) ||
        throw(DimensionMismatch("bad dimensions for output"))
    β == 1 || vscale!(y, β)
    α == 0 || _apply_sparse!(y, convert_multiplier(α, Ts, Tx),
                             coefs(S), rows(S), cols(S), x)
    return y
end

function apply!(α::Real,
                ::Type{Adjoint},
                S::SparseOperator{Ts,M,N},
                x::DenseArray{Tx,M},
                scratch::Bool,
                β::Real,
                y::DenseArray{<:Number,N}) where {Ts<:Number,Tx<:Number,M,N}
    size(x) == output_size(S) ||
        throw(DimensionMismatch("bad dimensions for input"))
    size(y) == input_size(S) ||
        throw(DimensionMismatch("bad dimensions for output"))
    β == 1 || vscale!(y, β)
    α == 0 || _apply_sparse!(y, convert_multiplier(α, Ts, Tx),
                             coefs(S), cols(S), rows(S), x)
    return y
end

function _apply_sparse!(y::DenseArray{<:Real},
                        α::Real,
                        C::DenseVector{<:Real},
                        I::DenseVector{Int},
                        J::DenseVector{Int},
                        x::DenseArray{<:Real})
    # We already known that α is non-zero.
    length(I) == length(J) == length(C) ||
        error("corrupted sparse operator structure")
    if α == 1
        for k in 1:length(C)
            @inbounds c, i, j = C[k], I[k], J[k]
            y[i] += c*x[j]
        end
    elseif α == -1
        for k in 1:length(C)
            @inbounds c, i, j = C[k], I[k], J[k]
            y[i] -= c*x[j]
        end
    else
        for k in 1:length(C)
            @inbounds c, i, j = C[k], I[k], J[k]
            y[i] += α*c*x[j]
        end
    end
end

function _apply_sparse!(y::DenseArray{<:Complex},
                        α::Real,
                        C::DenseVector{<:Complex},
                        I::DenseVector{Int},
                        J::DenseVector{Int},
                        x::DenseArray{<:Real})
    # We already known that α is non-zero.
    if α == 1
        for k in 1:length(C)
            @inbounds c, i, j = C[k], I[k], J[k]
            y[i] += c*x[j]
        end
    elseif α == -1
        for k in 1:length(C)
            @inbounds c, i, j = C[k], I[k], J[k]
            y[i] -= c*x[j]
        end
    else
        for k in 1:length(C)
            @inbounds c, i, j = C[k], I[k], J[k]
            y[i] += c*(α*x[j])
        end
    end
end

function _apply_sparse!(y::DenseArray{<:Complex},
                        α::Real,
                        C::DenseVector{<:Complex},
                        I::DenseVector{Int},
                        J::DenseVector{Int},
                        x::DenseArray{<:Complex})
    # We already known that α is non-zero.
    if α == 1
        for k in 1:length(C)
            @inbounds c, i, j = C[k], I[k], J[k]
            y[i] += c*x[j]
        end
    elseif α == -1
        for k in 1:length(C)
            @inbounds c, i, j = C[k], I[k], J[k]
            y[i] -= c*x[j]
        end
    else
        for k in 1:length(C)
            @inbounds c, i, j = C[k], I[k], J[k]
            y[i] += α*c*x[j]
        end
    end
end

function check_sizes(::Type{<:Union{Direct,InverseAdjoint}},
                     A::Mapping, x::AbstractArray, y::AbstractArray)
end

function check_sizes(::Type{<:Union{Adjoint,Inverse}},
                     A::Mapping, x::AbstractArray, y::AbstractArray)
end
