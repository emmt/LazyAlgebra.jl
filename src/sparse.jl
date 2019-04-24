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
# Copyright (c) 2017-2019 Éric Thiébaut.
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
    For efficiency reasons, sparse operators are currently limited to *flat*
    Julia arrays because they can be indexed linearly with no loss of
    performances.  If `C`, `I` and/or `J` are not flat arrays, they will be
    automatically converted to regular arrays.

See also [`isflatarray`](@ref).
"""
struct SparseOperator{T,M,N,
                      Tc<:AbstractVector{T},
                      Ti<:AbstractVector{Int},
                      Tj<:AbstractVector{Int}} <: LinearMapping
    outdims::NTuple{M,Int}
    inpdims::NTuple{N,Int}
    C::Tc
    I::Ti
    J::Tj
    samedims::Bool

    # The inner constructor checks whether arguments are indeed flat arrays,
    # it is not meant to be directly called.
    function SparseOperator{T,M,N,Tc,Ti,Tj}(
        outdims::NTuple{M,Int},
        inpdims::NTuple{N,Int},
        C::Tc,
        I::Ti,
        J::Tj
    ) where {T,M,N,
             Tc<:AbstractVector{T},
             Ti<:AbstractVector{Int},
             Tj<:AbstractVector{Int}}
        @assert isflatarray(C, I, J)
        samedims = (M == N && outdims == inpdims)
        return new{T,M,N,Tc,Ti,Tj}(outdims, inpdims, C, I, J, samedims)
    end
end

@callable SparseOperator

# helper to call inner constructor
function _sparseoperator(outdims::NTuple{M,Int},
                         inpdims::NTuple{N,Int},
                         C::Tc,
                         I::Ti,
                         J::Tj) where {T,M,N,
                                       Tc<:AbstractVector{T},
                                       Ti<:AbstractVector{Int},
                                       Tj<:AbstractVector{Int}}
    return SparseOperator{T,M,N,Tc,Ti,Tj}(outdims, inpdims, C, I, J)
end

function SparseOperator(outdims::NTuple{M,Integer},
                        inpdims::NTuple{N,Integer},
                        C::AbstractVector{T},
                        I::AbstractVector{<:Integer},
                        J::AbstractVector{<:Integer}) where {T,M,N}
    return _sparseoperator(map(Int, outdims),
                           map(Int, inpdims),
                           flatvector(T,   C),
                           flatvector(Int, I),
                           flatvector(Int, J))
end

function SparseOperator(A::AbstractArray{T,N}, n::Integer = 1) where {T,N}
    1 ≤ n < N || throw(ArgumentError("bad number of of leading dimensions"))
    has_standard_indexing(A) || throw(ArgumentError("array has non-standard indices"))
    dims = size(A)
    outdims = dims[1:n]
    inpdims = dims[n+1:end]
    nrows = prod(outdims)
    ncols = prod(inpdims)
    nz = 0
    @inbounds for k in eachindex(A)
        if A[k] != zero(T)
            nz += 1
        end
    end
    C = Array{T}(undef, nz)
    I = Array{Int}(undef, nz)
    J = Array{Int}(undef, nz)
    i = j = l = 0
    @inbounds for k in eachindex(A)
        if i < nrows
            i += 1
        else
            i = 1
            j += 1
        end
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

function is_same_mapping(A::SparseOperator{T,M,N,Tc,Ti,Tj},
                         B::SparseOperator{T,M,N,Tc,Ti,Tj}) where {T,M,N,
                                                                   Tc,Ti,Tj}
    return (is_same_mutable_object(coefs(A), coefs(B)) &&
            is_same_mutable_object(rows(A), rows(B)) &&
            is_same_mutable_object(cols(A), cols(B)) &&
            input_size(A) == input_size(B) &&
            output_size(A) == output_size(B))
end

EndomorphismType(S::SparseOperator) = (S.samedims ? Endomorphism : Morphism)

_bad_input_dimensions() = throw(DimensionMismatch("bad input dimensions"))
_bad_output_dimensions() = throw(DimensionMismatch("bad output dimensions"))

_bad_input_indexing() =
    throw(ArgumentError("input array has non-standard indices"))
_bad_output_indexing() =
    throw(ArgumentError("output array has non-standard indices"))

function vcreate(::Type{Direct},
                 S::SparseOperator{Ts,M,N},
                 x::AbstractArray{Tx,N},
                 scratch::Bool=false) where {Ts<:Real,Tx<:Real,M,N}
    # In-place operation is not possible so we simply ignore the scratch flag.
    size(x) == input_size(S) || _bad_input_dimensions()
    return Array{promote_type(Ts,Tx)}(undef, output_size(S))
end

function vcreate(::Type{Adjoint},
                 S::SparseOperator{Ts,M,N},
                 x::AbstractArray{Tx,M},
                 scratch::Bool=false) where {Ts<:Real,Tx<:Real,M,N}
    # In-place operation is not possible so we simply ignore the scratch flag.
    size(x) == output_size(S) || _bad_input_dimensions()
    return Array{promote_type(Ts,Tx)}(undef, input_size(S))
end

function apply!(α::Real,
                ::Type{Direct},
                S::SparseOperator{Ts,M,N},
                x::AbstractArray{Tx,N},
                scratch::Bool,
                β::Real,
                y::AbstractArray{Ty,M}) where {Ts<:Number,Tx<:Number,
                                               Ty<:Number,M,N}
    size(x) == input_size(S)  || _bad_input_dimensions()
    has_standard_indexing(x)  || _bad_input_indexing()
    size(y) == output_size(S) || _bad_output_dimensions()
    has_standard_indexing(y)  || _bad_output_indexing()
    β == 1 || vscale!(y, β)
    α == 0 || _apply_sparse!(y,
                             convert_multiplier(α, promote_type(Ts, Tx), Ty),
                             coefs(S), rows(S), cols(S), x)
    return y
end

function apply!(α::Real,
                ::Type{Adjoint},
                S::SparseOperator{Ts,M,N},
                x::AbstractArray{Tx,M},
                scratch::Bool,
                β::Real,
                y::AbstractArray{Ty,N}) where {Ts<:Number,Tx<:Number,
                                               Ty<:Number,M,N}
    size(x) == output_size(S) || _bad_input_dimensions()
    has_standard_indexing(x)  || _bad_input_indexing()
    size(y) == input_size(S)  || _bad_output_dimensions()
    has_standard_indexing(y)  || _bad_output_indexing()
    β == 1 || vscale!(y, β)
    α == 0 || _apply_sparse!(y, convert_multiplier(α, promote_type(Ts, Tx), Ty),
                             coefs(S), cols(S), rows(S), x)
    return y
end

function _apply_sparse!(y::AbstractArray{<:Real},
                        α::Real,
                        C::AbstractVector{<:Real},
                        I::AbstractVector{Int},
                        J::AbstractVector{Int},
                        x::AbstractArray{<:Real})
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

function _apply_sparse!(y::AbstractArray{<:Complex},
                        α::Real,
                        C::AbstractVector{<:Complex},
                        I::AbstractVector{Int},
                        J::AbstractVector{Int},
                        x::AbstractArray{<:Real})
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

function _apply_sparse!(y::AbstractArray{<:Complex},
                        α::Real,
                        C::AbstractVector{<:Complex},
                        I::AbstractVector{Int},
                        J::AbstractVector{Int},
                        x::AbstractArray{<:Complex})
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
