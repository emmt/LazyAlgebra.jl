#
# cropping.jl -
#
# Provide zero-padding and cropping operators.
#
#-------------------------------------------------------------------------------
#
# This file is part of LazyAlgebra (https://github.com/emmt/LazyAlgebra.jl)
# released under the MIT "Expat" license.
#
# Copyright (c) 2019-2022, Éric Thiébaut.
#

module Cropping

# FIXME: add simplifying rules:
#   Z'*Z = Id (not Z*Z' = Id)  crop zero-padded array is identity

export
    CroppingOperator,
    ZeroPaddingOperator,
    default_offset

using ArrayTools
using ..Foundations
using ..LazyAlgebra
using ..LazyAlgebra: bad_argument, bad_size
import ..LazyAlgebra: Adjoint, apply!, vcreate,
    input_size, input_ndims, output_size, output_ndims, output_eltype

"""
    CroppingOperator(outdims, inpdims, offset=default_offset(outdims,inpdims))

yields a linear map which implements cropping of arrays of size `inpdims` to
produce arrays of size `outdims`. By default, the output array is centered with
respect to the inpput array (using the same conventions as `fftshift`).
Optional argument `offset` can be used to specify a different relative
position. If `offset` is given, the output value at multi-dimensional index `i`
is given by input value at index `j = i + offset`.

The adjoint of a cropping operator is a zero-padding operator which can be
directly built by [`ZeroPaddingOperator`](@ref).

"""
struct CroppingOperator{N} <: LinearMapping
    outdims::Dims{N} # cropped dimensions
    inpdims::Dims{N} # input dimensions
    offset::CartesianIndex{N} # offset of cropped region w.r.t. input array
    function CroppingOperator{N}(outdims::Dims{N},
                                 inpdims::Dims{N}) where {N}
        @inbounds for d in 1:N
            1 ≤ outdims[d] || error("invalid cropped dimension(s)")
            outdims[d] ≤ inpdims[d] || error("cropped dimensions must be less or equal uncropped ones")
        end
        offset = default_offset(inpdims, outdims)
        return new{N}(outdims, inpdims, offset)
    end
    function CroppingOperator{N}(outdims::Dims{N},
                                 inpdims::Dims{N},
                                 offset::CartesianIndex{N}) where {N}
        @inbounds for d in 1:N
            1 ≤ outdims[d] || error("invalid cropped dimension(s)")
            outdims[d] ≤ inpdims[d] || error("cropped dimensions must be less or equal uncropped ones")
            0 ≤ offset[d] ≤ inpdims[d] - outdims[d] || error("out of range offset(s)")
        end
        return new{N}(outdims, inpdims, offset)
    end
end

@callable CroppingOperator

const ZeroPaddingOperator{N} = Adjoint{CroppingOperator{N}}

offset(A::CroppingOperator) = getfield(A, :offset)
offset(A::ZeroPaddingOperator) = offset(parent(A))
input_ndims(::Type{<:CroppingOperator{N}}) where {N} = N
output_ndims(::Type{<:CroppingOperator{N}}) where {N} = N
input_size(A::CroppingOperator) = getfield(A, :inpdims)
output_size(A::CroppingOperator) = getfield(A, :outdims)
cropped_size(A::CroppingOperator) = output_size(A)
cropped_size(A::ZeroPaddingOperator) = input_size(A)

# Union of acceptable types for the offset.
const Offset = Union{CartesianIndex,Integer,Tuple{Vararg{Integer}}}

CroppingOperator(outdims::ArraySize, inpdims::ArraySize) =
    CroppingOperator(to_size(outdims), to_size(inpdims))

CroppingOperator(outdims::ArraySize, inpdims::ArraySize, offset::Offset) =
    CroppingOperator(to_size(outdims), to_size(inpdims), CartesianIndex(offset))

CroppingOperator(::Tuple{Vararg{Int}}, ::Tuple{Vararg{Int}}) =
    error("numbers of output and input dimensions must be equal")

CroppingOperator(::Tuple{Vararg{Int}}, ::Tuple{Vararg{Int}}, ::CartesianIndex) =
    error("numbers of output and input dimensions and offsets must be equal")

CroppingOperator(outdims::Dims{N}, inpdims::Dims{N}) where {N} =
    CroppingOperator{N}(outdims, inpdims)

CroppingOperator(outdims::Dims{N}, inpdims::Dims{N}, offset::CartesianIndex{N}) where {N} =
    CroppingOperator{N}(outdims, inpdims, offset)

output_eltype(::Type{<:CroppingOperator}, ::Type{x}) where {x} = eltype(x)

function vcreate(α::Number,
                 A::Union{CroppingOperator{N},ZeroPaddingOperator{N}},
                 x::AbstractArray{<:Any,N},
                 scratch::Bool) where {N}
    dims = output_size(A)
    T = output_eltype(α, A, x)
    if scratch && isa(x, Array{T,N}) && size(x) == dims
        return x
    else
        return Array{T,N}(undef, dims)
    end
end

function apply!(α::Number,
                A::Union{CroppingOperator{N},ZeroPaddingOperator{N}},
                x::AbstractArray{<:Any,N},
                scratch::Bool,
                β::Number,
                y::AbstractArray{<:Any,N}) where {N}
    # Check arguments.
    has_standard_indexing(x) ||
        bad_argument("input array has non-standard indexing")
    size(x) == input_size(A) ||
        bad_size("bad input array dimensions")
    has_standard_indexing(y) ||
        bad_argument("output array has non-standard indexing")
    size(y) == output_size(A) ||
        bad_size("bad output array dimensions")

    # Apply operator unsafely.
    unsafe_apply!(α, A, x, β, y)
    return y
end

# Apply cropping operation.
#
#     for I in R
#         J = I + K
#         y[I] = α*x[J] + β*y[I]
#     end
#
function unsafe_apply!(α::Number,
                       A::CroppingOperator{N},
                       x::AbstractArray{<:Any,N},
                       β::Number,
                       y::AbstractArray{<:Any,N}) where {N}
    if iszero(α)
        isone(β) || vscale!(y, β)
    else
        k = offset(A)
        I = CartesianIndices(cropped_size(A))
        if isone(α)
            if iszero(β)
                @inbounds @simd for i in I
                    y[i] = x[i + k]
                end
            elseif isone(β)
                @inbounds @simd for i in I
                    y[i] += x[i + k]
                end
            else
                beta = promote_multiplier(β, eltype(y))
                @inbounds @simd for i in I
                    y[i] = x[i + k] + beta*y[i]
                end
            end
        else
            alpha = promote_multiplier(α, eltype(x))
            if iszero(β)
                @inbounds @simd for i in I
                    y[i] = alpha*x[i + k]
                end
            elseif isone(β)
                @inbounds @simd for i in I
                    y[i] += alpha*x[i + k]
                end
            else
                beta = promote_multiplier(β, eltype(y))
                @inbounds @simd for i in I
                    y[i] = alpha*x[i + k] + beta*y[i]
                end
            end
        end
    end
end

# Apply zero-padding operation.
#
#     for i in I
#         y[i + k] = α*x[i] + β*y[i + k]
#     end
#     # Plus y[i + k] *= β outside common region R
#
function usafe_apply!(α::Number,
                      A::ZeroPaddingOperator{N},
                      x::AbstractArray{<:Any,N},
                      β::Number,
                      y::AbstractArray{<:Any,N}) where {N}
    isone(β) || vscale!(y, β)
    if !iszero(α)
        k = offset(A)
        I = CartesianIndices(cropped_size(A))
        if isone(α)
            if iszero(β)
                @inbounds @simd for i in I
                    y[i + k] = x[i]
                end
            else
                @inbounds @simd for i in I
                    y[i + k] += x[i]
                end
            end
        else
            alpha = promote_multiplier(α, eltype(x))
            if iszero(β)
                @inbounds @simd for i in I
                    y[i + k] = alpha*x[i]
                end
            else
                @inbounds @simd for i in I
                    y[i + k] += alpha*x[i]
                end
            end
        end
    end
    return y
end

"""
    ZeroPaddingOperator(outdims, inpdims, offset=default_offset(outdims,inpdims))

yields a linear map which implements zero-padding of arrays of size `inpdims`
to produce arrays of size `outdims`.  By default, the input array is centered
with respect to the output array (using the same conventions as `fftshift`).
Optional argument `offset` can be used to specify a different relative
position.  If `offset` is given, the input value at multi-dimensional index `j`
is copied at index `i = j + offset` in the result.

A zero-padding operator is implemented as the adjoint of a cropping operator.

See also: [`CroppingOperator`](@ref).

"""
ZeroPaddingOperator(outdims, inpdims) =
    Adjoint(CroppingOperator(inpdims, outdims))
ZeroPaddingOperator(outdims, inpdims, offset) =
    Adjoint(CroppingOperator(inpdims, outdims, offset))

"""
    default_offset(dim1,dim2)

yields the index offset such that the centers (in the same sense as assumed by
`fftshift`) of dimensions of lengths `dim1` and `dim2` are coincident. If `off
= default_offset(dim1,dim2)` and `i2` is the index along `dim2`, then the index
along `dim1` is `i1 = i2 + off`.

"""
default_offset(dim1::Integer, dim2::Integer) =
    (Int(dim1) >> 1) - (Int(dim2) >> 1)
default_offset(dims1::NTuple{N,Integer}, dims2::NTuple{N,Integer}) where {N} =
    CartesianIndex(map(default_offset, dims1, dims2))

end # module
