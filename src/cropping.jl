#
# cropping.jl -
#
# Implement cropping and zero-padding operators.
#
#-----------------------------------------------------------------------------------------
#
# This file is part of LazyAlgebra (https://github.com/emmt/LazyAlgebra.jl) released under
# the MIT "Expat" license.
#
# Copyright (c), 2019-2025, Éric Thiébaut.
#

# FIXME: add simplifying rules:
#   Z'*Z = Id (not Z*Z' = Id)  crop zero-padded array is identity

"""
    A = CroppingOperator(I, J, k = default_cropping_offset(I, J))

builds a linear operator which implements cropping of arrays of shape `J` to produce
arrays of shape `I`. By default, the output array is centered with respect to the input
one (using the same conventions as `fftshift`). Optional argument `k` is to specify a
different relative position; `k` may be an integer (to assume the same offset in all
dimensions), a tuple of integers, or a Cartesian index. For an array `x` of shape `J`, the
result of `A*x` is an array `y` of shape `I` defined by:

```julia
∀ i ∈ I, y[i] = x[i + k]
```

The adjoint and pseudo-inverse of a cropping operator is a zero-padding operator.

See also [`ZeroPaddingOperator`](@ref).

"""
CroppingOperator(I::RelaxedArrayShape{N}, J::RelaxedArrayShape{N}) where {N} =
    CroppingOperator(as_array_axes(I), as_array_axes(J))

CroppingOperator(I::ArrayAxes{N}, J::ArrayAxes{N}) where {N} =
    CroppingOperator(I, J, default_cropping_offset(I, J))

CroppingOperator(I::RelaxedArrayShape{N}, J::RelaxedArrayShape{N}, k::CartesianIndex{N}) where {N} =
    CroppingOperator(as_array_axes(I), as_array_axes(J), k)

CroppingOperator(I::RelaxedArrayShape{N}, J::RelaxedArrayShape{N}, k::NTuple{N,Integer}) where {N} =
    CroppingOperator(I, J, CartesianIndex(map(as(Int), k)))

# Accessors and operator API for the cropping operator.
output_axes(A::CroppingOperator) = getfield(A, :I)
input_axes( A::CroppingOperator) = getfield(A, :J)
offset(     A::CroppingOperator) = getfield(A, :k)

output_eltype(::Type{<:CroppingOperator}, ::Type{T}) where {T} = T

function unsafe_vmul!(α::Number, A::CroppingOperator{N}, x::AbstractArray{<:Any,N},
                      β::Number, y::AbstractArray{<:Any,N}) where {N}
    k = offset(A)
    I = CartesianIndices(axes(y)) # also output_axes(A)
    if isone(α)
        if iszero(β)
            @inbounds @fastmath @simd for i in I
                y[i] = x[i + k]
            end
        elseif isone(β)
            @inbounds @fastmath @simd for i in I
                y[i] += x[i + k]
            end
        else
            @inbounds @fastmath @simd for i in I
                y[i] = x[i + k] + β*y[i]
            end
        end
    else
        if iszero(β)
            @inbounds @fastmath @simd for i in I
                y[i] = α*x[i + k]
            end
        elseif isone(β)
            @inbounds @fastmath @simd for i in I
                y[i] += α*x[i + k]
            end
        else
            @inbounds @fastmath @simd for i in I
                y[i] = α*x[i + k] + β*y[i]
            end
        end
    end
    nothing
end

# Accessors and operator API for the zero-padding operator which is stored as the adjoint
# of the cropping operator.
output_axes(A::ZeroPaddingOperator) = input_axes(A[])
input_axes( A::ZeroPaddingOperator) = output_axes(A[])
offset(     A::ZeroPaddingOperator) = offset(A[])

output_eltype(::Type{<:ZeroPaddingOperator}, ::Type{T}) where {T} = T

function unsafe_vmul!(α::Number, A::ZeroPaddingOperator{N}, x::AbstractArray{<:Any,N},
                      β::Number, y::AbstractArray{<:Any,N}) where {N}
    fix_vmul_output!(y, β)
    k = offset(A)
    J = CartesianIndices(axes(x)) # also input_axes(A)
    if isone(α)
        if iszero(β)
            @inbounds @fastmath @simd for j in J
                y[j + k] = x[j]
            end
        else
            @inbounds @fastmath @simd for j in J
                y[j + k] += x[j]
            end
        end
    else
        if iszero(β)
            @inbounds @fastmath @simd for j in J
                y[j + k] = α*x[j]
            end
        else
            @inbounds @fastmath @simd for j in J
                y[j + k] += α*x[j]
            end
        end
    end
    nothing
end

# Fix output `y` of `vmul!` so that it can be used as `y[i] += (α*A*x)[i]`
fix_vmul_output!(y::AbstractArray, β::Number) = fix_vmul_output!(β, y)
function fix_vmul_output!(β::Number, y::AbstractArray)
    if !isone(β)
        if iszero(β)
            vzero!(y)
        else
            unsafe_vscale!(y, β)
        end
    end
    return y
end

"""
    A = ZeroPaddingOperator(I, J, k = default_zeropadding_offset(I, J))

builds a linear operator which implements zero-padding of arrays of shape `J` to produce
arrays of shape `I`. By default, the input array is centered with respect to the output
array (using the same conventions as `fftshift`). Optional argument `k` is to specify a
different relative position; `k` may be an integer (to assume the same offset in all
dimensions), a tuple of integers, or a Cartesian index. For an array `x` of shape `J`, the
result of `A*x` is an array `y` of shape `I` defined by:

```julia
∀ i ∈ I, y[i] = x[i - k]    if i - k ∈ J
              = 0           else
```

A zero-padding operator is implemented as the adjoint of a cropping operator.

See also [`CroppingOperator`](@ref).

"""
ZeroPaddingOperator(I, J) = Adjoint(CroppingOperator(J, I))
ZeroPaddingOperator(I, J, k) = Adjoint(CroppingOperator(J, I, k))

"""
    LazyAlgebra.default_cropping_offset(I, J)

yields the offset for the cropping operator such that the centers (in the same sense as
assumed by `fftshift`) of the output and input arrays of respective shapes `I` and `J` are
coincident in a cropping operation.

"""
default_cropping_offset(out::eltype(ArrayShape), inp::eltype(ArrayShape)) =
    offset_to_center(inp) - offset_to_center(out)

default_cropping_offset(out::ArrayShape{N}, inp::ArrayShape{N}) where {N} =
    CartesianIndex(map(default_cropping_offset, out, inp))

"""
    LazyAlgebra.default_zeropadding_offset(I, J)

yields the offset for the zero-padding operator such that the centers (in the same sense
as assumed by `fftshift`) of the output and input arrays of respective shapes `I` and `J`
are coincident in a zero-padding operation.

"""
default_zeropadding_offset(I, J) = default_cropping_offset(J, I)

# `offset_to_center` yields the offset to the center relative to the first index and using
# the same conventions as `fftshift`. It is assumed that the argument is a valid dimension
# length or array axis.
offset_to_center(dim::Integer) = Int(dim) >> 1
offset_to_center(rng::AbstractUnitRange{<:Integer}) = offset_to_center(length(rng))

function Base.show(io::IO, A::CroppingOperator)
    write(io, "Crop(")
    print_shape(io, input_axes(A))
    #print_axes(io, map((r, k) -> (first(r) + k):(last(r) + k),
    #                   output_axes(A), Tuple(offset(A))))
    write(io, " -> ")
    print_shape(io, output_axes(A))
    write(io, " with offset ")
    show(io, Tuple(offset(A)))
    write(io, ')')
end

function Base.show(io::IO, A::ZeroPaddingOperator)
    write(io, "ZeroPad(")
    print_shape(io, input_axes(A))
    #print_axes(io, map((r, k) -> (first(r) - k):(last(r) - k),
    #                   output_axes(A), Tuple(offset(A))))
    write(io, " -> ")
    print_shape(io, output_axes(A))
    write(io, " with offset ")
    show(io, Tuple(offset(A)))
    write(io, ')')
end
