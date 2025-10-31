# Implement cropping and zero-padding operators.

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
function CroppingOperator(I::RelaxedArrayShape{N}, J::RelaxedArrayShape{N}) where {N}
    I = as_array_axes(I)
    J = as_array_axes(J)
    k = default_cropping_offset(I, J)
    CroppingOperator(I, J, k)
end

const CroppingOffset{N} = Union{Integer,NTuple{N,Integer},CartesianIndex{N}}

function CroppingOperator(I::RelaxedArrayShape{N}, J::RelaxedArrayShape{N},
                          k::CroppingOffset{N}) where {N}
    I = as_array_axes(I)
    J = as_array_axes(J)
    k = to_cropping_offset(Val(N), k)
    CroppingOperator(I, J, k)
end

to_cropping_offset(::Val{N}, k::CartesianIndex{N}) where {N} = k
to_cropping_offset(::Val{N}, k::NTuple{N,Integer}) where {N} =
    CartesianIndex(map(as(Int), k))
to_cropping_offset(::Val{N}, k::Integer) where {N} =
    CartesianIndex(ntuple(Returns(Int(k)), Val(N)))

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

function check_cropping_axis(I::AbstractUnitRange{Int},
                             J::AbstractUnitRange{Int},
                             k::Int)
    i_first = first(I)
    i_last  = last(I)
    j_first = first(J)
    j_last  = last(J)
    i_first ≤ i_last || throw_bad_argument("inner region must not be empty")
    ((j_first ≤ i_first + k) & (i_last + k ≤ j_last)) || throw_bad_argument(
        "inner region is not within outer one")
    nothing
end

# Testing for equality. Note that `isequal` amounts to calling `==` by default.
Base.:(==)(A::CroppingOperator{N}, B::CroppingOperator{N}) where {N} =
    A === B || (A.i == B.i && A.j == B.j && A.k == B.k)

# Accessors and operator API for the cropping and zero-padding operators.

offset(A::CroppingOperator) = getfield(A, :k)
offset(A::ZeroPaddingOperator) = offset(A')

output_eltype(::Type{<:CroppingOperator}, ::Type{x}) where {x<:AbstractArray} = eltype(x)
output_eltype(::Type{<:ZeroPaddingOperator}, ::Type{x}) where {x<:AbstractArray} = eltype(x)

InputShape(::Type{<:CroppingOperator{N}}) where {N} = HasInputShape{N}()
input_shape(A::CroppingOperator) = getfield(A, :j)

OutputShape(::Type{<:CroppingOperator{N}}) where {N} = HasOutputShape{N}()
output_shape(A::CroppingOperator) = getfield(A, :i)

for S in (:CroppingOperator, :ZeroPaddingOperator)
    @eval output_eltype(::Type{<:$S{N}}, ::Type{x}) where {T,N,x<:AbstractArray{T,N}} =
        float(T)
end

function unsafe_vmul!(α::Number, A::CroppingOperator{N}, x::AbstractArray{<:Any,N},
                      β::Number, y::AbstractArray{<:Any,N}) where {N}
    k = offset(A)
    I = CartesianIndices(axes(y)) # axes(y) = output_axes(A)
    @inbounds @fastmath @simd for i in I
        y[i] = α*x[i + k] + β*y[i]
    end
    return y
end

function unsafe_vmul!(α::Number, A::ZeroPaddingOperator{N}, x::AbstractArray{<:Any,N},
                      β::Number, y::AbstractArray{<:Any,N}) where {N}
    # Scale or zero-fill y depending on the value of β.
    isone(β) || unsafe_vscale!(y, β)

    # "Copy" x to inner region of y.
    k = offset(A)
    J = CartesianIndices(axes(x)) # axes(x) = input_axes(A')
    if iszero(β)
        @inbounds @fastmath @simd for j in J
            y[j + k] = α*x[j]
        end
    else
        @inbounds @fastmath @simd for j in J
            y[j + k] += α*x[j]
        end
    end
    return y
end

"""
    LazyAlgebra.default_cropping_offset(I, J)

yields the offset for the cropping operator such that the centers (in the same sense as
assumed by `fftshift`) of the inner and outer regions of respective shapes `I` and `J` are
coincident in a cropping operation.

"""
default_cropping_offset(inner::eltype(ArrayShape), outer::eltype(ArrayShape)) =
    offset_to_center(outer) - offset_to_center(inner)

default_cropping_offset(inner::ArrayShape{N}, outer::ArrayShape{N}) where {N} =
    CartesianIndex(map(default_cropping_offset, inner, outer))

# `offset_to_center` yields the offset to the center relative to the first index and using
# the same conventions as `fftshift`. It is assumed that the argument is a valid dimension
# length or array axis.
offset_to_center(dim::Integer) = Int(dim) >> 1
offset_to_center(rng::AbstractUnitRange{<:Integer}) = offset_to_center(length(rng))

function Base.show(io::IO, A::CroppingOperator)
    write(io, "Crop(")
    print_shape(io, input_shape(A))
    write(io, " -> ")
    print_shape(io, output_shape(A))
    write(io, " with offset ")
    show(io, Tuple(offset(A)))
    write(io, ')')
end

function Base.show(io::IO, A::ZeroPaddingOperator)
    write(io, "ZeroPad(")
    print_shape(io, input_shape(A))
    write(io, " -> ")
    print_shape(io, output_shape(A))
    write(io, " with offset ")
    show(io, Tuple(offset(A)))
    write(io, ')')
end
