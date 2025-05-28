# Implement generalized matrix and matrix-vector product in LazyAlgebra.

"""
    A = PseudoMatrix{T,M}(arr)
    A = PseudoMatrix{T}(arr, Dims{M})
    A = PseudoMatrix(arr, Dims{M})

build a linear operator `A` whose coefficients are given by a multi-dimensional array
`arr` and whose behavior generalizes the definition of the matrix-vector product.

Type parameter `T` is the element type of the stored coefficients. If `arr` has a
different type of element than `T` it is automatically converted, otherwise the
pseudo-matrix `A` shares its coefficients with `arr`. The array storing the coefficients
of `A` can be retrieved by `parent(A)`. If `T` is unspecified, `T = eltype(arr)` is
assumed.

Type parameter `M` is the number of consecutive leading dimensions of `arr` corresponding
to the *rows* of the pseudo-matrix `A`. An expression like `y = A*x` requires that the
axes of `x` match the `ndims(arr) - M` trailing axes of `arr` and yields a result `y`
whose axes are the `M` leading axes of `arr`.

If `arr` is a matrix (i.e., a 2-dimensional abstract array), then `Operator(arr)`
is a shortcut to `PseudoMatrix(arr,Dims{1})`.

Replacing `Dims{M}` by a colon `:` or type parameters `{T,M}` by `{T,Colon}` yields a
*flexible* pseudo-matrix whose number of row dimensions is not fixed. See
[`FlexibleMatrix`](@ref) for a more convenient constructor.

See also [`FlexibleMatrix`](@ref), [`Operator`](@ref), [`vmul`](@ref), and
[`vmul!`](@ref).

"""
PseudoMatrix(arr::AbstractArray{T}, ::Type{<:Dims{M}}) where {T,M} = PseudoMatrix{T,M}(arr)
PseudoMatrix(arr::AbstractArray{T}, ::Colon) where {T} = PseudoMatrix{T,Colon}(arr)
PseudoMatrix{T,M}(arr::AbstractArray) where {T,M} =
    PseudoMatrix{T,M}(as(AbstractArray{T}, arr))

"""
    A = FlexibleMatrix{T=eltype(arr)}(arr)
    A = PseudoMatrix{T=eltype(arr),Colon}(arr)
    A = PseudoMatrix{T=eltype(arr)}(arr, :)

build a *flexible matrix* `A` that is a linear operator whose coefficients are given by
the multi-dimensional array `arr` and whose behavior generalizes the definition of the
matrix-vector product. Expression like `y = A*x` requires that the axes of `x` match the
trailing axes of `arr` and yields a result `y` whose axes are the remaining leading axes
of `arr`, such that `axes(arr) == (axes(y)..., axes(x)...)` holds. Applying the adjoint of
`A` as in `y = A'*x` requires that the dimensions of `x` match the leading dimension of
`arr` and yields a result `y` whose dimensions are the remaining trailing dimensions of
`arr`, such that `axes(arr) == (axes(x)..., axes(y)...)` holds.

Type parameter `T` is the element type of the stored coefficients. If `arr` has a
different type of element than `T` it is automatically converted, otherwise the flexible
matrix `A` shares its coefficients with `arr`. The array storing the coefficients of `A`
can be retrieved by `parent(A)`. If `T` is unspecified, `T = eltype(arr)` is assumed.

`FlexibleMatrix{T}` is an alias for [`PseudoMatrix{T,Colon}`](@ref PseudoMatrix).

See also [`PseudoMatrix`](@ref), [`Operator`](@ref), [`vmul`](@ref), and [`vmul!`](@ref).

"""
FlexibleMatrix(arr::AbstractArray) = PseudoMatrix(arr, :)
FlexibleMatrix{T}(arr::AbstractArray) where {T} = PseudoMatrix{T}(arr, :)

# Accessors.
Base.parent(A::PseudoMatrix) = getfield(A, :parent)

# Traits.
Base.eltype(::Type{<:PseudoMatrix{T}}) where {T} = T

InputShape(::Type{<:PseudoMatrix{T,M,N}}) where {T,M,N} =
    M !== Colon ? HasInputShape{N-M}() : InputShapeUnknown()

OutputShape(::Type{<:PseudoMatrix{T,M,N}}) where {T,M,N} =
    M !== Colon ? HasOutputShape{M}() : OutputShapeUnknown()

input_axes(A::PseudoMatrix{T,M,N}) where {T,M,N} =
    M !== Colon ? axes(parent(A))[M+1:N] : error(
        "input axes are not known in advance for flexible general matrices")

output_axes(A::PseudoMatrix{T,M,N}) where {T,M,N} =
    M !== Colon ? axes(parent(A))[1:M] : error(
        "output axes are not known in advance for flexible general matrices")

function output_axes(A::Union{G,Adjoint{G},Inverse{G},InverseAdjoint{G}},
                     x_axes::ArrayAxes{L}) where {T,L,N,G<:FlexibleMatrix{T,N}}
    0 ≤ L ≤ N || throw(DimensionMismatch("input array has too many dimensions"))
    if A isa Union{FlexibleMatrix,InverseAdjoint{<:FlexibleMatrix}}
        R = axes(parent(A isa FlexibleMatrix ? A : parent(parent(A))))
        I = R[1:N-L]
        J = R[N-L+1:N]
    else
        R = axes(parent(parent(A)))
        I = R[L+1:N]
        J = R[1:L]
    end
    check_input_axes(x_axes, J)
    return I
end

function unsafe_vmul!(α::Number, A::PseudoMatrix, x::AbstractArray,
                      β::Number, y::AbstractArray)
    C = parent(A) # array storing the coefficients
    I = CartesianIndices(axes(y))
    J = CartesianIndices(axes(x))
    isone(β) || unsafe_vscale!(y, β)
    @inbounds for j in J
        αxⱼ = α*x[j]
        if !iszero(αxⱼ)
            @inbounds @fastmath @simd for i in I
                y[i] += C[i,j]*αxⱼ
            end
        end
    end
    return y
end

function unsafe_vmul!(α::Number, A::Adjoint{<:PseudoMatrix}, x::AbstractArray,
                      β::Number, y::AbstractArray)
    C = parent(A') # array storing the coefficients
    I = CartesianIndices(axes(x))
    J = CartesianIndices(axes(y))
    @inbounds for j in J
        s = 0*zero(eltype(C))*zero(eltype(x))
        @inbounds @fastmath @simd for i in I
            s += conj(C[i,j])*x[i]
        end
        y[j] = α*s + β*y[j]
    end
    return y
end

# Precision of pseudo-matrices and flexible matrices.
get_precision(::Type{T}) where {T<:PseudoMatrix} = get_precision(eltype(T))
_with_precision(::Type{T}, A::PseudoMatrix{<:Any,M}) where {T<:AbstractFloat,M} =
    PseudoMatrix(_with_precision(T, parent(A)), M === Colon ? Colon() : Dims{M})
