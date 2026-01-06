# Implement generalized matrix and matrix-vector product in LazyAlgebra.

"""
    A = PseudoMatrix{T,L}(arr)
    A = PseudoMatrix{T}(arr, Dims{L})
    A = PseudoMatrix(arr, Dims{L})

Build a linear operator `A` whose coefficients are given by a multi-dimensional array
`arr` and whose behavior generalizes the definition of the matrix-vector product.

Type parameter `T` is the element type of the stored coefficients. If `arr` has a
different type of element than `T` it is automatically converted, otherwise the
pseudo-matrix `A` shares its coefficients with `arr`. The array storing the coefficients
of `A` can be retrieved by `parent(A)`. If `T` is unspecified, `T = eltype(arr)` is
assumed.

Type parameter `L` is the number of consecutive *leading dimensions* of `arr` considered
as the *row index* of the pseudo-matrix `A`, the remaining consecutive trailing dimensions
being considered as the *column index* of the pseudo-matrix `A`. In other words, an
expression like `y = A*x` implies that the axes of `x` match the `ndims(arr) - L` trailing
axes of `arr` and that the axes of the result `y` are the `L` leading axes of `arr`.

If `arr` is a matrix (i.e., a 2-dimensional abstract array), then `Operator(arr)`
is a shortcut to `PseudoMatrix(arr,Dims{1})`.

Replacing `Dims{L}` by a colon `:` or type parameters `{T,L}` by `{T,:}` yields a
*flexible* pseudo-matrix whose number of row dimensions is not fixed. See
[`FlexibleMatrix`](@ref) for a more convenient constructor.

See also [`FlexibleMatrix`](@ref), [`Operator`](@ref), [`vmul`](@ref), and
[`vmul!`](@ref).

"""
PseudoMatrix(arr::AbstractArray{T}, ::Type{<:Dims{L}}) where {T,L} = PseudoMatrix{T,L}(arr)
PseudoMatrix(arr::AbstractArray{T}, ::Colon) where {T} = PseudoMatrix{T,:}(arr)
PseudoMatrix{T,L}(arr::AbstractArray) where {T,L} =
    PseudoMatrix{T,L}(as(AbstractArray{T}, arr))

"""
    A = FlexibleMatrix{T=eltype(arr)}(arr)
    A = PseudoMatrix{T=eltype(arr),Colon}(arr)
    A = PseudoMatrix{T=eltype(arr)}(arr, :)

Build a *flexible matrix* `A` that is a linear operator whose coefficients are given by
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

# Testing for equality.
for cmp in (:(==), :isequal)
    @eval begin
        function Base.$cmp(A::PseudoMatrix{<:Any,L,<:AbstractArray{<:Any,N}},
                           B::PseudoMatrix{<:Any,L,<:AbstractArray{<:Any,N}}) where {L,N}
            return A === B || $cmp(parent(A), parent(B))
        end
    end
end

# Traits.
Base.eltype(::Type{<:PseudoMatrix{T}}) where {T} = T

"""
    unveil(A::Operator) -> B::Operator

Return the bare operator embedded in `A`. If `A` is a bare operator, `A` is returned;
otherwise, `parent` is recursively called.

"""
unveil(A::Union{Adjoint,Transpose,Conjugate,Inverse}) = unveil(parent(A))
unveil(A::Operator) = A

InputShape( ::Type{<:FlexibleMatrix}) = InputShapeUnknown()
OutputShape(::Type{<:FlexibleMatrix}) = OutputShapeUnknown()

input_shape(A::FlexibleMatrix) = error(
    "input shape is not known in advance for flexible general matrices")

output_shape(A::FlexibleMatrix) = error(
    "output shape is not known in advance for flexible general matrices")

function output_axes(A::Union{G, Adjoint{G}, Transpose{G}, Conjugate{G}, Inverse{G},
                              InverseAdjoint{G}, InverseTranspose{G}, InverseConjugate{G}},
                     x_axes::ArrayAxes{N}) where {T,P,G<:FlexibleMatrix{T,P},N}
    MN = ndims(P) # total number of dimensions
    N ≤ MN || throw_dimension_mismatch("input array has too many dimensions")
    IJ = axes(parent(unveil(A)))
    if A isa Union{FlexibleMatrix, Conjugate, InverseAdjoint, InverseTranspose}
        I = IJ[1:MN-N]
        J = IJ[MN-N+1:MN]
    else
        I = IJ[N+1:MN]
        J = IJ[1:N]
    end
    check_input_axes(x_axes, J)
    return I
end

InputShape( ::Type{PseudoMatrix{T,L,P}}) where {T,L,P} = HasInputShape{ndims(P)-L}()
OutputShape(::Type{PseudoMatrix{T,L,P}}) where {T,L,P} = HasOutputShape{L}()

input_shape( A::PseudoMatrix{T,L,P}) where {T,L,P} = axes(parent(A))[L+1:ndims(P)]
output_shape(A::PseudoMatrix{T,L,P}) where {T,L,P} = axes(parent(A))[1:L]

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
    C = parent(parent(A)) # array storing the coefficients
    I = CartesianIndices(axes(x))
    J = CartesianIndices(axes(y))
    t = zero(eltype(C))*zero(eltype(x))
    T = typeof(t + t) # type of accumulator
    @inbounds for j in J
        s = zero(T)
        @inbounds @fastmath @simd for i in I
            s += conj(C[i,j])*x[i]
        end
        y[j] = α*s + β*y[j]
    end
    return y
end

# Precision of pseudo-matrices and flexible matrices.
TypeUtils.get_precision(::Type{A}) where {A<:PseudoMatrix} = get_precision(eltype(A))
TypeUtils.adapt_precision(::Type{T}, A::PseudoMatrix{T,L}) where {T<:TypeUtils.Precision,L} = A
TypeUtils.adapt_precision(::Type{T}, A::PseudoMatrix{<:Any,L}) where {T<:TypeUtils.Precision,L} =
    PseudoMatrix(adapt_precision(T, parent(A)), L isa Colon ? Colon() : Dims{L})
