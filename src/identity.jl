#
# identity.jl -
#
# Implement identity and uniform scaling.
#
#------------------------------------------------------------------------------------------

"""
    Identity(shape = :)

yields the identity operator for arrays of given `shape`. If `shape` is a colon (the
default), any array shape is considered as compatible. The singleton `Identity(:)` is
exported by `LazyAlgebra` as the `Id` alias.

The `LinearAlgebra` module of the standard library exports a constant `I` which also
corresponds to the identity (but in the sense of a matrix). When `I` is combined with any
`LazyAlgebra` operator, it is recognized as an alias of `Id`. So that, for instance,
`I/A`, `A\\I`, `Id/A` and `A\\Id` all yield `inv(A)` for any `LazyAlgebra` mapping `A`.

"""
Identity() = _Identity(:)
Identity(shape::eltype(ArrayShape)...) = Identity(shape)
Identity(shape::ArrayShape) = _Identity(as_array_shape(shape))

Base.show(io::IO, A::typeof(Id)) = write(io, "Id")
function Base.show(io::IO, A::Identity)
    write(io, "Identity(")
    print_shape(io, A.shape)
    write(io, ')')
end

# Implement API of operators for the identity.
output_eltype(::Type{<:Identity}, ::Type{X}) where {X<:AbstractArray} = float(eltype(X))
output_axes(A::UniversalIdentity, shape::ArrayAxes) = shape

InputShape(::Type{<:Identity}) = InputShapeUnknown()
InputShape(::Type{<:ShapedIdentity{N}}) where {N} = HasInputShape{N}()

OutputShape(::Type{<:Identity}) = OutputShapeUnknown()
OutputShape(::Type{<:ShapedIdentity{N}}) where {N} = HasOutputShape{N}()

for f in (:output_axes, :input_axes)
    @eval begin
        $f(A::Identity{<:ArrayAxes}) = A.shape
        $f(A::Identity{<:Dims}) = as_array_axes(A.shape)
        #$f(A::Identity{<:Tuple{}}) = ()
    end
end
for f in (:output_size, :input_size)
    @eval begin
        $f(A::Identity{<:Dims}) = A.shape
        #$f(A::Identity{<:Tuple{}}) = ()
    end
end

unsafe_vmul!(α::Number, A::Identity, x::AbstractArray, β::Number, y::AbstractArray) =
    unsafe_vcombine!(y, α, x, β, y)
unsafe_vmul!(dst::AbstractArray, α::Number, A::typeof(Id), x::AbstractArray) =
    unsafe_scale!(dst, α, x)

# Set precision for identity operators.
set_precision(::Type{T}, A::Identity) where {T<:AbstractFloat} = A

# Taking the adjoint or the inverse of the identity (whatever the i/o shape) does
# nothing.
Adjoint(A::Identity) = A
Inverse(A::Identity) = A

# Special rules for the universal identity which can be automatically simplified at
# construction/compile time (this is not the case of the shaped identity whose shape must
# be checked against that of the other arguments).
#
Prod(A::typeof(Id), B::typeof(Id)) = Id
Prod(A::Operator,   B::typeof(Id)) = A
Prod(A::typeof(Id), B::Operator  ) = B
#
Sum(A::typeof(Id),                B::typeof(Id)               ) = 2Id
Sum(A::Prod{<:Number,typeof(Id)}, B::typeof(Id)               ) = (A[1] + 1) * Id
Sum(A::typeof(Id),                B::Prod{<:Number,typeof(Id)}) = (B[1] + 1) * Id
Sum(A::Prod{<:Number,typeof(Id)}, B::Prod{<:Number,typeof(Id)}) = (A[1] + B[1]) * Id
