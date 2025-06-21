# Implement identity and uniform scaling.

"""
    Identity(shape = :)

yields the identity operator for arrays of given `shape`. If `shape` is a colon (the
default), any array shape is considered as compatible. The singleton `Identity(:)` is
exported by `LazyAlgebra` as the [`Id`](@ref) alias.

The `LinearAlgebra` module of the standard library exports a constant `I` which also
corresponds to the identity (but for usual matrices). When `I` is combined with any
`LazyAlgebra` operator, it is recognized as an alias of `Id`. So that, for instance,
`I/A`, `A\\I`, `Id/A` and `A\\Id` all yield `inv(A)` for any `LazyAlgebra` mapping `A`.

"""
Identity() = _Identity(:)
Identity(shape::eltype(ArrayShape)...) = Identity(shape)
Identity(shape::ArrayShape) = _Identity(as_array_shape(shape))

# MIME"text/plain" is for the REPL.
Base.show(io::IO, ::MIME"text/plain", A::Identity) = show(io, A)
Base.show(io::IO, A::typeof(Id)) = write(io, "Id")
function Base.show(io::IO, A::Identity)
    write(io, "Identity(")
    print_shape(io, A.shape)
    write(io, ')')
end

# Testing for equality. Note that `isequal` amounts to calling `==` by default.
Base.:(==)(A::Identity{Colon}, B::Identity{Colon}) = true
Base.:(==)(A::Identity{<:Dims{N}}, B::Identity{<:Dims{N}}) where {N} =
    A === B || A.shape == B.shape
Base.:(==)(A::Identity{<:ArrayAxes{N}}, B::Identity{<:ArrayAxes{N}}) where {N} =
    A === B || A.shape == B.shape
Base.:(==)(A::Identity{<:NTuple{N}}, B::Identity{<:NTuple{N}}) where {N} =
    input_axes(A) == input_axes(B)

# Get diagonal of identity.
LinearAlgebra.diag(A::UniversalIdentity) = Array{typeof(𝟙),0}(undef)
LinearAlgebra.diag(A::ShapedIdentity) = new_array(typeof(𝟙), input_axes(A))

# Implement API of operators for the identity.
output_eltype(::Type{<:Identity}, ::Type{X}) where {X<:AbstractArray} = float(eltype(X))
output_axes(A::UniversalIdentity, shape::ArrayAxes) = shape

InputShape(::Type{<:Identity}) = InputShapeUnknown()
InputShape(::Type{<:ShapedIdentity{N}}) where {N} = HasInputShape{N}()

OutputShape(::Type{<:Identity}) = OutputShapeUnknown()
OutputShape(::Type{<:ShapedIdentity{N}}) where {N} = HasOutputShape{N}()

input_axes(A::Identity{<:ArrayAxes}) = A.shape
input_size(A::Identity{<:Dims}) = A.shape
input_axes(A::Identity{<:Dims}) = as_array_axes(input_size(A))
input_axes(A::Identity{<:Tuple{}}) = ()

# Output has the same shape as input for the identity.
output_axes(A::Identity) = input_axes(A)
output_size(A::Identity) = input_size(A)

unsafe_vmul!(α::Number, A::Identity, x::AbstractArray, β::Number, y::AbstractArray) =
    unsafe_vcombine!(α, x, β, y)

# Taking the adjoint or the inverse of the identity (whatever the i/o shape) does
# nothing.
Adjoint(A::Identity) = A
Inverse(A::Identity) = A

# Special rules for the universal identity which can be automatically simplified at
# construction/compile time (this is not the case of the shaped identity whose shape must
# be checked against that of the other arguments).
#
Prod(A::typeof(Id), B::typeof(Id)) = Id
for T in (:Operator, :(Prod{<:Operator}), :(Prod{<:Number}))
    @eval begin
        Prod(A::$T, B::typeof(Id)) = A
        Prod(A::typeof(Id), B::$T) = B
    end
end
#
Sum(A::typeof(Id),                B::typeof(Id)               ) = 2Id
Sum(A::Prod{<:Number,typeof(Id)}, B::typeof(Id)               ) = (A[1] + 𝟙) * Id
Sum(A::typeof(Id),                B::Prod{<:Number,typeof(Id)}) = (B[1] + 𝟙) * Id
Sum(A::Prod{<:Number,typeof(Id)}, B::Prod{<:Number,typeof(Id)}) = (A[1] + B[1]) * Id
