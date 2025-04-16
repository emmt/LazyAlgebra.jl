#
# identity.jl -
#
# Implement identity and uniform scaling.
#
#-----------------------------------------------------------------------------------------
#
# This file is part of LazyAlgebra (https://github.com/emmt/LazyAlgebra.jl) released under
# the MIT "Expat" license.
#
# Copyright (c) 2017-2025, Éric Thiébaut.
#

identical(::Identity, ::Identity) = true

@callable Identity

# Implement API of operators for the identity.
output_eltype(::Type{Identity}, ::Type{X}) where {X} = float(X)
output_axes(A::Identity, I::ArrayAxes) = I
create_output(α::Number, ::Identity, x::AbstractArray) =
    similar(x, prod_type(typeof(α), eltype(x)))
unsafe_apply!(α::Number, A::Identity, x::AbstractArray, β::Number, y::AbstractArray) =
    unsafe_vcombine!(y, α, x, β, y)
unsafe_apply!(dst::AbstractArray, α::Number, A::Identity, x::AbstractArray) =
    unsafe_scale!(dst, α, x)

# Special rules for the identity.
#
Adjoint(::Identity) = Id
Inverse(::Identity) = Id
#
Prod(A::Identity, B::Identity) = Id
Prod(A::Operator, B::Identity) = A
Prod(A::Identity, B::Operator) = B
#
Sum(A::Identity,  B::Identity) = 2Id
Sum(A::Prod{<:Number,Identity}, B::Identity)                = (A.left + 1) * Id
Sum(A::Identity,                B::Prod{<:Number,Identity}) = (B.left + 1) * Id
Sum(A::Prod{<:Number,Identity}, B::Prod{<:Number,Identity}) = (A.left + B.left) * Id

# Traits.
SelfAdjointType(::Identity) = SelfAdjoint()
MorphismType(::Identity) = Endomorphism()
DiagonalType(::Identity) = DiagonalOperator()

# Rules to automatically convert `LinearAlgebra.UniformScaling` into `λ*Id` when combined
# with any `LazyAlgebra` operator.
Operator(A::LinearAlgebra.UniformScaling) = A.λ * Id
for op in (:(+), :(-), :(*), :(∘), :(/), Symbol("\\"))
    @eval begin
        Base.$op(A::LinearAlgebra.UniformScaling, B::Operator) = $op(Operator(A), B)
        Base.$op(A::Operator, B::LinearAlgebra.UniformScaling) = $op(A, Operator(B))
    end
end
