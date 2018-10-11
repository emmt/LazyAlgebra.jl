#
# mappings.jl -
#
# Provide basic mappings.
#
#-------------------------------------------------------------------------------
#
# This file is part of LazyAlgebra (https://github.com/emmt/LazyAlgebra.jl)
# released under the MIT "Expat" license.
#
# Copyright (c) 2017-2018 Éric Thiébaut.
#

#------------------------------------------------------------------------------
# IDENTITY AND UNIFORM SCALING

is_same_mapping(::Identity, ::Identity) = true

@callable Identity

const I = Identity()

# Traits:
SelfAdjointType(::Identity) = SelfAdjoint
MorphismType(::Identity) = Endomorphism
DiagonalType(::Identity) = DiagonalMapping

# Never let the inverse, adjoint or inverse-adjoint of the identity yeild
# something else than identity.
inv(::Identity) = I
adjoint(::Identity) = I

# Extend multiplication for the (scaled) identity.  It is important to account
# for all possible cases.  If all cases are covered, extend the left and right
# divison is not needed.
*(::Identity, ::Identity) = I
for T in (Scaled{Identity}, Mapping)
    @eval begin
        *(A::Identity, B::$T) = B
        *(A::$T, B::Identity) = A
    end
end
*(A::Scaled{Identity}, B::Scaled{Identity}) = (A.sc*B.sc)*I
*(A::Scaled{Identity}, B::Mapping) = A.sc*B
*(A::LinearMapping, B::Scaled{Identity}) = B.sc*A
*(A::Mapping, B::Scaled{Identity}) =
    islinear(A) ? B.sc*A : Composition(A, B)

# Extend addition for the (scaled) identity.  Extending subtraction is not
# necessary.
+(::Identity, ::Identity) = 2*I
+(A::Identity, B::Scaled{Identity}) = (1 + B.sc)*I
+(A::Scaled{Identity}, B::Identity) = (A.sc + 1)*I
+(A::Scaled{Identity}, B::Scaled{Identity}) = (A.sc + B.sc)*I

# Extend equality for the (scaled) identity for a small gain in performances.
==(::Identity, ::Identity) = true
==(::Identity, A::Scaled{Identity}) = (A.sc == one(A.sc))
==(A::Scaled{Identity}, ::Identity) = (A.sc == one(A.sc))
==(A::Scaled{Identity}, B::Scaled{Identity}) = (A.sc == B.sc)

apply(::Type{<:Operations}, ::Identity, x, scratch::Bool=false) = x

vcreate(::Type{<:Operations}, ::Identity, x, scratch::Bool=false) =
    (scratch ? x : vcreate(x))

apply!(α::Real, ::Type{<:Operations}, ::Identity, x, ::Bool, β::Real, y) =
    vcombine!(y, α, x, β, y)

# Rules to automatically convert UniformScaling from standard library module
# LinearAlgebra into λ*I.  For other operators, there is no needs to extend ⋅
# (\cdot) and ∘ (\circ) as they are already converted in calls to *.  But in
# the case of UniformScaling, we must explicitly do that for * and for ∘ (not
# for ⋅ which is replaced by a * by existing rules).
simplify(A::UniformScaling) = A.λ*I
for op in (:(+), :(-), :(*), :(∘), :(/), :(\))
    @eval begin
        Base.$op(A::UniformScaling, B::Mapping) = $op(simplify(A), B)
        Base.$op(A::Mapping, B::UniformScaling) = $op(A, simplify(B))
    end
end

"""
```julia
UniformScalingOperator(α)
```

creates a uniform scaling linear mapping whose effects is to multiply its
argument by the scalar `α`.  This is the same as `α*Identity()`.

!!! note
    This has been deprecated.  Use `α*Identity()` instead.

See also: [`NonuniformScalingOperator`](@ref).

"""
UniformScalingOperator

@deprecate UniformScalingOperator(α::Number) α*Identity()

isinvertible(A::Scaled{Identity}) = (isfinite(A.sc) && A.sc != zero(A.sc))

ensureinvertible(A::Scaled{Identity}) =
    isinvertible(A) ||
    throw(SingularSystem("Uniform scaling operator is singular"))

function inv(A::Scaled{Identity})
    ensureinvertible(A)
    return (1/A.sc)*I
end

#------------------------------------------------------------------------------
# NON-UNIFORM SCALING

"""
```julia
NonuniformScalingOperator(A)
```

creates a nonuniform scaling linear mapping whose effects is to apply
elementwise multiplication of its argument by the scaling factors `A`.
This mapping can be thought as a *diagonal* operator.

See also: [`Diag`](@ref).

"""
struct NonuniformScalingOperator{T} <: LinearMapping
    diag::T
end

@callable NonuniformScalingOperator

# Traits:
MorphismType(::NonuniformScalingOperator) = Endomorphism
DiagonalType(::NonuniformScalingOperator) = DiagonalMapping
SelfAdjointType(A::NonuniformScalingOperator) =
    _selfadjointtype(eltype(contents(A)), A)
_selfadjointtype(::Type{<:Real}, ::NonuniformScalingOperator) =
    SelfAdjoint
_selfadjointtype(::Type{<:Complex}, ::NonuniformScalingOperator) =
    NonSelfAdjoint

contents(A::NonuniformScalingOperator) = A.diag
LinearAlgebra.diag(A::NonuniformScalingOperator) = A.diag

function is_same_mapping(A::NonuniformScalingOperator{T},
                         B::NonuniformScalingOperator{T}) where {T}
    return is_same_mutable_object(diag(A), diag(B))
end

function inv(A::NonuniformScalingOperator{<:AbstractArray{T,N}}
             ) where {T<:AbstractFloat, N}
    q = A.diag
    r = similar(q)
    @inbounds @simd for i in eachindex(q, r)
        r[i] = one(T)/q[i]
    end
    return NonuniformScalingOperator(r)
end

for pfx in (:input, :output)
    @eval begin

        function $(Symbol(pfx,"_type"))(
            ::NonuniformScalingOperator{<:AbstractArray{T,N}}
        ) where {T,N}
            return Array{T,N}
        end

        function $(Symbol(pfx,"_eltype"))(
            ::NonuniformScalingOperator{<:AbstractArray{T,N}}
        ) where {T<:AbstractFloat,N}
            return T
        end

        function $(Symbol(pfx,"_eltype"))(
            ::NonuniformScalingOperator{<:AbstractArray{T,N}}
        ) where {T, N}
            return float(T)
        end

        function $(Symbol(pfx,"_size"))(
            A::NonuniformScalingOperator{<:AbstractArray}
        )
            return size(A.diag)
        end

        function $(Symbol(pfx,"_size"))(
            A::NonuniformScalingOperator{<:AbstractArray},
            args...
        )
            return size(A.diag, args...)
        end

        function $(Symbol(pfx,"_ndims"))(
            ::NonuniformScalingOperator{<:AbstractArray{T,N}}
        ) where {T, N}
            return N
        end

    end
end

"""
```julia
@axpby!(I, a, xi, b, yi)
```

yields the code to perform `y[i] = a*x[i] + b*y[i]` for each index `i ∈ I` with
`x[i]` and `y[i]` respectively given by expression `xi` and `yi`.  The
expression is evaluated efficiently considering the particular values of `a`
and `b`.

***Important*** It is assumed that all axes in `I` are within bounds, that
`a` is nonzero and that `a` and `b` have correct types.

"""
macro axpby!(i, I, a, xi, b, yi)
    esc(_compile_axpby!(i, I, a, xi, b, yi))
end

function _compile_axpby!(i::Symbol, I, a, xi, b, yi)
    quote
        if $a == 1
            if $b == 0
                @inbounds @simd for $i in $I
                    $yi = $xi
                end
            elseif $b == 1
                @inbounds @simd for $i in $I
                    $yi += $xi
                end
            elseif $b == -1
                @inbounds @simd for $i in $I
                    $yi = $xi - $yi
                end
            else
                @inbounds @simd for $i in $I
                    $yi = $xi + $b*$yi
                end
            end
        elseif $a == -1
            if $b == 0
                @inbounds @simd for $i in $I
                    $yi = -$xi
                end
            elseif $b == 1
                @inbounds @simd for $i in $I
                    $yi -= $xi
                end
            elseif $b == -1
                @inbounds @simd for $i in $I
                    $yi = -$xi - $yi
                end
            else
                @inbounds @simd for $i in $I
                    $yi = $b*$yi - $xi
                end
            end
        else
            if $b == 0
                @inbounds @simd for $i in $I
                    $yi = $a*$xi
                end
            elseif $b == 1
                @inbounds @simd for $i in $I
                    $yi += $a*$xi
                end
            elseif $b == -1
                @inbounds @simd for $i in $I
                    $yi = $a*$xi - $yi
                end
            else
                @inbounds @simd for $i in $I
                    $yi = $a*$xi + $b*$yi
                end
            end
        end
    end
end

function apply!(α::Real,
                ::Type{P},
                W::NonuniformScalingOperator{<:AbstractArray{Tw,N}},
                x::AbstractArray{Tx,N},
                scratch::Bool,
                β::Real,
                y::AbstractArray{Ty,N}) where {P<:Operations,
                                               Tw<:AbstractFloat,
                                               Tx<:AbstractFloat,
                                               Ty<:AbstractFloat,N}
    w = contents(W)
    @assert axes(w) == axes(x) == axes(y)
    if α == 0
        rmul!(y, β)
    else
        a = convert_multiplier(α, Tw, Tx) # FIXME: force float if there is a division?
        b = convert_multiplier(β, Ty)
        I = eachindex(w, x, y)
        if P === Direct || P === Adjoint
            @axpby!(i, I, a, w[i]*x[i], b, y[i])
        elseif P === Inverse || P === InverseAdjoint
            @axpby!(i, I, a, x[i]/w[i], b, y[i])
        end
    end
    return y
end

function apply!(α::Real,
                ::Type{P},
                W::NonuniformScalingOperator{<:AbstractArray{Complex{Tw},N}},
                x::AbstractArray{Complex{Tx},N},
                scratch::Bool,
                β::Real,
                y::AbstractArray{Complex{Ty},N}) where {P<:Operations,
                                                        Tw<:AbstractFloat,
                                                        Tx<:AbstractFloat,
                                                        Ty<:AbstractFloat,N}
    w = contents(W)
    @assert axes(w) == axes(x) == axes(y)
    if α == 0
        rmul!(y, β)
    else
        a = convert_multiplier(α, Tw, Tx) # FIXME: force float if there is a division?
        b = convert_multiplier(β, Ty)
        I = eachindex(w, x, y)
        if P === Direct
            # FIXME: expressions can be further optimized...
            @axpby!(i, I, a, w[i]*x[i], b, y[i])
        elseif P === Adjoint
            @axpby!(i, I, a, conj(w[i])*x[i], b, y[i])
        elseif P === Inverse
            @axpby!(i, I, a, x[i]/w[i], b, y[i])
        elseif P === InverseAdjoint
            @axpby!(i, I, a, x[i]/conj(w[i]), b, y[i])
        end
    end
    return y
end

function vcreate(::Type{<:Operations},
                 W::NonuniformScalingOperator{<:AbstractArray{Tw,N}},
                 x::AbstractArray{Tx,N},
                 scratch::Bool=false) where {Tw,Tx,N}
    inds = axes(W.diag)
    @assert axes(x) == inds
    T = promote_type(Tw, Tx)
    return (scratch && Tx == T ? x : similar(Array{T}, inds))
end

#------------------------------------------------------------------------------
# RANK-1 OPERATORS

"""

A `RankOneOperator` is defined by two *vectors* `u` and `v` and created by:

```julia
A = RankOneOperator(u, v)
```

and behaves as if `A = u⋅v'`; that is:

```julia
A*x  = vscale(vdot(v, x)), u)
A'*x = vscale(vdot(u, x)), v)
```

See also: [`SymmetricRankOneOperator`](@ref), [`LinearMapping`](@ref),
          [`apply!`](@ref), [`vcreate`](@ref).

"""
struct RankOneOperator{U,V} <: LinearMapping
    u::U
    v::V
end

@callable RankOneOperator

function apply!(α::Real, ::Type{Direct}, A::RankOneOperator, x, scratch::Bool,
                β::Real, y)
    return _apply_rank_one_operator!(α, A.u, A.v, x, β, y)
end

function apply!(α::Real, ::Type{Adjoint}, A::RankOneOperator, x, scratch::Bool,
                β::Real, y)
    return _apply_rank_one_operator!(α, A.v, A.u, x, β, y)
end

function _apply_rank_one_operator!(α::Real, u, v, x, β::Real, y)
    if α == 0
        # Lazily assume that y has correct type, dimensions, etc.
        vscale!(y, β)
    else
        vcombine!(y, α*vdot(v, x), u, β, y)
    end
    return y
end


# Lazily assume that x has correct type, dimensions, etc.
# FIXME: optimize when scratch=true
vcreate(::Type{Direct}, A::RankOneOperator, x, scratch::Bool=false) =
    vcreate(A.v)
vcreate(::Type{Adjoint}, A::RankOneOperator, x, scratch::Bool=false) =
    vcreate(A.u)

input_type(A::RankOneOperator{U,V}) where {U,V} = V
input_ndims(A::RankOneOperator) = ndims(A.v)
input_size(A::RankOneOperator) = size(A.v)
input_size(A::RankOneOperator, d...) = size(A.v, d...)
input_eltype(A::RankOneOperator) = eltype(A.v)

output_type(A::RankOneOperator{U,V}) where {U,V} = U
output_ndims(A::RankOneOperator) = ndims(A.u)
output_size(A::RankOneOperator) = size(A.u)
output_size(A::RankOneOperator, d...) = size(A.u, d...)
output_eltype(A::RankOneOperator) = eltype(A.u)

function is_same_mapping(A::RankOneOperator{U,V},
                         B::RankOneOperator{U,V}) where {U,V}
    return (is_same_mutable_object(A.u, B.u) &&
            is_same_mutable_object(A.v, B.v))
end

"""

A `SymmetricRankOneOperator` is defined by a *vector* `u` and created by:

```julia
A = SymmetricRankOneOperator(u)
```

and behaves as if `A = u⋅u'`; that is:

```julia
A*x = A'*x = vscale(vdot(u, x)), u)
```

See also: [`RankOneOperator`](@ref), [`LinearMapping`](@ref),
          [`Trait`](@ref) [`apply!`](@ref), [`vcreate`](@ref).

"""
struct SymmetricRankOneOperator{U} <: LinearMapping
    u::U
end

@callable SymmetricRankOneOperator

# Traits:
MorphismType(::SymmetricRankOneOperator) = Endomorphism
SelfAdjointType(::SymmetricRankOneOperator) = SelfAdjoint

function apply!(α::Real, ::Type{<:Union{Direct,Adjoint}},
                A::SymmetricRankOneOperator, x, scratch::Bool, β::Real, y)
    return _apply_rank_one_operator!(α, A.u, A.u, x, β, y)
end

function vcreate(::Type{<:Union{Direct,Adjoint}},
                 A::SymmetricRankOneOperator, x, scratch::Bool=false)
    # Lazily assume that x has correct type, dimensions, etc.
    return (scratch ? x : vcreate(x))
end

input_type(A::SymmetricRankOneOperator{U}) where {U} = U
input_ndims(A::SymmetricRankOneOperator) = ndims(A.u)
input_size(A::SymmetricRankOneOperator) = size(A.u)
input_size(A::SymmetricRankOneOperator, d...) = size(A.u, d...)
input_eltype(A::SymmetricRankOneOperator) = eltype(A.u)

output_type(A::SymmetricRankOneOperator{U}) where {U} = U
output_ndims(A::SymmetricRankOneOperator) = ndims(A.u)
output_size(A::SymmetricRankOneOperator) = size(A.u)
output_size(A::SymmetricRankOneOperator, d...) = size(A.u, d...)
output_eltype(A::SymmetricRankOneOperator) = eltype(A.u)

function is_same_mapping(A::SymmetricRankOneOperator{U},
                         B::SymmetricRankOneOperator{U}) where {U}
    return is_same_mutable_object(A.u, B.u)
end

#------------------------------------------------------------------------------
# GENERALIZED MATRIX AND MATRIX-VECTOR PRODUCT

"""
```julia
GeneralMatrix(A)
```

creates a linear mapping given a multi-dimensional array `A` whose interest is
to generalize the definition of the matrix-vector product without calling
`reshape` to change the dimensions.

For instance, assuming that `G = GeneralMatrix(A)` with `A` a regular array,
then `y = G*x` requires that the dimensions of `x` match the trailing
dimensions of `A` and yields a result `y` whose dimensions are the remaining
leading dimensions of `A`, such that `axes(A) = (axes(y)...,
axes(x)...)`.  Applying the adjoint of `G` as in `y = G'*x` requires that
the dimensions of `x` match the leading dimension of `A` and yields a result
`y` whose dimensions are the remaining trailing dimensions of `A`, such that
`axes(A) = (axes(x)..., axes(y)...)`.

See also: [`reshape`](@ref).

"""
struct GeneralMatrix{T<:AbstractArray} <: LinearMapping
    arr::T
end

@callable GeneralMatrix

contents(A) = A.arr # FIXME: coefs(A) ?, rows(A), cols(A)/colums(A)

# Make a GeneralMatrix behaves like an ordinary array.
eltype(A::GeneralMatrix) = eltype(A.arr)
length(A::GeneralMatrix) = length(A.arr)
ndims(A::GeneralMatrix) = ndims(A.arr)
axes(A::GeneralMatrix) = axes(A.arr)
size(A::GeneralMatrix) = size(A.arr)
size(A::GeneralMatrix, inds...) = size(A.arr, inds...)
getindex(A::GeneralMatrix, inds...) = getindex(A.arr, inds...)
setindex!(A::GeneralMatrix, x, inds...) = setindex!(A.arr, x, inds...)
stride(A::GeneralMatrix, k) = stride(A.arr, k)
strides(A::GeneralMatrix) = strides(A.arr)
eachindex(A::GeneralMatrix) = eachindex(A.arr)

function is_same_mapping(A::GeneralMatrix{T},
                         B::GeneralMatrix{T}) where {T}
    return is_same_mutable_object(A.arr, B.arr)
end

function apply!(α::Real,
                P::Type{<:Operations},
                A::GeneralMatrix{<:AbstractArray{<:AbstractFloat}},
                x::AbstractArray{<:AbstractFloat},
                scratch::Bool,
                β::Real,
                y::AbstractArray{<:AbstractFloat})
    return apply!(α, P, A.arr, x, scratch, β, y)
end

function vcreate(P::Type{<:Operations},
                 A::GeneralMatrix{<:AbstractArray{<:AbstractFloat}},
                 x::AbstractArray{<:AbstractFloat},
                 scratch::Bool=false)
    return vcreate(P, A.arr, x, scratch)
end

# FIXME: code all other cases for apply and apply!, do this by meta-code

function apply(A::AbstractArray{<:Number},
               x::AbstractArray{<:Number},
               scratch::Bool=false)
    return apply(Direct, A, x, scratch)
end

function apply(P::Type{<:Operations},
               A::AbstractArray{<:Number},
               x::AbstractArray{<:Number},
               scratch::Bool=false)
    return apply!(1, P, A, x, scratch, 0, vcreate(P, A, x, scratch))
end

function apply!(y::AbstractArray{<:Number},
                A::AbstractArray{<:Number},
                x::AbstractArray{<:Number})
    return apply!(1, Direct, A, x, false, 0, y)
end

function apply!(y::AbstractArray{<:Number},
                P::Type{<:Operations},
                A::AbstractArray{<:Number},
                x::AbstractArray{<:Number})
    return apply!(1, P, A, x, false, 0, y)
end

# By default, use pure Julia code for the generalized matrix-vector product.
function apply!(α::Real,
                P::Type{<:Union{Direct,InverseAdjoint}},
                A::AbstractArray{<:Real},
                x::AbstractArray{<:Real},
                scratch::Bool,
                β::Real,
                y::AbstractArray{<:Real})
    axes(A) == (axes(y)..., axes(x)...) ||
        throw(DimensionMismatch("`x` and/or `y` have axes incompatible with `A`"))
    return _apply!(α, P, A, x, β, y)
end

function apply!(α::Real,
                P::Type{<:Union{Adjoint,Inverse}},
                A::AbstractArray{<:Real},
                x::AbstractArray{<:Real},
                scratch::Bool,
                β::Real,
                y::AbstractArray{<:Real})
    axes(A) == (axes(x)..., axes(y)...) ||
        throw(DimensionMismatch("`x` and/or `y` have axes incompatible with `A`"))
    return _apply!(α, P, A, x, β, y)
end

function vcreate(P::Type{<:Union{Direct,InverseAdjoint}},
                 A::AbstractArray{Ta,Na},
                 x::AbstractArray{Tx,Nx},
                 scratch::Bool=false) where {Ta,Na,Tx,Nx}
    Ainds = axes(A)
    xinds = axes(x)
    Ny = Na - Nx
    (Nx < Na && xinds == Ainds[Ny+1:end]) ||
        throw(DimensionMismatch("the axes of `x` do not match the trailing axes of `A`"))
    Ty = promote_type(Ta, Tx)
    yinds = Ainds[1:Ny]
    return (scratch && Nx == Ny && xinds == yinds ? x :
            similar(Array{Ty}, yinds))
end

function vcreate(P::Type{<:Union{Adjoint,Inverse}},
                 A::AbstractArray{Ta,Na},
                 x::AbstractArray{Tx,Nx},
                 scratch::Bool=false) where {Ta,Na,Tx,Nx}
    Ainds = axes(A)
    xinds = axes(x)
    Ny = Na - Nx
    (Nx < Na && xinds == Ainds[1:Nx]) ||
        throw(DimensionMismatch("the axes of `x` do not match the leading axes of `A`"))
    yinds = Ainds[Nx+1:end]
    Ty = promote_type(Ta, Tx)
    return (scratch && Nx == Ny && xinds == yinds ? x :
            similar(Array{Ty}, yinds))
end

# Pure Julia code implementations.

function _apply!(α::Real,
                 ::Type{Direct},
                 A::AbstractArray{Ta},
                 x::AbstractArray{Tx},
                 β::Real,
                 y::AbstractArray{Ty}) where {Ta,Tx,Ty}
    if β != 1
        vscale!(y, β)
    end
    if α != 0
        # Loop through the coefficients of A assuming column-major storage
        # order.
        alpha = convert_multiplier(α, Ta, Tx)
        I, J = fastrange(axes(y)), fastrange(axes(x))
        @inbounds for j in J
            xj = alpha*x[j]
            if xj != zero(xj)
                @simd for i in I
                    y[i] += A[i,j]*xj
                end
            end
        end
    end
    return y
end

function _apply!(y::AbstractArray{Ty},
                 ::Type{Adjoint},
                 A::AbstractArray{Ta},
                 x::AbstractArray{Tx}) where {Ta,Tx,Ty}
    return _apply!(promote_type(Ty, Ta, Tx), y, Adjoint, A, x)
end

function _apply!(α::Real,
                 ::Type{Adjoint},
                 A::AbstractArray{Ta},
                 x::AbstractArray{Tx},
                 β::Real,
                 y::AbstractArray{Ty}) where {Ta,Tx,Ty}
    if α == 0
        vscale!(y, β)
    else
        # Loop through the coefficients of A assuming column-major storage
        # order.
        T = promote_type(Ta, Tx)
        alpha = convert_multiplier(α, T)
        I, J = fastrange(axes(x)), fastrange(axes(y))
        if β == 0
            @inbounds for j in J
                local s::T = zero(T)
                @simd for i in I
                    s += A[i,j]*x[i]
                end
                y[j] = alpha*s
            end
        else
            beta = convert_multiplier(β, Ty)
            @inbounds for j in J
                local s::T = zero(T)
                @simd for i in I
                    s += A[i,j]*x[i]
                end
                y[j] = alpha*s + beta*y[j]
            end
        end
    end
    return y
end

#------------------------------------------------------------------------------
# HESSIAN AND HALF HESSIAN

is_same_mapping(A::Hessian{T}, B::Hessian{T}) where {T} =
    is_same_mapping(A.obj, B.obj)

is_same_mapping(A::HalfHessian{T}, B::HalfHessian{T}) where {T} =
    is_same_mapping(A.obj, B.obj)

# Default method for allocating the result for Hessian and HalfHessian linear
# mappings.
vcreate(::Type{Direct}, ::Union{Hessian,HalfHessian}, x) = vcreate(x)
