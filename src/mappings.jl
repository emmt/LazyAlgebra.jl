#
# mappings.jl -
#
# Provide basic mappings.
#
#-------------------------------------------------------------------------------
#
# This file is part of the LazyAlgebra package released under the MIT "Expat"
# license.
#
# Copyright (c) 2017-2018 Éric Thiébaut.
#

#------------------------------------------------------------------------------
# IDENTITY

"""
```julia
Identity()
```

yields the identity linear mapping.  Beware that the purpose of this mapping
is to be as efficient as possible, hence the result of applying this mapping
may be the same as the input argument.

"""
struct Identity <: LinearMapping; end

@callable Identity

const I = Identity()

const Identities = Union{Identity,Adjoint{Identity},Inverse{Identity},
                         InverseAdjoint{Identity}}

# Traits:
selfadjointtype(::Identities) = SelfAdjoint
morphismtype(::Identities) = Endomorphism
diagonaltype(::Identities) = DiagonalMapping
inplacetype(::Type{<:Operations}, ::Identities) = InPlace

Base.inv(::Identities) = I
adjoint(::Identities) = I
*(::Identities, A::Mapping) = A
*(A::Mapping, ::Identities) = A
*(::Identities, ::Identities) = I

apply(::Type{<:Operations}, ::Identity, x) = x

apply!(α::Real, ::Type{<:Operations}, ::Identity, x, β::Real, y) =
    vcombine!(y, α, x, β, y)

vcreate(::Type{<:Operations}, ::Identity, x) = vcreate(x)

#------------------------------------------------------------------------------
# UNIFORM SCALING

"""
```julia
UniformScalingOperator(α)
```

creates a uniform scaling linear mapping whose effects is to multiply its
argument by the scalar `α`.

See also: [`NonuniformScalingOperator`](@ref).

"""
struct UniformScalingOperator <: LinearMapping
    α::Scalar
end

@callable UniformScalingOperator

# Traits:
selfadjointtype(::UniformScalingOperator) = SelfAdjoint
morphismtype(::UniformScalingOperator) = Endomorphism
diagonaltype(::UniformScalingOperator) = DiagonalMapping
inplacetype(::Type{<:Operations}, ::UniformScalingOperator) = InPlace

isinvertible(A::UniformScalingOperator) = (isfinite(A.α) && A.α != zero(Scalar))

ensureinvertible(A::UniformScalingOperator) =
    isinvertible(A) || throw(
        SingularSystem("Uniform scaling operator is singular"))

function Base.inv(A::UniformScalingOperator)
    ensureinvertible(A)
    return UniformScalingOperator(one(Scalar)/A.α)
end

function apply!(α::Real, ::Type{<:Union{Direct,Adjoint}},
                A::UniformScalingOperator, x, β::Real, y)
    return vcombine!(y, α*A.α, x, β, y)
end

function apply!(α::Real, ::Type{<:Union{Inverse,InverseAdjoint}},
                A::UniformScalingOperator, x, β::Real, y)
    ensureinvertible(A)
    return vcombine!(y, α/A.α, x, β, y)
end

function vcreate(::Type{<:Operations},
                 A::UniformScalingOperator,
                 x::AbstractArray{T,N}) where {T<:Real,N}
    return similar(Array{float(T)}, axes(x))
end

vcreate(::Type{<:Operations}, A::UniformScalingOperator, x) =
    vcreate(x)

#------------------------------------------------------------------------------
# NON-UNIFORM SCALING

"""
```julia
NonuniformScalingOperator(A)
```

creates a nonuniform scaling linear mapping whose effects is to apply
elementwise multiplication of its argument by the scaling factors `A`.
This mapping can be thought as a *diagonal* operator.

See also: [`UniformScalingOperator`](@ref).

"""
struct NonuniformScalingOperator{T} <: LinearMapping
    diag::T
end

@callable NonuniformScalingOperator

# Traits:
morphismtype(::NonuniformScalingOperator) = Endomorphism
diagonaltype(::NonuniformScalingOperator) = DiagonalMapping
inplacetype(::Type{<:Operations}, ::NonuniformScalingOperator) = InPlace
selfadjointtype(A::NonuniformScalingOperator) =
    _selfadjointtype(eltype(contents(A)), A)
_selfadjointtype(::Type{<:Real}, ::NonuniformScalingOperator) =
    SelfAdjoint
_selfadjointtype(::Type{<:Complex}, ::NonuniformScalingOperator) =
    NonSelfAdjoint

contents(A::NonuniformScalingOperator) = A.diag
LinearAlgebra.diag(A::NonuniformScalingOperator) = A.diag

function Base.inv(A::NonuniformScalingOperator{<:AbstractArray{T,N}}
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
        T = promote_type(Tw, Tx, Ty)
        a, b = convert(T, α), convert(T, β)
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
        T = promote_type(Tw, Tx, Ty)
        a, b = convert(T, α), convert(T, β)
        I = eachindex(w, x, y)
        if P === Direct
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
                 x::AbstractArray{Tx,N}) where {Tw<:AbstractFloat,
                                                Tx<:AbstractFloat,N}
    inds = axes(W.diag)
    @assert axes(x) == inds
    T = promote_type(Tw, Tx)
    return similar(Array{T}, inds)
end

function vcreate(::Type{<:Operations},
                 W::NonuniformScalingOperator{<:AbstractArray{Complex{Tw},N}},
                 x::AbstractArray{Complex{Tx},N}) where {Tw<:AbstractFloat,
                                                         Tx<:AbstractFloat,N}
    inds = axes(W.diag)
    @assert axes(x) == inds
    T = promote_type(Tw, Tx)
    return similar(Array{Complex{T}}, inds)
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

function apply!(α::Real, ::Type{Direct}, A::RankOneOperator, x, β::Real, y)
    if α == 0
        # Lazily assume that y has correct type, dimensions, etc.
        vscale!(y, β)
    else
        vcombine!(y, α*vdot(A.v, x), A.u, β, y)
    end
    return y
end

function apply!(α::Real, ::Type{Adjoint}, A::RankOneOperator, x, β::Real, y)
    if α == 0
        # Lazily assume that y has correct type, dimensions, etc.
        vscale!(y, β)
    else
        vcombine!(y, α*vdot(A.u, x), A.v, β, y)
    end
    return y
end

# Lazily assume that x has correct type, dimensions, etc.
vcreate(::Type{Direct}, A::RankOneOperator, x) = vcreate(A.v)
vcreate(::Type{Adjoint}, A::RankOneOperator, x) = vcreate(A.u)

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
morphismtype(::SymmetricRankOneOperator) = Endomorphism
inplacetype(::Type{<:Operations}, ::SymmetricRankOneOperator) = InPlace
selfadjointtype(A::SymmetricRankOneOperator) = SelfAdjoint

function apply!(α::Real, ::Type{<:Union{Direct,Adjoint}},
                A::SymmetricRankOneOperator, x, β::Real, y)
    if α == 0
        # Lazily assume that y has correct type, dimensions, etc.
        vscale!(y, β)
    else
        vcombine!(y, α*vdot(A.u, x), A.u, β, y)
    end
    return y
end

function vcreate(::Type{<:Union{Direct,Adjoint}},
                 A::SymmetricRankOneOperator, x)
    # Lazily assume that x has correct type, dimensions, etc.
    vcreate(A.u)
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

contents(A) = A.arr

# Make a GeneralMatrix behaves like an ordinary array.
Base.eltype(A::GeneralMatrix) = eltype(A.arr)
Base.length(A::GeneralMatrix) = length(A.arr)
Base.ndims(A::GeneralMatrix) = ndims(A.arr)
@static if isdefined(Base, :axes)
    Base.axes(A::GeneralMatrix) = axes(A.arr)
else
    import Compat: axes
    Base.indices(A::GeneralMatrix) = indices(A.arr)
end
Base.size(A::GeneralMatrix) = size(A.arr)
Base.size(A::GeneralMatrix, inds...) = size(A.arr, inds...)
Base.getindex(A::GeneralMatrix, inds...) = getindex(A.arr, inds...)
Base.setindex!(A::GeneralMatrix, x, inds...) = setindex!(A.arr, x, inds...)
Base.stride(A::GeneralMatrix, k) = stride(A.arr, k)
Base.strides(A::GeneralMatrix) = strides(A.arr)
Base.eachindex(A::GeneralMatrix) = eachindex(A.arr)

function apply!(α::Real,
                ::Type{P},
                A::GeneralMatrix{<:AbstractArray{<:AbstractFloat}},
                x::AbstractArray{<:AbstractFloat},
                β::Real,
                y::AbstractArray{<:AbstractFloat}) where {P<:Operations}
    return apply!(α, P, A.arr, x, β, y)
end

function vcreate(::Type{P},
                 A::GeneralMatrix{<:AbstractArray{<:AbstractFloat}},
                 x::AbstractArray{<:AbstractFloat}) where {P<:Operations}
    return vcreate(P, A.arr, x)
end

function apply(A::AbstractArray{<:Real},
               x::AbstractArray{<:Real})
    return apply(Direct, A, x)
end

function apply(::Type{P},
               A::AbstractArray{<:Real},
               x::AbstractArray{<:Real}) where {P<:Operations}
    return apply!(one(Scalar), P, A, x, zero(Scalar), vcreate(P, A, x))
end

function apply!(y::AbstractArray{<:Real},
                A::AbstractArray{<:Real},
                x::AbstractArray{<:Real})
    return apply!(y, Direct, A, x)
end

function apply!(y::AbstractArray{<:Real},
                ::Type{P},
                A::AbstractArray{<:Real},
                x::AbstractArray{<:Real}) where {P<:Operations}
    return apply!(one(Scalar), P, A, x, zero(Scalar), y)
end

# By default, use pure Julia code for the generalized matrix-vector product.
function apply!(α::Real,
                ::Type{P},
                A::AbstractArray{<:Real},
                x::AbstractArray{<:Real},
                β::Real,
                y::AbstractArray{<:Real}) where {P<:Union{Direct,
                                                          InverseAdjoint}}
    if axes(A) != (axes(y)..., axes(x)...)
        throw(DimensionMismatch("`x` and/or `y` have axes incompatible with `A`"))
    end
    return _apply!(α, P, A, x, β, y)
end

function apply!(α::Real,
                ::Type{P},
                A::AbstractArray{<:Real},
                x::AbstractArray{<:Real},
                β::Real,
                y::AbstractArray{<:Real}) where {P<:Union{Adjoint,Inverse}}
    if axes(A) != (axes(x)..., axes(y)...)
        throw(DimensionMismatch("`x` and/or `y` have axes incompatible with `A`"))
    end
    return _apply!(α, P, A, x, β, y)
end

function vcreate(::Type{P},
                 A::AbstractArray{Ta,Na},
                 x::AbstractArray{Tx,Nx}) where {Ta<:AbstractFloat, Na,
                                                 Tx<:AbstractFloat, Nx,
                                                 P<:Union{Direct,
                                                          InverseAdjoint}}
    inds = axes(A)
    Ny = Na - Nx
    if Nx ≥ Na || axes(x) != inds[Ny+1:end]
        throw(DimensionMismatch("the axes of `x` do not match the trailing axes of `A`"))
    end
    Ty = promote_type(Ta, Tx)
    return similar(Array{Ty}, inds[1:Ny])
end

function vcreate(::Type{P},
                 A::AbstractArray{Ta,Na},
                 x::AbstractArray{Tx,Nx}) where {Ta<:AbstractFloat, Na,
                                                 Tx<:AbstractFloat, Nx,
                                                 P<:Union{Adjoint,Inverse}}
    inds = axes(A)
    Ny = Na - Nx
    if Nx ≥ Na || axes(x) != inds[1:Nx]
        throw(DimensionMismatch("the axes of `x` do not match the leading axes of `A`"))
    end
    Ty = promote_type(Ta, Tx)
    return similar(Array{Ty}, inds[Nx+1:end])
end


# Pure Julia code implementations.

function _apply!(α::Real,
                 ::Type{Direct},
                 A::AbstractArray{Ta},
                 x::AbstractArray{Tx},
                 β::Real,
                 y::AbstractArray{Ty}) where {Ta<:Real, Tx<:Real, Ty<:Real}
    if β != 1
        vscale!(y, β)
    end
    if α != 0
        # Loop through the coefficients of A assuming column-major storage
        # order.
        T = promote_type(Ta, Tx, Ty)
        alpha = convert(T, α)
        I, J = CartesianIndices(axes(y)), CartesianIndices(axes(x))
        @inbounds for j in J
            xj = alpha*convert(T, x[j])
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
                 x::AbstractArray{Tx}) where {Ta<:Real, Tx<:Real, Ty<:Real}
    return _apply!(promote_type(Ty, Ta, Tx), y, Adjoint, A, x)
end

function _apply!(α::Real,
                 ::Type{Adjoint},
                 A::AbstractArray{Ta},
                 x::AbstractArray{Tx},
                 β::Real,
                 y::AbstractArray{Ty}) where {Ta<:Real, Tx<:Real, Ty<:Real}
    if α == 0
        vscale!(y, β)
    else
        # Loop through the coefficients of A assuming column-major storage
        # order.
        T = promote_type(Ta, Tx, Ty)
        alpha = convert(T, α)
        I, J = CartesianIndices(axes(x)), CartesianIndices(axes(y))
        if β == 0
            @inbounds for j in J
                local s::T = zero(T)
                @simd for i in I
                    s += A[i,j]*x[i]
                end
                y[j] = alpha*s
            end
        else
            beta = convert(T, β)
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

# Default method for allocating the result for Hessian and HalfHessian linear
# mappings.
vcreate(::Type{Direct}, ::Union{Hessian,HalfHessian}, x) = vcreate(x)
