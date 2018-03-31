#
# operators.jl -
#
# Methods for linear mappings.
#
#-------------------------------------------------------------------------------
#
# This file is part of the LazyAlgebra package released under the MIT "Expat"
# license.
#
# Copyright (c) 2017-2018 Éric Thiébaut.
#

# Basic methods:
"""
```julia
input_type([P=Direct,] A)
output_type([P=Direct,] A)
```

yield the (preferred) types of the input and output arguments of the operation
`P` with mapping `A`.  If `A` operates on Julia arrays, the element type,
list of dimensions, `i`-th dimension and number of dimensions for the input and
output are given by:

    input_eltype([P=Direct,] A)          output_eltype([P=Direct,] A)
    input_size([P=Direct,] A)            output_size([P=Direct,] A)
    input_size([P=Direct,] A, i)         output_size([P=Direct,] A, i)
    input_ndims([P=Direct,] A)           output_ndims([P=Direct,] A)

Only `input_size(A)` and `output_size(A)` have to be implemented.

Also see: [`vcreate`](@ref), [`apply!`](@ref), [`LinearMapping`](@ref),
[`Operations`](@ref).

"""
function input_type end

for sfx in (:size, :eltype, :ndims, :type),
    pfx in (:output, :input)

    fn1 = Symbol(pfx, "_", sfx)

    for P in (Direct, Adjoint, Inverse, InverseAdjoint)

         fn2 = Symbol(P == Adjoint || P == Inverse ?
                     (pfx == :output ? :input : :output) : pfx, "_", sfx)

        #println("$fn1($P) -> $fn2")

        # Provide basic methods for the different operations and for tagged
        # mappings.
        @eval begin

            if $(P != Direct)
                $fn1(A::$P{<:LinearMapping}) = $fn2(A.op)
            end

            $fn1(::Type{$P}, A::LinearMapping) = $fn2(A)

            if $(sfx == :size)
                if $(P != Direct)
                    $fn1(A::$P{<:LinearMapping}, dim...) =
                        $fn2(A.op, dim...)
                end
                $fn1(::Type{$P}, A::LinearMapping, dim...) =
                    $fn2(A, dim...)
            end
        end
    end

    # Link documentation for the basic methods.
    @eval begin
        if $(fn1 != :input_type)
            @doc @doc(:input_type) $fn1
        end
    end

end

# Provide default methods for `$(sfx)_size(A, dim...)` and `$(sfx)_ndims(A)`.
for pfx in (:input, :output)
    pfx_size = Symbol(pfx, "_size")
    pfx_ndims = Symbol(pfx, "_ndims")
    @eval begin

        $pfx_ndims(A::LinearMapping) = length($pfx_size(A))

        $pfx_size(A::LinearMapping, dim) = $pfx_size(A)[dim]

        function $pfx_size(A::LinearMapping, dim...)
            dims = $pfx_size(A)
            ntuple(i -> dims[dim[i]], length(dim))
        end

    end
end


"""
```julia
is_applicable_in_place([P,] A, x)
```

yields whether mapping `A` is applicable *in-place* for performing operation
`P` with argument `x`, that is with the result stored into the argument `x`.
This can be used to spare allocating ressources.

See also: [`LinearMapping`](@ref), [`apply!`](@ref).

"""
is_applicable_in_place(::Type{<:Operations}, A::LinearMapping, x) = false
is_applicable_in_place(A::LinearMapping, x) =
    is_applicable_in_place(Direct, A, x)

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
struct Identity <: SelfAdjointOperator; end

is_applicable_in_place(::Type{<:Operations}, ::Identity, x) = true

Base.inv(A::Identity) = A

apply(::Type{<:Operations}, ::Identity, x) = x

apply!(α::Scalar, ::Type{<:Operations}, ::Identity, x, β::Scalar, y) =
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
struct UniformScalingOperator <: SelfAdjointOperator
    α::Float64
end

is_applicable_in_place(::Type{<:Operations}, ::UniformScalingOperator, x) = true

isinvertible(A::UniformScalingOperator) = (isfinite(A.α) && A.α != 0.0)

ensureinvertible(A::UniformScalingOperator) =
    isinvertible(A) || throw(
        SingularSystem("Uniform scaling operator is singular"))

function Base.inv(A::UniformScalingOperator)
    ensureinvertible(A)
    return UniformScalingOperator(1.0/A.α)
end

function apply!(α::Scalar, ::Type{<:Union{Direct,Adjoint}},
                A::UniformScalingOperator, x, β::Scalar, y)
    return vcombine!(y, α*A.α, x, β, y)
end

function apply!(α::Scalar, ::Type{<:Union{Inverse,InverseAdjoint}},
                A::UniformScalingOperator, x, β::Scalar, y)
    ensureinvertible(A)
    return vcombine!(y, α/A.α, x, β, y)
end

function vcreate(::Type{<:Operations},
                 A::UniformScalingOperator,
                 x::AbstractArray{T,N}) where {T<:Real,N}
    return similar(Array{float(T)}, indices(x))
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
struct NonuniformScalingOperator{T} <: SelfAdjointOperator
    scl::T
end

is_applicable_in_place(::Type{<:Operations}, ::NonuniformScalingOperator, x) =
    true

function Base.inv(A::NonuniformScalingOperator{<:AbstractArray{T,N}}
                  ) where {T<:AbstractFloat, N}
    q = A.scl
    r = similar(q)
    @inbounds @simd for i in eachindex(q, r)
        r[i] = one(T)/q[i]
    end
    return NonuniformScalingOperator(r)
end

function apply!(α::Scalar,
                ::Type{<:Union{Direct,Adjoint}},
                A::NonuniformScalingOperator{<:AbstractArray{<:AbstractFloat,N}},
                x::AbstractArray{<:AbstractFloat,N},
                β::Scalar,
                y::AbstractArray{<:AbstractFloat,N}) where {N}
    w = A.scl
    @assert indices(w) == indices(x) == indices(y)
    T = promote_type(eltype(w), eltype(x), eltype(y))
    if α == one(α)
        if β == zero(β)
            @inbounds @simd for i in eachindex(w, x, y)
                y[i] = w[i]*x[i]
            end
        elseif β == one(β)
            @inbounds @simd for i in eachindex(w, x, y)
                y[i] = w[i]*x[i] + y[i]
            end
        elseif β == -one(β)
            @inbounds @simd for i in eachindex(w, x, y)
                y[i] = w[i]*x[i] - y[i]
            end
        else
            beta = convert(T, β)
            @inbounds @simd for i in eachindex(w, x, y)
                y[i] = w[i]*x[i] + beta*y[i]
            end
        end
    elseif α == zero(α)
        vscale!(y, β)
    elseif α == -one(α)
        if β == zero(β)
            @inbounds @simd for i in eachindex(w, x, y)
                y[i] = -w[i]*x[i]
            end
        elseif β == one(β)
            @inbounds @simd for i in eachindex(w, x, y)
                y[i] = y[i] - w[i]*x[i]
            end
        elseif β == -one(β)
            @inbounds @simd for i in eachindex(w, x, y)
                y[i] = -w[i]*x[i] - y[i]
            end
        else
            beta = convert(T, β)
            @inbounds @simd for i in eachindex(w, x, y)
                y[i] = beta*y[i] - w[i]*x[i]
            end
        end
    else
        alpha = convert(T, β)
        if β == zero(β)
            @inbounds @simd for i in eachindex(w, x, y)
                y[i] = alpha*w[i]*x[i]
            end
        elseif β == one(β)
            @inbounds @simd for i in eachindex(w, x, y)
                y[i] = alpha*w[i]*x[i] + y[i]
            end
        elseif β == -one(β)
            @inbounds @simd for i in eachindex(w, x, y)
                y[i] = alpha*w[i]*x[i] - y[i]
            end
        else
            beta = convert(T, β)
            @inbounds @simd for i in eachindex(w, x, y)
                y[i] = alpha*w[i]*x[i] + beta*y[i]
            end
        end
    end
    return y
end

function apply!(α::Scalar,
                ::Type{<:Union{Inverse,InverseAdjoint}},
                A::NonuniformScalingOperator{<:AbstractArray{<:AbstractFloat,N}},
                x::AbstractArray{<:AbstractFloat,N},
                β::Scalar,
                y::AbstractArray{<:AbstractFloat,N}) where {N}
    w = A.scl
    @assert indices(w) == indices(x) == indices(y)
    T = promote_type(eltype(w), eltype(x), eltype(y))
    if α == one(α)
        if β == zero(β)
            @inbounds @simd for i in eachindex(w, x, y)
                y[i] = x[i]/w[i]
            end
        elseif β == one(β)
            @inbounds @simd for i in eachindex(w, x, y)
                y[i] = x[i]/w[i] + y[i]
            end
        elseif β == -one(β)
            @inbounds @simd for i in eachindex(w, x, y)
                y[i] = x[i]/w[i] - y[i]
            end
        else
            beta = convert(T, β)
            @inbounds @simd for i in eachindex(w, x, y)
                y[i] = x[i]/w[i] + beta*y[i]
            end
        end
    elseif α == zero(α)
        vscale!(y, β)
    elseif α == -one(α)
        if β == zero(β)
            @inbounds @simd for i in eachindex(w, x, y)
                y[i] = -x[i]/w[i]
            end
        elseif β == one(β)
            @inbounds @simd for i in eachindex(w, x, y)
                y[i] = y[i] - x[i]/w[i]
            end
        elseif β == -one(β)
            @inbounds @simd for i in eachindex(w, x, y)
                y[i] = -x[i]/w[i] - y[i]
            end
        else
            beta = convert(T, β)
            @inbounds @simd for i in eachindex(w, x, y)
                y[i] = beta*y[i] - x[i]/w[i]
            end
        end
    else
        alpha = convert(T, β)
        if β == zero(β)
            @inbounds @simd for i in eachindex(w, x, y)
                y[i] = alpha*x[i]/w[i]
            end
        elseif β == one(β)
            @inbounds @simd for i in eachindex(w, x, y)
                y[i] = alpha*x[i]/w[i] + y[i]
            end
        elseif β == -one(β)
            @inbounds @simd for i in eachindex(w, x, y)
                y[i] = alpha*x[i]/w[i] - y[i]
            end
        else
            beta = convert(T, β)
            @inbounds @simd for i in eachindex(w, x, y)
                y[i] = alpha*x[i]/w[i] + beta*y[i]
            end
        end
    end
    return y
end

function vcreate(::Type{<:Operations},
                 A::NonuniformScalingOperator{<:AbstractArray{Ta,N}},
                 x::AbstractArray{Tx,N}) where {Ta<:AbstractFloat,
                                                Tx<:AbstractFloat, N}
    inds = indices(A.scl)
    @assert indices(x) == inds
    T = promote_type(Ta, Tx)
    return similar(Array{T}, inds)
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

function apply!(α::Scalar, ::Type{Direct}, A::RankOneOperator, x,
                β::Scalar, y)
    if α == zero(α)
        # Lazily assume that y has correct type, dimensions, etc.
        vscale!(y, β)
    else
        vcombine!(y, α*vdot(A.v, x), A.u, β, y)
    end
    return y
end

function apply!(α::Scalar, ::Type{Adjoint}, A::RankOneOperator, x,
                β::Scalar, y)
    if α == zero(α)
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
          [`SelfAdjointOperator`](@ref) [`apply!`](@ref), [`vcreate`](@ref).

"""
struct SymmetricRankOneOperator{U} <: SelfAdjointOperator
    u::U
end

is_applicable_in_place(::Type{<:Operations}, ::SymmetricRankOneOperator) = true

function apply!(α::Scalar, ::Type{P}, A::SymmetricRankOneOperator, x,
                β::Scalar, y) where {P<:Union{Direct,Adjoint}}
    if α == zero(α)
        # Lazily assume that y has correct type, dimensions, etc.
        vscale!(y, β)
    else
        vcombine!(y, α*vdot(A.u, x), A.u, β, y)
    end
    return y
end

function vcreate(::Type{P}, A::SymmetricRankOneOperator,
                 x) where {P<:Union{Direct,Adjoint}}
    # Lazily assume that x has correct type, dimensions, etc.
    vcreate(A.u)
end

input_type(A::SymmetricRankOneOperator{U}) where {U} = U
input_ndims(A::SymmetricRankOneOperator) = ndims(A.u)
input_size(A::SymmetricRankOneOperator) = size(A.u)
input_size(A::SymmetricRankOneOperator, d...) = size(A.u, d...)
input_eltype(A::SymmetricRankOneOperator) = eltype(A.u)

# FIXME: this should be automatically done for SelfAdjointOperators?
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
leading dimensions of `A`, such that `indices(A) = (indices(y)...,
indices(x)...)`.  Applying the adjoint of `G` as in `y = G'*x` requires that
the dimensions of `x` match the leading dimension of `A` and yields a result
`y` whose dimensions are the remaining trailing dimensions of `A`, such that
`indices(A) = (indices(x)..., indices(y)...)`.

See also: [`reshape`](@ref).

"""
struct GeneralMatrix{T<:AbstractArray} <: LinearMapping
    arr::T
end

# Make a GeneralMatrix behaves like an ordinary array.
Base.eltype(A::GeneralMatrix) = eltype(A.arr)
Base.length(A::GeneralMatrix) = length(A.arr)
Base.ndims(A::GeneralMatrix) = ndims(A.arr)
Base.indices(A::GeneralMatrix) = indices(A.arr)
Base.size(A::GeneralMatrix) = size(A.arr)
Base.size(A::GeneralMatrix, inds...) = size(A.arr, inds...)
Base.getindex(A::GeneralMatrix, inds...) = getindex(A.arr, inds...)
Base.setindex!(A::GeneralMatrix, x, inds...) = setindex!(A.arr, x, inds...)
Base.stride(A::GeneralMatrix, k) = stride(A.arr, k)
Base.strides(A::GeneralMatrix) = strides(A.arr)
Base.eachindex(A::GeneralMatrix) = eachindex(A.arr)

function apply!(α::Scalar,
                ::Type{P},
                A::GeneralMatrix{<:AbstractArray{<:AbstractFloat}},
                x::AbstractArray{<:AbstractFloat},
                β::Scalar,
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

# By default, use pure Julia code for the generalized matrix-vector product.
function apply!(α::Scalar,
                ::Type{P},
                A::AbstractArray{<:Real},
                x::AbstractArray{<:Real},
                β::Scalar,
                y::AbstractArray{<:Real}) where {P<:Union{Direct,
                                                          InverseAdjoint}}
    if indices(A) != (indices(y)..., indices(x)...)
        throw(DimensionMismatch("`x` and/or `y` have indices incompatible with `A`"))
    end
    return _apply!(α, P, A, x, β, y)
end

function apply!(α::Scalar,
                ::Type{P},
                A::AbstractArray{<:Real},
                x::AbstractArray{<:Real},
                β::Scalar,
                y::AbstractArray{<:Real}) where {P<:Union{Adjoint,Inverse}}
    if indices(A) != (indices(x)..., indices(y)...)
        throw(DimensionMismatch("`x` and/or `y` have indices incompatible with `A`"))
    end
    return _apply!(α, P, A, x, β, y)
end

function vcreate(::Type{P},
                 A::AbstractArray{Ta,Na},
                 x::AbstractArray{Tx,Nx}) where {Ta<:AbstractFloat, Na,
                                                 Tx<:AbstractFloat, Nx,
                                                 P<:Union{Direct,
                                                          InverseAdjoint}}
    inds = indices(A)
    Ny = Na - Nx
    if Nx ≥ Na || indices(x) != inds[Ny+1:end]
        throw(DimensionMismatch("the dimensions of `x` do not match the trailing dimensions of `A`"))
    end
    Ty = promote_type(Ta, Tx)
    return similar(Array{Ty}, inds[1:Ny])
end

function vcreate(::Type{P},
                 A::AbstractArray{Ta,Na},
                 x::AbstractArray{Tx,Nx}) where {Ta<:AbstractFloat, Na,
                                                 Tx<:AbstractFloat, Nx,
                                                 P<:Union{Adjoint,Inverse}}
    inds = indices(A)
    Ny = Na - Nx
    if Nx ≥ Na || indices(x) != inds[1:Nx]
        throw(DimensionMismatch("the dimensions of `x` do not match the leading dimensions of `A`"))
    end
    Ty = promote_type(Ta, Tx)
    return similar(Array{Ty}, inds[Ny+1:end])
end


# Pure Julia code implementations.

function _apply!(α::Scalar,
                 ::Type{Direct},
                 A::AbstractArray{Ta},
                 x::AbstractArray{Tx},
                 β::Scalar,
                 y::AbstractArray{Ty}) where {Ta<:Real, Tx<:Real, Ty<:Real}
    if β != one(β)
        vscale!(y, β)
    end
    if α != zero(α)
        # Loop through the coefficients of A assuming column-major storage
        # order.
        T = promote_type(Ta, Tx, Ty)
        alpha = convert(T, α)
        I, J = CartesianRange(indices(y)), CartesianRange(indices(x))
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

function _apply!(α::Scalar,
                 ::Type{Adjoint},
                 A::AbstractArray{Ta},
                 x::AbstractArray{Tx},
                 β::Scalar,
                 y::AbstractArray{Ty}) where {Ta<:Real, Tx<:Real, Ty<:Real}
    if α == zero(α)
        vscale!(y, β)
    else
        # Loop through the coefficients of A assuming column-major storage
        # order.
        T = promote_type(Ta, Tx, Ty)
        alpha = convert(T, α)
        I, J = CartesianRange(indices(x)), CartesianRange(indices(y))
        if β == zero(β)
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
# HALF HESSIAN

"""

`HalfHessian(A)` is a container to be interpreted as the linear mapping
representing the second derivatives (times 1/2) of some objective function at
some point both represented by `A` (which can be anything).  Given `H =
HalfHessian(A)`, the contents `A` is retrieved by `contents(H)`.

For a simple quadratic objective function like:

```
f(x) = ‖D⋅x‖²
```

the half-Hessian is:

```
H = D'⋅D
```

As the half-Hessian is symmetric, a single method `apply!` has to be
implemented to apply the direct and adjoint of the mapping, the signature of
the method is:

```julia
apply!(y::T, ::Type{Direct}, H::HalfHessian{typeof(A)}, x::T)
```

where `y` is overwritten by the result of applying `H` (or its adjoint) to the
argument `x`.  Here `T` is the relevant type of the variables.  Similarly, to
allocate a new object to store the result of applying the mapping, it is
sufficient to implement the method:

```julia
vcreate(::Type{Direct}, H::HalfHessian{typeof(A)}, x::T)
```

See also: [`LinearMapping`][@ref).

"""
struct HalfHessian{T} <: SelfAdjointOperator
    obj::T
end

"""
```julia
contents(C)
```

yields the contents of the container `C`.  A *container* is any type which
implements the `contents` method.

"""
contents(H::HalfHessian) = H.obj
