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

# Traits:
SelfAdjointType(::Identity) = SelfAdjoint()
MorphismType(::Identity) = Endomorphism()
DiagonalType(::Identity) = DiagonalMapping()

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

#------------------------------------------------------------------------------
# SYMBOLIC MAPPINGS (FOR TESTS)

struct SymbolicMapping{T} <: Mapping end
struct SymbolicLinearMapping{T} <: LinearMapping end
SymbolicMapping(id::AbstractString) = SymbolicMapping(Symbol(id))
SymbolicMapping(id::Symbol) = SymbolicMapping{Val{id}}()
SymbolicLinearMapping(id::AbstractString) = SymbolicLinearMapping(Symbol(id))
SymbolicLinearMapping(x::Symbol) = SymbolicLinearMapping{Val{x}}()

show(io::IO, A::SymbolicMapping{Val{T}}) where {T} = print(io, T)
show(io::IO, A::SymbolicLinearMapping{Val{T}}) where {T} = print(io, T)

is_same_mapping(::SymbolicMapping{T}, ::SymbolicMapping{T}) where {T} = true
is_same_mapping(::SymbolicLinearMapping{T}, ::SymbolicLinearMapping{T}) where {T} = true

#------------------------------------------------------------------------------
# NON-UNIFORM SCALING

"""
```julia
NonuniformScalingOperator(A)
```

creates a non-uniform scaling linear mapping whose effect is to apply
elementwise multiplication of its argument by the scaling factors `A`.  This
mapping can be thought as a *diagonal* operator.

The method [`Diag`](@ref) is a shortcut to build a non-uniform scaling operator.

The scaling factors of a non-uniform scaling operator can be retrieved with the
[`diag`](@ref) method:

```julia
W = Diag(A)
diag(W) === A  # this is true
```

!!! note
    Beware of the differences between the [`Diag`](@ref) (with an uppercase
    'D') and [`diag`](@ref) (with an lowercase 'd') methods.

"""
struct NonuniformScalingOperator{T} <: LinearMapping
    diag::T
end

@callable NonuniformScalingOperator

# Traits:
MorphismType(::NonuniformScalingOperator) = Endomorphism()
DiagonalType(::NonuniformScalingOperator) = DiagonalMapping()
SelfAdjointType(A::NonuniformScalingOperator) =
    _selfadjointtype(eltype(contents(A)), A)
_selfadjointtype(::Type{<:Real}, ::NonuniformScalingOperator) =
    SelfAdjoint()
_selfadjointtype(::Type{<:Complex}, ::NonuniformScalingOperator) =
    NonSelfAdjoint()

"""
```
Diag(A)
```

yields a non-uniform scaling linear mapping whose effect is to apply
elementwise multiplication of its argument by the scaling factors `A`.  This
mapping can be thought as a *diagonal* operator.

See also: [`NonuniformScalingOperator`](@ref), [`diag`](@ref).

"""
Diag(A) = NonuniformScalingOperator(A)

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
        a = convert_multiplier(α, promote_type(Tw, Tx), Ty)
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
        a = convert_multiplier(α, promote_type(Tw, Tx), Ty)
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
MorphismType(::SymmetricRankOneOperator) = Endomorphism()
SelfAdjointType(::SymmetricRankOneOperator) = SelfAdjoint()

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

function apply!(α::Number,
                P::Type{<:Operations},
                A::GeneralMatrix{<:AbstractArray{<:GenMult.Floats}},
                x::AbstractArray{<:GenMult.Floats},
                scratch::Bool,
                β::Number,
                y::AbstractArray{<:GenMult.Floats})
    return apply!(α, P, A.arr, x, scratch, β, y)
end

function vcreate(P::Type{<:Operations},
                 A::GeneralMatrix{<:AbstractArray{<:GenMult.Floats}},
                 x::AbstractArray{<:GenMult.Floats},
                 scratch::Bool=false)
    return vcreate(P, A.arr, x, scratch)
end

for (T, L) in ((:Direct, 'N'), (:Adjoint, 'C'))
    @eval begin
        function apply!(α::Number,
                        ::Type{$T},
                        A::AbstractArray{<:GenMult.Floats},
                        x::AbstractArray{<:GenMult.Floats},
                        scratch::Bool,
                        β::Number,
                        y::AbstractArray{<:GenMult.Floats})
            return lgemv!(α, $L, A, x, β, y)
        end
    end
end

# To have apply and apply! methods callable with an array (instead of a
# mapping), we have to provide the different possibilities.

apply(A::AbstractArray, x::AbstractArray, scratch::Bool=false) =
    apply(Direct, A, x, scratch)

apply(P::Type{<:Operations}, A::AbstractArray, x::AbstractArray, scratch::Bool=false) =
    apply!(1, P, A, x, scratch, 0, vcreate(P, A, x, scratch))

apply!(y::AbstractArray, A::AbstractArray, x::AbstractArray) =
    apply!(1, Direct, A, x, false, 0, y)

apply!(y::AbstractArray, P::Type{<:Operations}, A::AbstractArray, x::AbstractArray) =
    apply!(1, P, A, x, false, 0, y)

function vcreate(P::Type{<:Union{Direct,InverseAdjoint}},
                 A::AbstractArray{Ta,Na},
                 x::AbstractArray{Tx,Nx},
                 scratch::Bool=false) where {Ta,Na,Tx,Nx}
    # Non-transposed matrix.  Trailing dimensions of X must match those of A,
    # leading dimensions of A are those of the result.  Whatever the scratch
    # parameter, a new array is returned as the operation cannot be done
    # in-place.
    @noinline incompatible_dimensions() =
        throw(DimensionMismatch("the indices of `x` do not match the trailing indices of `A`"))
    1 ≤ Nx < Na || incompatible_dimensions()
    Ny = Na - Nx
    @inbounds for d in 1:Nx
        axes(x, d) == axes(A, Ny + d) || incompatible_dimensions()
    end
    shape = ntuple(d -> axes(A, d), Val(Ny))
    return similar(A, promote_type(Ta, Tx), shape)
end

function vcreate(P::Type{<:Union{Adjoint,Inverse}},
                 A::AbstractArray{Ta,Na},
                 x::AbstractArray{Tx,Nx},
                 scratch::Bool=false) where {Ta,Na,Tx,Nx}
    # Transposed matrix.  Leading dimensions of X must match those of A,
    # trailing dimensions of A are those of the result.  Whatever the scratch
    # parameter, a new array is returned as the operation cannot be done
    # in-place.
    @noinline incompatible_dimensions() =
        throw(DimensionMismatch("the indices of `x` do not match the leading indices of `A`"))
    1 ≤ Nx < Na || incompatible_dimensions()
    Ny = Na - Nx
    @inbounds for d in 1:Nx
        axes(x, d) == axes(A, d) || incompatible_dimensions()
    end
    shape = ntuple(d -> axes(A, Nx + d), Val(Ny))
    return similar(A, promote_type(Ta, Tx), shape)
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
