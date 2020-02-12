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
# Copyright (c) 2017-2020 Éric Thiébaut.
#

#------------------------------------------------------------------------------
# IDENTITY AND UNIFORM SCALING

are_same_mappings(::Identity, ::Identity) = true

@callable Identity

# Traits:
SelfAdjointType(::Identity) = SelfAdjoint()
MorphismType(::Identity) = Endomorphism()
DiagonalType(::Identity) = DiagonalMapping()

apply(::Type{<:Operations}, ::Identity, x, scratch::Bool=false) = x

vcreate(::Type{<:Operations}, ::Identity, x, scratch::Bool) =
    (scratch ? x : vcreate(x)) # FIXME: should always return x?

apply!(α::Number, ::Type{<:Operations}, ::Identity, x, ::Bool, β::Number, y) =
    vcombine!(y, α, x, β, y)

# Rules to automatically convert UniformScaling from standard library module
# LinearAlgebra into λ*I.  For other operators, there is no needs to extend ⋅
# (\cdot) and ∘ (\circ) as they are already converted in calls to *.  But in
# the case of UniformScaling, we must explicitly do that for * and for ∘ (not
# for ⋅ which is replaced by a * by existing rules).
simplify(A::UniformScaling) = A.λ*I
for op in (:(+), :(-), :(*), :(∘), :(/), Symbol("\\"))
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

are_same_mappings(::T, ::T) where {T<:SymbolicMapping} = true
are_same_mappings(::T, ::T) where {T<:SymbolicLinearMapping} = true

#------------------------------------------------------------------------------
# NON-UNIFORM SCALING

"""
```julia
NonuniformScaling(A)
```

creates a non-uniform scaling linear mapping whose effect is to apply
elementwise multiplication of its argument by the scaling factors `A`.  This
mapping can be thought as a *diagonal* operator.

The method [`Diag`](@ref) is a shortcut to build a non-uniform scaling operator
and the scaling factors of a non-uniform scaling operator retrieved with the
[`diag`](@ref) method:

```julia
W = Diag(A)
diag(W) === A  # this is true
```

!!! note
    Beware of the differences between the [`Diag`](@ref) (with an uppercase
    'D') and [`diag`](@ref) (with an lowercase 'd') methods.

"""
struct NonuniformScaling{T} <: LinearMapping
    diag::T
end

@deprecate NonuniformScalingOperator NonuniformScaling
@callable NonuniformScaling

# Traits:
MorphismType(::NonuniformScaling) = Endomorphism()
DiagonalType(::NonuniformScaling) = DiagonalMapping()
SelfAdjointType(A::NonuniformScaling) =
    _selfadjointtype(eltype(coefficients(A)), A)
_selfadjointtype(::Type{<:Real}, ::NonuniformScaling) =
    SelfAdjoint()
_selfadjointtype(::Type{<:Complex}, ::NonuniformScaling) =
    NonSelfAdjoint()

"""
```
Diag(A)
```

yields a non-uniform scaling linear mapping whose effect is to apply
elementwise multiplication of its argument by the scaling factors `A`.  This
mapping can be thought as a *diagonal* operator.

See also: [`NonuniformScaling`](@ref), [`diag`](@ref).

"""
Diag(A) = NonuniformScaling(A)

coefficients(A::NonuniformScaling) = A.diag
LinearAlgebra.diag(A::NonuniformScaling) = coefficients(A)

# FIXME: This is nearly the default implementation.
are_same_mappings(A::T, B::T) where {T<:NonuniformScaling} =
    coefficients(A) === coefficients(B)

function inv(A::NonuniformScaling{<:AbstractArray{T,N}}
             ) where {T<:AbstractFloat, N}
    q = coefficients(A)
    r = similar(q)
    @inbounds @simd for i in eachindex(q, r)
        r[i] = one(T)/q[i]
    end
    return NonuniformScaling(r)
end

eltype(::Type{<:NonuniformScaling{<:AbstractArray{T,N}}}) where {T, N} = T

input_ndims(::NonuniformScaling{<:AbstractArray{T,N}}) where {T, N} = N
input_size(A::NonuniformScaling{<:AbstractArray}) = size(coefficients(A))
input_size(A::NonuniformScaling{<:AbstractArray}, i) =
    size(coefficients(A), i)

output_ndims(::NonuniformScaling{<:AbstractArray{T,N}}) where {T, N} = N
output_size(A::NonuniformScaling{<:AbstractArray}) = size(coefficients(A))
output_size(A::NonuniformScaling{<:AbstractArray}, i) =
    size(coefficients(A), i)

# Simplify left multiplication (and division) by a scalar.
# FIXME: α = zero(α) should be treated specifically
*(α::Number, A::NonuniformScaling)::NonuniformScaling =
    (α == one(α) ? A : NonuniformScaling(vscale(α, coefficients(A))))

# Extend composition of diagonal operators.
*(A::NonuniformScaling, B::NonuniformScaling) =
    NonuniformScaling(vproduct(coefficients(A), coefficients(B)))

function apply!(α::Number,
                ::Type{P},
                W::NonuniformScaling{<:AbstractArray{Tw,N}},
                x::AbstractArray{Tx,N},
                scratch::Bool,
                β::Number,
                y::AbstractArray{Ty,N}) where {P<:Operations,
                                               Tw<:Floats,
                                               Tx<:Floats,
                                               Ty<:Floats,N}
    w = coefficients(W)
    I = safe_indices(w, x, y)
    if α == 0
        vscale!(y, β)
    elseif β == 0
        if α == 1
            _apply_diagonal!(P, axpby_yields_x, I, 1, w, x, 0, y)
        else
            a = promote_multiplier(α, Tw, Tx)
            _apply_diagonal!(P, axpby_yields_ax, I, a, w, x, 0, y)
        end
    elseif β == 1
        if α == 1
            _apply_diagonal!(P, axpby_yields_xpy, I, 1, w, x, 1, y)
        else
            a = promote_multiplier(α, Tw, Tx)
            _apply_diagonal!(P, axpby_yields_axpy, I, a, w, x, 1, y)
        end
    else
        b = promote_multiplier(β, Ty)
        if α == 1
            _apply_diagonal!(P, axpby_yields_xpby, I, 1, w, x, b, y)
        else
            a = promote_multiplier(α, Tw, Tx)
            _apply_diagonal!(P, axpby_yields_axpby, I, a, w, x, b, y)
        end
    end
    return y
end

function _apply_diagonal!(::Type{Direct}, axpby::Function, I,
                          α, w, x, β, y)
    @inbounds @simd for i in I
        y[i] = axpby(α, w[i]*x[i], β, y[i])
    end
end

function _apply_diagonal!(::Type{Adjoint}, axpby::Function, I,
                          α, w, x, β, y)
    @inbounds @simd for i in I
        y[i] = axpby(α, conj(w[i])*x[i], β, y[i])
    end
end

function _apply_diagonal!(::Type{Inverse}, axpby::Function, I,
                          α, w, x, β, y)
    @inbounds @simd for i in I
        y[i] = axpby(α, x[i]/w[i], β, y[i])
    end
end

function _apply_diagonal!(::Type{InverseAdjoint}, axpby::Function, I,
                          α, w, x, β, y)
    @inbounds @simd for i in I
        y[i] = axpby(α, x[i]/conj(w[i]), β, y[i])
    end
end

function vcreate(::Type{<:Operations},
                 W::NonuniformScaling{<:AbstractArray{Tw,N}},
                 x::AbstractArray{Tx,N},
                 scratch::Bool) where {Tw,Tx,N}
    inds = same_axes(coefficients(W), x)
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

function apply!(α::Number, ::Type{Direct}, A::RankOneOperator,
                x, scratch::Bool, β::Number, y)
    return _apply_rank_one!(α, A.u, A.v, x, β, y)
end

function apply!(α::Number, ::Type{Adjoint}, A::RankOneOperator,
                x, scratch::Bool, β::Number, y)
    return _apply_rank_one!(α, A.v, A.u, x, β, y)
end

function _apply_rank_one!(α::Number, u, v, x, β::Number, y)
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
vcreate(::Type{Direct}, A::RankOneOperator, x, scratch::Bool) =
    vcreate(A.v)
vcreate(::Type{Adjoint}, A::RankOneOperator, x, scratch::Bool) =
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

are_same_mappings(A::T, B::T) where {T<:RankOneOperator} =
    (A.u === B.u && A.v === B.v)

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

function apply!(α::Number, ::Type{<:Union{Direct,Adjoint}},
                A::SymmetricRankOneOperator, x, scratch::Bool, β::Number, y)
    return _apply_rank_one!(α, A.u, A.u, x, β, y)
end

function vcreate(::Type{<:Union{Direct,Adjoint}},
                 A::SymmetricRankOneOperator, x, scratch::Bool)
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

are_same_mappings(A::T, B::T) where {T<:SymmetricRankOneOperator} =
    (A.u === B.u)

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

coefficients(A) = A.arr

# Make a GeneralMatrix behaves like an ordinary array.
eltype(A::GeneralMatrix) = eltype(coefficients(A))
length(A::GeneralMatrix) = length(coefficients(A))
ndims(A::GeneralMatrix) = ndims(coefficients(A))
axes(A::GeneralMatrix) = axes(coefficients(A))
size(A::GeneralMatrix) = size(coefficients(A))
size(A::GeneralMatrix, i...) = size(coefficients(A), i...)
getindex(A::GeneralMatrix, i...) = getindex(coefficients(A), i...)
setindex!(A::GeneralMatrix, x, i...) = setindex!(coefficients(A), x, i...)
stride(A::GeneralMatrix, k) = stride(coefficients(A), k)
strides(A::GeneralMatrix) = strides(coefficients(A))
eachindex(A::GeneralMatrix) = eachindex(coefficients(A))

are_same_mappings(A::T, B::T) where {T<:GeneralMatrix} =
    (coefficients(A) === coefficients(B))

function apply!(α::Number,
                P::Type{<:Operations},
                A::GeneralMatrix{<:AbstractArray{<:GenMult.Floats}},
                x::AbstractArray{<:GenMult.Floats},
                scratch::Bool,
                β::Number,
                y::AbstractArray{<:GenMult.Floats})
    return apply!(α, P, coefficients(A), x, scratch, β, y)
end

function vcreate(P::Type{<:Operations},
                 A::GeneralMatrix{<:AbstractArray{<:GenMult.Floats}},
                 x::AbstractArray{<:GenMult.Floats},
                 scratch::Bool)
    return vcreate(P, coefficients(A), x, scratch)
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

apply(A::AbstractArray, x::AbstractArray, scratch::Bool) =
    apply(Direct, A, x, scratch)

apply(P::Type{<:Operations}, A::AbstractArray, x::AbstractArray, scratch::Bool) =
    apply!(1, P, A, x, scratch, 0, vcreate(P, A, x, scratch))

apply!(y::AbstractArray, A::AbstractArray, x::AbstractArray) =
    apply!(1, Direct, A, x, false, 0, y)

apply!(y::AbstractArray, P::Type{<:Operations}, A::AbstractArray, x::AbstractArray) =
    apply!(1, P, A, x, false, 0, y)

function vcreate(P::Type{<:Union{Direct,InverseAdjoint}},
                 A::AbstractArray{Ta,Na},
                 x::AbstractArray{Tx,Nx},
                 scratch::Bool) where {Ta,Na,Tx,Nx}
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
                 scratch::Bool) where {Ta,Na,Tx,Nx}
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
