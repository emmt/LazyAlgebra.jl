#
# operators.jl -
#
# Provide basic operators.
#
#-----------------------------------------------------------------------------------------
#
# This file is part of LazyAlgebra (https://github.com/emmt/LazyAlgebra.jl) released under
# the MIT "Expat" license.
#
# Copyright (c) 2017-2025, Éric Thiébaut.
#

const AnyVariant{A} = Union{A,Adjoint{A},Inverse{A},InverseAdjoint{A}}

#------------------------------------------------------------------------------
# IDENTITY AND UNIFORM SCALING


#------------------------------------------------------------------------------
# SYMBOLIC MAPPINGS (FOR TESTS)

struct SymbolicOperator <: Operator
    name::Symbol
end

SymbolicOperator(name::AbstractString) = SymbolicOperator(Symbol(id))

show(io::IO, A::SymbolicOperator) = print(io, A.name)

identical(A::SymbolicOperator, B::SymbolicOperator) = A.name === B.name

#------------------------------------------------------------------------------
# NON-UNIFORM SCALING

"""
    A = Diag(w)

yields a non-uniform scaling linear mapping (of type `NonuniformScaling`) whose
effect is to apply elementwise multiplication of its argument by the scaling
factors `w`. This operator can be thought as a generalized *diagonal* operator.

The `LinearAlgebra.diag` method (exported by `using LazyAlgebra`) can be called to
retrieve the scaling factors:

    using LinearAlgebra
    W = Diag(A)
    diag(W) === A  # this is true

!!! note
    Beware of the differences between the [`Diag`](@ref) (with an uppercase 'D') and
    `diag` (with an lowercase 'd') methods.

"""
struct Diag{D<:AbstractArray} <: Operator
    diag::D
end

@callable Diag

struct LazyMap{T,N,L,F,A<:AbstractArray{<:Any,N}} <: AbstractArray{T,N}
    func::F
    arr::A
    LazyMap{T}(func::F, arr::A) where {T,N,F<:Function,A<:AbstractArray{<:Any,N}} =
        new{T,N,IndexStyle(A)==IndexLinear(),F,A}(func, arr)
end
LazyMap(func::Function, arr::AbstractArray) =
    LazyMap{Base.promote_op(func, eltype(arr))}(func, arr)

Base.length(A::LazyMap) = length(A.arr)
Base.size(A::LazyMap) = size(A.arr)
Base.axes(A::LazyMap) = axes(A.arr)
for (L, S, Idecl, Icall) in ((false, :IndexCartesian, :(I::Vararg{Int,N}), :(I...)),
                             (true,  :IndexLinear,    :(i::Int),           :(i)))
    @eval begin
        Base.IndexStyle(::Type{<:LazyMap{T,N,$L}}) where {T,N} = $S()
        @inline function Base.getindex(A::LazyMap{T,N,$L}, $Idecl) where {T,N}
            @boundscheck checkbounds(A, $Icall)
            return as(T, A.func(@inbounds(getindex(A.arr, $Icall))))
        end
        @inline function Base.setindex!(A::LazyMap{T,N,$L}, x, $Idecl) where {T,N}
            @boundscheck checkbounds(A, $Icall)
            error("attempt to write read-only array")
            return A
        end
    end
end

# Traits.
Base.eltype(::Type{<:Union{A,Adjoint{A}}}) where {D,A<:Diag{D}} = eltype(D)
Base.eltype(::Type{<:Union{Inverse{A},InverseAdjoint{A}}}) where {D,A<:Diag{D}} =
    float(eltype(D))

MorphismType(::Diag) = Endomorphism()
DiagonalType(::Diag) = DiagonalOperator()
SelfAdjointType(A::Diag) =
    _selfadjointtype(eltype(coefficients(A)), A)
_selfadjointtype(::Type{<:Real}, ::Diag) =
    SelfAdjoint()
_selfadjointtype(::Type{<:Complex}, ::Diag) =
    NonSelfAdjoint()

# Accessors.
coefficients(A::Diag) = diag(A)
LinearAlgebra.diag(A::Diag) = A.diag
LinearAlgebra.diag(A::Adjoint{<:Diag}) = LazyMap(conj, diag(unveil(A)))
LinearAlgebra.diag(A::Inverse{<:Diag}) = LazyMap(inv, diag(unveil(A)))
LinearAlgebra.diag(A::InverseAdjoint{<:Diag}) = LazyMap(inv∘conj, diag(unveil(A)))

identical(A::T, B::T) where {T<:Diag} =
    coefficients(A) === coefficients(B)

# FIXME: simplify the product of diagonal operators, theirs inverse/adjoint, etc..
simplify(A::Inverse{<:Diag}) = Diag(map(inv, diag(A)))
simplify(A::Scaled{<:Diag}) =
    # FIXME: α = 0 should be treated specifically
    Diag(map(Base.Fix1(*,multiplier(A)), diag(unscaled(A))))

output_axes(A::AnyVariant{Diag}, x::AbstractArray) =
    (rngs = axes(diag(unveil(A)))) == axes(x) ? rngs : throw(DimensionMismatch(
        "argument has incompatible indices"))

conj_mul(w, x) = conj(w)*x
conj_ldiv(w, x) = conj(w)\x

function unsafe_apply!(α::Number, A::Diag, x::AbstractArray,
                       β::Number, y::AbstractArray)
    unsafe_map!(α, *, diag(A), x, β, y)
end

function unsafe_apply!(α::Number, A::Adjoint{<:Diag}, x::AbstractArray,
                       β::Number, y::AbstractArray)
    unsafe_map!(α, conj_mul, diag(unveil(A)), x, β, y)
end

function unsafe_apply!(α::Number, A::Inverse{<:Diag}, x::AbstractArray,
                       β::Number, y::AbstractArray)
    unsafe_map!(α, \, diag(unveil(A)), x, β, y)
end

function unsafe_apply!(α::Number, A::InverseAdjoint{<:Diag}, x::AbstractArray,
                       β::Number, y::AbstractArray)
    unsafe_map!(α, conj_ldiv, diag(unveil(A)), x, β, y)
end

"""
    LazyAlgebra.unsafe_map!(dst, α, f, w, x)

overwrite `dst` with `dst[i] = α*f(w[i], x[i])` assuming that `dst`, `w`, and `x` have the
same axes, and that multiplier `α` has suitable types and is non-zero.

"""
function unsafe_map!(dst::AbstractArray,
                     α::Number,
                     f::Function,
                     w::AbstractArray,
                     x::AbstractArray)
    if α == one(α)
        @inbounds @simd for i in eachindex(dst, w, x)
            dst[i] = f(w[i], x[i])
        end
    else
        @inbounds @simd for i in eachindex(dst, w, x)
            dst[i] = α*f(w[i], x[i])
        end
    end
    nothing
end

"""
    LazyAlgebra.unsafe_map!(α, f, w, x, β, y)

overwrite `y` with `y[i] = α*f(w[i], x[i]) + β*y[i])` assuming that `w`, `x`, and `y` have
the same axes, and that multipliers `α` and `β` have suitable types and are both non-zero.

"""
function unsafe_map!(α::Number,
                     f::Function,
                     w::AbstractArray,
                     x::AbstractArray,
                     β::Number,
                     y::AbstractArray)
    if β == zero(β)
        unsafe_map!(y, α, f, w, x)
    elseif β == one(β)
        if α == one(α)
            @inbounds @simd for i in eachindex(w, x)
                y[i] += f(w[i], x[i])
            end
        elseif α == -one(α)
            @inbounds @simd for i in eachindex(w, x)
                y[i] -= f(w[i], x[i])
            end
        else
            @inbounds @simd for i in eachindex(w, x)
                y[i] += α*f(w[i], x[i])
            end
        end
    elseif β == -one(β)
        if α == one(α)
            @inbounds @simd for i in eachindex(w, x)
                y[i] = f(w[i], x[i]) - y[i]
            end
        else
            @inbounds @simd for i in eachindex(w, x)
                y[i] = α*f(w[i], x[i]) - y[i]
            end
        end
    else
        if α == one(α)
            @inbounds @simd for i in eachindex(w, x)
                y[i] = f(w[i], x[i]) + β*y[i]
            end
        else
            @inbounds @simd for i in eachindex(w, x)
                y[i] = α*f(w[i], x[i]) + β*y[i]
            end
        end
    end
    nothing
end

#------------------------------------------------------------------------------
# RANK-1 OPERATORS

"""
    RankOneOperator(u, v) -> A

yields the rank one linear operator `A = u⋅v'` defined by the two *vectors* `u` and `v`
and behaving as:

    A*x  -> vdot(v, x) * u
    A'*x -> vdot(u, x) * v

See also: [`SymmetricRankOneOperator`](@ref), [`Operator`](@ref),
          [`apply!`](@ref), [`vcreate`](@ref).

"""
struct RankOneOperator{U<:AbstractArray,V<:AbstractArray} <: Operator
    u::U
    v::V
end

@callable RankOneOperator

function apply!(α::Number, A::RankOneOperator,
                x, scratch::Bool, β::Number, y)
    return _apply_rank_one!(α, A.u, A.v, x, β, y)
end

function apply!(α::Number, A::Adjoint{<:RankOneOperator},
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
#FIXME: vcreate(::Type{Direct}, A::RankOneOperator, x, scratch::Bool) = vcreate(A.v)
#FIXME: vcreate(::Type{Adjoint}, A::RankOneOperator, x, scratch::Bool) = vcreate(A.u)

# Traits.
for func in (:(Base.eltype), :input_ndims, :output_ndims)
    @eval $func(A::RankOneOperator) = $func(typeof(A))
end
Base.eltype(::Type{<:RankOneOperator{U,V}}) where {U,V} =
    float(prod_type(eltype(U), eltype(V)))
input_ndims(::Type{<:RankOneOperator{U,V}}) where {U,V} = ndims(V)
output_ndims(::Type{<:RankOneOperator{U,V}}) where {U,V} = ndims(U)

input_size(A::RankOneOperator) = size(A.v)
input_size(A::RankOneOperator, d...) = size(A.v, d...)
input_eltype(A::RankOneOperator) = eltype(A.v)

output_size(A::RankOneOperator) = size(A.u)
output_size(A::RankOneOperator, d...) = size(A.u, d...)

identical(A::T, B::T) where {T<:RankOneOperator} =
    ((A.u === B.u)&(A.v === B.v))

"""
    SymmetricRankOneOperator(u) -> A

yields the symmetric rank one operator `A = u⋅u'` defined by the *vector* `u`
and behaving as follows:

    A'*x -> A*x
    A*x  -> vscale(vdot(u, x)), u)

See also: [`RankOneOperator`](@ref), [`Operator`](@ref),
          [`Trait`](@ref) [`apply!`](@ref), [`vcreate`](@ref).

"""
struct SymmetricRankOneOperator{U} <: Operator
    u::U
end

@callable SymmetricRankOneOperator

# Traits:
MorphismType(::SymmetricRankOneOperator) = Endomorphism()
SelfAdjointType(::SymmetricRankOneOperator) = SelfAdjoint()

# FIXME: function apply!(α::Number, ::Type{<:Union{Direct,Adjoint}},
# FIXME:                 A::SymmetricRankOneOperator, x, scratch::Bool, β::Number, y)
# FIXME:     return _apply_rank_one!(α, A.u, A.u, x, β, y)
# FIXME: end
# FIXME:
# FIXME: function vcreate(::Type{<:Union{Direct,Adjoint}},
# FIXME:                  A::SymmetricRankOneOperator, x, scratch::Bool)
# FIXME:     # Lazily assume that x has correct type, dimensions, etc.
# FIXME:     return (scratch ? x : vcreate(x))
# FIXME: end

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

identical(A::T, B::T) where {T<:SymmetricRankOneOperator} =
    (A.u === B.u)

#------------------------------------------------------------------------------
# GENERALIZED MATRIX AND MATRIX-VECTOR PRODUCT

"""
    GeneralMatrix(A)

creates a linear operator whose coefficients are given by a multi-dimensional array `A`
and which generalizes the definition of the matrix-vector product.

For instance, assuming that `G = GeneralMatrix(A)` with `A` a regular array, then `y =
G*x` requires that the dimensions of `x` match the trailing dimensions of `A` and yields a
result `y` whose dimensions are the remaining leading dimensions of `A`, such that
`axes(A) = (axes(y)..., axes(x)...)`. Applying the adjoint of `G` as in `y = G'*x`
requires that the dimensions of `x` match the leading dimension of `A` and yields a result
`y` whose dimensions are the remaining trailing dimensions of `A`, such that `axes(A) =
(axes(x)..., axes(y)...)`.

"""
struct GeneralMatrix{T,L,A<:AbstractArray{T,L}} <: Operator
    arr::A
    GeneralMatrix(arr::A) where {T,L,A<:AbstractArray{T,L}} = new{T,L,A}(arr)
end
GeneralMatrix{T}(arr) where {T} = GeneralMatrix(as(AbstractArray{T}, arr))

@callable GeneralMatrix

# Traits.
Base.eltype(::Type{<:Union{A,Adjoint{A}}}) where {T,A<:GeneralMatrix{T}} = T

# Accessors.
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

identical(A::T, B::T) where {T<:GeneralMatrix} =
    (coefficients(A) === coefficients(B))

# FIXME: function apply!(α::Number,
# FIXME:                 P::Type{<:Operations},
# FIXME:                 A::GeneralMatrix{<:AbstractArray{<:GenMult.Floats}},
# FIXME:                 x::AbstractArray{<:GenMult.Floats},
# FIXME:                 scratch::Bool,
# FIXME:                 β::Number,
# FIXME:                 y::AbstractArray{<:GenMult.Floats})
# FIXME:     return apply!(α, P, coefficients(A), x, scratch, β, y)
# FIXME: end

function output_axes(A::Union{G,InverseAdjoint{G}},
                     x::AbstractArray{<:Any,N}) where {N,L,
                                                       G<:GeneralMatrix{<:Any,L}}
    rngs = axes(coefficients(unveil(A)))
    N ≤ L && axes(x) == rngs[L-N+1:L] || throw(DimensionMismatch(
        "axes of argument do not match trailing dimensions of generalized matrix coefficients"))
    return rngs[1:L-N]
end

function output_axes(A::Union{Adjoint{G},Inverse{G}},
                     x::AbstractArray{<:Any,M}) where {M,L,
                                                       G<:GeneralMatrix{<:Any,L}}
    rngs = axes(coefficients(unveil(A)))
    M ≤ L && axes(x) == rngs[1:M] || throw(DimensionMismatch(
        "axes of argument do not match leading dimensions of generalized matrix coefficients"))
    return rngs[M+1:L]
end

function unsafe_apply!(α::Number,
                       A::GeneralMatrix,
                       x::AbstractArray,
                       β::Number,
                       y::AbstractArray)
    C = coefficients(unveil(A))
    I = CartesianIndices(axes(y))
    J = CartesianIndices(axes(x))
    vscale!(y, β)
    @inbounds for j in J
        αxⱼ = α*x[j]
        if αxⱼ != zero(αxⱼ)
            @simd for i in I
                y[i] += C[i,j]*αxⱼ
            end
        end
    end
    nothing
end

function unsafe_apply!(α::Number,
                       A::Adjoint{<:GeneralMatrix},
                       x::AbstractArray,
                       β::Number,
                       y::AbstractArray)
    C = coefficients(unveil(A))
    I = CartesianIndices(axes(x))
    J = CartesianIndices(axes(y))
    @inbounds for j in J
        s = 0*zero(eltype(C))*zero(eltype(x))
        @simd for i in I
            s += conj(C[i,j])*x[i]
        end
        if β == zero(β)
            y[j] = α*s
        else
            y[j] = β*y[j] + α*s
        end
    end
    nothing
end


# FIXME: function vcreate(P::Type{<:Operations},
# FIXME:                  A::GeneralMatrix{<:AbstractArray{<:GenMult.Floats}},
# FIXME:                  x::AbstractArray{<:GenMult.Floats},
# FIXME:                  scratch::Bool)
# FIXME:     return vcreate(P, coefficients(A), x, scratch)
# FIXME: end
# FIXME:
# FIXME: for (T, L) in ((:Direct, 'N'), (:Adjoint, 'C'))
# FIXME:     @eval begin
# FIXME:         function apply!(α::Number,
# FIXME:                         ::Type{$T},
# FIXME:                         A::AbstractArray{<:GenMult.Floats},
# FIXME:                         x::AbstractArray{<:GenMult.Floats},
# FIXME:                         scratch::Bool,
# FIXME:                         β::Number,
# FIXME:                         y::AbstractArray{<:GenMult.Floats})
# FIXME:             return lgemv!(α, $L, A, x, β, y)
# FIXME:         end
# FIXME:     end
# FIXME: end

# To have apply and apply! methods callable with an array (instead of a
# mapping), we have to provide the different possibilities.

# FIXME: apply(A::AbstractArray, x::AbstractArray, scratch::Bool) =
# FIXME:     apply(Direct, A, x, scratch)
# FIXME:
# FIXME: apply(P::Type{<:Operations}, A::AbstractArray, x::AbstractArray, scratch::Bool) =
# FIXME:     apply!(1, P, A, x, scratch, 0, vcreate(P, A, x, scratch))
# FIXME:
# FIXME: apply!(y::AbstractArray, A::AbstractArray, x::AbstractArray) =
# FIXME:     apply!(1, Direct, A, x, false, 0, y)
# FIXME:
# FIXME: apply!(y::AbstractArray, P::Type{<:Operations}, A::AbstractArray, x::AbstractArray) =
# FIXME:     apply!(1, P, A, x, false, 0, y)
# FIXME:
# FIXME: function vcreate(P::Type{<:Union{Direct,InverseAdjoint}},
# FIXME:                  A::AbstractArray{Ta,Na},
# FIXME:                  x::AbstractArray{Tx,Nx},
# FIXME:                  scratch::Bool) where {Ta,Na,Tx,Nx}
# FIXME:     # Non-transposed matrix.  Trailing dimensions of X must match those of A,
# FIXME:     # leading dimensions of A are those of the result.  Whatever the scratch
# FIXME:     # parameter, a new array is returned as the operation cannot be done
# FIXME:     # in-place.
# FIXME:     @noinline incompatible_dimensions() =
# FIXME:         bad_size("the indices of `x` do not match the trailing indices of `A`")
# FIXME:     1 ≤ Nx < Na || incompatible_dimensions()
# FIXME:     Ny = Na - Nx
# FIXME:     xinds = axes(x)
# FIXME:     Ainds = axes(A)
# FIXME:     @inbounds for d in 1:Nx
# FIXME:         xinds[d] == Ainds[Ny + d] || incompatible_dimensions()
# FIXME:     end
# FIXME:     shape = ntuple(d -> Ainds[d], Val(Ny)) # faster than Ainds[1:Ny]
# FIXME:     return similar(A, promote_type(Ta, Tx), shape)
# FIXME: end
# FIXME:
# FIXME: function vcreate(P::Type{<:Union{Adjoint,Inverse}},
# FIXME:                  A::AbstractArray{Ta,Na},
# FIXME:                  x::AbstractArray{Tx,Nx},
# FIXME:                  scratch::Bool) where {Ta,Na,Tx,Nx}
# FIXME:     # Transposed matrix.  Leading dimensions of X must match those of A,
# FIXME:     # trailing dimensions of A are those of the result.  Whatever the scratch
# FIXME:     # parameter, a new array is returned as the operation cannot be done
# FIXME:     # in-place.
# FIXME:     @noinline incompatible_dimensions() =
# FIXME:         bad_size("the indices of `x` do not match the leading indices of `A`")
# FIXME:     1 ≤ Nx < Na || incompatible_dimensions()
# FIXME:     Ny = Na - Nx
# FIXME:     xinds = axes(x)
# FIXME:     Ainds = axes(A)
# FIXME:     @inbounds for d in 1:Nx
# FIXME:         xinds[d] == Ainds[d] || incompatible_dimensions()
# FIXME:     end
# FIXME:     shape = ntuple(d -> Ainds[Nx + d], Val(Ny)) # faster than Ainds[Nx+1:end]
# FIXME:     return similar(A, promote_type(Ta, Tx), shape)
# FIXME: end
