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

identical(::Identity, ::Identity) = true

@callable Identity

# Traits:
SelfAdjointType(::Type{<:Identity}) = SelfAdjoint()
MorphismType(::Type{<:Identity}) = Endomorphism()
DiagonalType(::Type{<:Identity}) = DiagonalMapping()

const Identities = Union{Identity,
                         Adjoint{Identity},
                         Inverse{true,Identity},
                         InverseAdjoint{Identity}}

output_eltype(::Type{A}, ::Type{x}) where {A<:Identities,x} = float(eltype(x))

function vcreate(α::Number, ::Identities, x, scratch::Bool)
    T = output_eltype(α, Id, x)
    return (scratch && x isa Array{T}) ? x : vcreate(x, T)
end

apply!(α::Number, ::Identities, x, scratch::Bool, β::Number, y) =
    vcombine!(y, α, x, β, y)

simplify(::Identities) = Id
simplify(A::UniformScaling) = Mapping(A)

# Rules to automatically convert UniformScaling from standard library module
# LinearAlgebra into λ*Id.  For other operators, there is no needs to extend ⋅
# (\cdot) and ∘ (\circ) as they are already converted in calls to *.  But in
# the case of UniformScaling, we must explicitly do that for * and for ∘ (not
# for ⋅ which is replaced by a * by existing rules).
for op in (:(+), :(-), :(*), :(∘), :(/), Symbol("\\"))
    @eval begin
        Base.$op(A::UniformScaling, B::Mapping) = $op(Mapping(A), B)
        Base.$op(A::Mapping, B::UniformScaling) = $op(A, Mapping(B))
    end
end

#------------------------------------------------------------------------------
# SYMBOLIC MAPPINGS (FOR TESTS)

struct SymbolicMapping{L,S} <: Mapping{L} end

# Alias for symbolic linear mapping.
const SymbolicLinearMapping{S} = SymbolicMapping{true,S}

# Constructors.
SymbolicMapping(id) = SymbolicMapping{false}(id)
SymbolicMapping{L}(id::AbstractString) where {L} = SymbolicMapping{L}(Symbol(id))
SymbolicMapping{L}(id::Symbol) where {L} = SymbolicMapping{L,Val{id}}()

show(io::IO, A::SymbolicMapping{L,Val{S}}) where {L,S} = print(io, S)

identical(::T, ::T) where {T<:SymbolicMapping} = true

#------------------------------------------------------------------------------
# NON-UNIFORM SCALING

"""
    Diag(A) -> NonuniformScaling(A)

yields a non-uniform scaling linear mapping (of type `NonuniformScaling`) whose
effect is to apply elementwise multiplication of its argument by the scaling
factors `A`.  This mapping can be thought as a *diagonal* operator.

The `diag` method in `LinearAlgebra` can be called to retrieve the scaling
factors:

    using LinearAlgebra
    W = Diag(A)
    diag(W) === A  # this is true

!!! note
    Beware of the differences between the [`Diag`](@ref) (with an uppercase
    'D') and [`diag`](@ref) (with an lowercase 'd') methods.

"""
struct NonuniformScaling{T} <: LinearMapping
    diag::T
end

const Diag{T} = NonuniformScaling{T}

@callable NonuniformScaling

# Traits:
MorphismType(::Type{<:NonuniformScaling}) = Endomorphism() # FIXME: only if unitless coefficients
DiagonalType(::Type{<:NonuniformScaling}) = DiagonalMapping()
SelfAdjointType(::Type{<:NonuniformScaling{<:AbstractArray{<:Real}}}) =
    SelfAdjoint() # FIXME: check this...

coefficients(A::NonuniformScaling) = getfield(A, :diag)
LinearAlgebra.diag(A::NonuniformScaling) = coefficients(A)

identical(A::T, B::T) where {T<:NonuniformScaling} =
    coefficients(A) === coefficients(B)

Base.eltype(::Type{NonuniformScaling{T}}) where {T} = eltype(T)

input_ndims(::NonuniformScaling{<:AbstractArray{T,N}}) where {T, N} = N
input_size(A::NonuniformScaling{<:AbstractArray}) = size(coefficients(A))
input_size(A::NonuniformScaling{<:AbstractArray}, i) =
    size(coefficients(A), i)

output_ndims(::NonuniformScaling{<:AbstractArray{T,N}}) where {T, N} = N
output_size(A::NonuniformScaling{<:AbstractArray}) = size(coefficients(A))
output_size(A::NonuniformScaling{<:AbstractArray}, i) =
    size(coefficients(A), i)

# Simplify left multiplication (and division) by a scalar.
# FIXME: α = 0 should be treated specifically
# FIXME: This should only be done by `optimize`.
*(α::Number, A::NonuniformScaling)::NonuniformScaling =
    (α == 1 ? A : NonuniformScaling(vscale(α, coefficients(A))))

# Extend composition of diagonal operators.
# FIXME: This should only be done by `optimize`.
*(A::NonuniformScaling, B::NonuniformScaling) =
    NonuniformScaling(vproduct(coefficients(A), coefficients(B)))

function simplify(A::inv(NonuniformScaling{<:AbstractArray}))
    q = coefficients(A)
    r = similar(q, typeof(inv(oneunits(eltype(q)))))
    @inbounds @simd for i in eachindex(q, r)
        r[i] = inv(q[i])
    end
    return NonuniformScaling(r)
end

function simplify(A::NonuniformScaling{<:AbstractArray{Ta,N}}, *,
                  B::NonuniformScaling{<:AbstractArray{Tb,N}}) where {Ta,Tb,N}
    A_diag = diag(A)
    B_diag = diag(B)
    axes(A_diag) == axes(B_diag) || error("FIXME:")
    return NonuniformScaling(A_diag .* B_diag)
end

operation(::Mapping) = identity
operation(::Adjoint) = adjoint
operation(::Inverse) = inv
operation(::Inverse{true,<:Adjoint}) = inv∘adjoint
operation(::Adjoint{<:Inverse{true}}) = inv∘adjoint

const NonuniformScalings{T} = Union{NonuniformScaling{T},
                                    Adjoint{<:NonuniformScaling{T}},
                                    Inverse{true,<:NonuniformScaling{T}},
                                    Inverse{true,Adjoint{<:NonuniformScaling{T}}},
                                    Adjoint{Inverse{true,<:NonuniformScaling{T}}}}

function apply!(α::Number,
                W::NonuniformScalings{<:AbstractArray{Tw,N}},
                x::AbstractArray{Tx,N},
                scratch::Bool,
                β::Number,
                y::AbstractArray{Ty,N}) where {Tw,Tx,Ty,N}
    w = coefficients(W)
    I = all_indices(w, x, y)
    P = operation(W)
    if iszero(α)
        vscale!(y, β)
    elseif iszero(β)
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

function _apply_diagonal!(::typeof(identity), axpby::Function, I,
                          α, w, x, β, y)
    @inbounds @simd for i in I
        y[i] = axpby(α, w[i]*x[i], β, y[i])
    end
end

function _apply_diagonal!(::typeof(adjoint), axpby::Function, I,
                          α, w, x, β, y)
    @inbounds @simd for i in I
        y[i] = axpby(α, conj(w[i])*x[i], β, y[i])
    end
end

function _apply_diagonal!(::typeof(inv), axpby::Function, I,
                          α, w, x, β, y)
    @inbounds @simd for i in I
        y[i] = axpby(α, x[i]/w[i], β, y[i])
    end
end

function _apply_diagonal!(::typeof(inv∘adjoint), axpby::Function, I,
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
    RankOneOperator(u, v) -> A

yields the rank one linear operator `A = u⋅v'` defined by the two *vectors* `u`
and `v` and behaving as:

    A*x  -> vscale(vdot(v, x)), u)
    A'*x -> vscale(vdot(u, x)), v)

See also: [`SymmetricRankOneOperator`](@ref), [`LinearMapping`](@ref),
          [`apply!`](@ref), [`vcreate`](@ref).

"""
struct RankOneOperator{U,V} <: LinearMapping
    u::U
    v::V
end

@callable RankOneOperator

@generated Base.eltype(::Type{<:RankOneOperator{U,V}}) where {U,V} =
    :($(typeof(zero(eltype(U))*zero(eltype(V)))))

# Lazily assume that x has correct type, dimensions, etc.
# FIXME: optimize when scratch=true

vcreate(α::Number, A::RankOneOperator, x, scratch::Bool) = similar(A.v, output_eltype(α, A, x))
vcreate(α::Number, A::Adjoint{<:RankOneOperator}, x, scratch::Bool) = similar(A.u, output_eltype(α, A, x))

apply!(α::Number, A::RankOneOperator, x, scratch::Bool, β::Number, y) =
    _apply_rank_one!(α, A.u, A.v, x, β, y)

apply!(α::Number, A::Adjoint{<:RankOneOperator}, x, scratch::Bool, β::Number, y) =
    _apply_rank_one!(α, A.v, A.u, x, β, y)

function _apply_rank_one!(α::Number, u, v, x, β::Number, y)
    if iszero(α)
        # Lazily assume that y has correct type, dimensions, etc.
        vscale!(y, β)
    else
        vcombine!(y, α*vdot(v, x), u, β, y)
    end
    return y
end

# FIXME: input_type(A::RankOneOperator{U,V}) where {U,V} = V
input_ndims(::Type{<:RankOneOperator{U,V}}) where {U,V} = ndims(V)
input_size(A::RankOneOperator) = size(A.v)
input_size(A::RankOneOperator, d...) = size(A.v, d...)
# FIXME: input_eltype(A::RankOneOperator) = eltype(A.v)

# FIXME: output_type(A::RankOneOperator{U,V}) where {U,V} = U
output_ndims(::Type{<:RankOneOperator{U,V}}) where {U,V} = ndims(U)
output_size(A::RankOneOperator) = size(A.u)
output_size(A::RankOneOperator, d...) = size(A.u, d...)
# FIXME: output_eltype(A::RankOneOperator) = eltype(A.u)

identical(A::T, B::T) where {T<:RankOneOperator} = ((A.u === B.u)&(A.v === B.v))

"""
    SymmetricRankOneOperator(u) -> A

yields the symmetric rank one operator `A = u⋅u'` defined by the *vector* `u`
and behaving as follows:

    A'*x -> A*x
    A*x  -> vscale(vdot(u, x)), u)

See also: [`RankOneOperator`](@ref), [`LinearMapping`](@ref),
          [`Trait`](@ref) [`apply!`](@ref), [`vcreate`](@ref).

"""
struct SymmetricRankOneOperator{U} <: LinearMapping
    u::U
end

@callable SymmetricRankOneOperator

# Traits:
MorphismType(::Type{<:SymmetricRankOneOperator}) = Endomorphism() # FIXME: not true with units
SelfAdjointType(::Type{<:SymmetricRankOneOperator}) = SelfAdjoint()

@generated Base.eltype(::Type{<:SymmetricRankOneOperator{U}}) where {U} =
    :($(typeof(zero(eltype(U))^2)))

# Automatic simplifications rules.
adjoint(A::SymmetricRankOneOperator) = A

apply!(α::Number, A::SymmetricRankOneOperator, x, scratch::Bool, β::Number, y) =
    _apply_rank_one!(α, A.u, A.u, x, β, y)
apply!(α::Number, A::Adjoint{<:SymmetricRankOneOperator}, x, scratch::Bool, β::Number, y) =
    apply!(α, parent(A), x, scratch, β, y)

# Lazily assume that x has correct dimensions.
function vcreate(α::Number, A::SymmetricRankOneOperator, x, scratch::Bool)
    T = output_eltype(α, A, x)
    return (scratch && x isa Array{T,ndims(A.u)}) ? x : similar(A.u, T)
end
vcreate(α::Number, A::Adjoint{<:SymmetricRankOneOperator}, x, scratch::Bool) =
    vcreate(α, parent(A), x)

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

creates a linear mapping whose coefficients are given by a multi-dimensional
array `A` and which generalizes the definition of the matrix-vector product
without calling `reshape` to change the dimensions.

For instance, assuming that `G = GeneralMatrix(A)` with `A` a regular array,
then `y = G*x` requires that the dimensions of `x` match the trailing
dimensions of `A` and yields a result `y` whose dimensions are the remaining
leading dimensions of `A`, such that `axes(A) = (axes(y)..., axes(x)...)`.
Applying the adjoint of `G` as in `y = G'*x` requires that the dimensions of
`x` match the leading dimension of `A` and yields a result `y` whose dimensions
are the remaining trailing dimensions of `A`, such that `axes(A) = (axes(x)...,
axes(y)...)`.

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

identical(A::T, B::T) where {T<:GeneralMatrix} =
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
        bad_size("the indices of `x` do not match the trailing indices of `A`")
    1 ≤ Nx < Na || incompatible_dimensions()
    Ny = Na - Nx
    xinds = axes(x)
    Ainds = axes(A)
    @inbounds for d in 1:Nx
        xinds[d] == Ainds[Ny + d] || incompatible_dimensions()
    end
    shape = ntuple(d -> Ainds[d], Val(Ny)) # faster than Ainds[1:Ny]
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
        bad_size("the indices of `x` do not match the leading indices of `A`")
    1 ≤ Nx < Na || incompatible_dimensions()
    Ny = Na - Nx
    xinds = axes(x)
    Ainds = axes(A)
    @inbounds for d in 1:Nx
        xinds[d] == Ainds[d] || incompatible_dimensions()
    end
    shape = ntuple(d -> Ainds[Nx + d], Val(Ny)) # faster than Ainds[Nx+1:end]
    return similar(A, promote_type(Ta, Tx), shape)
end
