#
# finitedifferences.jl -
#
# Implement rules for basic operations involving mappings.
#
#-------------------------------------------------------------------------------
#
# This file is part of LazyAlgebra (https://github.com/emmt/LazyAlgebra.jl)
# released under the MIT "Expat" license.
#
# Copyright (c) 2017-2018 Éric Thiébaut.
#

module FiniteDifferences

export
    SimpleFiniteDifferences


using Compat
using ..Coder
using  ...LazyAlgebra
import ...LazyAlgebra: vcreate, apply!, HalfHessian
using  ...LazyAlgebra: fastrange, @callable

# Define operator D which implements simple finite differences.  Make it
# callable.
struct SimpleFiniteDifferences <: LinearMapping end
@callable SimpleFiniteDifferences

# Extend the vcreate() and apply!() methods for these operators.  The apply!()
# method does all the checking and, then, calls a private method specialized
# for the considered dimensionality.

function vcreate(::Type{Direct}, ::SimpleFiniteDifferences,
                 x::AbstractArray{T,N}) where {T<:Real,N}
    return Array{T}(undef, (N, size(x)...))
end

function vcreate(::Type{Adjoint}, ::SimpleFiniteDifferences,
                 x::AbstractArray{T,N}) where {T<:Real,N}
    N ≥ 2 ||
        throw(DimensionMismatch("argument must have at least 2 dimensions"))
    dims = size(x)
    dims[1] == N - 1 ||
        throw(DimensionMismatch("first dimension should be $(N-1)"))
    return Array{T}(undef, dims[2:end])
end

# FIXME: The following is not absolutely needed because HalfHessian is
#        automatically an Endomorphism.
for P in (Direct, Adjoint)
    @eval function vcreate(::Type{$P},
                           ::HalfHessian{SimpleFiniteDifferences},
                           x::AbstractArray{T,N}) where {T<:Real,N}
        return Array{T}(undef, size(x))
    end
end

function apply!(α::Real, ::Type{<:Direct}, ::SimpleFiniteDifferences,
                x::AbstractArray{Tx,Nx}, β::Real,
                y::AbstractArray{Ty,Ny}) where {Tx<:Real,Nx,Ty<:Real,Ny}
    Ny == Nx + 1 ||
        throw(DimensionMismatch("incompatible number of dimensions"))
    ydims = size(y)
    ydims[1]== Nx ||
        throw(DimensionMismatch("first dimension of destination must be $Nx"))
    ydims[2:end] == size(x) ||
        throw(DimensionMismatch("dimensions 2:end of destination must be $(size(x))"))
    if α == 0
        vscale!(y, β)
    else
        T = promote_type(Tx, Ty)
        _apply_D!(convert(T, α), x, convert(T, β), y)
    end
    return y
end

function apply!(α::Real, ::Type{<:Adjoint}, ::SimpleFiniteDifferences,
                x::AbstractArray{Tx,Nx}, β::Real,
                y::AbstractArray{Ty,Ny}) where {Tx<:Real,Nx,Ty<:Real,Ny}
    Ny == Nx - 1 ||
        throw(DimensionMismatch("incompatible number of dimensions"))
    xdims = size(x)
    xdims[1]== Ny ||
        throw(DimensionMismatch("first dimension of source must be $Ny"))
    xdims[2:end] == size(y) ||
        throw(DimensionMismatch("dimensions 2:end of source must be $(size(y))"))
    vscale!(y, β)
    if α != 0
        T = promote_type(Tx, Ty)
        _apply_Dt!(convert(T, α), x, convert(T, β), y)
    end
    return y
end

function apply!(α::Real, ::Type{<:Union{Direct,Adjoint}},
                ::HalfHessian{SimpleFiniteDifferences},
                x::AbstractArray{Tx,Nx}, β::Real,
                y::AbstractArray{Ty,Ny}) where {Tx<:Real,Nx,Ty<:Real,Ny}
    Ny == Nx ||
        throw(DimensionMismatch("incompatible number of dimensions"))
    size(x) == size(y) ||
        throw(DimensionMismatch("source and destination must have the same dimensions"))
    vscale!(y, β)
    if α != 0
        T = promote_type(Tx, Ty)
        _apply_DtD!(convert(T, α), x, convert(T, β), y)
    end
    return y
end

offset(::Type{CartesianIndex{N}}, d::Integer, s::Integer=1) where {N} =
    CartesianIndex{N}(ntuple(i -> (i == d ? s : 0), N))

@generated function _apply_D!(a::T, x::AbstractArray{<:Real,N},
                              b::T, y::AbstractArray{<:Real,Np1}
                              ) where {N,Np1,T<:Real}
    # We know that a ≠ 0.
    @assert Np1 == N + 1
    D = generate_symbols("d", N)
    I = generate_symbols("i", N)
    S = generate_symbols("s", N)
    common = (
        [:( $(I[d]) = min(i + $(S[d]), imax) ) for d in 1:N]...,
        :(  xi = x[i]                        ),
        [:( $(D[d]) = x[$(I[d])] - xi        ) for d in 1:N]...
    )
    return encode(
        :(  inds = fastrange(size(x))                 ),
        :(  imax = last(inds)                         ),
        [:( $(S[d]) = $(offset(CartesianIndex{N}, d)) ) for d in 1:N]...,
        :inbounds,
        (
            :if, :(b == 0),
            (
                :simd_for, :(i in inds),
                (
                    common...,
                    [:( y[$d,i] = a*$(D[d]) ) for d in 1:N]...
                )
            ),
            :elseif, :(b == 1),
            (
                :simd_for, :(i in inds),
                (
                    common...,
                    [:( y[$d,i] += a*$(D[d]) ) for d in 1:N]...
                )
            ),
            :else,
            (
                :simd_for, :(i in inds),
                (
                    common...,
                    [:( y[$d,i] = a*$(D[d]) + b*y[$d,i] ) for d in 1:N]...
                )
            )
        )
    )
end

@generated function _apply_Dt!(a::T, x::AbstractArray{<:Real,Np1},
                               b::T, y::AbstractArray{<:Real,N}
                               ) where {N,Np1,T<:Real}
    # We know that a ≠ 0 and that y has been pre-multiplied by b.
    @assert Np1 == N + 1
    D = generate_symbols("d", N)
    I = generate_symbols("i", N)
    S = generate_symbols("s", N)
    common = [:( $(I[d]) = min(i + $(S[d]), imax) ) for d in 1:N]
    return encode(
        :(  inds = fastrange(size(y))                 ),
        :(  imax = last(inds)                         ),
        [:( $(S[d]) = $(offset(CartesianIndex{N}, d)) ) for d in 1:N]...,
        :inbounds,
        (
            :if, :(a == 1),
            (
                :simd_for, :(i in inds),
                (
                    common...,
                    [:( $(D[d]) = x[$d,i]                 ) for d in 1:N]...,
                    :(  y[i] -= $(encode_sum_of_terms(D)) ),
                    [:( y[$(I[d])] += $(D[d])             ) for d in 1:N]...
                ),
            ),
            :else,
            (
                :simd_for, :(i in inds),
                (
                    common...,
                    [:( $(D[d]) = a*x[$d,i]               ) for d in 1:N]...,
                    :(  y[i] -= $(encode_sum_of_terms(D)) ),
                    [:( y[$(I[d])] += $(D[d])             ) for d in 1:N]...
                ),
            )
        )
    )
end

@generated function _apply_DtD!(a::T, x::AbstractArray{<:Real,N},
                                b::T, y::AbstractArray{<:Real,N}
                                ) where {N,T<:Real}
    # We know that a ≠ 0 and that y has been pre-multiplied by b.
    D = generate_symbols("d", N)
    I = generate_symbols("i", N)
    S = generate_symbols("s", N)
    common = (
        [:( $(I[d]) = min(i + $(S[d]), imax) ) for d in 1:N]...,
        :(  xi = x[i]                        ))
    return encode(
        :(  inds = fastrange(size(x))                 ),
        :(  imax = last(inds)                         ),
        [:( $(S[d]) = $(offset(CartesianIndex{N}, d)) ) for d in 1:N]...,
        :inbounds,
        (
            :if, :(a == 1),
            (
                :simd_for, :(i in inds),
                (
                    common...,
                    [:( $(D[d]) = x[$(I[d])] - xi         ) for d in 1:N]...,
                    :(  y[i] -= $(encode_sum_of_terms(D)) ),
                    [:( y[$(I[d])] += $(D[d])             ) for d in 1:N]...
                ),
            ),
            :else,
            (
                :simd_for, :(i in inds),
                (
                    common...,
                    [:( $(D[d]) = a*(x[$(I[d])] - xi)     ) for d in 1:N]...,
                    :(  y[i] -= $(encode_sum_of_terms(D)) ),
                    [:( y[$(I[d])] += $(D[d])             ) for d in 1:N]...
                ),
            )
        )
    )
end



end # module
