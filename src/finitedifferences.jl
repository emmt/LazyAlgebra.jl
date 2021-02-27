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
# Copyright (c) 2017-2020 Éric Thiébaut.
#

module FiniteDifferences

export
    SimpleFiniteDifferences

using ..Coder
using ..Foundations
using ..LazyAlgebra
using ..LazyAlgebra:
    @callable, Floats, bad_argument, bad_size
import ..LazyAlgebra:
    vcreate, apply!, identical
using ArrayTools
import Base: show, *

# Define operator D which implements simple finite differences.  Make it
# callable.
struct SimpleFiniteDifferences <: LinearMapping end
@callable SimpleFiniteDifferences

const Diff = SimpleFiniteDifferences

identical(::SimpleFiniteDifferences, ::SimpleFiniteDifferences) = true

show(io::IO, ::SimpleFiniteDifferences) = print(io, "Diff")
show(io::IO, ::Gram{SimpleFiniteDifferences}) = print(io, "(Diff'⋅Diff)")

# Extend the vcreate() and apply!() methods for these operators.  The apply!()
# method does all the checking and, then, calls a private method specialized
# for the considered dimensionality.

function vcreate(::Type{Direct},
                 ::SimpleFiniteDifferences,
                 x::AbstractArray{T,N},
                 scratch::Bool) where {T<:Floats,N}
    # In-place operation never possible, so ignore the scratch flag.
    return Array{T}(undef, (N, size(x)...))
end

function vcreate(::Type{Adjoint},
                 ::SimpleFiniteDifferences,
                 x::AbstractArray{T,N},
                 scratch::Bool) where {T<:Floats,N}
    # In-place operation never possible, so ignore the scratch flag.
    N ≥ 2 ||
        bad_size("argument must have at least 2 dimensions")
    dims = size(x)
    dims[1] == N - 1 ||
        bad_size("first dimension should be ", N - 1)
    return Array{T}(undef, dims[2:end])
end

function vcreate(::Type{Direct},
                 ::Gram{SimpleFiniteDifferences},
                 x::AbstractArray{T,N},
                 scratch::Bool) where {T<:Real,N}
    (scratch && isa(x, Array{T,N})) ? x : Array{T,N}(undef, size(x))
end

# A Gram operator is Hermitian by construction.
vcreate(::Type{Adjoint}, A::Gram{SimpleFiniteDifferences}, x, scratch::Bool) =
    vcreate(Direct, A, x, scratch)

function apply!(α::Number,
                ::Type{<:Direct},
                ::SimpleFiniteDifferences,
                x::AbstractArray{Tx,Nx},
                scratch::Bool,
                β::Number,
                y::AbstractArray{Ty,Ny}) where {Tx<:Floats,Nx,Ty<:Floats,Ny}
    Ny == Nx + 1 ||
        bad_size("incompatible number of dimensions")
    ydims = size(y)
    ydims[1]== Nx ||
        bad_size("first dimension of destination must be ", Nx)
    ydims[2:end] == size(x) ||
        bad_size("dimensions 2:end of destination must be ", size(x))
    if α == 0
        vscale!(y, β)
    else
        _apply_D!(promote_multiplier(α, Tx), x, promote_multiplier(β, Ty), y)
    end
    return y
end

function apply!(α::Number,
                ::Type{<:Adjoint},
                ::SimpleFiniteDifferences,
                x::AbstractArray{Tx,Nx},
                scratch::Bool,
                β::Number,
                y::AbstractArray{Ty,Ny}) where {Tx<:Floats,Nx,Ty<:Floats,Ny}
    Ny == Nx - 1 ||
        bad_size("incompatible number of dimensions")
    xdims = size(x)
    xdims[1]== Ny ||
        bad_size("first dimension of source must be ", Ny)
    xdims[2:end] == size(y) ||
        bad_size("dimensions 2:end of source must be ", size(y))
    β == 1 || vscale!(y, β)
    α == 0 || _apply_Dt!(promote_multiplier(α, Tx), x, y)
    return y
end

function apply!(α::Number,
                ::Type{Direct},
                ::Gram{SimpleFiniteDifferences},
                x::AbstractArray{Tx,Nx},
                scratch::Bool,
                β::Number,
                y::AbstractArray{Ty,Ny}) where {Tx<:Floats,Nx,Ty<:Floats,Ny}
    Ny == Nx ||
        bad_size("incompatible number of dimensions")
    size(x) == size(y) ||
        bad_size("source and destination must have the same dimensions")
    β == 1 || vscale!(y, β)
    α == 0 || _apply_DtD!(promote_multiplier(α, Tx), x, y)
    return y
end

offset(::Type{CartesianIndex{N}}, d::Integer, s::Integer=1) where {N} =
    CartesianIndex{N}(ntuple(i -> (i == d ? s : 0), N))

@generated function _apply_D!(α::Number, x::AbstractArray{<:Floats,N},
                              β::Number, y::AbstractArray{<:Floats,Np1}
                              ) where {N,Np1}
    # We know that α ≠ 0.
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
        :(  inds = cartesian_indices(x)               ),
        :(  imax = last(inds)                         ),
        [:( $(S[d]) = $(offset(CartesianIndex{N}, d)) ) for d in 1:N]...,
        :inbounds,
        (
            :if, :(β == 0),
            (
                :simd_for, :(i in inds),
                (
                    common...,
                    [:( y[$d,i] = α*$(D[d]) ) for d in 1:N]...
                )
            ),
            :elseif, :(β == 1),
            (
                :simd_for, :(i in inds),
                (
                    common...,
                    [:( y[$d,i] += α*$(D[d]) ) for d in 1:N]...
                )
            ),
            :else,
            (
                :simd_for, :(i in inds),
                (
                    common...,
                    [:( y[$d,i] = α*$(D[d]) + β*y[$d,i] ) for d in 1:N]...
                )
            )
        )
    )
end

@generated function _apply_Dt!(α::Number, x::AbstractArray{<:Floats,Np1},
                               y::AbstractArray{<:Floats,N}) where {N,Np1}
    # We know that α ≠ 0 and that y has been pre-multiplied by β.
    @assert Np1 == N + 1
    D = generate_symbols("d", N)
    I = generate_symbols("i", N)
    S = generate_symbols("s", N)
    common = [:( $(I[d]) = min(i + $(S[d]), imax) ) for d in 1:N]
    return encode(
        :(  inds = cartesian_indices(y)               ),
        :(  imax = last(inds)                         ),
        [:( $(S[d]) = $(offset(CartesianIndex{N}, d)) ) for d in 1:N]...,
        :inbounds,
        (
            :if, :(α == 1),
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
                    [:( $(D[d]) = α*x[$d,i]               ) for d in 1:N]...,
                    :(  y[i] -= $(encode_sum_of_terms(D)) ),
                    [:( y[$(I[d])] += $(D[d])             ) for d in 1:N]...
                ),
            )
        )
    )
end

@generated function _apply_DtD!(α::Number, x::AbstractArray{<:Floats,N},
                                y::AbstractArray{<:Floats,N}) where {N}
    # We know that α ≠ 0 and that y has been pre-multiplied by β.
    D = generate_symbols("d", N)
    I = generate_symbols("i", N)
    S = generate_symbols("s", N)
    common = (
        [:( $(I[d]) = min(i + $(S[d]), imax) ) for d in 1:N]...,
        :(  xi = x[i]                        ))
    return encode(
        :(  inds = cartesian_indices(x)               ),
        :(  imax = last(inds)                         ),
        [:( $(S[d]) = $(offset(CartesianIndex{N}, d)) ) for d in 1:N]...,
        :inbounds,
        (
            :if, :(α == 1),
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
                    [:( $(D[d]) = α*(x[$(I[d])] - xi)     ) for d in 1:N]...,
                    :(  y[i] -= $(encode_sum_of_terms(D)) ),
                    [:( y[$(I[d])] += $(D[d])             ) for d in 1:N]...
                ),
            )
        )
    )
end

end # module
