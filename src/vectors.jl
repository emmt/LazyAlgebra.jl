#
# vectors.jl -
#
# Implement basic operations for *vectors*.  Here arrays of any rank are
# considered as *vectors*, the only requirements are that, when combining
# *vectors*, they have the same type and dimensions.  These methods are
# intended to be used for numerical optimization and thus, for now,
# elements must be real (not complex).
#
#-------------------------------------------------------------------------------
#
# This file is part of the MockAlgebra package released under the MIT "Expat"
# license.
#
# Copyright (c) 2017-2018 Éric Thiébaut.
#

"""
```julia
vnorm2([T,] v)
```

yields the Euclidean (L2) norm of `v`.  The floating point type of the result
can be imposed by optional argument `T`.  Also see [`vnorm1`](@ref) and
[`vnorminf`](@ref).

"""
function vnorm2(::Type{T},
                v::AbstractArray{<:Real,N}) where {T<:AbstractFloat,N}
    local s::T = zero(T)
    @inbounds @simd for i in eachindex(v)
        s += v[i]*v[i]
    end
    return sqrt(s)
end

vnorm2(v::AbstractArray{T,N}) where {T<:AbstractFloat,N} = vnorm2(T, v)
vnorm2(v::AbstractArray{T,N}) where {T<:Real,N} = vnorm2(float(T), x)
vnorm2(x) = sqrt(vdot(x, x))
vnorm2(::Type{T}, x) where {T<:AbstractFloat} = sqrt(vdot(T, x, x))

"""
```julia
vnorm1([T,] v)
```

yields the L1 norm of `v`, that is the sum of the absolute values of its
elements.  The floating point type of the result can be imposed by optional
argument `T`.  Also see [`vnorm2`](@ref) and [`vnorminf`](@ref).

"""
function vnorm1(::Type{T},
                v::AbstractArray{<:Real,N}) where {T<:AbstractFloat,N}
    local s::T = zero(T)
    @inbounds @simd for i in eachindex(v)
        s += abs(v[i])
    end
    return s
end

vnorm1(v::AbstractArray{T,N}) where {T<:AbstractFloat,N} = vnorm1(T, v)
vnorm1(v::AbstractArray{T,N}) where {T<:Real,N} = vnorm1(float(T), x)


"""
```julia
vnorminf([T,] v)
```

yields the infinite norm of `v`, that is the maximum absolute value of its
elements.  The floating point type of the result can be imposed by optional
argument `T`.  Also see [`vnorm1`](@ref) and [`vnorm2`](@ref).

"""
function vnorminf(::Type{F},
                  v::AbstractArray{T,N}) where {F<:AbstractFloat,T<:Real,N}
    local s::T = zero(T)
    @inbounds @simd for i in eachindex(v)
        s = max(s, abs(v[i]))
    end
    return F(s)
end
function vnorminf(::Type{F},
                  v::AbstractArray{T,N}) where {F<:AbstractFloat,T<:Unsigned,N}
    local s::T = zero(T)
    @inbounds @simd for i in eachindex(v)
        s = max(s, v[i])
    end
    return F(s)
end

vnorminf(v::AbstractArray{T,N}) where {T<:AbstractFloat,N} = vnorminf(T, v)
vnorminf(v::AbstractArray{T,N}) where {T<:Real,N} = vnorminf(float(T), x)

#------------------------------------------------------------------------------

"""
```julia
vcreate(x)
```

yields a new variable instance similar to `x`.  If `x` is an array, the
element type of the result is a floating-point type.

Also see [`similar`](@ref).

"""
vcreate(x::AbstractArray{T,N}) where {T<:AbstractFloat,N} = similar(x, T)
vcreate(x::AbstractArray{T,N}) where {T,N} = similar(x, float(T))

#------------------------------------------------------------------------------

"""

```julia
zerofill!(A) -> A
```

fills `A` with zeros and returns it.

"""
zerofill!(A::AbstractArray{T,N}) where {T, N} = fill!(A, zero(T))

"""
```julia
vscale!(dst, α, src) -> dst
```

overwrites `dst` with `α*src` and returns `dst`.  Computations are done at
the numerical precision of `src`.

"""
function vscale!(dst::AbstractArray{Td,N},
                 alpha::Real,
                 src::AbstractArray{Ts,N}) where {Td<:AbstractFloat,
                                                  Ts<:AbstractFloat,N}
    @assert indices(src) == indices(dst)
    if alpha == 0
        zerofill!(dst)
    elseif alpha == -1
        @inbounds @simd for i in eachindex(dst, src)
            dst[i] = -src[i]
        end
    elseif alpha == 1
        copy!(dst, src)
    else
        α = convert(Ts, alpha)
        @inbounds @simd for i in eachindex(dst, src)
            dst[i] = α*src[i]
        end
    end
    return dst
end

#--- INNER PRODUCT ------------------------------------------------------------

"""
```julia
vdot([T,] [w,] x, y)
```

yields the inner product of `x` and `y`; that is, the sum of `x[i]*y[i]` or, if
`w` is specified, the sum of `w[i]*x[i]*y[i]`, for all indices `i`.  Optional
argument `T` is the floating point type of the result.

"""
function vdot(::Type{T},
              x::AbstractArray{<:Real,N},
              y::AbstractArray{<:Real,N}) where {T<:Real, N}
    return julia_vdot(T, x, y)
end

function vdot(x::AbstractArray{Tx,N},
              y::AbstractArray{Ty,N}) where {Tx<:Real, Ty<:Real, N}
    return vdot(float(promote_type(Tx, Ty)), x, y)
end

function julia_vdot(x::AbstractArray{Tx,N},
                    y::AbstractArray{Ty,N}) where {Tx<:Real, Ty<:Real, N}
    if indices(x) != indices(y)
        throw(DimensionMismatch("`x` and `y` must have the same indices"))
    end
    T = float(promote_type(Tx, Ty))$
    local s::T = zero(T)
    @inbounds @simd for i in eachindex(x, y)
        s += x[i]*y[i]
    end
    return s
end

function julia_vdot(::Type{T},
                    x::AbstractArray{<:Real,N},
                    y::AbstractArray{<:Real,N}) where {T<:AbstractFloat, N}
    if indices(x) != indices(y)
        throw(DimensionMismatch("`x` and `y` must have the same indices"))
    end
    s = zero(T)
    @inbounds @simd for i in eachindex(x, y)
        s += convert(T, x[i])*convert(T, y[i])
    end
    return s
end

function vdot(w::AbstractArray{Tw,N},
              x::AbstractArray{Tx,N},
              y::AbstractArray{Ty,N}) where {Tw <: Real, Tx <: Real,
                                             Ty <: Real, N}
    if !(indices(w) == indices(x) == indices(y))
        throw(DimensionMismatch("`w`, `x` and `y` must have the same indices"))
    end
    T = float(promote_type(Tw, Tx, Ty))
    local s::T = zero(T)
    @inbounds @simd for i in eachindex(w, x, y)
        s += w[i]*x[i]*y[i]
    end
    return s
end

function vdot(::Type{T},
              w::AbstractArray{<:Real,N},
              x::AbstractArray{<:Real,N},
              y::AbstractArray{<:Real,N}) where {T<:AbstractFloat, N}
    if !(indices(w) == indices(x) == indices(y))
        throw(DimensionMismatch("`w`, `x` and `y` must have the same indices"))
    end
    s = zero(T)
    @inbounds @simd for i in eachindex(w, x, y)
        s += convert(T, w[i])*convert(T, x[i])*convert(T, y[i])
    end
    return s
end

#--- VECTOR UPDATE ------------------------------------------------------------

"""
```julia
vupdate!(y, α, x)
```
overwrites `y` with `α*x + y` and returns `y`.

"""
function vupdate!(y::AbstractArray{T,N},
                  alpha::Real,
                  x::AbstractArray{T,N}) where {T <: AbstractFloat, N}
    return julia_vupdate!(y, convert(T, alpha), x)
end

# This version is to use pure Julia code.
function julia_vupdate!(y::AbstractArray{Ty,N},
                        alpha::Ta,
                        x::AbstractArray{Tx,N}
                        ) where {Ty <: AbstractFloat, Ta <: Real,
                                Tx <: AbstractFloat, N}
    if indices(x) != indices(y)
        throw(DimensionMismatch("`x` and `y` must have the same indices"))
    end
    @inbounds @simd for i in eachindex(x, y)
        y[i] += alpha*x[i]
    end
    return y
end

#--- MATRIX-VECTOR PRODUCT ----------------------------------------------------

function apply(A::AbstractArray{<:Real},
               x::AbstractArray{<:Real})
    return apply(Direct, A, x)
end

function apply(::Type{Op},
               A::AbstractArray{<:Real},
               x::AbstractArray{<:Real}) where {Op<:Operations}
    return apply!(vcreate(Op, A, x), Op, A, x)
end

# By default, use pure Julia code for the generalized matrix-vector product.
function apply!(y::AbstractArray{<:Real},
                ::Type{Op},
                A::AbstractArray{<:Real},
                x::AbstractArray{<:Real}) where {Op<:Union{Adjoint,Direct}}
    return julia_apply!(y, Op, A, x)
end

function vcreate(::Type{Direct},
                   A::AbstractArray{Ta,Na},
                   x::AbstractArray{Tx,Nx}) where {Ta<:AbstractFloat, Na,
                                                   Tx<:AbstractFloat, Nx}
    inds = indices(A)
    Ny = Na - Nx
    if Nx ≥ Na || indices(x) != inds[Ny+1:end]
        throw(DimensionMismatch("the dimensions of `x` do not match the trailing dimensions of `A`"))
    end
    Ty = promote_type(Ta, Tx)
    return similar(Array{Ty}, inds[1:Ny])
end

function vcreate(::Type{Adjoint},
                   A::AbstractArray{Ta,Na},
                   x::AbstractArray{Tx,Nx}) where {Ta<:AbstractFloat, Na,
                                                   Tx<:AbstractFloat, Nx}
    inds = indices(A)
    Ny = Na - Nx
    if Nx ≥ Na || indices(x) != inds[1:Nx]
        throw(DimensionMismatch("the dimensions of `x` do not match the leading dimensions of `A`"))
    end
    Ty = promote_type(Ta, Tx)
    return similar(Array{Ty}, inds[Ny+1:end])
end


# Pure Julia code implementations.

function julia_apply!(y::AbstractArray{<:Real},
                      ::Type{Direct},
                      A::AbstractArray{<:Real},
                      x::AbstractArray{<:Real},
                      clr::Bool = true)
    if indices(A) != (indices(y)..., indices(x)...)
        throw(DimensionMismatch("`x` and/or `y` have indices incompatible with `A`"))
    end
    # Loop through the coefficients of A assuming column-major storage order.
    clr && zerofill!(y)
    I, J = CartesianRange(indices(y)), CartesianRange(indices(x))
    @inbounds for j in J
        @simd for i in I
            y[i] += A[i,j]*x[j]
        end
    end
    return y
end

function julia_apply!(y::AbstractArray{Ty},
                      ::Type{Adjoint},
                      A::AbstractArray{Ta},
                      x::AbstractArray{Tx}) where {Ta<:Real, Tx<:Real, Ty<:Real}
    return julia_apply!(promote_type(Ty, Ta, Tx), y, Adjoint, A, x)
end

function julia_apply!(::Type{T},
                      y::AbstractArray{<:Real},
                      ::Type{Adjoint},
                      A::AbstractArray{<:Real},
                      x::AbstractArray{<:Real}) where {T<:Real}
    if indices(A) != (indices(x)..., indices(y)...)
        throw(DimensionMismatch("`x` and/or `y` have indices incompatible with `A`"))
    end
    # Loop through the coefficients of A assuming column-major storage order.
    I, J = CartesianRange(indices(x)), CartesianRange(indices(y))
    @inbounds for j in J
        local s::T = zero(T)
        @simd for i in I
            s += A[i,j]*x[i]
        end
        y[j] = s
    end
    return y
end
