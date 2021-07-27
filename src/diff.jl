#
# diff.jl -
#
# Implement finite differences operators.
#
#-------------------------------------------------------------------------------
#
# This file is part of LazyAlgebra (https://github.com/emmt/LazyAlgebra.jl)
# released under the MIT "Expat" license.
#
# Copyright (c) 2017-2021 Éric Thiébaut.
#

module FiniteDifferences

export Diff

using MayOptimize
using LazyAlgebra
using LazyAlgebra.Foundations
import LazyAlgebra: apply!, vcreate, identical

using Base: @propagate_inbounds
import Base: show

const ArrayAxis = AbstractUnitRange{Int}
const ArrayAxes{N} = NTuple{N,ArrayAxis}

"""
    limits(r) -> (first(r), last(r))

yields the first and last value of the unit-range `r`.

"""
limits(r::AbstractUnitRange) = (first(r), last(r))

"""
    Diff([opt::MayOptimize.Vectorize,] n=1, dims=:)

yields a linear mapping that computes a finite difference approximation of the
`n`-order derivative along the dimension(s) specified by `dims`.  Arguments
`dims` is an integer, a tuple or a vector of integers specifying along which
dimension(s) to apply the operator or `:` to specify all dimensions.  If
multiple dimensions are specified, the result is as if the operator is applied
separately on the specified dimension(s).

Optional argument `opt` is the optimization level and may be specified as the
first or last argument.  By default, `opt` is assumed to be `Vectorize`,
however depending on the dimensions of the array, the dimensions of interest
and on the machine, setting `opt` to `InBounds` may be more efficient.

If `dims` is a scalar, the result, say `y`, of applying the finite difference
operator to an array, say `x`, has the same axes as `x`.  Otherwise and even
though `x` has a single dimension or `dims` is a 1-tuple, `y` has one more
dimension than `x`, the last dimension of `y` is used to store the finite
differences along each dimensions specified by `dims` and the leading
dimensions of `y` are the same as the dimensions of `x`.

More specifically, the operator created by `Diff` implements **forward finite
differences** with **flat boundary conditions**, that is to say extrapolated
entries are assumed equal to the nearest entry.

"""
struct Diff{L,D,O<:OptimLevel} <: LinearMapping end
# L = level of differentiation
# D = list of dimensions along which compute the differences
# O = optimization level

# Constructors.
function Diff(n::Integer = 1,
              dims::Union{Colon,Integer,Tuple{Vararg{Integer}},
                          AbstractVector{<:Integer}}=Colon(),
              opt::Type{<:OptimLevel} = Vectorize)
    return Diff{to_int(n), to_dims(dims), opt}()
end

function Diff(opt::Type{<:OptimLevel}, n::Integer = 1,
              dims::Union{Colon,Integer,Tuple{Vararg{Integer}},
                          AbstractVector{<:Integer}}=Colon())
    return Diff{to_int(n), to_dims(dims), opt}()
end

function Diff(n::Integer, opt::Type{<:OptimLevel})
    return Diff{to_int(n), Colon, opt}()
end

# Make a finite difference operator callable.
@callable Diff

# Two finite difference operators are identical if they have the same level of
# differentiation and list of dimensions along which compute the differences.
# Their optimization levels may be different.
identical(::Diff{L,D}, ::Diff{L,D}) where {L,D} = true

# Print operator in such a way that is similar to how the operator would be
# created in Julia.
show(io::IO, ::Diff{L,D,Opt}) where {L,D,Opt} =
    print(io, "Diff(", L, ',', (D === Colon ? ":" : D),',',
          (Opt === Debug ? "Debug" :
           Opt === InBounds ? "InBounds" :
           Opt === Vectorize ? "Vectorize" : Opt), ')')

"""
    differentiation_order(A)

yields the differentiation order of finite difference operator `A` (argument
can also be a type).

"""
differentiation_order(::Type{<:Diff{L,D,Opt}}) where {L,D,Opt} = L

"""
    dimensions_of_interest(A)

yields the list of dimensions of interest of finite difference operator `A`
(argument can also be a type).

"""
dimensions_of_interest(::Type{<:Diff{L,D,Opt}}) where {L,D,Opt} = D

"""
    optimization_level(A)

yields the optimization level for applying finite difference operator `A`
(argument can also be a type).

"""
optimization_level(::Type{<:Diff{L,D,Opt}}) where {L,D,Opt} = Opt

for f in (:differentiation_order,
          :dimensions_of_interest,
          :optimization_level)
    @eval begin
        $f(A::Diff) = $f(typeof(A))
        $f(A::Gram{<:Diff}) = $f(typeof(A))
        $f(::Type{<:Gram{T}}) where {T<:Diff} = $f(T)
    end
end

# Convert argument to `Int`.
to_int(x::Int) = x
to_int(x::Integer) = Int(x)

# Convert argument to the type parameter which specifies the list of dimensions
# of interest.
to_dims(::Colon) = Colon
to_dims(x::Int) = x
to_dims(x::Integer) = to_int(x)
to_dims(x::Tuple{Vararg{Int}}) = x
to_dims(x::Tuple{Vararg{Integer}}) = map(to_int, x)
to_dims(x::AbstractVector{<:Integer}) = to_dims((x...,))

# Drop list of dimensions from type to avoid unecessary specializations.
anydims(::Diff{L,D,P}) where {L,D,P} = Diff{L,Any,P}()
anydims(::Gram{Diff{L,D,P}}) where {L,D,P} = gram(Diff{L,Any,P}())

# Applying a separable operator is split in several stages:
#
# 1. Check arguments (so that avoiding bound checking should be safe) and deal
#    with the trivial cases α = 0 or no dimension of interest to apply the
#    operation (to simplify subsequent stages).
#
# 2. If α is non-zero, dispatch on dimension(s) along which to apply the
#    operation and on the specific values of the multipliers α and β.
#
# The second stage may be split in several sub-stages.

# Declare all possible signatures (not using unions) to avoid ambiguities.
for (P,A) in ((:Direct,  :Diff),
              (:Adjoint, :Diff),
              (:Direct,  :(Gram{<:Diff})))
    @eval function apply!(α::Number,
                          P::Type{$P},
                          A::$A,
                          x::AbstractArray,
                          scratch::Bool,
                          β::Number,
                          y::AbstractArray)
        inds, ndims = check_arguments(P, A, x, y)
        if α == 0 || ndims < 1
            # Get rid of this stupid case!
            vscale!(y, β)
        else
            # Call unsafe_apply! to dispatch on the dimensions of interest and on
            # the values of the multipliers.
            unsafe_apply!(α, P, A, x, β, y, inds)
        end
        return y
    end
end

# FIXME: This should not be necessary.
function apply!(α::Number,
                ::Type{<:Adjoint},
                A::Gram{<:Diff},
                x::AbstractArray,
                scratch::Bool,
                β::Number,
                y::AbstractArray)
    apply!(α, Direct, A, x, scratch, β, y)
end

function vcreate(::Type{Direct},
                 A::Diff{L,D,P},
                 x::AbstractArray{T,N},
                 scratch::Bool) where {L,D,P,T,N}
    if D === Colon
        return Array{T}(undef, size(x)..., N)
    elseif isa(D, Tuple{Vararg{Int}})
        return Array{T}(undef, size(x)..., length(D))
    elseif isa(D, Int)
        # if L === 1 && scratch && isa(x, Array)
        #    # First order finite difference along a single dimension.
        #    # Operation could be done in-place but we must preserve
        #    # type-stability.
        #    return x
        #else
        #    return Array{T}(undef, size(x))
        #end
        return Array{T}(undef, size(x))
    else
        error("invalid list of dimensions")
    end
end

function vcreate(::Type{Adjoint},
                 A::Diff{L,D,P},
                 x::AbstractArray{T,N},
                 scratch::Bool) where {L,D,P,T,N}
    # Checking the validity of the argument dimensions is done by applying the
    # opererator.  In-place operation never possible, so ignore the scratch
    # flag.
    if D === Colon || isa(D, Tuple{Vararg{Int}})
        return Array{T}(undef, size(x)[1:N-1])
    elseif isa(D, Int)
        return Array{T}(undef, size(x))
    else
        error("invalid list of dimensions")
    end
end

#------------------------------------------------------------------------------
# CHECKING OF ARGUMENTS

"""
    check_arguments(P, A, x, y) -> inds, ndims

checks that arguments `x` and `y` are valid for applying `P(A)`, with `A` a
separable operator, to `x` and store the result in `y`.  The result is a
2-tuple, `inds` is the axes that the arguments have in common and `ndims` is
the number of dimensions of interest.

If this function returns normally, the caller may safely assume that index
bound checking is not needed; hence, this function must throw an exception if
the dimensions/indices of `x` and `y` are not compatible or if the dimensions
of interest in `A` are out of range.  This function may also throw an exception
if the element types of `x` and `y` are not compatible.

This method must be specialized for the different types of separable operators.

"""
function check_arguments(P::Type{<:Union{Direct,Adjoint}},
                         A::Union{Diff{L,D},Gram{<:Diff{L,D}}},
                         x::AbstractArray,
                         y::AbstractArray) where {L,D}
    inds = check_axes(P, A, axes(x), axes(y))
    ndims = check_dimensions_of_interest(D, length(inds))
    return inds, ndims
end

function check_axes(P::Type{<:Union{Direct,Adjoint}},
                    A::Diff{L,D},
                    xinds::ArrayAxes,
                    yinds::ArrayAxes) where {L,D}
    if D === Colon || isa(D, Dims)
        if P === Direct
            length(yinds) == length(xinds) + 1 ||
                throw_dimension_mismatch("output array must have one more dimension than input array")
            N = (D === Colon ? length(xinds) : length(D))
            yinds[end] == 1:N ||
                throw_dimension_mismatch("last axis of output array must be 1:", N)
            yinds[1:end-1] == xinds ||
                throw_dimension_mismatch("leading axes must be identical")
            return xinds
        else
            length(yinds) == length(xinds) - 1 ||
                throw_dimension_mismatch("output array must have one less dimension than input array")
            N = (D === Colon ? length(yinds) : length(D))
            xinds[end] == 1:N ||
                throw_dimension_mismatch("last axis of input array must be 1:", N)
            xinds[1:end-1] == yinds ||
                throw_dimension_mismatch("leading axes must be identical")
            return yinds
        end
    elseif isa(D, Int)
        xinds == yinds || throw_dimension_mismatch("array axes must be identical")
        return xinds
    else
        throw(ArgumentError("invalid dimensions of interest"))
    end
end

function check_axes(P::Type{<:Operations},
                    A::Gram{<:Diff},
                    xinds::ArrayAxes,
                    yinds::ArrayAxes)
    xinds == yinds || throw_dimension_mismatch("array axes must be identical")
    return xinds
end

check_dimensions_of_interest(::Type{Colon}, ndims::Int) = ndims

check_dimensions_of_interest(dim::Int, ndims::Int) = begin
    1 ≤ dim ≤ ndims ||
        throw_dimension_mismatch("out of range dimension ", dim,
                                 "for ", ndims,"-dimensional arrays")
    return 1
end

check_dimensions_of_interest(dims::Dims{N}, ndims::Int) where {N} = begin
    for dim in dims
        1 ≤ dim ≤ ndims ||
            throw_dimension_mismatch("out of range dimension ", dim,
                                     "for ", ndims,"-dimensional arrays")
    end
    return N
end

throw_dimension_mismatch(str::String) = throw(DimensionMismatch(str))

@noinline throw_dimension_mismatch(args...) =
    throw_dimension_mismatch(string(args...))

#------------------------------------------------------------------------------

# Apply the operation along all dimensions of interest but one dimension at a
# time and knowing that α is not zero.
@generated function unsafe_apply!(α::Number,
                                  ::Type{P},
                                  A::Diff{L,D},
                                  x::AbstractArray,
                                  β::Number,
                                  y::AbstractArray,
                                  inds::ArrayAxes{N}) where {L,D,N,
                                                             P<:Union{Direct,
                                                                      Adjoint}}
    # Allocate empty vector of statements.
    exprs = Expr[]

    # Discard type parameter specifying the dimensions of interest to avoid
    # specialization on this parameter.
    push!(exprs, :(B = anydims(A)))

    # Dispatch on dimensions of interest.
    if isa(D, Int)
        # Arrays x and y have the same dimensions.
        push!(exprs, :(unsafe_apply!(α, P, B, x, β, y,
                                     inds[1:$(D-1)],
                                     inds[$D],
                                     inds[$(D+1):$N],
                                     CartesianIndex())))
    elseif D === Colon || isa(D, Dims)
        # One of x or y (depending on whether the direct or the adjoint
        # operator is applied) has an extra leading dimension used to store the
        # result computed along a given dimension.
        keep_beta = true # initially scale y by β
        dims = (D === Colon ? (1:N) : D)
        for l in 1:length(dims)
            d = dims[l]
            push!(exprs, :(unsafe_apply!(α, P, B, x,
                                         $(keep_beta ? :β : 1), y,
                                         inds[1:$(d-1)],
                                         inds[$d],
                                         inds[$(d+1):$N],
                                         CartesianIndex($l))))
            keep_beta = (P === Direct && A <: Diff)
        end
    else
        # This should never happen.
        return quote
            error("invalid list of dimensions of interest")
        end
    end

    return quote
        $(Expr(:meta, :inline))
        $(exprs...)
        nothing
    end
end

@generated function unsafe_apply!(α::Number,
                                  ::Type{P},
                                  A::Gram{<:Diff{L,D}},
                                  x::AbstractArray,
                                  β::Number,
                                  y::AbstractArray,
                                  inds::ArrayAxes{N}) where {L,D,N,
                                                             P<:Direct}
    # Allocate empty vector of statements.
    exprs = Expr[]

    # Discard type parameter specifying the dimensions of interest to avoid
    # specialization on this parameter.
    push!(exprs, :(B = anydims(A)))

    # Dispatch on dimensions of interest.  Arrays x and y have the same
    # dimensions and there is no last index `l` to specify.
    if isa(D, Int)
        push!(exprs, :(unsafe_apply!(α, P, B, x, β, y,
                                     inds[1:$(D-1)],
                                     inds[$D],
                                     inds[$(D+1):$N])))
    elseif D === Colon || isa(D, Dims)
        # β is set to 1 after first dimension of interest.
        dims = (D === Colon ? (1:N) : D)
        for l in 1:length(dims)
            d = dims[l]
            push!(exprs, :(unsafe_apply!(α, P, B, x,
                                         $(l == 1 ? :β : 1), y,
                                         inds[1:$(d-1)],
                                         inds[$d],
                                         inds[$(d+1):$N])))
        end
    else
        # This should never happen.
        return quote
            error("invalid list of dimensions of interest")
        end
    end

    return quote
        $(Expr(:meta, :inline))
        $(exprs...)
        nothing
    end
end

# Dispatch on multipliers values (α is not zero).
function unsafe_apply!(alpha::Number,
                       P::Type{<:Operations},
                       A::Union{Diff{L,Any,Opt},
                                Gram{Diff{L,Any,Opt}}},
                       x::AbstractArray,
                       beta::Number,
                       y::AbstractArray,
                       I::ArrayAxes,
                       J::ArrayAxis,
                       K::ArrayAxes,
                       l::CartesianIndex) where {L,Opt}
    if alpha == 1
        if beta == 0
            unsafe_apply!(axpby_yields_x,     1, P, A, x, 0, y, I, J, K, l)
        elseif beta == 1
            unsafe_apply!(axpby_yields_xpy,   1, P, A, x, 1, y, I, J, K, l)
        else
            β = promote_multiplier(beta, y)
            unsafe_apply!(axpby_yields_xpby,  1, P, A, x, β, y, I, J, K, l)
        end
    else
        α = promote_multiplier(alpha, y)
        if beta == 0
            unsafe_apply!(axpby_yields_ax,    α, P, A, x, 0, y, I, J, K, l)
        elseif beta == 1
            unsafe_apply!(axpby_yields_axpy,  α, P, A, x, 1, y, I, J, K, l)
        else
            β = promote_multiplier(beta, y)
            unsafe_apply!(axpby_yields_axpby, α, P, A, x, β, y, I, J, K, l)
        end
    end
    nothing
end

# Dispatch on multipliers values (α is not zero) for Gram compositions of a
# finite difference operator.
function unsafe_apply!(alpha::Number,
                       P::Type{<:Operations},
                       A::Gram{<:Diff},
                       x::AbstractArray,
                       beta::Number,
                       y::AbstractArray,
                       I::ArrayAxes,
                       J::ArrayAxis,
                       K::ArrayAxes)
    if alpha == 1
        if beta == 0
            unsafe_apply!(axpby_yields_x,     1, P, A, x, 0, y, I, J, K)
        elseif beta == 1
            unsafe_apply!(axpby_yields_xpy,   1, P, A, x, 1, y, I, J, K)
        else
            β = promote_multiplier(beta, y)
            unsafe_apply!(axpby_yields_xpby,  1, P, A, x, β, y, I, J, K)
        end
    else
        α = promote_multiplier(alpha, y)
        if beta == 0
            unsafe_apply!(axpby_yields_ax,    α, P, A, x, 0, y, I, J, K)
        elseif beta == 1
            unsafe_apply!(axpby_yields_axpy,  α, P, A, x, 1, y, I, J, K)
        else
            β = promote_multiplier(beta, y)
            unsafe_apply!(axpby_yields_axpby, α, P, A, x, β, y, I, J, K)
        end
    end
    nothing
end

#------------------------------------------------------------------------------
#
# The operator D implementing 1st order forward finite difference with flat
# boundary conditions and its adjoint D' are given by:
#
#     D = [ -1   1   0   0
#            0  -1   1   0
#            0   0  -1   1
#            0   0   0   0];
#
#     D' = [ -1   0   0   0
#             1  -1   0   0
#             0   1  -1   0
#             0   0   1   0];
#
# The row (for D) and column (for D') of zeros are to preserve the size.  This
# is needed for multi-dimensional arrays when derivatives along each dimension
# are stored into a single array.
#
# Apply 1st order finite differences along 1st dimension:
#
function unsafe_apply!(f::Function,
                       α::Number,
                       ::Type{Direct},
                       A::Diff{1,Any,Opt},
                       x::AbstractArray,
                       β::Number,
                       y::AbstractArray,
                       I::Tuple{},
                       J::ArrayAxis,
                       K::ArrayAxes,
                       l::CartesianIndex) where {Opt}
    T = real(eltype(x))
    jmin, jmax = limits(J)
    if jmin ≤ jmax
        @maybe_inbounds Opt for k in CartesianIndices(K)
            @maybe_vectorized Opt for j in jmin:jmax-1
                z = x[j+1,k] - x[j,k]
                y[j,k,l] = f(α, z, β, y[j,k,l])
            end
            let j = jmax, z = zero(T)
                y[j,k,l] = f(α, z, β, y[j,k,l])
            end
        end
    end
    nothing
end
#
# Apply 1st order finite differences along 2nd and subsequent dimensions:
#
function unsafe_apply!(f::Function,
                       α::Number,
                       ::Type{Direct},
                       A::Diff{1,Any,Opt},
                       x::AbstractArray,
                       β::Number,
                       y::AbstractArray,
                       I::ArrayAxes,
                       J::ArrayAxis,
                       K::ArrayAxes,
                       l::CartesianIndex) where {Opt}
    T = real(eltype(x))
    jmin, jmax = limits(J)
    if jmin ≤ jmax
        @maybe_inbounds Opt for k in CartesianIndices(K)
            for j in jmin:jmax-1
                @maybe_vectorized Opt for i in CartesianIndices(I)
                    z = x[i,j+1,k] - x[i,j,k]
                    y[i,j,k,l] = f(α, z, β, y[i,j,k,l])
                end
            end
            let j = jmax, z = zero(T)
                @maybe_vectorized Opt for i in CartesianIndices(I)
                    y[i,j,k,l] = f(α, z, β, y[i,j,k,l])
                end
            end
        end
    end
    nothing
end
#
# Apply adjoint of 1st order finite differences along 1st dimension:
#
function unsafe_apply!(f::Function,
                       α::Number,
                       ::Type{Adjoint},
                       A::Diff{1,Any,Opt},
                       x::AbstractArray,
                       β::Number,
                       y::AbstractArray,
                       I::Tuple{},
                       J::ArrayAxis,
                       K::ArrayAxes,
                       l::CartesianIndex) where {Opt}
    T = real(eltype(x))
    jmin, jmax = limits(J)
    if jmin < jmax
        @maybe_inbounds Opt for k in CartesianIndices(K)
            let j = jmin
                z = -x[j,k,l]
                y[j,k] = f(α, z, β, y[j,k])
            end
            @maybe_vectorized Opt for j in jmin+1:jmax-1
                z = x[j-1,k,l] - x[j,k,l]
                y[j,k] = f(α, z, β, y[j,k])
            end
            let j = jmax
                z = x[j-1,k,l]
                y[j,k] = f(α, z, β, y[j,k])
            end
        end
    elseif jmin == jmax && β != 1
        let j = jmin, z = zero(T)
            @maybe_vectorized Opt for k in CartesianIndices(K)
                y[j,k] = f(α, z, β, y[j,k])
            end
        end
    end
    nothing
end
#
# Apply adjoint of 1st order finite differences along 2nd and subsequent
# dimensions:
#
function unsafe_apply!(f::Function,
                       α::Number,
                       ::Type{Adjoint},
                       A::Diff{1,Any,Opt},
                       x::AbstractArray,
                       β::Number,
                       y::AbstractArray,
                       I::ArrayAxes,
                       J::ArrayAxis,
                       K::ArrayAxes,
                       l::CartesianIndex) where {Opt}
    T = real(eltype(x))
    jmin, jmax = limits(J)
    if jmin < jmax
        @maybe_inbounds Opt for k in CartesianIndices(K)
            let j = jmin
                @maybe_vectorized Opt for i in CartesianIndices(I)
                    z = -x[i,j,k,l]
                    y[i,j,k] = f(α, z, β, y[i,j,k])
                end
            end
            for j in jmin+1:jmax-1
                @maybe_vectorized Opt for i in CartesianIndices(I)
                    z = x[i,j-1,k,l] - x[i,j,k,l]
                    y[i,j,k] = f(α, z, β, y[i,j,k])
                end
            end
            let j = jmax
                @maybe_vectorized Opt for i in CartesianIndices(I)
                    z = x[i,j-1,k,l]
                    y[i,j,k] = f(α, z, β, y[i,j,k])
                end
            end
        end
    elseif jmin == jmax && β != 1
        let j = jmin, z = zero(T)
            @maybe_inbounds Opt for k in CartesianIndices(K)
                @maybe_vectorized Opt for i in CartesianIndices(I)
                    y[i,j,k] = f(α, z, β, y[i,j,k])
                end
            end
        end
    end
    nothing
end
#
# The Gram composition D'*D of the 1st order forward finite differences D with
# flat boundary conditions writes:
#
#     D'*D = [  1  -1   0   0   0
#              -1   2  -1   0   0
#               0  -1   2  -1   0
#               0   0  -1   2  -1
#               0   0   0  -1   1 ]
#
# Apply D'*D along 1st dimension:
#
function unsafe_apply!(f::Function,
                       α::Number,
                       ::Type{<:Union{Direct,Adjoint}},
                       A::Gram{Diff{1,Any,Opt}},
                       x::AbstractArray,
                       β::Number,
                       y::AbstractArray,
                       I::Tuple{},
                       J::ArrayAxis,
                       K::ArrayAxes) where {Opt}
    T = real(eltype(x))
    jmin, jmax = limits(J)
    if jmin < jmax
        @maybe_inbounds Opt for k in CartesianIndices(K)
            let j = jmin
                z = x[j,k] - x[j+1,k]
                y[j,k] = f(α, z, β, y[j,k])
            end
            @maybe_vectorized Opt for j in jmin+1:jmax-1
                z = T(2)*x[j,k] - (x[j-1,k] + x[j+1,k])
                y[j,k] = f(α, z, β, y[j,k])
            end
            let j = jmax
                z = x[j,k] - x[j-1,k]
                y[j,k] = f(α, z, β, y[j,k])
            end
        end
    elseif jmin == jmax && β != 1
        let j = jmin, z = zero(T)
            @maybe_vectorized Opt for k in CartesianIndices(K)
                y[j,k] = f(α, z, β, y[j,k])
            end
        end
    end
    nothing
end
#
# Apply  D'*D along 2nd and subsequent dimensions:
#
function unsafe_apply!(f::Function,
                       α::Number,
                       ::Type{<:Union{Direct,Adjoint}},
                       A::Gram{Diff{1,Any,Opt}},
                       x::AbstractArray,
                       β::Number,
                       y::AbstractArray,
                       I::ArrayAxes,
                       J::ArrayAxis,
                       K::ArrayAxes) where {Opt}
    T = real(eltype(x))
    jmin, jmax = limits(J)
    if jmin < jmax
        @maybe_inbounds Opt for k in CartesianIndices(K)
            let j = jmin
                @maybe_vectorized Opt for i in CartesianIndices(I)
                    z = x[i,j,k] - x[i,j+1,k]
                    y[i,j,k] = f(α, z, β, y[i,j,k])
                end
            end
            for j in jmin+1:jmax-1
                @maybe_vectorized Opt for i in CartesianIndices(I)
                    z = T(2)*x[i,j,k] - (x[i,j-1,k] + x[i,j+1,k])
                    y[i,j,k] = f(α, z, β, y[i,j,k])
                end
            end
            let j = jmax
                @maybe_vectorized Opt for i in CartesianIndices(I)
                    z = x[i,j,k] - x[i,j-1,k]
                    y[i,j,k] = f(α, z, β, y[i,j,k])
                end
            end
        end
    elseif jmin == jmax && β != 1
        let j = jmin, z = zero(T)
            @maybe_inbounds Opt for k in CartesianIndices(K)
                @maybe_vectorized Opt for i in CartesianIndices(I)
                    y[i,j,k] = f(α, z, β, y[i,j,k])
                end
            end
        end
    end
    nothing
end

#------------------------------------------------------------------------------
#
# 2nd order finite differences with flat boundary conditions are computed by:
#
#     D = [-1   1   0   0   0   0
#           1  -2   1   0   0   0
#           0   1  -2   1   0   0
#           0   0   1  -2   1   0
#           0   0   0   1  -2   1
#           0   0   0   0   1  -1]
#
# Remarks:
#
#  - Applying this operator on a single dimension is self-adjoint.
#
#  - For a single dimension, this operator is the opposite of the Gram
#    composition of 1st order finite differences (backward or forward).
#
# Apply 2nd order finite differences along 1st dimension:
#
function unsafe_apply!(f::Function,
                       α::Number,
                       ::Type{Direct},
                       A::Diff{2,Any,Opt},
                       x::AbstractArray,
                       β::Number,
                       y::AbstractArray,
                       I::Tuple{},
                       J::ArrayAxis,
                       K::ArrayAxes,
                       l::CartesianIndex) where {Opt}
    T = real(eltype(x))
    jmin, jmax = limits(J)
    if jmin < jmax
        @maybe_inbounds Opt for k in CartesianIndices(K)
            let j = jmin
                z = x[j+1,k] - x[j,k]
                y[j,k,l] = f(α, z, β, y[j,k,l])
            end
            @maybe_vectorized Opt for j in jmin+1:jmax-1
                z = x[j-1,k] + x[j+1,k] - T(2)*x[j,k]
                y[j,k,l] = f(α, z, β, y[j,k,l])
            end
            let j = jmax
                z = x[j-1,k] - x[j,k]
                y[j,k,l] = f(α, z, β, y[j,k,l])
            end
        end
    elseif jmin == jmax && β != 1
        let j = jmin, z = zero(T)
            @maybe_vectorized Opt for k in CartesianIndices(K)
                y[j,k,l] = f(α, z, β, y[j,k,l])
            end
        end
    end
    nothing
end
#
# Apply 2nd order finite differences along 2nd and subsequent dimensions:
#
function unsafe_apply!(f::Function,
                       α::Number,
                       ::Type{Direct},
                       A::Diff{2,Any,Opt},
                       x::AbstractArray,
                       β::Number,
                       y::AbstractArray,
                       I::ArrayAxes,
                       J::ArrayAxis,
                       K::ArrayAxes,
                       l::CartesianIndex) where {Opt}
    T = real(eltype(x))
    jmin, jmax = limits(J)
    if jmin < jmax
        @maybe_inbounds Opt for k in CartesianIndices(K)
            let j = jmin
                @maybe_vectorized Opt for i in CartesianIndices(I)
                    z = x[i,j+1,k] - x[i,j,k]
                    y[i,j,k,l] = f(α, z, β, y[i,j,k,l])
                end
            end
            for j in jmin+1:jmax-1
                @maybe_vectorized Opt for i in CartesianIndices(I)
                    # Other possibility:
                    # z = (x[i,j-1,k] - x[i,j,k]) + (x[i,j+1,k] - x[i,j,k])
                    z = x[i,j-1,k] + x[i,j+1,k] - T(2)*x[i,j,k]
                    y[i,j,k,l] = f(α, z, β, y[i,j,k,l])
                end
            end
            let j = jmax
                @maybe_vectorized Opt for i in CartesianIndices(I)
                    z = x[i,j-1,k] - x[i,j,k]
                    y[i,j,k,l] = f(α, z, β, y[i,j,k,l])
                end
            end
        end
    elseif jmin == jmax && β != 1
        let j = jmin, z = zero(T)
            @maybe_inbounds Opt for k in CartesianIndices(K)
                @maybe_vectorized Opt for i in CartesianIndices(I)
                    y[i,j,k,l] = f(α, z, β, y[i,j,k,l])
                end
            end
        end
    end
    nothing
end
#
# Apply adjoint of 2nd order finite differences along 1st dimension:
#
function unsafe_apply!(f::Function,
                       α::Number,
                       ::Type{Adjoint},
                       A::Diff{2,Any,Opt},
                       x::AbstractArray,
                       β::Number,
                       y::AbstractArray,
                       I::Tuple{},
                       J::ArrayAxis,
                       K::ArrayAxes,
                       l::CartesianIndex) where {Opt}
    T = real(eltype(x))
    jmin, jmax = limits(J)
    if jmin < jmax
        @maybe_inbounds Opt for k in CartesianIndices(K)
            let j = jmin
                z = x[j+1,k,l] - x[j,k,l]
                y[j,k] = f(α, z, β, y[j,k])
            end
            @maybe_vectorized Opt for j in jmin+1:jmax-1
                # Other possibility:
                # z = (x[j-1,k,l] - x[j,k,l]) + (x[j+1,k,l] - x[j,k,l])
                z = x[j-1,k,l] + x[j+1,k,l] - T(2)*x[j,k,l]
                y[j,k] = f(α, z, β, y[j,k])
            end
            let j = jmax
                z = x[j-1,k,l] - x[j,k,l]
                y[j,k] = f(α, z, β, y[j,k])
            end
        end
    elseif jmin == jmax && β != 1
        let j = jmin, z = zero(T)
            @maybe_vectorized Opt for k in CartesianIndices(K)
                y[j,k] = f(α, z, β, y[j,k])
            end
        end
    end
    nothing
end
#
# Apply 2nd order finite differences along 2nd and subsequent dimensions:
#
function unsafe_apply!(f::Function,
                       α::Number,
                       ::Type{Adjoint},
                       A::Diff{2,Any,Opt},
                       x::AbstractArray,
                       β::Number,
                       y::AbstractArray,
                       I::ArrayAxes,
                       J::ArrayAxis,
                       K::ArrayAxes,
                       l::CartesianIndex) where {Opt}
    T = real(eltype(x))
    jmin, jmax = limits(J)
    if jmin < jmax
        @maybe_inbounds Opt for k in CartesianIndices(K)
            let j = jmin
                @maybe_vectorized Opt for i in CartesianIndices(I)
                    z = x[i,j+1,k,l] - x[i,j,k,l]
                    y[i,j,k] = f(α, z, β, y[i,j,k])
                end
            end
            for j in jmin+1:jmax-1
                @maybe_vectorized Opt for i in CartesianIndices(I)
                    z = x[i,j-1,k,l] + x[i,j+1,k,l] - T(2)*x[i,j,k,l]
                    y[i,j,k] = f(α, z, β, y[i,j,k])
                end
            end
            let j = jmax
                @maybe_vectorized Opt for i in CartesianIndices(I)
                    z = x[i,j-1,k,l] - x[i,j,k,l]
                    y[i,j,k] = f(α, z, β, y[i,j,k])
                end
            end
        end
    elseif jmin == jmax && β != 1
        let j = jmin, z = zero(T)
            @maybe_inbounds Opt for k in CartesianIndices(K)
                @maybe_vectorized Opt for i in CartesianIndices(I)
                    y[i,j,k] = f(α, z, β, y[i,j,k])
                end
            end
        end
    end
    nothing
end
#
# The Gram composition of 2nd order finite differences writes:
#
#    D'*D = [ 2  -3   1   0   0   0             (1)
#            -3   6  -4   1   0   0             (2)
#             1  -4   6  -4   1   0             (3)
#             0   1  -4   6  -4   1             (3)
#             0   0   1  -4   6  -3             (4)
#             0   0   0   1  -3   2]            (5)
#
# The above is for len ≥ 4, with len is the length of the dimension of
# interest, omitting the Eq. (5) for len = 4 and repeating Eq. (5) as necessary
# for the central rows for n ≥ 5.  For len = 3:
#
#    D'*D = [ 2  -3   1                         (1)
#            -3   6  -3                         (6)
#             1  -3   2]                        (5)
#
# For len = 2:
#
#    D'*D = [ 2  -2                             (7)
#            -2   2]                            (8)
#
# For len = 1, D = 0 and D'*D = 0 (the null 1×1 operator).
#
# Methods to apply the rows of D'D ():
#
# - Eq. (1), first row when len ≥ 3:
#
@inline @propagate_inbounds D2tD2_1(x::AbstractArray, j::Int, k) = begin
    T = real(eltype(x))
    T(2)*x[j,k] - T(3)*x[j+1,k] + x[j+2,k]
end
@inline @propagate_inbounds D2tD2_1(x::AbstractArray, i, j::Int, k) = begin
    T = real(eltype(x))
    T(2)*x[i,j,k] - T(3)*x[i,j+1,k] + x[i,j+2,k]
end
#
# - Eq. (2), second row when len ≥ 4:
#
@inline @propagate_inbounds D2tD2_2(x::AbstractArray, j::Int, k) = begin
    T = real(eltype(x))
    T(6)*x[j,k] - T(3)*x[j-1,k] - T(4)*x[j+1,k] + x[j+2,k]
end
@inline @propagate_inbounds D2tD2_2(x::AbstractArray, i, j::Int, k) = begin
    T = real(eltype(x))
    T(6)*x[i,j,k] - T(3)*x[i,j-1,k] - T(4)*x[i,j+1,k] + x[i,j+2,k]
end
#
# - Eq. (3), central rows when len ≥ 5:
#
@inline @propagate_inbounds D2tD2_3(x::AbstractArray, j::Int, k) = begin
    T = real(eltype(x))
    (x[j-2,k] + x[j+2,k]) + T(6)*x[j,k] - T(4)*(x[j-1,k] + x[j+1,k])
end
@inline @propagate_inbounds D2tD2_3(x::AbstractArray, i, j::Int, k) = begin
    T = real(eltype(x))
    (x[i,j-2,k] + x[i,j+2,k]) + T(6)*x[i,j,k] - T(4)*(x[i,j-1,k] + x[i,j+1,k])
end
#
# - Eq. (4), before last row when len ≥ 4:
#
@inline @propagate_inbounds D2tD2_4(x::AbstractArray, j::Int, k) = begin
    T = real(eltype(x))
    T(6)*x[j,k] - T(3)*x[j+1,k] - T(4)*x[j-1,k] + x[j-2,k]
end
@inline @propagate_inbounds D2tD2_4(x::AbstractArray, i, j::Int, k) = begin
    T = real(eltype(x))
    T(6)*x[i,j,k] - T(3)*x[i,j+1,k] - T(4)*x[i,j-1,k] + x[i,j-2,k]
end
#
# - Eq. (5), last row when len ≥ 3:
#
@inline @propagate_inbounds D2tD2_5(x::AbstractArray, j::Int, k) = begin
    T = real(eltype(x))
    T(2)*x[j,k] - T(3)*x[j-1,k] + x[j-2,k]
end
@inline @propagate_inbounds D2tD2_5(x::AbstractArray, i, j::Int, k) = begin
    T = real(eltype(x))
    T(2)*x[i,j,k] - T(3)*x[i,j-1,k] + x[i,j-2,k]
end
#
# - Eq. (6), central row when len = 3:
#
@inline @propagate_inbounds D2tD2_6(x::AbstractArray, j::Int, k) = begin
    T = real(eltype(x))
    T(6)*x[j,k] - T(3)*(x[j-1,k] + x[j+1,k])
end
@inline @propagate_inbounds D2tD2_6(x::AbstractArray, i, j::Int, k) = begin
    T = real(eltype(x))
    T(6)*x[i,j,k] - T(3)*(x[i,j-1,k] + x[i,j+1,k])
end
#
# - Eq. (7), first row when len = 2:
#
@inline @propagate_inbounds D2tD2_7(x::AbstractArray, j::Int, k) = begin
    z = x[j,k] - x[j+1,k]
    return z + z
end
@inline @propagate_inbounds D2tD2_7(x::AbstractArray, i, j::Int, k) = begin
    z = x[i,j,k] - x[i,j+1,k]
    return z + z
end
#
# - Eq. (8), last row when len = 2:
#
@inline @propagate_inbounds D2tD2_8(x::AbstractArray, j::Int, k) = begin
    z = x[j,k] - x[j-1,k]
    return z + z
end
@inline @propagate_inbounds D2tD2_8(x::AbstractArray, i, j::Int, k) = begin
    z = x[i,j,k] - x[i,j-1,k]
    return z + z
end
#
# Apply Gram composition of 2nd order finite differences along 1st dimension:
#
function unsafe_apply!(f::Function,
                       α::Number,
                       ::Type{<:Union{Direct,Adjoint}},
                       A::Gram{Diff{2,Any,Opt}},
                       x::AbstractArray,
                       β::Number,
                       y::AbstractArray,
                       I::Tuple{},
                       J::ArrayAxis,
                       K::ArrayAxes) where {Opt}
    T = real(eltype(x))
    jmin, jmax = limits(J)
    len = length(J)
    if len ≥ 5
        @maybe_inbounds Opt for k in CartesianIndices(K)
            let j = jmin
                y[j,k] = f(α, D2tD2_1(x,j,k), β, y[j,k])
            end
            let j = jmin+1
                y[j,k] = f(α, D2tD2_2(x,j,k), β, y[j,k])
            end
            @maybe_vectorized Opt for j in jmin+2:jmax-2
                y[j,k] = f(α, D2tD2_3(x,j,k), β, y[j,k])
            end
            let j = jmax-1
                y[j,k] = f(α, D2tD2_4(x,j,k), β, y[j,k])
            end
            let j = jmax
                y[j,k] = f(α, D2tD2_5(x,j,k), β, y[j,k])
            end
        end
    elseif len == 4
        @maybe_vectorized Opt for k in CartesianIndices(K)
            let j = jmin
                y[j,k] = f(α, D2tD2_1(x,j,k), β, y[j,k])
            end
            let j = jmin+1
                y[j,k] = f(α, D2tD2_2(x,j,k), β, y[j,k])
            end
            let j = jmax-1
                y[j,k] = f(α, D2tD2_4(x,j,k), β, y[j,k])
            end
            let j = jmax
                y[j,k] = f(α, D2tD2_5(x,j,k), β, y[j,k])
            end
        end
    elseif len == 3
        @maybe_vectorized Opt for k in CartesianIndices(K)
            let j = jmin
                y[j,k] = f(α, D2tD2_1(x,j,k), β, y[j,k])
            end
            let j = jmin+1
                y[j,k] = f(α, D2tD2_6(x,j,k), β, y[j,k])
            end
            let j = jmax
                y[j,k] = f(α, D2tD2_5(x,j,k), β, y[j,k])
            end
        end
    elseif len == 2
        @maybe_vectorized Opt for k in CartesianIndices(K)
            let j = jmin
                y[j,k] = f(α, D2tD2_7(x,j,k), β, y[j,k])
            end
            let j = jmax
                y[j,k] = f(α, D2tD2_8(x,j,k), β, y[j,k])
            end
        end
    elseif len == 1 && β != 1
        let j = jmin, z = zero(T)
            @maybe_vectorized Opt for k in CartesianIndices(K)
                y[j,k] = f(α, z, β, y[j,k])
            end
        end
    end
    nothing
end
#
# Apply Gram composition of 2nd order finite differences along 2nd and
# subsequent dimensions:
#
function unsafe_apply!(f::Function,
                       α::Number,
                       ::Type{<:Union{Direct,Adjoint}},
                       A::Gram{Diff{2,Any,Opt}},
                       x::AbstractArray,
                       β::Number,
                       y::AbstractArray,
                       I::ArrayAxes,
                       J::ArrayAxis,
                       K::ArrayAxes) where {Opt}
    T = real(eltype(x))
    jmin, jmax = limits(J)
    len = length(J)
    if len ≥ 5
        @maybe_inbounds Opt for k in CartesianIndices(K)
            let j = jmin
                @maybe_vectorized Opt for i in CartesianIndices(I)
                    y[i,j,k] = f(α, D2tD2_1(x,i,j,k), β, y[i,j,k])
                end
            end
            let j = jmin+1
                @maybe_vectorized Opt for i in CartesianIndices(I)
                    y[i,j,k] = f(α, D2tD2_2(x,i,j,k), β, y[i,j,k])
                end
            end
            for j in jmin+2:jmax-2
                @maybe_vectorized Opt for i in CartesianIndices(I)
                    y[i,j,k] = f(α, D2tD2_3(x,i,j,k), β, y[i,j,k])
                end
            end
            let j = jmax-1
                @maybe_vectorized Opt for i in CartesianIndices(I)
                    y[i,j,k] = f(α, D2tD2_4(x,i,j,k), β, y[i,j,k])
                end
            end
            let j = jmax
                @maybe_vectorized Opt for i in CartesianIndices(I)
                    y[i,j,k] = f(α, D2tD2_5(x,i,j,k), β, y[i,j,k])
                end
            end
        end
    elseif len == 4
        @maybe_inbounds Opt for k in CartesianIndices(K)
            let j = jmin
                @maybe_vectorized Opt for i in CartesianIndices(I)
                    y[i,j,k] = f(α, D2tD2_1(x,i,j,k), β, y[i,j,k])
                end
            end
            let j = jmin+1
                @maybe_vectorized Opt for i in CartesianIndices(I)
                    y[i,j,k] = f(α, D2tD2_2(x,i,j,k), β, y[i,j,k])
                end
            end
            let j = jmax-1
                @maybe_vectorized Opt for i in CartesianIndices(I)
                    y[i,j,k] = f(α, D2tD2_4(x,i,j,k), β, y[i,j,k])
                end
            end
            let j = jmax
                @maybe_vectorized Opt for i in CartesianIndices(I)
                    y[i,j,k] = f(α, D2tD2_5(x,i,j,k), β, y[i,j,k])
                end
            end
        end
    elseif len == 3
        @maybe_inbounds Opt for k in CartesianIndices(K)
            let j = jmin
                @maybe_vectorized Opt for i in CartesianIndices(I)
                    y[i,j,k] = f(α, D2tD2_1(x,i,j,k), β, y[i,j,k])
                end
            end
            let j = jmin+1
                @maybe_vectorized Opt for i in CartesianIndices(I)
                    y[i,j,k] = f(α, D2tD2_6(x,i,j,k), β, y[i,j,k])
                end
            end
            let j = jmax
                @maybe_vectorized Opt for i in CartesianIndices(I)
                    y[i,j,k] = f(α, D2tD2_5(x,i,j,k), β, y[i,j,k])
                end
            end
        end
    elseif len == 2
        @maybe_inbounds Opt for k in CartesianIndices(K)
            let j = jmin
                @maybe_vectorized Opt for i in CartesianIndices(I)
                    y[i,j,k] = f(α, D2tD2_7(x,i,j,k), β, y[i,j,k])
                end
            end
            let j = jmax
                @maybe_vectorized Opt for i in CartesianIndices(I)
                    y[i,j,k] = f(α, D2tD2_8(x,i,j,k), β, y[i,j,k])
                end
            end
        end
    elseif len == 1 && β != 1
        let j = jmin, z = zero(T)
            @maybe_inbounds Opt for k in CartesianIndices(K)
                @maybe_vectorized Opt for i in CartesianIndices(I)
                    y[i,j,k] = f(α, z, β, y[i,j,k])
                end
            end
        end
    end
    nothing
end

end # module
