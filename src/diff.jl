"""
    A = Diff{L=1,D=Colon}()

Return a linear mapping that computes a finite difference approximation of the `L`-order
derivative along the dimension(s) specified by `D`. Parameter `D` is an `Int`, a tuple of
`Int`s, or `Colon` for differentiating respectively along a single dimension, several
dimensions, or all dimensions.

Currently, only 1st or 2nd order finite difference (`L=1` or `L=2`) are implemented. If `L`
is unspecified, `A` will compute 1st order finite differences.

If `D` is unspecified, `A` will compute finite differences along all dimensions.

If `D` is a single `Int`, the result `y = A*x` of applying the finite difference operator to
an array `x` has the same axes as `x`. Otherwise and even though `x` has a single dimension
or `D` is a 1-tuple, `y` has one more dimension than `x`, the last dimension of `y` storing
the finite differences along each dimensions specified by `D` and the leading dimensions of
`y` are the same as the dimensions of `x`.

If multiple dimensions are specified, the result is as if the operator is applied separately
on the specified dimension(s).

More specifically, the operator created by `Diff` implements **forward finite differences**
with *flat boundary conditions*, that is to say extrapolated entries are assumed equal to
the nearest entry.

"""
Diff() = Diff{1}()
Diff{L}() where {L} = Diff{L,Colon}()

# Traits.
MatrixShape(::Type{<:Diff{1}}) = UpperTriangularShape()

# Two finite difference operators are identical if they have the same order of
# differentiation and list of dimensions along which compute the differences. This amounts
# to checking whether they have the same type. Note that `isequal` amounts to calling `==`
# by default.
Base.:(==)(A::T, B::T) where {T<:Diff} = true

# Print operator in such a way that is similar to how the operator would be created in
# Julia.
function Base.show(io::IO, ::Diff{L,D}) where {L,D}
    print(io, "Diff{", L, ',')
    if D === Colon
        print(io, "Colon")
    elseif D isa Tuple{Integer,Vararg{Integer}}
        print(io, "(")
        for (i,d) in enumerate(D)
            i > 1 && print(io, ",")
            print(io, d)
        end
        print(io, length(D) == 1 ? ",)" : ")")
    else
        print(io, D)
    end
    print(io, "}()")
    nothing
end

# Output element type for any variant of the finite difference operator.
function output_eltype(::Type{<:Union{D,Adjoint{D}}},
                       ::Type{x}) where {D<:Diff,x<:AbstractArray}
    # FIXME: At least a signed type should be returned.
    return float(eltype(x))
end

# Output axes for D'*D with D a finite difference operator.
output_axes(A::TwoProd{Adjoint{D},D}, axes_x::ArrayAxes) where {D<:Diff} = axes_x

# Output element type for other variant of the finite difference operator.
function output_axes(A::Union{Diff{L,D},Adjoint{<:Diff{L,D}}},
                     axes_x::ArrayAxes{N}) where {L,D,N}
    if D isa Int || D isa Dims
        # All dimensions of finite differentiation must be in range.
        for d in D
            1 ≤ d ≤ (A isa Adjoint ? N - 1 : N) || throw_bad_argument(
                "out of range dimension of finite differentiation")
        end
    elseif D !== Colon
        throw_assertion_error("unexpected dimension(s) of differentiation")
    end

    # Unless `D` is a scalar `Int`, output of finite difference has one more trailing
    # dimension equal to the number of dimensions along which to differentiate.
    if D isa Int
        return axes_x
    elseif A isa Adjoint
        N ≥ 1 || throw_dimension_mismatch("input array must have at least 1 dimension")
        nd = (D === Colon ? N-1 : length(D))
        axes_x[N] == 𝟙:nd || throw_dimension_mismatch(
            "last axis of input array must be 1:$nd, got $(axes_x[N])")
        return axes_x[1:N-1]
    else
        nd = (D === Colon ? N : length(D))
        return (axes_x..., 𝟙:nd)
    end
end

# Apply the operation along all dimensions of interest but one dimension at a time and
# knowing that α is not zero.
@generated function unsafe_vmul!(α::Number,
                                 A::Union{𝒟,Adjoint{𝒟},TwoProd{Adjoint{𝒟},𝒟}},
                                 x::AbstractArray{Tx,Nx},
                                 β::Number,
                                 y::AbstractArray{Ty,Ny}) where {L,D,𝒟<:Diff{L,D},Tx,Nx,Ty,Ny}
    # Minimal check to avoid compiling an invalid function.
    D === Colon || D isa Int || D isa Tuple{Vararg{Int}} || throw_assertion_error(
        "invalid list of dimension(s) of differentiation")

    # Start with empty vector of statements.
    code = Expr[]

    # Make sure x[...] delivers a value of the correct type.
    T = output_eltype(A, x)
    T === Tx || push!(code, :(x = as_eltype($T, x)))

    # Discard type parameter specifying the dimensions of interest to avoid specialization
    # on this parameter.
    op = :(Diff{$L,:any}())
    if A <: Adjoint
        push!(code, :(B = $op'))
    elseif A <: Prod # Gram
        push!(code, :(B = $op'*$op))
    else
        push!(code, :(B = $op))
    end

    # Define `rngs` to be the axes of x or y (whichever is the longest list) and set `N`
    # such that `rngs[1:N]` is the list of common axes while, except for Gram, `rngs[N+1]`
    # is the axis storing the differences along the dimension(s) of interest.
    if A <: Adjoint
        push!(code, :(rngs = axes(x)))
        N = Ny
    else
        push!(code, :(rngs = axes(y)))
        N = Nx
    end

    # Dispatch on dimension(s) of interest.
    for (i, d) in enumerate(D === Colon ? (1:N) : D)
        # Checking that `d ∈ 1:N` has no extra cost at run-time and avoid compiling an
        # invalid function. This is an assertion error because it should have been detected
        # sooner.
        d ∈ 1:N || return quote
            throw_assertion_error("out of range dimension(s) of differentiation")
        end
        if A <: Prod # Gram
            args = ()
        elseif D isa Int
            args = (:(CartesianIndex()),)
        else
            # One of x or y (depending on whether the direct or the adjoint operator is
            # applied) has an extra leading dimension used to store the result computed
            # along a given dimension.
            args = (:(CartesianIndex(rngs[$(N+1)][$i])),)
        end
        push!(code, :(_Diff.unsafe_vmul!(α, B, x,
                                         $(i == 1 || A <: Diff ? :β : :(one(β))), y,
                                         rngs[1:$(d-1)],
                                         rngs[$d],
                                         rngs[$(d+1):$N], $(args...))))
    end

    return quote
        $(Expr(:meta, :inline))
        $(code...)
        return y
    end
end

"""
Private module for finite differences.
"""
module _Diff

using TypeUtils
using Base: @propagate_inbounds
using ..LazyAlgebra: Adjoint, Diff, Prod, TwoProd

"""
    limits(r) -> (first(r), last(r))

yields the first and last value of the unit-range `r`.

"""
limits(r::AbstractUnitRange) = (first(r), last(r))

#------------------------------------------------------------ 1st order finite differences -
#
# The operator D implementing 1st order forward finite difference with flat boundary
# conditions and its adjoint D' are given by:
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
# The last row (for D) and column (for D') of zeros are to preserve the size. This is needed
# for multi-dimensional arrays when derivatives along each dimension are stored into a
# single array.
#
function unsafe_vmul!(α::Number,
                      A::Diff{1,:any},
                      x::AbstractArray,
                      β::Number,
                      y::AbstractArray,
                      I::ArrayAxes,
                      J::eltype(ArrayAxes),
                      K::ArrayAxes,
                      l::CartesianIndex)
    # Assumptions:
    # (1) `f` is chosen according to the specific values of multipliers `α` and `β`;
    # (2) element type of `x` is such that expressions `x[i] - x[j]` and `-x[i]` yield a
    #     correct result.
    jmin, jmax = limits(J)
    if jmin ≤ jmax
        if I isa Tuple{} # apply along 1st dimension
            @inbounds @fastmath for k in CartesianIndices(K)
                @simd for j in jmin:jmax-1
                    z = x[j+1,k] - x[j,k]
                    y[j,k,l] = α*z + β*y[j,k,l]
                end
                let j = jmax, z = zero(real(eltype(x)))
                    y[j,k,l] = α*z + β*y[j,k,l]
                end
            end
        else # apply along 2nd and subsequent dimensions
            @inbounds @fastmath for k in CartesianIndices(K)
                for j in jmin:jmax-1
                    @simd for i in CartesianIndices(I)
                        z = x[i,j+1,k] - x[i,j,k]
                        y[i,j,k,l] = α*z + β*y[i,j,k,l]
                    end
                end
                let j = jmax, z = zero(real(eltype(x)))
                    @simd for i in CartesianIndices(I)
                        y[i,j,k,l] = α*z + β*y[i,j,k,l]
                    end
                end
            end
        end
    end
    nothing
end

function unsafe_vmul!(α::Number,
                      A::Adjoint{<:Diff{1,:any}},
                      x::AbstractArray,
                      β::Number,
                      y::AbstractArray,
                      I::ArrayAxes,
                      J::eltype(ArrayAxes),
                      K::ArrayAxes,
                      l::CartesianIndex)
    jmin, jmax = limits(J)
    if jmin < jmax
        if I isa Tuple{} # apply along 1st dimension
            @inbounds @fastmath for k in CartesianIndices(K)
                let j = jmin
                    z = -x[j,k,l]
                    y[j,k] = α*z + β*y[j,k]
                end
                @simd for j in jmin+1:jmax-1
                    z = x[j-1,k,l] - x[j,k,l]
                    y[j,k] = α*z + β*y[j,k]
                end
                let j = jmax
                    z = x[j-1,k,l]
                    y[j,k] = α*z + β*y[j,k]
                end
            end
        else # apply along 2nd and subsequent dimensions
            @inbounds @fastmath for k in CartesianIndices(K)
                let j = jmin
                    @simd for i in CartesianIndices(I)
                        z = -x[i,j,k,l]
                        y[i,j,k] = α*z + β*y[i,j,k]
                    end
                end
                for j in jmin+1:jmax-1
                    @simd for i in CartesianIndices(I)
                        z = x[i,j-1,k,l] - x[i,j,k,l]
                        y[i,j,k] = α*z + β*y[i,j,k]
                    end
                end
                let j = jmax
                    @simd for i in CartesianIndices(I)
                        z = x[i,j-1,k,l]
                        y[i,j,k] = α*z + β*y[i,j,k]
                    end
                end
            end
        end
    elseif jmin == jmax && β != 1
        let j = jmin, z = zero(real(eltype(x)))
            if I isa Tuple{} # apply along 1st dimension
                @inbounds @fastmath @simd for k in CartesianIndices(K)
                    y[j,k] = α*z + β*y[j,k]
                end
            else # apply along 2nd and subsequent dimensions
                @inbounds @fastmath for k in CartesianIndices(K)
                    @simd for i in CartesianIndices(I)
                        y[i,j,k] = α*z + β*y[i,j,k]
                    end
                end
            end
        end
    end
    nothing
end
#
# The Gram composition D'*D of the 1st order forward finite differences D with flat boundary
# conditions writes:
#
#     D'*D = [  1  -1   0   0   0
#              -1   2  -1   0   0
#               0  -1   2  -1   0
#               0   0  -1   2  -1
#               0   0   0  -1   1 ]
#
function unsafe_vmul!(α::Number,
                      A::TwoProd{Adjoint{𝒟},𝒟}, # Gram
                      x::AbstractArray,
                      β::Number,
                      y::AbstractArray,
                      I::ArrayAxes,
                      J::eltype(ArrayAxes),
                      K::ArrayAxes) where {𝒟<:Diff{1,:any}}
    jmin, jmax = limits(J)
    if jmin < jmax
        T = real_type(eltype(x))
        two = as(T, 2)
        @inbounds @fastmath for k in CartesianIndices(K)
            if I isa Tuple{} # apply D'*D along 1st dimension
                let j = jmin
                    z = x[j,k] - x[j+1,k]
                    y[j,k] = α*z + β*y[j,k]
                end
                @simd for j in jmin+1:jmax-1
                    z = two*x[j,k] - (x[j-1,k] + x[j+1,k])
                    y[j,k] = α*z + β*y[j,k]
                end
                let j = jmax
                    z = x[j,k] - x[j-1,k]
                    y[j,k] = α*z + β*y[j,k]
                end
            else # apply D'*D along 2nd and subsequent dimensions
                let j = jmin
                    @simd for i in CartesianIndices(I)
                        z = x[i,j,k] - x[i,j+1,k]
                        y[i,j,k] = α*z + β*y[i,j,k]
                    end
                end
                for j in jmin+1:jmax-1
                    @simd for i in CartesianIndices(I)
                        z = two*x[i,j,k] - (x[i,j-1,k] + x[i,j+1,k])
                        y[i,j,k] = α*z + β*y[i,j,k]
                    end
                end
                let j = jmax
                    @simd for i in CartesianIndices(I)
                        z = x[i,j,k] - x[i,j-1,k]
                        y[i,j,k] = α*z + β*y[i,j,k]
                    end
                end
            end
        end
    elseif jmin == jmax && β != 1
        let j = jmin, z = zero(real(eltype(x)))
            if I isa Tuple{}
                @inbounds @fastmath @simd for k in CartesianIndices(K)
                    y[j,k] = α*z + β*y[j,k]
                end
            else
                @inbounds @fastmath for k in CartesianIndices(K)
                    @simd for i in CartesianIndices(I)
                        y[i,j,k] = α*z + β*y[i,j,k]
                    end
                end
            end
        end
    end
    nothing
end

#------------------------------------------------------------ 2nd order finite differences -
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
#  - For a single dimension, this operator is the opposite of the Gram composition of 1st
#    order finite differences (backward or forward).
#
# Apply 2nd order finite differences.
#
function unsafe_vmul!(α::Number,
                      A::Diff{2,:any},
                      x::AbstractArray,
                      β::Number,
                      y::AbstractArray,
                      I::ArrayAxes,
                      J::eltype(ArrayAxes),
                      K::ArrayAxes,
                      l::CartesianIndex)
    jmin, jmax = limits(J)
    if jmin < jmax
        T = real_type(eltype(x))
        two = as(T, 2)
        @inbounds @fastmath for k in CartesianIndices(K)
            if I isa Tuple{} # apply along 1st dimension
                let j = jmin
                    z = x[j+1,k] - x[j,k]
                    y[j,k,l] = α*z + β*y[j,k,l]
                end
                @simd for j in jmin+1:jmax-1
                    z = x[j-1,k] + x[j+1,k] - two*x[j,k]
                    y[j,k,l] = α*z + β*y[j,k,l]
                end
                let j = jmax
                    z = x[j-1,k] - x[j,k]
                    y[j,k,l] = α*z + β*y[j,k,l]
                end
            else # apply along 2nd and subsequent dimensions
                let j = jmin
                    @simd for i in CartesianIndices(I)
                        z = x[i,j+1,k] - x[i,j,k]
                        y[i,j,k,l] = α*z + β*y[i,j,k,l]
                    end
                end
                for j in jmin+1:jmax-1
                    @simd for i in CartesianIndices(I)
                        z = x[i,j-1,k] + x[i,j+1,k] - two*x[i,j,k]
                        y[i,j,k,l] = α*z + β*y[i,j,k,l]
                    end
                end
                let j = jmax
                    @simd for i in CartesianIndices(I)
                        z = x[i,j-1,k] - x[i,j,k]
                        y[i,j,k,l] = α*z + β*y[i,j,k,l]
                    end
                end
            end
        end
    elseif jmin == jmax && β != 1
        let j = jmin, z = zero(real(eltype(x)))
            if I isa Tuple{} # apply along 1st dimension
                @inbounds @fastmath @simd for k in CartesianIndices(K)
                    y[j,k,l] = α*z + β*y[j,k,l]
                end
            else # apply along 2nd and subsequent dimensions
                @inbounds @fastmath for k in CartesianIndices(K)
                    @simd for i in CartesianIndices(I)
                        y[i,j,k,l] = α*z + β*y[i,j,k,l]
                    end
                end
            end
        end
    end
    nothing
end
#
# Apply adjoint of 2nd order finite differences.
#
function unsafe_vmul!(α::Number,
                      A::Adjoint{<:Diff{2,:any}},
                      x::AbstractArray,
                      β::Number,
                      y::AbstractArray,
                      I::ArrayAxes,
                      J::eltype(ArrayAxes),
                      K::ArrayAxes,
                      l::CartesianIndex)
    jmin, jmax = limits(J)
    if jmin < jmax
        T = real_type(eltype(x))
        two = as(T, 2)
        @inbounds @fastmath for k in CartesianIndices(K)
            if I isa Tuple{} # apply along 1st dimension
                let j = jmin
                    z = x[j+1,k,l] - x[j,k,l]
                    y[j,k] = α*z + β*y[j,k]
                end
                @simd for j in jmin+1:jmax-1
                    z = x[j-1,k,l] + x[j+1,k,l] - two*x[j,k,l]
                    y[j,k] = α*z + β*y[j,k]
                end
                let j = jmax
                    z = x[j-1,k,l] - x[j,k,l]
                    y[j,k] = α*z + β*y[j,k]
                end
            else # apply along 2nd and subsequent dimensions
                let j = jmin
                    @simd for i in CartesianIndices(I)
                        z = x[i,j+1,k,l] - x[i,j,k,l]
                        y[i,j,k] = α*z + β*y[i,j,k]
                    end
                end
                for j in jmin+1:jmax-1
                    @simd for i in CartesianIndices(I)
                        z = x[i,j-1,k,l] + x[i,j+1,k,l] - two*x[i,j,k,l]
                        y[i,j,k] = α*z + β*y[i,j,k]
                    end
                end
                let j = jmax
                    @simd for i in CartesianIndices(I)
                        z = x[i,j-1,k,l] - x[i,j,k,l]
                        y[i,j,k] = α*z + β*y[i,j,k]
                    end
                end
            end
        end
    elseif jmin == jmax && β != 1
        let j = jmin, z = zero(real(eltype(x)))
            if I isa Tuple{} # apply along 1st dimension
                @simd for k in CartesianIndices(K)
                    y[j,k] = α*z + β*y[j,k]
                end
            else # apply along 2nd and subsequent dimensions
                @inbounds @fastmath for k in CartesianIndices(K)
                    @simd for i in CartesianIndices(I)
                        y[i,j,k] = α*z + β*y[i,j,k]
                    end
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
# The above is for `len ≥ 4`, with `len` the length of the dimension of interest, omitting
# the Eq. (5) for `len = 4` and repeating Eq. (5) as necessary for the central rows for `n ≥
# 5`. For len = 3:
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
# Methods to apply the rows of D'*D:
#
# - Eq. (1), first row when len ≥ 3:
#
@propagate_inbounds function D2tD2_1(x::AbstractArray,
                                     i::CartesianIndex, j::Int, k::CartesianIndex)
    T = real_type(eltype(x))
    T(2)*x[i,j,k] - T(3)*x[i,j+1,k] + x[i,j+2,k]
end
#
# - Eq. (2), second row when len ≥ 4:
#
@propagate_inbounds function D2tD2_2(x::AbstractArray,
                                     i::CartesianIndex, j::Int, k::CartesianIndex)
    T = real_type(eltype(x))
    T(6)*x[i,j,k] - T(3)*x[i,j-1,k] - T(4)*x[i,j+1,k] + x[i,j+2,k]
end
#
# - Eq. (3), central rows when len ≥ 5:
#
@propagate_inbounds function D2tD2_3(x::AbstractArray,
                                     i::CartesianIndex, j::Int, k::CartesianIndex)
    T = real_type(eltype(x))
    (x[i,j-2,k] + x[i,j+2,k]) + T(6)*x[i,j,k] - T(4)*(x[i,j-1,k] + x[i,j+1,k])
end
#
# - Eq. (4), before last row when len ≥ 4:
#
@propagate_inbounds function D2tD2_4(x::AbstractArray,
                                     i::CartesianIndex, j::Int, k::CartesianIndex)
    T = real_type(eltype(x))
    T(6)*x[i,j,k] - T(3)*x[i,j+1,k] - T(4)*x[i,j-1,k] + x[i,j-2,k]
end
#
# - Eq. (5), last row when len ≥ 3:
#
@propagate_inbounds function D2tD2_5(x::AbstractArray,
                                     i::CartesianIndex, j::Int, k::CartesianIndex)
    T = real_type(eltype(x))
    T(2)*x[i,j,k] - T(3)*x[i,j-1,k] + x[i,j-2,k]
end
#
# - Eq. (6), central row when len = 3:
#
@propagate_inbounds function D2tD2_6(x::AbstractArray,
                                     i::CartesianIndex, j::Int, k::CartesianIndex)
    T = real_type(eltype(x))
    T(6)*x[i,j,k] - T(3)*(x[i,j-1,k] + x[i,j+1,k])
end
#
# - Eq. (7), first row when len = 2:
#
@propagate_inbounds function D2tD2_7(x::AbstractArray,
                                     i::CartesianIndex, j::Int, k::CartesianIndex)
    z = x[i,j,k] - x[i,j+1,k]
    return z + z
end
#
# - Eq. (8), last row when len = 2:
#
@propagate_inbounds function D2tD2_8(x::AbstractArray,
                                     i::CartesianIndex, j::Int, k::CartesianIndex)
    z = x[i,j,k] - x[i,j-1,k]
    return z + z
end
#
# Apply Gram composition of 2nd order finite differences.
#
function unsafe_vmul!(α::Number,
                      A::TwoProd{Adjoint{𝒟},𝒟}, # Gram,
                      x::AbstractArray,
                      β::Number,
                      y::AbstractArray,
                      I::ArrayAxes,
                      J::eltype(ArrayAxes),
                      K::ArrayAxes) where {𝒟<:Diff{2,:any}}
    jmin, jmax = limits(J)
    len = length(J)
    if len ≥ 5
        if I isa Tuple{} # apply along 1st dimension
            let i = CartesianIndex()
                @inbounds @fastmath for k in CartesianIndices(K)
                    let j = jmin
                        y[i,j,k] = α*D2tD2_1(x,i,j,k) + β*y[i,j,k]
                    end
                    let j = jmin+1
                        y[i,j,k] = α*D2tD2_2(x,i,j,k) + β*y[i,j,k]
                    end
                    @simd for j in jmin+2:jmax-2
                        y[i,j,k] = α*D2tD2_3(x,i,j,k) + β*y[i,j,k]
                    end
                    let j = jmax-1
                        y[i,j,k] = α*D2tD2_4(x,i,j,k) + β*y[i,j,k]
                    end
                    let j = jmax
                        y[i,j,k] = α*D2tD2_5(x,i,j,k) + β*y[i,j,k]
                    end
                end
            end
        else # apply along 2nd and subsequent dimensions
            @inbounds @fastmath for k in CartesianIndices(K)
                let j = jmin
                    @simd for i in CartesianIndices(I)
                        y[i,j,k] = α*D2tD2_1(x,i,j,k) + β*y[i,j,k]
                    end
                end
                let j = jmin+1
                    @simd for i in CartesianIndices(I)
                        y[i,j,k] = α*D2tD2_2(x,i,j,k) + β*y[i,j,k]
                    end
                end
                for j in jmin+2:jmax-2
                    @simd for i in CartesianIndices(I)
                        y[i,j,k] = α*D2tD2_3(x,i,j,k) + β*y[i,j,k]
                    end
                end
                let j = jmax-1
                    @simd for i in CartesianIndices(I)
                        y[i,j,k] = α*D2tD2_4(x,i,j,k) + β*y[i,j,k]
                    end
                end
                let j = jmax
                    @simd for i in CartesianIndices(I)
                        y[i,j,k] = α*D2tD2_5(x,i,j,k) + β*y[i,j,k]
                    end
                end
            end
        end
    elseif len == 4
        if I isa Tuple{} # apply along 1st dimension
            let i = CartesianIndex()
                @inbounds @fastmath @simd for k in CartesianIndices(K)
                    let j = jmin
                        y[i,j,k] = α*D2tD2_1(x,i,j,k) + β*y[i,j,k]
                    end
                    let j = jmin+1
                        y[i,j,k] = α*D2tD2_2(x,i,j,k) + β*y[i,j,k]
                    end
                    let j = jmax-1
                        y[i,j,k] = α*D2tD2_4(x,i,j,k) + β*y[i,j,k]
                    end
                    let j = jmax
                        y[i,j,k] = α*D2tD2_5(x,i,j,k) + β*y[i,j,k]
                    end
                end
            end
        else # apply along 2nd and subsequent dimensions
            @inbounds @fastmath for k in CartesianIndices(K)
                let j = jmin
                    @simd for i in CartesianIndices(I)
                        y[i,j,k] = α*D2tD2_1(x,i,j,k) + β*y[i,j,k]
                    end
                end
                let j = jmin+1
                    @simd for i in CartesianIndices(I)
                        y[i,j,k] = α*D2tD2_2(x,i,j,k) + β*y[i,j,k]
                    end
                end
                let j = jmax-1
                    @simd for i in CartesianIndices(I)
                        y[i,j,k] = α*D2tD2_4(x,i,j,k) + β*y[i,j,k]
                    end
                end
                let j = jmax
                    @simd for i in CartesianIndices(I)
                        y[i,j,k] = α*D2tD2_5(x,i,j,k) + β*y[i,j,k]
                    end
                end
            end
        end
     elseif len == 3
        if I isa Tuple{} # apply along 1st dimension
            let i = CartesianIndex()
                @simd for k in CartesianIndices(K)
                    let j = jmin
                        y[i,j,k] = α*D2tD2_1(x,i,j,k) + β*y[i,j,k]
                    end
                    let j = jmin+1
                        y[i,j,k] = α*D2tD2_6(x,i,j,k) + β*y[i,j,k]
                    end
                    let j = jmax
                        y[i,j,k] = α*D2tD2_5(x,i,j,k) + β*y[i,j,k]
                    end
                end
            end
        else # apply along 2nd and subsequent dimensions
            @inbounds @fastmath for k in CartesianIndices(K)
                let j = jmin
                    @simd for i in CartesianIndices(I)
                        y[i,j,k] = α*D2tD2_1(x,i,j,k) + β*y[i,j,k]
                    end
                end
                let j = jmin+1
                    @simd for i in CartesianIndices(I)
                        y[i,j,k] = α*D2tD2_6(x,i,j,k) + β*y[i,j,k]
                    end
                end
                let j = jmax
                    @simd for i in CartesianIndices(I)
                        y[i,j,k] = α*D2tD2_5(x,i,j,k) + β*y[i,j,k]
                    end
                end
            end
        end
    elseif len == 2
        if I isa Tuple{} # apply along 1st dimension
            let i = CartesianIndex()
                @inbounds @fastmath @simd for k in CartesianIndices(K)
                    let j = jmin
                        y[i,j,k] = α*D2tD2_7(x,i,j,k) + β*y[i,j,k]
                    end
                    let j = jmax
                        y[i,j,k] = α*D2tD2_8(x,i,j,k) + β*y[i,j,k]
                    end
                end
            end
        else # apply along 2nd and subsequent dimensions
            @inbounds @fastmath for k in CartesianIndices(K)
                let j = jmin
                    @simd for i in CartesianIndices(I)
                        y[i,j,k] = α*D2tD2_7(x,i,j,k) + β*y[i,j,k]
                    end
                end
                let j = jmax
                    @simd for i in CartesianIndices(I)
                        y[i,j,k] = α*D2tD2_8(x,i,j,k) + β*y[i,j,k]
                    end
                end
            end
        end
    elseif len == 1 && β != 1
        if I isa Tuple{} # apply along 1st dimension
            let i = CartesianIndex(), j = jmin, z = zero(real(eltype(x)))
                @inbounds @fastmath @simd for k in CartesianIndices(K)
                    y[i,j,k] = α*z + β*y[i,j,k]
                end
            end
        else # apply along 2nd and subsequent dimensions
            let j = jmin, z = zero(real(eltype(x)))
                @inbounds @fastmath for k in CartesianIndices(K)
                    @simd for i in CartesianIndices(I)
                        y[i,j,k] = α*z + β*y[i,j,k]
                    end
                end
            end
        end
    end
    nothing
end

end # module _Diff
