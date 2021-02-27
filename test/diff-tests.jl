#
# diff-tests.jl -
#
# Tests for finite difference operators.
#
module TestingLazyAlgebraDiff

using Test
using MayOptimize
using LazyAlgebra
using LazyAlgebra.Foundations
using LazyAlgebra.FiniteDifferences: limits, ArrayAxis, ArrayAxes,
    differentiation_order, dimensions_of_interest, optimization_level

include("common.jl")

@inline colons(n::Integer) = ntuple(x -> Colon(), max(Int(n), 0))

# Slice along the last dimension.
slice(A::AbstractArray{T,N}, d::Int) where {T,N} =
    view(A, colons(N - 1)..., d)

# Reference implementation of finite differences.

gram_diff_ref(order::Integer, A::AbstractArray; dim::Int=1) =
    diff_ref(order, diff_ref(order, A; dim=dim, adj=false); dim=dim, adj=true)

diff_ref(order::Integer, A::AbstractArray; kwds...) =
    (order == 1 ? diff1_ref(A; kwds...) :
     order == 2 ? diff2_ref(A; kwds...) :
     error("invalid derivative order ", order))

diff1_ref(A::AbstractArray; kwds...) = diff1_ref!(similar(A), A; kwds...)

diff2_ref(A::AbstractArray; kwds...) = diff2_ref!(similar(A), A; kwds...)

function diff1_ref!(dst::AbstractArray{<:Any,N},
                    src::AbstractArray{<:Any,N};
                    dim::Int=1,
                    adj::Bool=false) where {N}
    inds = axes(src)
    @assert axes(dst) == inds
    @assert 1 ≤ dim ≤ N
    I = colons(dim - 1)
    J = inds[dim]
    K = colons(N - dim)
    diff1_ref!(dst, src, I, J, K, adj)
end

function diff2_ref!(dst::AbstractArray{<:Any,N},
                    src::AbstractArray{<:Any,N};
                    dim::Int=1,
                    adj::Bool=false) where {N}
    inds = axes(src)
    @assert axes(dst) == inds
    @assert 1 ≤ dim ≤ N
    I = colons(dim - 1)
    J = inds[dim]
    K = colons(N - dim)
    diff2_ref!(dst, src, I, J, K)
end

#
# Code to apply 1st order finite difference operator D or its adjoint D' (given
# below) along a given dimensions of an array.
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
function diff1_ref!(dst::AbstractArray{<:Any,N},
                    src::AbstractArray{<:Any,N},
                    I::Tuple{Vararg{Colon}},
                    J::ArrayAxis,
                    K::Tuple{Vararg{Colon}},
                    adj::Bool) where {N}
    # Forward 1st order finite differences assuming flat boundary conditions
    # along a given dimension.
    len = length(J)
    if len > 1
        j_first = first(J)
        j_last = last(J)
        if adj
            map!(-,
                 view(dst, I..., j_first, K...),
                 view(src, I..., j_first, K...))
            if len > 2
                map!(-,
                     view(dst, I..., (j_first+1):(j_last-1), K...),
                     view(src, I..., (j_first  ):(j_last-2), K...),
                     view(src, I..., (j_first+1):(j_last-1), K...))
            end
            copyto!(view(dst, I..., j_last,   K...),
                    view(src, I..., j_last-1, K...))
        else
            map!(-,
                 view(dst, I..., (j_first  ):(j_last-1), K...),
                 view(src, I..., (j_first+1):(j_last  ), K...),
                 view(src, I..., (j_first  ):(j_last-1), K...))
            fill!(view(dst, I..., j_last, K...), zero(eltype(dst)))
        end
    else
        fill!(dst, zero(eltype(dst)))
    end
    return dst
end

#
# 2nd order finite differences D2 with flat boundary conditions is given by:
#
#     D2 = [-1   1   0   0
#            1  -2   1   0
#            0   1  -2   1
#            0   0   1  -1]
#
# Note that this operator is self-adjoint, so there is no `adj` argument.
#
function diff2_ref!(dst::AbstractArray{<:Any,N},
                    src::AbstractArray{<:Any,N},
                    I::Tuple{Vararg{Colon}},
                    J::ArrayAxis,
                    K::Tuple{Vararg{Colon}}) where {N}
    # Forward 1st order finite differences assuming flat boundary conditions
    # along a given dimension.
    len = length(J)
    if len ≥ 2
        j_first = first(J)
        j_last = last(J)
        map!((b, c) -> c - b,
             view(dst, I..., j_first,   K...),
             view(src, I..., j_first,   K...),
             view(src, I..., j_first+1, K...))
        if len > 2
            map!((a, b, c) -> (c - b) - (b - a),
                 view(dst, I..., (j_first+1):(j_last-1), K...),
                 view(src, I..., (j_first  ):(j_last-2), K...),
                 view(src, I..., (j_first+1):(j_last-1), K...),
                 view(src, I..., (j_first+2):(j_last  ), K...))
        end
        map!((a, b) -> a - b,
             view(dst, I..., j_last,   K...),
             view(src, I..., j_last-1, K...),
             view(src, I..., j_last,   K...))
    else
        fill!(dst, zero(eltype(dst)))
    end
    return dst
end

@testset "Finite differences" begin
    # First test the correctness of the result compared to the above reference
    # implementation and use the saclar product to check the adjoint.  Use
    # integer values for exact computations, over a reduced range to avoid
    # overflows and have exact floating-point representation.
    vmin = -(vmax = Float64(7_000))
    vals = vmin:vmax
    @testset "Differentiation order = $order" for order in 1:2
        # Dimensions to test: 1, 2 and any ≥ 3 for 1st order derivatives, 1:4
        # and any ≥ 5 for 2nd order derivatives.
        dimlist = (order == 1 ? (1, 2, 5) : ((1:4)..., 6))
        sizes = Tuple{Vararg{Int}}[]
        for dim1 in dimlist
            push!(sizes, (dim1,))
        end
        for dim1 in dimlist, dim2 in dimlist
            push!(sizes, (dim1,dim2))
        end
        for dim1 in dimlist, dim2 in dimlist, dim3 in dimlist
            push!(sizes, (dim1,dim2,dim3))
        end
        for dims in sizes
            # Random values (an copies to check that input variables do not
            # change).
            x = rand(vals, dims)
            y = rand(vals, (size(x)..., ndims(x)))
            xsav = copy(x)
            ysav = copy(y)
            atol = zero(eltype(x))
            rtol = 16*eps(eltype(x))
            # Apply along all dimensions specified as a colon.
            D_all = Diff(order,:,Debug)
            @test differentiation_order(D_all) === order
            @test dimensions_of_interest(D_all) === Colon
            @test optimization_level(D_all) === Debug
            D_all_x = D_all*x;   @test x == xsav
            Dt_all_y = D_all'*y; @test y == ysav
            @test vdot(x, Dt_all_y) == vdot(y, D_all_x)
            test_api(Direct, D_all, x, y; atol=atol, rtol=rtol)
            test_api(Adjoint, D_all, x, y; atol=atol, rtol=rtol)
            DtD_all = D_all'*D_all
            @test DtD_all === gram(D_all)
            @test isa(DtD_all, Gram{typeof(D_all)})
            @test differentiation_order(DtD_all) === order
            @test dimensions_of_interest(DtD_all) === dimensions_of_interest(D_all)
            @test optimization_level(DtD_all) === optimization_level(D_all)
            @test DtD_all*x == D_all'*D_all_x; @test x == xsav
            test_api(Direct, DtD_all, x, x; atol=atol, rtol=rtol)
            # Apply along all dimensions specified as a list.
            D_lst = Diff(order,((1:ndims(x))...,),Debug)
            @test differentiation_order(D_lst) === order
            @test dimensions_of_interest(D_lst) === ((1:ndims(x))...,)
            @test optimization_level(D_lst) === Debug
            D_lst_x = D_lst*x;   @test x == xsav
            Dt_lst_y = D_lst'*y; @test y == ysav
            @test vdot(x, Dt_lst_y) == vdot(y, D_lst_x)
            test_api(Direct, D_lst, x, y; atol=atol, rtol=rtol)
            test_api(Adjoint, D_lst, x, y; atol=atol, rtol=rtol)
            DtD_lst = D_lst'*D_lst
            @test DtD_lst === gram(D_lst)
            @test isa(DtD_lst, Gram{typeof(D_lst)})
            @test differentiation_order(DtD_lst) === order
            @test dimensions_of_interest(DtD_lst) === dimensions_of_interest(D_lst)
            @test optimization_level(DtD_lst) === optimization_level(D_lst)
            @test DtD_lst*x == D_lst'*D_lst_x
            test_api(Direct, DtD_lst, x, x; atol=atol, rtol=rtol)
            if ndims(x) > 1
                # Apply along all dimensions in reverse order specified as a
                # range and to avoid all dimensions at a time version.
                D_rev = Diff(order,ndims(x):-1:1,Debug)
                @test differentiation_order(D_rev) === order
                @test dimensions_of_interest(D_rev) === ((ndims(x):-1:1)...,)
                @test optimization_level(D_rev) === Debug
                D_rev_x = D_rev*x;   @test x == xsav
                Dt_rev_y = D_rev'*y; @test y == ysav
                @test vdot(x, Dt_rev_y) == vdot(y, D_rev_x)
                test_api(Direct, D_rev, x, y; atol=atol, rtol=rtol)
                test_api(Adjoint, D_rev, x, y; atol=atol, rtol=rtol)
                DtD_rev = D_rev'*D_rev
                @test DtD_rev === gram(D_rev)
                @test isa(DtD_rev, Gram{typeof(D_rev)})
                @test differentiation_order(DtD_rev) === order
                @test dimensions_of_interest(DtD_rev) === dimensions_of_interest(D_rev)
                @test optimization_level(DtD_rev) === optimization_level(D_rev)
                @test DtD_rev*x == D_rev'*D_rev_x
                test_api(Direct, DtD_rev, x, x; atol=atol, rtol=rtol)
            end
            for d in 1:ndims(x)
                # Apply along a single dimension.
                D_one = Diff(order,d,Debug)
                @test differentiation_order(D_one) === order
                @test dimensions_of_interest(D_one) === d
                @test optimization_level(D_one) === Debug
                z = slice(y, d)
                D_ref_x = diff_ref(order, x, dim=d, adj=false)
                Dt_ref_z = diff_ref(order, z, dim=d, adj=true)
                @test D_one*x == D_ref_x;   @test x == xsav
                @test D_one'*z == Dt_ref_z; @test y == ysav # z is a view on y
                @test slice(D_all_x, d) == D_ref_x
                @test slice(D_lst_x, d) == D_ref_x
                if ndims(x) > 1
                    r = ndims(x) - d + 1
                    @test slice(D_rev_x, r) == D_ref_x
                end
                test_api(Direct, D_one, x, z; atol=atol, rtol=rtol)
                test_api(Adjoint, D_one, x, z; atol=atol, rtol=rtol)
                DtD_one = D_one'*D_one
                @test DtD_one === gram(D_one)
                @test isa(DtD_one, Gram{typeof(D_one)})
                @test differentiation_order(DtD_one) === order
                @test dimensions_of_interest(DtD_one) === dimensions_of_interest(D_one)
                @test optimization_level(DtD_one) === optimization_level(D_one)
                DtD_one_x = DtD_one*x; @test x == xsav
                @test DtD_one_x == gram_diff_ref(order, x, dim=d)
                @test DtD_one_x == D_one'*(D_one*x)
                test_api(Direct, DtD_one, x, x; atol=atol, rtol=rtol)
            end
        end
    end
end

nothing

end # module
