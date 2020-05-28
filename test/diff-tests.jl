#
# diff-tests.jl -
#
# Tests for finite differences.
#

using LazyAlgebra
using Test


"""
    random([T=Float64,], siz)

yield an array of size `siz` of pseudo random numbers of type `T` uniformly
distributed on `[-1/2,+1/2)`.

"""
random(siz::Integer...) = random(siz)
random(siz::Tuple{Vararg{Integer}}) = random(Float64, siz)
random(::Type{T}, siz::Integer...) where {T<:AbstractFloat} =
    random(T, siz)
random(::Type{T}, siz::Tuple{Vararg{Integer}}) where {T<:AbstractFloat} =
    random(T, map(Int, siz))
function random(::Type{T}, siz::NTuple{N,Int}) where {T<:AbstractFloat,N}
    A = rand(T, siz)
    b = one(T)/2
    @inbounds @simd for i in eachindex(A)
        A[i] -= b
    end
    return A
end

@testset "Finite differences" begin
    include("common.jl")
    types = (Float32, Float64)
    sizes = ((50,), (8,9), (4,5,6))
    D = SimpleFiniteDifferences()
    DtD = gram(D)
    for T in types, dims in sizes
        x = random(T, dims)
        xsav = vcopy(x)
        y = random(T, ndims(x), size(x)...)
        ysav = vcopy(y)
        z = random(T, size(x))
        zsav = vcopy(z)
        # Apply direct and adjoint of D "by-hand".
        Dx_truth = Array{T}(undef, size(y))
        Dty_truth = Array{T}(undef, size(x))
        fill!(Dty_truth, 0)
        if ndims(x) == 1
            Dx_truth[1,1:end-1] = x[2:end] - x[1:end-1]
            Dx_truth[1,end] = 0
            Dty_truth[2:end]   += y[1,1:end-1]
            Dty_truth[1:end-1] -= y[1,1:end-1]
        elseif ndims(x) == 2
            Dx_truth[1,1:end-1,:] = x[2:end,:] - x[1:end-1,:]
            Dx_truth[1,end,:] .= 0
            Dx_truth[2,:,1:end-1] = x[:,2:end] - x[:,1:end-1]
            Dx_truth[2,:,end] .= 0
            Dty_truth[2:end,:]   += y[1,1:end-1,:]
            Dty_truth[1:end-1,:] -= y[1,1:end-1,:]
            Dty_truth[:,2:end]   += y[2,:,1:end-1]
            Dty_truth[:,1:end-1] -= y[2,:,1:end-1]
        elseif ndims(x) == 3
            Dx_truth[1,1:end-1,:,:] = x[2:end,:,:] - x[1:end-1,:,:]
            Dx_truth[1,end,:,:] .= 0
            Dx_truth[2,:,1:end-1,:] = x[:,2:end,:] - x[:,1:end-1,:]
            Dx_truth[2,:,end,:] .= 0
            Dx_truth[3,:,:,1:end-1] = x[:,:,2:end] - x[:,:,1:end-1]
            Dx_truth[3,:,:,end] .= 0
            Dty_truth[2:end,:,:]   += y[1,1:end-1,:,:]
            Dty_truth[1:end-1,:,:] -= y[1,1:end-1,:,:]
            Dty_truth[:,2:end,:]   += y[2,:,1:end-1,:]
            Dty_truth[:,1:end-1,:] -= y[2,:,1:end-1,:]
            Dty_truth[:,:,2:end]   += y[3,:,:,1:end-1]
            Dty_truth[:,:,1:end-1] -= y[3,:,:,1:end-1]
        end
        Dx = D*x;     @test x == xsav
        Dty = D'*y;   @test y == ysav
        DtDx = DtD*x; @test x == xsav
        # There should be no differences between Dx and Dx_truth because
        # they are computed in the exact same way.  For Dty and Dty_truth,
        # the comparsion must be approximative.  For testing DtD against
        # D'*D, parenthesis are needed to avoid simplifications.
        atol, rtol = zero(T), 4*eps(T)
        @test vdot(y,Dx) ≈ vdot(Dty,x) atol=atol rtol=sqrt(eps(T))
        @test vnorm2(Dx - Dx_truth) == 0
        @test Dty ≈ Dty_truth atol=atol rtol=rtol norm=vnorm2
        @test DtDx ≈ D'*(D*x) atol=atol rtol=rtol norm=vnorm2
        test_api(Direct, D, x, y; atol=atol, rtol=rtol)
        test_api(Adjoint, D, x, y; atol=atol, rtol=rtol)
        test_api(Direct, DtD, x, z; atol=atol, rtol=rtol)
    end
end
nothing
