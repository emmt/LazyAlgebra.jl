#
# diff-tests.jl -
#
# Tests for finite differences.
#

using LazyAlgebra
using Test

@testset "Finite differences" begin
    types = (Float32, Float64)
    alphas = (0, 1, -1,  2.71, π)
    betas = (0, 1, -1, -1.33, Base.MathConstants.φ)
    sizes = ((50,), (8,9), (4,5,6))
    D = SimpleFiniteDifferences()
    DtD = gram(D)
    for T in types, dims in sizes
        x = randn(T, dims)
        xsav = vcopy(x)
        y = randn(T, ndims(x), size(x)...)
        ysav = vcopy(y)
        z = randn(T, size(x))
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
        for α in alphas,
            β in betas,
            scratch in (false, true)
            @test apply!(α, Direct, D, x, scratch, β, vcopy(y)) ≈
                T(α)*Dx + T(β)*y atol=atol rtol=rtol norm=vnorm2
            if scratch
                vcopy!(x, xsav)
            else
                @test x == xsav
            end
            @test apply!(α, Adjoint, D, y, scratch, β, vcopy(x)) ≈
                T(α)*Dty + T(β)*x atol=atol rtol=rtol norm=vnorm2
            if scratch
                vcopy!(y, ysav)
            else
                @test y == ysav
            end
            @test apply!(α, Direct, DtD, x, scratch, β, vcopy(z)) ≈
                T(α)*DtDx + T(β)*z atol=atol rtol=rtol norm=vnorm2
            if scratch
                vcopy!(x, xsav)
            else
                @test x == xsav
            end
        end
    end
end
nothing
