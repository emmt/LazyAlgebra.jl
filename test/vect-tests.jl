using LazyAlgebra
using Test
using Neutrals
using LinearAlgebra
using Unitful
using Unitful: km, cm, mm, μm, °, s
using TypeUtils

@testset "Vectorized operations in `LazyAlgebra`" begin
    @testset "Vector Norms" begin
        # Test vectors
        x = [1.0, -2.0, 3.0, -4.0]
        y = [2.0 + 1im, -1.0 - 2im]

        # Test vnorm1
        @test @inferred(vnorm1(x)) ≈ norm(x, 1)
        @test @inferred(vnorm1(y)) ≈ norm(y, 1)
        @test @inferred(vnorm1(Float32, x)) === adapt_precision(Float32, vnorm1(x))
        @test @inferred(vnorm1(Float32, y)) === adapt_precision(Float32, vnorm1(y))

        # Test vnorm2
        @test @inferred(vnorm2(x)) ≈ norm(x, 2)
        @test @inferred(vnorm2(y)) ≈ norm(y, 2)
        @test @inferred(vnorm2(Float32, x)) === adapt_precision(Float32, vnorm2(x))
        @test @inferred(vnorm2(Float32, y)) === adapt_precision(Float32, vnorm2(y))

        # Test vnorminf
        @test @inferred(vnorminf(x)) ≈ norm(x, Inf)
        @test @inferred(vnorminf(y)) ≈ norm(y, Inf)
        @test @inferred(vnorminf(Float32, x)) === adapt_precision(Float32, vnorminf(x))
        @test @inferred(vnorminf(Float32, y)) === adapt_precision(Float32, vnorminf(y))

        # Test with scalar inputs
        @test @inferred(vnorm1(-2.0)) == 2.0
        @test @inferred(vnorm1(-3.0 + 4.0im)) == 5.0
        @test @inferred(vnorm2(-2.0)) == 2.0
        @test @inferred(vnorm2(-3.0 + 4.0im)) == 5.0
        @test @inferred(vnorminf(-2.0)) == 2.0
        @test @inferred(vnorminf(-3.0 + 4.0im)) == 5.0
    end

    @testset "Inner Products" begin
        # Test real/complex vectors
        w = [0.5, 1.0, 2.0]  # weights
        x = [1.0, -2.0, 3.0]
        y = [2.0, 1.0, -1.0]
        xc = [1.0 + 1im, -2.0 - 2im, 3.0 + 4.0im]
        yc = [2.0 - 1im, 1.0 + 1im, -1.0 - 2.0im]
        ux = mm/s # units for x
        uy = s    # units for y

        # Test vdot without weights
        @test @inferred(vdot(x, y)) ≈ dot(x, y)
        @test @inferred(vdot(x.*ux, y)) == vdot(x, y)*ux
        @test @inferred(vdot(x, y.*uy)) == vdot(x, y)*uy
        @test @inferred(vdot(x.*ux, y.*uy)) == vdot(x, y)*ux*uy

        @test @inferred(vdot(xc, y)) ≈ dot(xc, y)
        @test @inferred(vdot(xc.*ux, y)) == vdot(xc, y)*ux
        @test @inferred(vdot(xc, y.*uy)) == vdot(xc, y)*uy
        @test @inferred(vdot(xc.*ux, y.*uy)) == vdot(xc, y)*ux*uy

        @test @inferred(vdot(x, yc)) ≈ dot(x, yc)
        @test @inferred(vdot(x.*ux, yc)) == vdot(x, yc)*ux
        @test @inferred(vdot(x, yc.*uy)) == vdot(x, yc)*uy
        @test @inferred(vdot(x.*ux, yc.*uy)) == vdot(x, yc)*ux*uy

        @test @inferred(vdot(xc, yc)) ≈ dot(xc, yc)
        @test @inferred(vdot(xc.*ux, yc)) == vdot(xc, yc)*ux
        @test @inferred(vdot(xc, yc.*uy)) == vdot(xc, yc)*uy
        @test @inferred(vdot(xc.*ux, yc.*uy)) == vdot(xc, yc)*ux*uy

        @test @inferred(vdot(Float32, x, y)) === Float32(vdot(x, y))

        # Test vdot with weights
        @test @inferred(vdot(w, x, y)) ≈ sum(w .* x .* y)
        @test @inferred(vdot(w, x.*ux, y)) == vdot(w, x, y)*ux
        @test @inferred(vdot(w, x, y.*uy)) == vdot(w, x, y)*uy
        @test @inferred(vdot(w, x.*ux, y.*uy)) == vdot(w, x, y)*ux*uy

        @test @inferred(vdot(w, xc, y)) ≈ sum(w .* conj.(xc) .* y)
        @test @inferred(vdot(w, xc.*ux, y)) == vdot(w, xc, y)*ux
        @test @inferred(vdot(w, xc, y.*uy)) == vdot(w, xc, y)*uy
        @test @inferred(vdot(w, xc.*ux, y.*uy)) == vdot(w, xc, y)*ux*uy

        @test @inferred(vdot(w, x, yc)) ≈ sum(w .* x .* yc)
        @test @inferred(vdot(w, x.*ux, yc)) == vdot(w, x, yc)*ux
        @test @inferred(vdot(w, x, yc.*uy)) == vdot(w, x, yc)*uy
        @test @inferred(vdot(w, x.*ux, yc.*uy)) == vdot(w, x, yc)*ux*uy

        @test @inferred(vdot(w, xc, yc)) ≈ sum(w .* conj.(xc) .* yc)
        @test @inferred(vdot(w, xc.*ux, yc)) == vdot(w, xc, yc)*ux
        @test @inferred(vdot(w, xc, yc.*uy)) == vdot(w, xc, yc)*uy
        @test @inferred(vdot(w, xc.*ux, yc.*uy)) == vdot(w, xc, yc)*ux*uy

        @test @inferred(vdot(Float32, w, x, y)) === Float32(vdot(w, x, y))

        # Test with scalar inputs
        @test @inferred(vdot(2.0, 3.0)) == 6.0
        @test @inferred(vdot(2.0, 3.0 + 1im)) == 6.0 + 2.0im
        @test @inferred(vdot(2.0 + 1im, 3.0)) == 6.0 - 3.0im
        @test @inferred(vdot(2.0 + 1im, 3.0 + 1im)) == conj(2.0 + 1im) * (3.0 + 1im)
    end

    @testset "Vector Operations" begin
        x = [5.0, -2.0,  3.0]
        y = [2.0,  4.0, -1.0]
        u = cm/s

        # Test vcopy and vcopy!
        z = similar(x)
        @test @inferred(vcopy(x)) == x
        @test @inferred(vcopy!(z, x)) === z
        @test z == x

        # Test vfill!
        @test @inferred(vfill!(z, 2.0)) === z
        @test all(z .== 2.0)

        # Test vzeros! and vzeros
        @test @inferred(vzeros!(z)) === z
        @test all(iszero, z)
        let t = @inferred(vzeros(x))
            @test typeof(t) === typeof(x)
            @test all(iszero, t)
        end
        @test @inferred(vzeros(x)) == fill(0.0, size(x))
        @test @inferred(vzeros(x.*u)) == fill(0.0*u, size(x))

        # Test vones
        @test @inferred(vones(x)) == fill(1.0, size(x))
        @test @inferred(vones(x.*u)) == fill(1.0*u, size(x))

        # Test vnans
        @test all(xy -> isequal(xy...), zip(@inferred(vnans(x)), fill(NaN, size(x))))
        @test all(xy -> isequal(xy...), zip(@inferred(vnans(x.*u)), fill(NaN*u, size(x))))

        # Test vscale and vscale!
        @testset "`vscale` and `vscale!` with α=$α" for α in (0, 1, -1, 2 #=, 𝟘, 𝟙, -𝟙 =#)
            @test @inferred(vscale(α, x)) == @inferred(vscale(x, α))
            @test @inferred(vscale(α, x)) ≈ α .* x
            @test @inferred(vscale(x, α)) ≈ x .* α
            @test @inferred(vscale!(z, α, x)) === z
            @test z ≈ α .* x
        end

        # Test vproduct and vproduct!
        @test @inferred(vproduct(x, y)) ≈ x .* y
        @test @inferred(vproduct!(z, x, y)) === z
        @test z ≈ x .* y

        # Test vupdate!
        @testset "`vupdate!` with α=$α" for α in (-1, 0, 1, 2)
            @test @inferred(vupdate!(vcopy!(z, y), α, x)) === z
            @test z ≈ y .+ α .* x
        end

        # Test vcombine!
        @testset "`vcombine` and `vcombine!` with α=$α and β=$β" for α in (-1, 0, 1, 2), β in (-1, 0, 1, 2)
            @test @inferred(vcombine(α, x, β, y)) ≈ α .* x .+ β .* y
            @test @inferred(vcombine!(z, α, x, β, y)) === z
            @test z ≈ α .* x .+ β .* y
            @test @inferred(vcombine!(α, x, β, vcopy!(z, y))) === z
            @test z ≈ α .* x .+ β .* y
        end

        # Test vswap!
        z1 = @inferred(vcopy(x))
        z2 = @inferred(vcopy(y))
        vswap!(z1, z2)
        @test z1 == y && z2 == x
    end

    @testset "Selected Indices Operations" begin
        x = [1.0, -2.0,  3.0, -4.0]
        y = [2.0,  1.0, -1.0,  5.0]
        sel = [1, 3]  # selected indices
        msk = ones(Bool, size(x))
        msk[sel] .= false

        # Test vdot with selected indices
        @test @inferred(vdot(sel, x, y)) ≈ sum(x[sel] .* y[sel])

        # Test vproduct! with selected indices
        z = @inferred(vzeros(x))
        @test @inferred(vproduct!(z, sel, x, y)) === z
        @test z[sel] ≈ x[sel] .* y[sel]
        @test all(iszero, z[msk])

        # Test vupdate! with selected indices
        @testset "`vupdate!` with `sel` and α=$α" for α in (-1, 0, 1, 2)
            z = @inferred(vcopy(y))
            @test @inferred(vupdate!(z, sel, α, x)) === z
            @test z[sel] ≈ y[sel] .+ α * x[sel]
            @test z[msk] == y[msk]
        end
    end
end
nothing
