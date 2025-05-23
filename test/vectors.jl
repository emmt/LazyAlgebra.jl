using LazyAlgebra
using Test
using Neutrals
using LinearAlgebra

@testset "Vectorized operations in `LazyAlgebra`" begin
    @testset "Vector Norms" begin
        # Test vectors
        x = [1.0, -2.0, 3.0, -4.0]
        y = [2.0 + 1im, -1.0 - 2im]

        # Test vnorm1
        @test @inferred(vnorm1(x)) ≈ sum(abs.(x))
        @test @inferred(vnorm1(y)) ≈ sum(abs.(y))
        let v = @inferred(vnorm1(Float32, x))
            @test v isa Float32
            @test v ≈ Float32(sum(abs.(x)))
        end

        # Test vnorm2
        @test @inferred(vnorm2(x)) ≈ sqrt(mapreduce(abs2, +, x))
        @test @inferred(vnorm2(y)) ≈ sqrt(mapreduce(abs2, +, y))
        let v = @inferred(vnorm2(Float32, x))
            @test v isa Float32
            @test v ≈ Float32(sqrt(mapreduce(abs2, +, x)))
        end

        # Test vnorminf
        @test @inferred(vnorminf(x)) ≈ mapreduce(abs, max, x)
        @test @inferred(vnorminf(y)) ≈ mapreduce(abs, max, y)
        let v = @inferred(vnorminf(Float32, x))
            @test v isa Float32
            @test v ≈ Float32(mapreduce(abs, max, x))
        end

        # Test with scalar inputs
        @test @inferred(vnorm1(-2.0)) == 2.0
        @test @inferred(vnorm1(-3.0 + 4.0im)) == 5.0
        @test @inferred(vnorm2(-2.0)) == 2.0
        @test @inferred(vnorm2(-3.0 + 4.0im)) == 5.0
        @test @inferred(vnorminf(-2.0)) == 2.0
        @test @inferred(vnorminf(-3.0 + 4.0im)) == 5.0
    end

    @testset "Inner Products" begin
        # Test real vectors
        x = [1.0, -2.0, 3.0]
        y = [2.0, 1.0, -1.0]
        w = [0.5, 1.0, 2.0]  # weights

        # Test vdot without weights
        @test @inferred(vdot(x, y)) ≈ dot(x, y)
        @test @inferred(vdot(Float32, x, y)) ≈ Float32(dot(x, y))

        # Test vdot with weights
        @test @inferred(vdot(w, x, y)) ≈ sum(w .* x .* y)
        @test @inferred(vdot(Float32, w, x, y)) ≈ Float32(sum(w .* x .* y))

        # Test complex vectors
        xc = [1.0 + 1im, -2.0 - 2im]
        yc = [2.0 - 1im, 1.0 + 1im]

        @test @inferred(vdot(xc, yc)) ≈ dot(xc, yc)
        @test @inferred(vdot(w[1:2], xc, yc)) ≈ sum(w[1:2] .* conj.(xc) .* yc)

        # Test with scalar inputs
        @test @inferred(vdot(2.0, 3.0)) == 6.0
        @test @inferred(vdot(2.0, 3.0 + 1im)) == 6.0
        @test @inferred(vdot(2.0 + 1im, 3.0)) == 6.0
        @test @inferred(vdot(2.0 + 1im, 3.0 + 1im)) == conj(2.0 + 1im) * (3.0 + 1im)
    end

    @testset "Vector Operations" begin
        x = [5.0, -2.0,  3.0]
        y = [2.0,  4.0, -1.0]

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

        # Test vones
        @test @inferred(vones(x)) == [1.0, 1.0, 1.0]

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
