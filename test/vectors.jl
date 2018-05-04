#
# vectors.jl -
#
# Tests for vectorized operations.
#

@testset "Vectors" begin
    types = (Float16, Float32, Float64)
    dims = (3,4,5)
    @testset "vnorm ($T)" for T in types
        v = randn(T, dims)
        @test vnorminf(a) == maximum(abs.(a))
        @test vnorm1(a) ≈ sum(abs.(a))
        @test vnorm2(a) ≈ sqrt(sum(a.*a))
    end
    @testset "vcopy, vswap ($T)" for T in types
        u = randn(T, dims)
        uc = vcopy(u)
        @test extrema(u - uc) == (0, 0)
        v = randn(T, dims)
        vc = vcopy!(vcreate(v), v)
        @test extrema(v - vc) == (0, 0)
        vswap!(u, v)
        @test extrema(u - vc) == extrema(v - uc) == (0, 0)
    end
    @testset "vfill ($T)" for T in types
        a = randn(T, dims)
        @test (0,0) == extrema(vfill!(a,0) - zeros(T,dims))
        a = randn(T, dims)
        @test (0,0) == extrema(vfill!(a,0) - vzero(a))
        a = randn(T, dims)
        @test (0,0) == extrema(vfill!(a,1) - ones(T,dims))
        a = randn(T, dims)
        @test (0,0) == extrema(vfill!(a,π) - fill!(similar(a), π))
        ac = vcopy(a)
        @test (0,0) == extrema(vzeros(a) - zeros(T,dims))
        @test (0,0) == extrema(a - ac)
        @test (0,0) == extrema(vones(a) - ones(T,dims))
        @test (0,0) == extrema(a - ac)
    end
    @testset "vscale" begin
        for T in types
            a = randn(T, dims)
            b = vcreate(a)
            for α in (0, -1, 1, π, 2.71)
                @test (0,0) == extrema(vscale!(b,α,a) - T(α)*a)
            end
        end
        for Ta in types, Tb in types
            a = randn(Ta, dims)
            b = Array{Tb}(dims)
            e = max(eps(Ta), eps(Tb))
            for α in (0, -1, 1, π, 2.71)
                @test maximum(abs.(vscale!(b,α,a) - α*a)) ≤ 4*eps
            end
        end
    end
    @testset "vupdate ($T)" for T in types
        a = randn(T, dims)
        d = randn(T, dims)
        atol, rtol = zero(T), sqrt(eps(T))
        for α in (0, -1, 1, π, 2.71)
            @test vupdate!(vcopy(a),α,d) ≈
                a + T(α)*d atol=atol rtol=rtol norm=vnorm2
        end
    end
    @testset "vproduct ($T)" for T in types
        a = randn(T, dims)
        b = randn(T, dims)
        c = vcreate(a)
        @test (0,0) == extrema(vproduct!(c,a,b) - (a .* b))
    end
    @testset "vcombine ($T)" for T in types
        a = randn(T, dims)
        b = randn(T, dims)
        d = vcreate(a)
        for α in (0, -1, 1, π,  2.71),
            β in (0, -1, 1, φ, -1.33)
            @test (0,0) == extrema(vcombine!(d,α,a,β,b) - (T(α)*a + T(β)*b))
        end
    end
end
