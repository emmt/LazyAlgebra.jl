#
#
# vectors.jl -
#
# Tests for vectorized operations.
#

@testset "Vectors" begin
    dims = (3,4,5)
    @testset "vfill ($T)" for T in (Float16, Float32, Float64)
        a = randn(T, dims)
        @test (0,0) == extrema(vfill!(a,0) - zeros(T,dims))
        @test (0,0) == extrema(vfill!(a,1) - ones(T,dims))
        @test (0,0) == extrema(vfill!(a,π) - fill!(similar(a), π))
    end
    @testset "vscale ($T)" for T in (Float16, Float32, Float64)
        a = randn(T, dims)
        b = vcreate(a)
        for α in (0, -1, 1, π, 2.71)
            @test (0,0) == extrema(vscale!(b,α,a) - T(α)*a)
        end
    end
    @testset "vupdate ($T)" for T in (Float16, Float32, Float64)
        a = randn(T, dims)
        d = randn(T, dims)
        for α in (0, -1, 1, π, 2.71)
            # FIXME: tighten this tolerance
            ϵ = sqrt(eps(T))*(maximum(abs.(a)) + abs(α)*maximum(abs.(d)))
            @test maxrelabsdif(vupdate!(vcopy(a),α,d), a + T(α)*d) ≤ ϵ
        end
    end
    @testset "vproduct ($T)" for T in (Float16, Float32, Float64)
        a = randn(T, dims)
        b = randn(T, dims)
        c = vcreate(a)
        @test (0,0) == extrema(vproduct!(c,a,b) - (a .* b))
    end
    @testset "vcombine ($T)" for T in (Float16, Float32, Float64)
        a = randn(T, dims)
        b = randn(T, dims)
        d = vcreate(a)
        for α in (0, -1, 1, π,  2.71),
            β in (0, -1, 1, φ, -1.33)
            @test (0,0) == extrema(vcombine!(d,α,a,β,b) - (T(α)*a + T(β)*b))
        end
    end
end
