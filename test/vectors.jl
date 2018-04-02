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
        α = 2.71
        @test (0,0) == extrema(vscale!(b, 0,a) - 0*a)
        @test (0,0) == extrema(vscale!(b, 1,a) - a)
        @test (0,0) == extrema(vscale!(b,-1,a) + a)
        @test (0,0) == extrema(vscale!(b, α,a) - T(α)*a)
        @test (0,0) == extrema(vscale!(b, π,a) - T(π)*a)
    end
    @testset "vupdate ($T)" for T in (Float16, Float32, Float64)
        a = randn(T, dims)
        d = randn(T, dims)
        b = vcreate(a)
        α = 2.71
        ϵ = eps(T)
        @test (0,0) == extrema(vupdate!(vcopy!(b,a), 0,d) - a)
        @test (0,0) == extrema(vupdate!(vcopy!(b,a), 1,d) - (a + d))
        @test (0,0) == extrema(vupdate!(vcopy!(b,a),-1,d) - (a - d))
        @test 10ϵ ≥ maxrelabsdif(vupdate!(vcopy!(b,a),α,d), a + T(α)*d)
        @test 10ϵ ≥ maxrelabsdif(vupdate!(vcopy!(b,a),π,d), a + T(π)*d)
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
        α =  2.71
        β = -1.33
        @test (0,0) == extrema(vcombine!(d, 0,a, 0,b) - zeros(T,dims))
        @test (0,0) == extrema(vcombine!(d, 0,a, 1,b) - b)
        @test (0,0) == extrema(vcombine!(d, 0,a,-1,b) + b)
        @test (0,0) == extrema(vcombine!(d, 0,a, β,b) - T(β)*b)
        @test (0,0) == extrema(vcombine!(d, 0,a, π,b) - T(π)*b)
        @test (0,0) == extrema(vcombine!(d, 1,a, 0,b) - a)
        @test (0,0) == extrema(vcombine!(d,-1,a, 0,b) + a)
        @test (0,0) == extrema(vcombine!(d, α,a, 0,b) - T(α)*a)
        @test (0,0) == extrema(vcombine!(d, π,a, 0,b) - T(π)*a)
        @test (0,0) == extrema(vcombine!(d, 1,a, 1,b) - (a + b))
        @test (0,0) == extrema(vcombine!(d,-1,a, 1,b) - (b - a))
        @test (0,0) == extrema(vcombine!(d, 1,a,-1,b) - (a - b))
        @test (0,0) == extrema(vcombine!(d,-1,a,-1,b) + (a + b))
        @test (0,0) == extrema(vcombine!(d, α,a, β,b) - (T(α)*a + T(β)*b))
        @test (0,0) == extrema(vcombine!(d, α,a, φ,b) - (T(α)*a + T(φ)*b))
        @test (0,0) == extrema(vcombine!(d, π,a, β,b) - (T(π)*a + T(β)*b))
        @test (0,0) == extrema(vcombine!(d, π,a, φ,b) - (T(π)*a + T(φ)*b))
    end
end
