#
# vectors.jl -
#
# Tests for vectorized operations.
#

isdefined(:LazyAlgebra) || include("../src/LazyAlgebra.jl")

module LazyAlgebraVectorsTests

using LazyAlgebra

@static if VERSION < v"0.7.0-DEV.2005"
    using Base.Test
else
    using Test
end

distance(a::Real, b::Real) = abs(a - b)

distance(a::NTuple{2,Real}, b::NTuple{2,Real}) =
    hypot(a[1] - b[1], a[2] - b[2])

distance(A::AbstractArray{Ta,N}, B::AbstractArray{Tb,N}) where {Ta,Tb,N} =
    maximum(abs.(A - B))

@testset "Vectors" begin
    types = (Float32, Float64)
    dims = (3,4,5)
    @testset "vnorm ($T)" for T in types
        v = randn(T, dims)
        @test vnorminf(v) == maximum(abs.(v))
        @test vnorm1(v) ≈ sum(abs.(v))
        @test vnorm2(v) ≈ sqrt(sum(v.*v))
    end
    @testset "vcopy, vswap ($T)" for T in types
        u = randn(T, dims)
        uc = vcopy(u)
        @test distance(u, uc) == 0
        v = randn(T, dims)
        vc = vcopy!(vcreate(v), v)
        @test distance(v, vc) == 0
        vswap!(u, v)
        @test distance(u, vc) == distance(v, uc) == 0
    end
    @testset "vfill ($T)" for T in types
        a = randn(T, dims)
        @test distance(vfill!(a,0), zeros(T,dims)) == 0
        a = randn(T, dims)
        @test distance(vfill!(a,0), vzero!(a)) == 0
        a = randn(T, dims)
        @test distance(vfill!(a,1), ones(T,dims)) == 0
        a = randn(T, dims)
        @test distance(vfill!(a,π), fill!(similar(a), π)) == 0
        ac = vcopy(a)
        @test distance(vzeros(a), zeros(T,dims)) == 0
        @test distance(a, ac) == 0
        @test distance(vones(a), ones(T,dims)) == 0
        @test distance(a, ac) == 0
    end
    @testset "vscale" begin
        for T in types
            a = randn(T, dims)
            b = vcreate(a)
            for α in (0, -1, 1, π, 2.71)
                d = T(α)*a
                @test distance(vscale(α,a), d) == 0
                @test distance(vscale!(b,α,a), d) == 0
            end
        end
        for Ta in types, Tb in types
            a = randn(Ta, dims)
            ac = vcopy(a)
            b = Array{Tb}(dims)
            e = max(eps(Ta), eps(Tb))
            for α in (0, -1, 1, π, 2.71)
                d = α*a
                @test distance(vscale!(b,α,a), d) ≤ 8e
                @test distance(vscale(α,a), d) ≤ 8e
                @test distance(a, ac) == 0
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
    @testset "vproduct ($Ta,$Tb)" for Ta in types, Tb in types
        a = randn(Ta, dims)
        b = randn(Tb, dims)
        c = vcreate(a)
        e = max(eps(Ta), eps(Tb))
        @test distance(vproduct!(c,a,b), (a .* b)) ≤ 2e
    end
    @testset "vcombine ($T)" for T in types
        a = randn(T, dims)
        b = randn(T, dims)
        d = vcreate(a)
        for α in (0, -1, 1, π,  2.71),
            β in (0, -1, 1, φ, -1.33)
            @test distance(vcombine!(d,α,a,β,b), (T(α)*a + T(β)*b)) == 0
        end
    end
end

end # module
