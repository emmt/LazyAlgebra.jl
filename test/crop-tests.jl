#
# crop-tests.jl -
#
# Tests for cropping and zero-padding.
#

using Test
using LazyAlgebra

@testset "Cropping and zero-padding" begin
    #
    # Private methods for testing.
    #
    include("common.jl")

    offset(outer::NTuple{N,Int}, inner::NTuple{N,Int}) where {N} =
        ntuple(i -> (outer[i]>>1) - (inner[i]>>1), Val(N))

    function crop(x::AbstractArray{T,N}, siz::NTuple{N,Int}) where {T,N}
        return crop!(Array{T,N}(undef, siz), x)
    end

    function crop(x::AbstractArray{T,N}, siz::NTuple{N,Int},
                  off::NTuple{N,Int}) where {T,N}
        return crop!(Array{T,N}(undef, siz), x, off)
    end

    function zeropad(x::AbstractArray{T,N}, siz::NTuple{N,Int}) where {T,N}
        return zeropad!(Array{T,N}(undef, siz), x)
    end

    function zeropad(x::AbstractArray{T,N}, siz::NTuple{N,Int},
                     off::NTuple{N,Int}) where {T,N}
        return zeropad!(Array{T,N}(undef, siz), x, off)
    end

    function subregionindices(sub::AbstractArray{<:Any,N},
                              big::AbstractArray{<:Any,N},
                              off::NTuple{N,Int}) where {N}
        I = CartesianIndices(sub) # indices in smallest region
        J = CartesianIndices(big) # indices in largest region
        k = CartesianIndex(off)   # offset index
        (first(J) ≤ first(I) + k && last(I) + k ≤ last(J)) ||
            error("out of range sub-region")
        return I, J, k
    end

    function crop!(y::AbstractArray{T,N},
                   x::AbstractArray{<:Any,N},
                   off::NTuple{N,Int} = offset(size(x), size(y))) where {T,N}
        I, J, k = subregionindices(y, x, off)
        @inbounds @simd for i ∈ I
            y[i] = x[i + k]
        end
        return y
    end

    function zeropad!(y::AbstractArray{T,N},
                      x::AbstractArray{<:Any,N},
                      off::NTuple{N,Int} = offset(size(y), size(x)),
                      init::Bool = false) where {T,N}
        I, J, k = subregionindices(x, y, off)
        init || fill!(y, zero(T))
        @inbounds @simd for i ∈ I
            y[i+k] = x[i]
        end
        return y
    end

    #
    # Tests for different sizes and element types.
    #
    for (osz, isz) in (((3,), (8,)),
                       ((4,), (8,)),
                       ((3,), (7,)),
                       ((4,), (7,)),
                       ((4,5), (6,7))),
        T in (Float64, Complex{Float32})
        x = rand(T, isz)
        y = rand(T, osz)
        C = CroppingOperator(osz, isz)
        off = ntuple(i->Int16(isz[i] - osz[i])>>1, length(isz))
        @test ZeroPaddingOperator(isz, osz) === Adjoint(C)
        @test ZeroPaddingOperator(isz, osz, off) === Adjoint(CroppingOperator(osz, isz, off))
        @test input_size(C) == isz
        @test output_size(C) == osz
        @test C*x == crop(x, osz)
        @test C'*y == zeropad(y, isz)
        test_api(Direct, C, x, y)
        test_api(Adjoint, C, x, y)
    end
end
nothing
