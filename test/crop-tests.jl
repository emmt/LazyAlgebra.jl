#
# crop-tests.jl -
#
# Tests for cropping and zero-padding.
#
module TestingLazyAlgebraCrop

using Test
using LazyAlgebra
using LazyAlgebra.LazyAlgebraLowLevel

@testset "Cropping and zero-padding" begin
    #
    # Private methods for testing.
    #

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
    # Miscellaneous tests.
    #
    @test_throws ErrorException CroppingOperator((2,3,), (2,3,4,))
    @test_throws ErrorException CroppingOperator((2,3,), (2,3,4,), (0,1,))

    #
    # Tests for different sizes and element types.
    #
    for (osz, isz) in (((3,), (8,)),
                       ((4,), (8,)),
                       ((3,), (7,)),
                       ((4,), (7,)),
                       ((4,5), (6,7))),
        T in (Float64, Complex{Float32})

        # Basics methods.
        R = real(T)
        C = CroppingOperator(osz, isz)
        off = ntuple(i->Int16(isz[i] - osz[i])>>1, length(isz))
        @test ZeroPaddingOperator(isz, osz) === Adjoint(C)
        @test ZeroPaddingOperator(isz, osz, off) === Adjoint(CroppingOperator(osz, isz, off))
        @test input_ndims(C) == length(isz)
        @test input_size(C) == isz
        @test all(i -> input_size(C,i) == isz[i], 1:length(isz))
        @test output_ndims(C) == length(osz)
        @test output_size(C) == osz
        @test all(i -> output_size(C,i) == osz[i], 1:length(osz))

        # Compare result of opertaor with reference implementation.
        x = rand(T, isz)
        xsav = vcopy(x)
        Cx = C*x
        @test x == xsav
        @test Cx == crop(x, osz)
        y = rand(T, osz)
        ysav = vcopy(y)
        Cty = C'*y
        @test y == ysav
        @test Cty == zeropad(y, isz)

        # Test various possibilities for apply!
        atol = 0
        rtol = eps(R)
        for α in (0, 1, -1,  2.71, π),
            β in (0, 1, -1, -1.33, Base.MathConstants.φ),
            scratch in (false, true)
            # Test operator.
            @test apply!(α, Direct, C, x, scratch, β, vcopy(y)) ≈
                R(α)*Cx + R(β)*y  atol=atol rtol=rtol
            if scratch
                vcopy!(x, xsav)
            else
                @test x == xsav
            end
            # Test  adjoint.
            @test apply!(α, Adjoint, C, y, scratch, β, vcopy(x)) ≈
                R(α)*Cty + R(β)*x  atol=atol rtol=rtol
            if scratch
                vcopy!(y, ysav)
            else
                @test y == ysav
            end
        end
    end
end
nothing

end # module
