module CroppingTests

using Test
using LazyAlgebra

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


let TYPES = (Float64, Complex{Float32}),
    ALPHAS = (0, 1, -1,  2.71, π),
    BETAS = (0, 1, -1, -1.33, Base.MathConstants.φ)

    @testset "Cropping and zero-padding" begin
        for (osz, isz) in (((3,), (8,)),
                           ((4,), (8,)),
                           ((3,), (7,)),
                           ((4,), (7,)),
                           ((4,5), (6,7))),
            T in TYPES
            x = rand(T, isz)
            y = rand(T, osz)
            C = CroppingOperator(osz, isz)
            @test input_size(C) == isz
            @test output_size(C) == osz
            @test C*x == crop(x, osz)
            @test C'*y == zeropad(y, isz)
        end
    end
end

end # module
