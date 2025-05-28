"""

# Tests for cropping and zero-padding

Typical usage:

    include("test/crop-tests.jl").runtests();

"""
module LazyAlgebraCropTests

using LazyAlgebra
using Random
using Test
using TypeUtils

include("common.jl")

infer_output_eltype(::Type{<:CroppingOperator}, ::Type{x}) where {x<:AbstractArray} =
    eltype(x)
infer_output_eltype(::Type{<:ZeroPaddingOperator}, ::Type{x}) where {x<:AbstractArray} =
    eltype(x)

# Private methods for testing.
default_offset(inner::ArrayShape{N}, outer::ArrayShape{N}) where {N} =
    ntuple(i -> (as_array_dim(outer[i]) ÷ 2) - (as_array_dim(inner[i]) ÷ 2), Val(N))

function crop(A::AbstractArray{T,N}, shape::ArrayShape{N},
              off::NTuple{N,Integer} = default_offset(shape, axes(A))) where {T,N}
    return crop!(new_array(T, shape), A, off)
end

function zeropad(A::AbstractArray{T,N}, shape::ArrayShape{N},
                 off::NTuple{N,Integer} = default_offset(axes(A), shape)) where {T,N}
    return zeropad!(new_array(T, shape), A, off)
end

function subregionindices(inner::AbstractArray{<:Any,N},
                          outer::AbstractArray{<:Any,N},
                          off::NTuple{N,Integer}) where {N}
    I = CartesianIndices(inner) # indices in smallest region
    J = CartesianIndices(outer) # indices in largest region
    k = CartesianIndex(off)     # offset index
    (first(J) ≤ first(I) + k && last(I) + k ≤ last(J)) || error("out of range sub-region")
    return I, J, k
end

function crop!(y::AbstractArray{T,N},
               x::AbstractArray{<:Any,N},
               off::NTuple{N,Integer} = default_offset(axes(x), axes(y))) where {T,N}
    I, J, k = subregionindices(y, x, off)
    @inbounds @simd for i ∈ I
        y[i] = x[i + k]
    end
    return y
end

function zeropad!(y::AbstractArray{T,N},
                  x::AbstractArray{<:Any,N},
                  off::NTuple{N,Integer} = default_offset(axes(y), axes(x)),
                  init::Bool = false) where {T,N}
    I, J, k = subregionindices(x, y, off)
    fill!(y, zero(eltype(y)))
    @inbounds @simd for i ∈ I
        y[i + k] = x[i]
    end
    return y
end

const DimsPair{N} = Tuple{Pair{Dims{N},Dims{N}}}

function runtests(; rng::AbstractRNG = MersenneTwister(314159),
                  alphas::Tuple{Vararg{Number}} = (-1, 0, 1, 3, -2 + 1im),
                  betas::Tuple{Vararg{Number}} = (-1, 0, 1, 2, π),
                  sizes = ((4,) => (7,),
                           (4,) => (8,),
                           (5,) => (7,),
                           (5,) => (8,),
                           (3, 4) => (4, 9),
                           (2, 3, 4) => (5, 3, 6)),
                  eltypes::Tuple{Vararg{Type}} = (Float64, Complex{Float32}),
                  # NOTE Tolerance must not be too tight if we mix single and double precision.
                  rtol = 4e-7)

    @testset "Cropping and zero-padding" begin
        @testset "dims=$(inner_dims)=>$(outer_dims)" for (inner_dims, outer_dims) in sizes
            N = length(outer_dims)
            off = default_offset(inner_dims, outer_dims)
            C = @inferred CroppingOperator(inner_dims, outer_dims)
            @test C === @inferred CroppingOperator(inner_dims, outer_dims, off)
            Z = @inferred ZeroPaddingOperator(outer_dims, inner_dims)
            @test Z === @inferred ZeroPaddingOperator(outer_dims, inner_dims, off)
            @test Z === C'
            @test C' === Z
            @test LazyAlgebra.InputShape(C) === LazyAlgebra.HasInputShape{N}()
            @test LazyAlgebra.OutputShape(C) === LazyAlgebra.HasOutputShape{N}()
            @test LazyAlgebra.InputShape(Z) === LazyAlgebra.HasInputShape{N}()
            @test LazyAlgebra.OutputShape(Z) === LazyAlgebra.HasOutputShape{N}()
            @test LazyAlgebra.input_axes(C) == as_array_axes(outer_dims)
            @test LazyAlgebra.output_axes(C) == as_array_axes(inner_dims)
            @test LazyAlgebra.input_axes(Z) == as_array_axes(inner_dims)
            @test LazyAlgebra.output_axes(Z) == as_array_axes(outer_dims)
            off1 = map((inner,outer) -> inner < outer ? 1 : 0, inner_dims, outer_dims)
            C1 = @inferred CroppingOperator(inner_dims, outer_dims, off1)
            Z1 = @inferred ZeroPaddingOperator(outer_dims, inner_dims, off1)
            @test Z1 === C1'
            @test Z1' === C1
            @testset "T=$T" for T in eltypes
                x = rand(rng, T, outer_dims)
                xsav = copy(x)
                Cx = C*x
                @test x == xsav
                @test eltype(Cx) === eltype(x)
                @test Cx == crop(x, inner_dims)
                @test Cx == crop(x, inner_dims, off)
                @test C1*x == crop(x, inner_dims, off1)
                y = rand(rng, T, inner_dims)
                ysav = copy(y)
                Zy = Z*y
                @test y == ysav
                @test eltype(Zy) === eltype(y)
                @test Zy == zeropad(y, outer_dims)
                @test Zy == zeropad(y, outer_dims, off)
                @test Z1*y == zeropad(y, outer_dims, off1)

                @testset "α*C*x with α=$α" for α in alphas
                    αCx = @inferred(vmul(α, C, x))
                    @test x == xsav
                    @test eltype(αCx) == infer_output_eltype(α,C,x)
                    @test αCx ≈ α*Cx rtol=rtol
                end

                @testset "α*C*x + β*y with α=$α and β=$β" for α in alphas, β in betas
                    z = similar(y, infer_output_eltype(α, C, x, β, y))
                    iszero(β) ? vnans!(z) : vcopy!(z, y) # fill with NaNs if values not to be used
                    @test @inferred(vmul!(α, C, x, β, z)) === z
                    @test x == xsav
                    @test z ≈ α*Cx + β*y rtol=rtol
                end

                @testset "α*C'*y with α=$α" for α in alphas
                    αZy = @inferred(vmul(α, Z, y))
                    @test y == ysav
                    @test eltype(αZy) == infer_output_eltype(α, Z, y)
                    @test αZy ≈ α*Zy rtol=rtol
                end

                @testset "α*C'*y + β*x with α=$α and β=$β" for α in alphas, β in betas
                    z = similar(x, infer_output_eltype(α, Z, y, β, x))
                    iszero(β) ? vnans!(z) : vcopy!(z, x) # fill with NaNs if values not to be used
                    @test @inferred(vmul!(α, Z, y, β, z)) === z
                    @test y == ysav
                    @test z ≈ α*Zy + β*x rtol=rtol
                end

            end
        end
    end
end

end # module
