"""

# Tests for diagonal operator

Typical usage:

    include("test/diag-tests.jl").runtests();

"""
module LazyAlgebraDiagTests

using LazyAlgebra
using Test
using Random
using Neutrals
using TypeUtils
using LinearAlgebra

include("common.jl")

function runtests(; rng::AbstractRNG = MersenneTwister(314159),
                  alphas::Tuple{Vararg{Number}} = (-1, 0, 1, 3, -2 + 1im),
                  betas::Tuple{Vararg{Number}} = (-1, 0, 1, 2, π),
                  sizes::Tuple{Vararg{Dims}} = ((3, 4), (2, 3, 4)),
                  eltypes::Tuple{Vararg{Type}} = (Float32, Float64, Complex{Float32}),
                  # NOTE Tolerance must not be too tight if we mix single and double precision.
                  rtol = 4e-7)

    @testset "Diagonal operator" begin
        @testset "Diagonal and identity" begin
            @test Diag(Id) === Id
            @test Diag(I) === I.λ*Id
            @test Diag(2Id) === 2*Id
            @test typeof(diag(Id)) <: Array{typeof(𝟙),0}
            @test typeof(diag(Identity(2,3))) <: Array{typeof(𝟙),2}
            @test size(diag(Identity(2,3))) == (2,3)
        end

        @testset "dims=$(dims), T=$T" for dims in sizes, T in eltypes
            # Generate array of coefficients making sure they are all non-zero to
            # implement inverse.
            w = shift_values!(-0.5, rand(rng, T, dims))
            w[iszero.(abs.(w))] .= 0.3

            # Build operator.
            D = @inferred(Diag(w))

            # Check operator properties.
            @test D isa Diag{typeof(w)}
            @test eltype(D) === eltype(w)
            @test LazyAlgebra.InputEltype(D) === LazyAlgebra.InputEltypeUnknown()
            @test LazyAlgebra.OutputEltype(D) === LazyAlgebra.OutputEltypeUnknown()
            @test LazyAlgebra.InputShape(D) === LazyAlgebra.HasInputShape{ndims(w)}()
            @test LazyAlgebra.OutputShape(D) === LazyAlgebra.HasOutputShape{ndims(w)}()
            @test LazyAlgebra.input_axes(D) == axes(w)
            # FIXME @test LazyAlgebra.input_size(D) === size(w)
            @test LazyAlgebra.output_axes(D) == axes(w)
            @test_throws Exception LazyAlgebra.output_axes(D, [1])
            # FIXME @test LazyAlgebra.output_size(D) === size(w)

            # Set precision.
            if real_type(eltype(D)) <: AbstractFloat
                @test get_precision(D) === real_type(eltype(D))
                @test @inferred(with_precision(real_type(eltype(D)), D)) === D
            else
                @test get_precision(D) === AbstractFloat
            end
            Tp = real(eltype(D)) === Float32 ? Float64 : Float32
            Dp = @inferred(with_precision(Tp, D))
            @test real(eltype(Dp)) === Tp
            @test diag(Dp) ≈ diag(D) rtol=rtol

            # Loop over possible element types for x and y.
            for (Tx,Ty) in ((Float64, Float32), (Complex{Float32}, Float64))
                # Input and output vectors.
                x = shift_values!(-0.5, rand(rng, Tx, size(w)))
                y = shift_values!(-0.5, rand(rng, Ty, size(w)))

                # Output sizes.
                @test LazyAlgebra.output_axes(D, x) == axes(w)
                @test LazyAlgebra.output_axes(D', x) == axes(w)
                @test LazyAlgebra.output_axes(inv(D), x) == axes(w)
                @test LazyAlgebra.output_axes(inv(D'), x) == axes(w)

                # Inferred types.
                @test @inferred(LazyAlgebra.output_eltype(    D,   x)) == infer_output_eltype(    D,   x)
                @test @inferred(LazyAlgebra.output_eltype(    D',  x)) == infer_output_eltype(    D',  x)
                @test @inferred(LazyAlgebra.output_eltype(inv(D),  x)) == infer_output_eltype(inv(D),  x)
                @test @inferred(LazyAlgebra.output_eltype(inv(D'), x)) == infer_output_eltype(inv(D'), x)

                # Apply direct operator.
                z0 = w.*x
                z = @inferred(vmul(D, x))
                @test eltype(z) == infer_output_eltype(D, x)
                @test axes(z) == axes(w)
                @test z ≈ z0 rtol=rtol
                @test z == D*x
                @test z == D(x)
                @test @inferred(vmul!(vnans!(z), D, x)) === z
                @test z ≈ z0 rtol=rtol

                # Apply adjoint of operator.
                z0 = conj.(w).*x
                z = @inferred(vmul(D', x))
                @test eltype(z) == infer_output_eltype(D', x)
                @test axes(z) == axes(w)
                @test z ≈ z0 rtol=rtol
                @test z == D'*x
                @test z == D'(x)
                @test @inferred(vmul!(vnans!(z), D', x)) === z
                @test z ≈ z0 rtol=rtol

                # Apply inverse of operator.
                z0 = w.\x
                z = @inferred(vmul(inv(D), x))
                @test eltype(z) == infer_output_eltype(inv(D), x)
                @test axes(z) == axes(w)
                @test z ≈ z0 rtol=rtol
                @test z == D\x
                @test z == inv(D)*x
                @test z == inv(D)(x)
                @test @inferred(vmul!(vnans!(z), inv(D), x)) === z
                @test z ≈ z0 rtol=rtol

                # Apply adjoint-inverse of operator.
                z0 = conj.(w).\x
                z = @inferred(vmul(inv(D'), x))
                @test eltype(z) == infer_output_eltype(inv(D'), x)
                @test axes(z) == axes(w)
                @test z ≈ z0 rtol=rtol
                @test z == D'\x
                @test z == inv(D')*x
                @test z == inv(D')(x)
                @test @inferred(vmul!(vnans!(z), inv(D'), x)) === z
                @test z ≈ z0 rtol=rtol

                @testset "α*D*x with α=$α" for α in alphas
                    z0 = α*w.*x
                    z = @inferred(vmul(α, D, x))
                    @test eltype(z) === infer_output_eltype(α, D, x)
                    @test axes(z) == axes(w)
                    @test z ≈ z0 rtol=rtol
                    @test @inferred(vmul!(vnans!(z), α, D, x)) === z
                    @test z ≈ z0 rtol=rtol
                end

                @testset "α*D'*x with α=$α" for α in alphas
                    z0 = α*conj.(w).*x
                    z = @inferred(vmul(α, D', x))
                    @test eltype(z) === infer_output_eltype(α, D', x)
                    @test axes(z) == axes(w)
                    @test z ≈ z0 rtol=rtol
                    @test @inferred(vmul!(vnans!(z), α, D', x)) === z
                    @test z ≈ z0 rtol=rtol
                end

                @testset "α*inv(D)*x with α=$α" for α in alphas
                    z0 = α*(w.\x)
                    z = @inferred(vmul(α, inv(D), x))
                    @test eltype(z) === infer_output_eltype(α, inv(D), x)
                    @test axes(z) == axes(w)
                    @test z ≈ z0 rtol=rtol
                    @test @inferred(vmul!(vnans!(z), α, inv(D), x)) === z
                    @test z ≈ z0 rtol=rtol
                end

                @testset "α*inv(D')*x with α=$α" for α in alphas
                    z0 = α*(conj.(w).\x)
                    z = @inferred(vmul(α, inv(D'), x))
                    @test eltype(z) === infer_output_eltype(α, inv(D'), x)
                    @test axes(z) == axes(w)
                    @test z ≈ z0 rtol=rtol
                    @test @inferred(vmul!(vnans!(z), α, inv(D'), x)) === z
                    @test z ≈ z0 rtol=rtol
                end

                @testset "α*D*x + β*y with α=$α and β=$β" for α in alphas, β in betas
                    z = similar(x, infer_output_eltype(α, D, x, β, y))
                    iszero(β) ? vnans!(z) : vcopy!(z, y) # fill with NaNs if values not to be used
                    @test @inferred(vmul!(α, D, x, β, z)) === z
                    @test z ≈ α*w.*x + β*y rtol=1e-7
                end

                @testset "α*D'*x + β*y with α=$α and β=$β" for α in alphas, β in betas
                    z = similar(x, infer_output_eltype(α, D', x, β, y))
                    iszero(β) ? vnans!(z) : vcopy!(z, y) # fill with NaNs if values not to be used
                    @test @inferred(vmul!(α, D', x, β, z)) === z
                    @test z ≈ α*conj.(w).*x + β*y rtol=1e-7
                end

                @testset "α*inv(D)*x + β*y with α=$α and β=$β" for α in alphas, β in betas
                    z = similar(x, infer_output_eltype(α, inv(D), x, β, y))
                    iszero(β) ? vnans!(z) : vcopy!(z, y) # fill with NaNs if values not to be used
                    @test @inferred(vmul!(α, inv(D), x, β, z)) === z
                    @test z ≈ α*(w.\x) + β*y rtol=1e-7
                end

                @testset "α*inv(D')*x + β*y with α=$α and β=$β" for α in alphas, β in betas
                    z = similar(x, infer_output_eltype(α, inv(D'), x, β, y))
                    iszero(β) ? vnans!(z) : vcopy!(z, y) # fill with NaNs if values not to be used
                    @test @inferred(vmul!(α, inv(D'), x, β, z)) === z
                    @test z ≈ α*(conj.(w).\x) + β*y rtol=1e-7
                end
            end
        end
    end
end
end # module

#isinteractive() && LazyAlgebraDiagTests.runtests()
