module TestingLazyAlgebraPseudoMatrices

using LazyAlgebra
using Test
using Random
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

    @testset "Pseudo- and flexible matrices" begin
        @testset "dims=$(dims), M=$(M < length(dims) ? M : :(:)), T=$T" for dims in sizes,
            T in eltypes, M in 1:length(dims)

            # Generate array of coefficients and pseudo- or flexible matrix.
            C = shift_values!(-0.2, rand(rng, T, dims))
            N = length(dims) - M # number of column dimensions, 0 for flexible matrix
            A = N ≥ 1 ?
                @inferred(PseudoMatrix(C, Dims{M})) :
                @inferred(FlexibleMatrix(C))

            # Check operator properties.
            @test A isa (N ≥ 1 ? PseudoMatrix : FlexibleMatrix)
            @test eltype(A) === eltype(C)
            if N < 1 # FlexibleMatrix
                @test LazyAlgebra.InputEltype(A) === LazyAlgebra.InputEltypeUnknown()
                @test LazyAlgebra.OutputEltype(A) === LazyAlgebra.OutputEltypeUnknown()
                @test LazyAlgebra.InputShape(A) === LazyAlgebra.InputShapeUnknown()
                @test LazyAlgebra.OutputShape(A) === LazyAlgebra.OutputShapeUnknown()
            else # PseudoMatrix
                @test LazyAlgebra.InputEltype(A) === LazyAlgebra.InputEltypeUnknown()
                @test LazyAlgebra.OutputEltype(A) === LazyAlgebra.OutputEltypeUnknown()
                @test LazyAlgebra.InputShape(A) === LazyAlgebra.HasInputShape{N}()
                @test LazyAlgebra.OutputShape(A) === LazyAlgebra.HasOutputShape{M}()
                @test LazyAlgebra.input_axes(A) == axes(C)[M+1:M+N]
                @test LazyAlgebra.output_axes(A) == axes(C)[1:M]
                # FIXME @test LazyAlgebra.input_size(A) === size(C)[M+1:M+N]
                # FIXME @test LazyAlgebra.output_size(A) === size(C)[1:M]
            end

            # Set precision.
            if T <: AbstractFloat
                @test get_precision(A) === T
                @test @inferred(with_precision(get_precision(A), A)) === A
                @test @inferred(with_precision(real(eltype(A)), A)) === A
            end
            Tp = real(eltype(A)) === Float32 ? Float64 : Float32
            Ap = @inferred(with_precision(Tp, A))
            @test real(eltype(Ap)) === Tp
            @test parent(Ap) ≈ parent(A) rtol=rtol

            # For a flexible matrix, loop over possible number of row dimensions.
            for l in (N < 1 ? (1:ndims(C)-1) : (M,))
                # Row and column axes.
                I = axes(C)[1:l]
                J = axes(C)[l+1:end]

                # Reshaped array equivalent to A.
                m = prod(map(length, I))
                n = prod(map(length, J))
                R = reshape(C, m, n)

                # Loop over possible element types for x and y.
                for (Tx,Ty) in ((Float64, Float32), (Complex{Float32}, Float64))
                    # Input and output vectors.
                    x = shift_values!(-0.5, rand(rng, Tx, as_array_size(J)))
                    y = shift_values!(-0.5, rand(rng, Ty, as_array_size(I)))

                    # Apply direct operator.
                    z = @inferred(vmul(A, x))
                    @test eltype(z) == infer_output_eltype(A, x)
                    @test axes(z) == I
                    @test flat(z) ≈ R*flat(x) rtol=rtol
                    @test z == A*x
                    @test z == A(x)
                    z0 = copy(z)
                    @test @inferred(vmul!(vnans!(z), A, x)) === z
                    @test z ≈ z0

                    # Apply adjoint operator.
                    z = @inferred(vmul(A', y))
                    @test eltype(z) == infer_output_eltype(A', y)
                    @test axes(z) == J
                    @test flat(z) ≈ R'*flat(y) rtol=rtol
                    @test z == A'*y
                    @test z == A'(y)
                    z0 = copy(z)
                    @test @inferred(vmul!(vnans!(z), A', y)) === z
                    @test z ≈ z0

                    @testset "α*A*x with α=$α" for α in alphas
                        z = @inferred(vmul(α, A, x))
                        @test eltype(z) == infer_output_eltype(α, A, x)
                        @test axes(z) == I
                        @test flat(z) ≈ α*(R*flat(x)) rtol=rtol
                        @test z == (α*A)*x
                        @test z == (α*A)(x)
                        z0 = copy(z)
                        @test @inferred(vmul!(vnans!(z), α, A, x)) === z
                        @test z ≈ z0
                    end

                    @testset "α*A'*y with α=$α" for α in alphas
                        z = @inferred(vmul(α, A', y))
                        @test eltype(z) == infer_output_eltype(α, A', y)
                        @test axes(z) == J
                        @test flat(z) ≈ α*(R'*flat(y)) rtol=rtol
                        @test z == (α*A')*y
                        @test z == (α*A')(y)
                        z0 = copy(z)
                        @test @inferred(vmul!(vnans!(z), α, A', y)) === z
                        @test z ≈ z0
                    end

                    @testset "α*A*x + β*y with α=$α and β=$β" for α in alphas, β in betas
                        z = similar(y, infer_output_eltype(α, A, x, β, y))
                        iszero(β) ? vnans!(z) : vcopy!(z, y) # fill with NaNs if values not to be used
                        @test @inferred(vmul!(α, A, x, β, z)) === z
                        @test flat(z) ≈ α*(R*flat(x)) + β*flat(y) rtol=rtol
                    end

                    @testset "α*A'*y + β*x with α=$α and β=$β" for α in alphas, β in betas
                        z = similar(x, infer_output_eltype(α, A', y, β, x))
                        iszero(β) ? vnans!(z) : vcopy!(z, x) # fill with NaNs if values not to be used
                        @test @inferred(vmul!(α, A', y, β, z)) === z
                        @test flat(z) ≈ α*(R'*flat(y)) + β*flat(x) rtol=rtol
                    end
                end
            end
        end
    end
end
end # module

#isinteractive() && TestingLazyAlgebraPseudoMatrices.runtests()
