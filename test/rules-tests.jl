using LazyAlgebra
using Test
using LinearAlgebra

@testset "Arithmetic rules" begin
    @testset "Multipliers" begin
        let multiplier_type = LazyAlgebra.multiplier_type
            @test @inferred(multiplier_type(BigFloat, Vector{Float32})) === Float32
            @test @inferred(multiplier_type(Float64, AbstractMatrix{Complex{Float32}})) === Float32
            @test @inferred(multiplier_type(Int, AbstractMatrix{Complex{Float32}})) === Float32
            @test @inferred(multiplier_type(Rational, AbstractMatrix{Complex{Float32}})) === Float32
            @test @inferred(multiplier_type(Complex{Float64}, AbstractMatrix{Float32})) === Complex{Float32}
            @test @inferred(multiplier_type(Complex{Int}, AbstractMatrix{Complex{Float32}})) === Complex{Float32}
            @test @inferred(multiplier_type(Complex{Float64}, AbstractMatrix{Complex{Float32}})) === Complex{Float32}
            @test @inferred(multiplier_type(Complex{Rational}, AbstractMatrix{Complex{Float32}})) === Complex{Float32}
            @test @inferred(multiplier_type(Int, AbstractVector{Int})) === Float64
        end
    end

    @testset "Arithmetic rules" begin
        A, B, C, D = SymbolicOperator.((:A, :B, :C, :D))
        @test A isa Operator
        @test A === A
        @test A !== B

        # Sum are stored in right-associativity order.
        # NOTE S[i] is not inferable unless compiled with a constant i as
        #      in first(S) and last(S).
        @testset "Sum of $N terms" for N in (2, 3, 4)
            S = N == 2 ? @inferred(A + B) :
                N == 3 ? @inferred(A + B + C) :
                N == 4 ? @inferred(A + B + C + D) : nothing
            @test S isa LazyAlgebra.Sum{typeof(A)}
            @test @inferred(length(S)) === 2
            @test @inferred(firstindex(S)) === 1
            @test @inferred(lastindex(S)) === 2
            S1, S2 = S
            @test @inferred(Tuple(S)) === (S1, S2)
            @test @inferred(first(S)) === S1
            @test @inferred( last(S)) === S2
            @test S[1] === S1
            @test S[2] === S2
            @test S[1] === A
            if N == 2
                @test S isa LazyAlgebra.Sum{typeof(A),typeof(B)}
                @test S[2] === B
            elseif N == 3
                @test S isa LazyAlgebra.Sum{typeof(A),LazyAlgebra.Sum{typeof(B),typeof(C)}}
                @test S[2] === B + C
                @test S[2][1] === B
                @test S[2][2] === C
            elseif N == 4
                @test S isa LazyAlgebra.Sum{typeof(A),LazyAlgebra.Sum{typeof(B),LazyAlgebra.Sum{typeof(C),typeof(D)}}}
                @test S[2] === B + C + D
                @test S[2][1] === B
                @test S[2][2] === C + D
                @test S[2][2][1] === C
                @test S[2][2][2] === D
            end
        end

        # Grouping sums makes no difference (thanks to building according to a given
        # associativity).
        @test (A + B) + C === A + B + C
        @test A + (B + C) === A + B + C
        @test A + (B + C) + D === A + B + C + D
        @test ((A + B) + C) + D === A + B + C + D
        @test A + (B + (C + D)) === A + B + C + D
        @test A + (B + C + D) === A + B + C + D
        @test A + (B + C) + D === A + B + C + D
        @test (A + B + C) + D === A + B + C + D

        # Sums and differences.
        @test A - B === A + (-1)*B
        @test A + B - C === A + B + (-1)*C
        @test A - B - C === A + (-1)*B + (-1)*C
        @test A - (B + C) === A + (-1)*(B + C)
        @test -A + B - (C + D) === (-1)*A + B + (-1)*(C + D)

        @testset "Product of $N terms" for N in (2, 3, 4)
            S = N == 2 ? @inferred(A * B) :
                N == 3 ? @inferred(A * B * C) :
                N == 4 ? @inferred(A * B * C * D) : nothing
            @test S isa LazyAlgebra.Prod{typeof(A)}
            @test @inferred(length(S)) === 2
            @test @inferred(firstindex(S)) === 1
            @test @inferred(lastindex(S)) === 2
            S1, S2 = S
            @test @inferred(Tuple(S)) === (S1, S2)
            @test @inferred(first(S)) === S1
            @test @inferred( last(S)) === S2
            @test S[1] === S1
            @test S[2] === S2
            @test S[1] === A
            if N == 2
                @test S isa LazyAlgebra.Prod{typeof(A),typeof(B)}
                @test S[2] === B
            elseif N == 3
                @test S isa LazyAlgebra.Prod{typeof(A),LazyAlgebra.Prod{typeof(B),typeof(C)}}
                @test S[2] === B * C
                @test S[2][1] === B
                @test S[2][2] === C
            elseif N == 4
                @test S isa LazyAlgebra.Prod{typeof(A),LazyAlgebra.Prod{typeof(B),LazyAlgebra.Prod{typeof(C),typeof(D)}}}
                @test S[2] === B * C * D
                @test S[2][1] === B
                @test S[2][2] === C * D
                @test S[2][2][1] === C
                @test S[2][2][2] === D
            end
        end

        # Grouping products makes no difference (thanks to building according to a given
        # associativity).
        @test (A * B) * C === A * B * C
        @test A * (B * C) === A * B * C
        @test A * (B * C) * D === A * B * C * D
        @test ((A * B) * C) * D === A * B * C * D
        @test A * (B * (C * D)) === A * B * C * D
        @test A * (B * C * D) === A * B * C * D
        @test A * (B * C) * D === A * B * C * D
        @test (A * B * C) * D === A * B * C * D

        # Composition of operators.
        @test A ∘ B === A * B
        @test A ∘ B ∘ C === A * B * C

        # Division of operators.
        @test A / B === A * inv(B)
        @test A \ B === inv(A) * B
        @test A / B * C === A * inv(B) * C
        @test A / (B * C) === A * inv(C) * inv(B)
        @test A \ B * C === inv(A) * B * C
        @test A * B / C * D === A * B * inv(C) * D
        @test A * B / (C * D) === A * B * inv(D) * inv(C)
        @test A * B \ C * D === inv(A * B) * C * D # FIXME not `A * inv(B) * C * D` due to Julia associative rules

        # Adjoint of a number
        @test @inferred(LazyAlgebra.Adjoint(42)) === 42
        @test @inferred(LazyAlgebra.Adjoint(4.0 - 2.0im)) === 4.0 + 2.0im
        # Adjoint of an operator
        @test A' isa LazyAlgebra.Adjoint
        @test A' === adjoint(A)
        @test @inferred(adjoint(A')) === A
        @test @inferred(parent(A')) === A
        @test @inferred(getindex(A')) === A
        @test A'[] === A
        @test A'' === A
        # Adjoint of a sum.
        @test (A + B)' === A' + B'
        @test (A + B)' isa LazyAlgebra.Sum
        @test Tuple((A + B)') === (A', B')
        @test (A + B + C + D)' === A' + B' + C' + D'
        @test (A + B + C + D)' isa LazyAlgebra.Sum
        @test (A + B + C + D)'[1] === A'
        @test (A + B + C + D)'[2][1] === B'
        @test (A + B + C + D)'[2][2][1] === C'
        @test (A + B + C + D)'[2][2][2] === D'
        # Adjoint of a product.
        @test (A * B)' === B' * A'
        @test (A * B)' isa LazyAlgebra.Prod
        @test Tuple((A * B)') === (B', A')
        @test (A * B * C * D)' === D' * C' * B' * A'
        @test (A * B * C * D)' isa LazyAlgebra.Prod
        @test (A * B * C * D)'[1] === D'
        @test (A * B * C * D)'[2][1] === C'
        @test (A * B * C * D)'[2][2][1] === B'
        @test (A * B * C * D)'[2][2][2] === A'

        # Inverse of a number
        @test @inferred(LazyAlgebra.Inverse(2)) === 1/2
        @test @inferred(LazyAlgebra.Inverse(3.0 - 2.0im)) ≈ (3.0 + 2.0im)/13.0
        # Inverse of an operator
        @test inv(A) isa LazyAlgebra.Inverse
        @test inv(A) === Id/A
        @test inv(A) === A\Id
        @test @inferred(inv(inv(A))) === A
        @test @inferred(parent(inv(A))) === A
        @test @inferred(getindex(inv(A))) === A
        @test inv(A)[] === A
        @test Id/inv(A) === A
        @test inv(A)\Id === A
        # Inverse of a sum.
        @test inv(A + B) isa LazyAlgebra.Inverse
        @test inv(A + B)[] === A + B
        @test inv(A + B + C + D) isa LazyAlgebra.Inverse
        @test inv(A + B + C + D)[] === A + B + C + D
        # Inverse of a product.
        @test inv(A * B) === inv(B) * inv(A)
        @test inv(A * B) isa LazyAlgebra.Prod
        @test Tuple(inv(A * B)) === (inv(B), inv(A))
        @test inv(A * B * C * D) === inv(D) * inv(C) * inv(B) * inv(A)
        @test inv(A * B * C * D) isa LazyAlgebra.Prod
        @test inv(A * B * C * D)[1] === inv(D)
        @test inv(A * B * C * D)[2][1] === inv(C)
        @test inv(A * B * C * D)[2][2][1] === inv(B)
        @test inv(A * B * C * D)[2][2][2] === inv(A)

        # Inverse-adjoint and adjoint-inverse
        @test inv(A)' === @inferred(adjoint(inv(A))) isa LazyAlgebra.Inverse{<:LazyAlgebra.Adjoint}
        @test @inferred(inv(A')) === @inferred(inv(adjoint(A))) isa LazyAlgebra.Inverse{<:LazyAlgebra.Adjoint}
        @test @inferred(parent(inv(A'))) === A'
        @test @inferred(parent(inv(A)')) === A'
        @test @inferred(parent(parent(inv(A')))) === A
        @test @inferred(parent(parent(inv(A)'))) === A
        @test @inferred(inv(inv(A'))) === A'
        @test @inferred(inv(inv(A)')) === A'
        @test @inferred(adjoint(inv(A'))) === inv(A)
        @test @inferred(adjoint(inv(A)')) === inv(A)

        # Gram operator.
        X = @inferred(Gram(A))
        @test @inferred(adjoint(X)) === X

        # Scalar times operator.
        @testset "Scalar (λ=$λ) times $X" for λ in (0x0, true, -1, 1//2, pi, 2.3f0, 2.0 - 3.0im), X in (A, A + B, A*B)
            @test λ*X === X*λ
            @test λ*X isa LazyAlgebra.Prod{typeof(λ),typeof(X)}
            @test @inferred(first(λ*X)) === λ
            @test @inferred( last(λ*X)) === X
            @test (λ*X)' === (X*λ)'
            @test @inferred(first((λ*X)')) === conj(λ)
            @test @inferred( last((X*λ)')) === X'
            @test inv(λ*X) === inv(X*λ)
            @test @inferred(first(inv(λ*X))) ≈ inv(λ)
            @test @inferred( last(inv(X*λ))) === inv(X)
            @test λ\X === X/λ
            @test @inferred(first(λ\X)) ≈ inv(λ)
            @test @inferred( last(λ\X)) === X
            @test (λ\X)' === (X/λ)'
            @test @inferred(first((λ\X)')) ≈ inv(conj(λ))
            @test @inferred( last((λ\X)')) === X'
            @test inv(λ\X) === inv(X/λ)
            @test @inferred(first(inv(λ\X))) ≈ λ
            @test @inferred( last(inv(λ\X))) === inv(X)
            @test inv((λ\X)') === inv((X/λ)')
            @test @inferred(first(inv((λ\X)'))) ≈ conj(λ)
            @test @inferred( last(inv((λ\X)'))) === inv(X')
        end

        # Left-factorization of scalar in products.
        α, β = 3//4, -2.0 + 3.0im
        X, Y = B + C*D, A - D
        @test A*α === α*A isa LazyAlgebra.Prod{typeof(α),typeof(A)}
        @test (A*B)*α === A*(B*α) === A*(α*B) === (A*α)*B === (α*A)*B === α*(A*B) isa LazyAlgebra.Prod{typeof(α),typeof(A*B)}
        @test (A*X)*α === A*(X*α) === A*(α*X) === (A*α)*X === (α*A)*X === α*(A*X) isa LazyAlgebra.Prod{typeof(α),typeof(A*X)}
        @test (A*B)/α === A*(B/α) === A*(α\B) === (A/α)*B === (α\A)*B === α\(A*B) isa LazyAlgebra.Prod{<:Number,typeof(A*B)}
        @test X*α + Y*β === α*X + β*Y isa LazyAlgebra.Sum{LazyAlgebra.Prod{typeof(α),typeof(X)},LazyAlgebra.Prod{typeof(β),typeof(Y)}}
        @test (X*α)*(Y*β) === (α*X)*(β*Y) === (α*β)*(X*Y) isa LazyAlgebra.Prod{<:Number,typeof(X*Y)}

        # Showing expressions.
        @test string(A) == "A"
        @test string(A + B) == "A + B"
        @test string(A * B) == "A*B"
        @test string(A * B + C - D) == "A*B + C - D"
        @test string(A * (B + C - D)) == "A*(B + C - D)"
        @test string(A * (B + C * D)) == "A*(B + C*D)"
        @test string(2A * (B + (-2 + 1im)C * D)) == "2*A*(B + (-2 + 1im)*C*D)"
        @test string(-4A * (1*B - 3C * D)) == "-4*A*(B - 3*C*D)"

        # Unary plus and minus.
        @testset "Unary plus and minus on $X" for X in (A, A + B, A + B*C, A*(B + C*D))
            @test +X === X
            @test -X === (-1)*X isa LazyAlgebra.Prod{Int,typeof(X)}
        end

        # Only type-stable simplifications are applied by constructors.
        @test 2A != A + A isa LazyAlgebra.Sum{typeof(A),typeof(A)}
        @test 0A != A - A === A + (-1)*A isa LazyAlgebra.Sum{typeof(A),<:LazyAlgebra.Prod{<:Number,typeof(A)}}
        @test 2Id === Id + Id isa LazyAlgebra.Prod{<:Number,typeof(Id)}
        @test 4Id === Id + 3Id isa LazyAlgebra.Prod{<:Number,typeof(Id)}
        @test Id - Id === 0*Id isa LazyAlgebra.Prod{<:Number,typeof(Id)}

        # Neutral element for the addition.
        @test zero(A) === 0*A
        @test zero(π*A) === zero(A)
        @test !iszero(A)
        @test !iszero(A - A) # because automatic simplifications must be type-stable
        @test iszero(Id - Id) # this simplification is type-stable
        @test !iszero(π*A)
        @test iszero(0*A)

        # Neutral element for the composition.
        @test one(A) === Id
        @test one(π*A + B*C) === Id
        @test !isone(A)
        @test !isone(π*A + B*C)
        @test isone(Id)
        @test isone(Identity(3,4,5))

    end

    @testset "Identity" begin
        A, B, C, D = SymbolicOperator.((:A, :B, :C, :D))
        # Universal identity.
        @test isone(Id)
        @test Id' === Id
        @test inv(Id) === Id
        @test Id*Id === Id
        @test Id*A === A
        @test A*Id === A
        @test Id\A === A
        @test A/Id === A
        @test A*Id*B === A*B
        @test A\Id === inv(A)
        @test Id/A === inv(A)
        @test (A + B*C)\Id === inv(A + B*C)
        @test Id/(A + B*C) === inv(A + B*C)
        @test iszero(Id - Id)
        @test Id - Id + 3Id - Id === 2Id

        # Operator API for universal identity.
        x = [1.0 -3.0 0.0; 2.0 5.0 -1.0]
        @test LazyAlgebra.InputEltype(Id) === LazyAlgebra.InputEltypeUnknown()
        @test LazyAlgebra.OutputEltype(Id) === LazyAlgebra.OutputEltypeUnknown()
        @test LazyAlgebra.InputShape(Id) === LazyAlgebra.InputShapeUnknown()
        @test LazyAlgebra.OutputShape(Id) === LazyAlgebra.OutputShapeUnknown()
        @test_throws Exception LazyAlgebra.input_axes(Id)
        @test_throws Exception LazyAlgebra.output_axes(Id)
        @test @inferred(LazyAlgebra.output_axes(Id, x)) === axes(x)
        @test @inferred(LazyAlgebra.output_eltype(Id, x)) === eltype(x)
        @test @inferred(LazyAlgebra.output_eltype(2 - 1im, Id, x)) === Complex{Float64}
        @test @inferred(LazyAlgebra.output_eltype(2.0 - 1.0im, Id, Float32.(x))) === Complex{Float32}

        # Uniform scaling operator from LinearAlgebra.
        @test Operator(UniformScaling(π)) === π*Id
        @test Operator(I) === I.λ*Id
        @test A*I === I.λ*A
        @test A∘I === I.λ*A
        @test I*A === I.λ*A
        @test I∘A === I.λ*A
        @test A/I   isa LazyAlgebra.Prod{<:Number,typeof(A)}
        @test A/I*B isa LazyAlgebra.Prod{<:Number,typeof(A*B)}
        @test I\A   isa LazyAlgebra.Prod{<:Number,typeof(A)}
        @test A*I\B isa LazyAlgebra.Prod{<:Number,typeof(A\B)}
        @test Id + I === 2Id
        @test Id - I === 0Id
        @test I + Id === 2Id
        @test I - Id === 0Id

        # Shaped identity.
        shape1 = (2,3,4)
        n1 = length(shape1)
        I1 = @inferred(Identity(shape1))
        @test @inferred(Identity(shape1...)) === I1
        @test @inferred(Identity(map(Base.OneTo, shape1))) === I1
        shape2 = (-1:2,3:5)
        n2 = length(shape2)
        I2 = @inferred(Identity(shape2))
        @test @inferred(Identity(shape2...)) === I2
        @test LazyAlgebra.InputEltype(I1) === LazyAlgebra.InputEltypeUnknown()
        @test LazyAlgebra.InputEltype(I2) === LazyAlgebra.InputEltypeUnknown()
        @test LazyAlgebra.OutputEltype(I1) === LazyAlgebra.OutputEltypeUnknown()
        @test LazyAlgebra.OutputEltype(I2) === LazyAlgebra.OutputEltypeUnknown()
        @test LazyAlgebra.InputShape(I1) === LazyAlgebra.HasInputShape{n1}()
        @test LazyAlgebra.InputShape(I2) === LazyAlgebra.HasInputShape{n2}()
        @test LazyAlgebra.OutputShape(I1) === LazyAlgebra.HasOutputShape{n1}()
        @test LazyAlgebra.OutputShape(I2) === LazyAlgebra.HasOutputShape{n2}()
        @test @inferred(LazyAlgebra.input_axes(I1)) === map(Base.OneTo, shape1)
        @test @inferred(LazyAlgebra.input_axes(I2)) === shape2
        @test @inferred(LazyAlgebra.output_axes(I1)) === map(Base.OneTo, shape1)
        @test @inferred(LazyAlgebra.output_axes(I2)) === shape2
        #@test @inferred(LazyAlgebra.input_size(I1)) === shape1
        #@test @inferred(LazyAlgebra.input_size(I1)) === map(length, shape2)
        #@test @inferred(LazyAlgebra.output_size(I1)) === shape1
        #@test @inferred(LazyAlgebra.output_size(I2)) === map(length, shape2)
    end
end
nothing
