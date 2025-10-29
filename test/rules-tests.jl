using LazyAlgebra
using Test
using LinearAlgebra
using Neutrals

using LazyAlgebra: Adjoint, Transpose, Inverse, Prod, Sum, Scaled, divide, inverse

function plain(x)
    io = IOBuffer()
    show(io, MIME"text/plain"(), x)
    return String(take!(io))
end

@testset "Arithmetic rules" begin
    @testset "Arithmetic rules" begin
        A, B, C, D = SymbolicOperator.((:A, :B, :C, :D))
        @test typeof(A) <: Operator
        @test A === A
        @test A !== B

        # Sum are stored in right-associativity order.
        # NOTE S[i] is not inferable unless compiled with a constant i as
        #      in first(S) and last(S).
        @testset "Sum of $N terms" for N in (2, 3, 4)
            S = N == 2 ? @inferred(A + B) :
                N == 3 ? @inferred(A + B + C) :
                N == 4 ? @inferred(A + B + C + D) : nothing
            @test typeof(S) <: Sum{typeof(A)}
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
                @test typeof(S) <: Sum{typeof(A),typeof(B)}
                @test S[2] === B
            elseif N == 3
                @test typeof(S) <: Sum{typeof(A),Sum{typeof(B),typeof(C)}}
                @test S[2] === B + C
                @test S[2][1] === B
                @test S[2][2] === C
            elseif N == 4
                @test typeof(S) <: Sum{typeof(A),Sum{typeof(B),Sum{typeof(C),typeof(D)}}}
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
        @test A - B === A + (-𝟙)*B
        @test A + B - C === A + B + (-𝟙)*C
        @test A - B - C === A + (-𝟙)*B + (-𝟙)*C
        @test A - (B + C) === A + (-𝟙)*(B + C)
        @test -A + B - (C + D) === (-𝟙)*A + B + (-𝟙)*(C + D)

        @testset "Product of $N terms" for N in (2, 3, 4)
            S = N == 2 ? @inferred(A * B) :
                N == 3 ? @inferred(A * B * C) :
                N == 4 ? @inferred(A * B * C * D) : nothing
            @test typeof(S) <: Prod{typeof(A)}
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
                @test typeof(S) <: Prod{typeof(A),typeof(B)}
                @test S[2] === B
            elseif N == 3
                @test typeof(S) <: Prod{typeof(A),Prod{typeof(B),typeof(C)}}
                @test S[2] === B * C
                @test S[2][1] === B
                @test S[2][2] === C
            elseif N == 4
                @test typeof(S) <: Prod{typeof(A),Prod{typeof(B),Prod{typeof(C),typeof(D)}}}
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

        # Adjoint of an operator
        @test typeof(A') <: Adjoint
        @test A' === adjoint(A)
        @test @inferred(adjoint(A')) === A
        @test @inferred(parent(A')) === A
        @test @inferred(getindex(A')) === A
        @test A'[] === A
        @test A'' === A
        # Adjoint of a sum.
        @test (A + B)' === A' + B'
        @test typeof((A + B)') <: Sum
        @test Tuple((A + B)') === (A', B')
        @test (A + B + C + D)' === A' + B' + C' + D'
        @test typeof((A + B + C + D)') <: Sum
        @test (A + B + C + D)'[1] === A'
        @test (A + B + C + D)'[2][1] === B'
        @test (A + B + C + D)'[2][2][1] === C'
        @test (A + B + C + D)'[2][2][2] === D'
        # Adjoint of a product.
        @test (A * B)' === B' * A'
        @test typeof((A * B)') <: Prod
        @test Tuple((A * B)') === (B', A')
        @test (A * B * C * D)' === D' * C' * B' * A'
        @test typeof((A * B * C * D)') <: Prod
        @test (A * B * C * D)'[1] === D'
        @test (A * B * C * D)'[2][1] === C'
        @test (A * B * C * D)'[2][2][1] === B'
        @test (A * B * C * D)'[2][2][2] === A'

        # Transpose of an operator
        @test typeof(transpose(A)) <: Transpose
        @test @inferred(transpose(transpose(A))) === A
        @test @inferred(parent(transpose(A))) === A
        @test @inferred(getindex(transpose(A))) === A
        @test transpose(A)[] === A
        # Transpose of a sum.
        @test transpose(A + B) === transpose(A) + transpose(B)
        @test typeof(transpose(A + B)) <: Sum
        @test Tuple(transpose(A + B)) === (transpose(A), transpose(B))
        @test transpose(A + B + C + D) === transpose(A) + transpose(B) + transpose(C) + transpose(D)
        @test typeof(transpose(A + B + C + D)) <: Sum
        @test transpose(A + B + C + D)[1] === transpose(A)
        @test transpose(A + B + C + D)[2][1] === transpose(B)
        @test transpose(A + B + C + D)[2][2][1] === transpose(C)
        @test transpose(A + B + C + D)[2][2][2] === transpose(D)
        # Transpose of a product.
        @test transpose(A * B) === transpose(B) * transpose(A)
        @test typeof(transpose(A * B)) <: Prod
        @test Tuple(transpose(A * B)) === (transpose(B), transpose(A))
        @test transpose(A * B * C * D) === transpose(D) * transpose(C) * transpose(B) * transpose(A)
        @test typeof(transpose(A * B * C * D)) <: Prod
        @test transpose(A * B * C * D)[1] === transpose(D)
        @test transpose(A * B * C * D)[2][1] === transpose(C)
        @test transpose(A * B * C * D)[2][2][1] === transpose(B)
        @test transpose(A * B * C * D)[2][2][2] === transpose(A)

        # Inverse of a number
        @test @inferred(inverse(2)) === 1//2
        @test @inferred(inverse(3.0 - 2.0im)) ≈ (3.0 + 2.0im)/13.0
        # Inverse of an operator
        @test typeof(inv(A)) <: Inverse
        @test @inferred(inv(inv(A))) === A
        @test @inferred(parent(inv(A))) === A
        @test @inferred(getindex(inv(A))) === A
        @test inv(A)[] === A
        # Inverse of a sum.
        @test typeof(inv(A + B)) <: Inverse
        @test inv(A + B)[] === A + B
        @test typeof(inv(A + B + C + D)) <: Inverse
        @test inv(A + B + C + D)[] === A + B + C + D
        # Inverse of a product.
        @test inv(A * B) === inv(B) * inv(A)
        @test typeof(inv(A * B)) <: Prod
        @test Tuple(inv(A * B)) === (inv(B), inv(A))
        @test inv(A * B * C * D) === inv(D) * inv(C) * inv(B) * inv(A)
        @test typeof(inv(A * B * C * D)) <: Prod
        @test inv(A * B * C * D)[1] === inv(D)
        @test inv(A * B * C * D)[2][1] === inv(C)
        @test inv(A * B * C * D)[2][2][1] === inv(B)
        @test inv(A * B * C * D)[2][2][2] === inv(A)

        # Inverse-adjoint and adjoint-inverse
        @test inv(A)' === @inferred(adjoint(inv(A)))
        @test @inferred(inv(A')) === @inferred(inv(adjoint(A)))
        @test typeof(inv(A)') <: Inverse{<:Adjoint}
        @test typeof(inv(A')) <: Inverse{<:Adjoint}
        @test @inferred(parent(inv(A'))) === A'
        @test @inferred(parent(inv(A)')) === A'
        @test @inferred(parent(parent(inv(A')))) === A
        @test @inferred(parent(parent(inv(A)'))) === A
        @test @inferred(inv(inv(A'))) === A'
        @test @inferred(inv(inv(A)')) === A'
        @test @inferred(adjoint(inv(A'))) === inv(A)
        @test @inferred(adjoint(inv(A)')) === inv(A)

        # Scalar times operator.
        @testset "Scalar (λ=$λ) times $X" for λ in (0x0, true, 𝟙, -1, 1//2, pi, 2.3f0, 2.0 - 3.0im), X in (A, A + B, A*B)
            @test @inferred(λ*X) === @inferred(X*λ)
            @test typeof(λ*X) <: Scaled{typeof(λ),typeof(X)}
            @test  first(λ*X) === λ
            @test   last(λ*X) === X
            #
            @test @inferred(X/λ) === @inferred(λ\X)
            @test typeof(X/λ) <: Scaled{<:Number,typeof(X)}
            @test  first(X/λ) === inverse(λ)
            @test   last(X/λ) === X
            #
            @test @inferred(adjoint((λ*X))) === (λ*X)'
            @test @inferred(adjoint((X*λ))) === (X*λ)'
            @test @inferred(conj(λ)*X') ===  (λ*X)'
            @test typeof((λ*X)') <: Scaled{<:Number,typeof(X')}
            @test  first((λ*X)') === conj(λ)
            @test   last((λ*X)') === X'
            #
            @test @inferred(inv(λ*X)) === @inferred(inverse(λ)*inv(X))
            @test @inferred(inv(X*λ)) === @inferred(inverse(λ)*inv(X))
            @test typeof(inv(λ*X)) <: Scaled{<:Number,typeof(inv(X))}
            @test  first(inv(λ*X)) === inverse(λ)
            @test   last(inv(λ*X)) === inv(X)
            #
            @test (X/λ)' === @inferred(adjoint((X/λ)))
            @test (λ\X)' === @inferred(adjoint((X/λ)))
            @test typeof((X/λ)') <: Scaled{<:Number,typeof(X')}
            @test  first((X/λ)') === inverse(conj(λ))
            @test   last((X/λ)') === X'
            #
            @test @inferred(inv(λ\X)) === @inferred(inv(X/λ))
            @test typeof(inv(X/λ)) <: Scaled{<:Number,typeof(inv(X))}
            @test  first(inv(X/λ)) ≈ λ
            @test   last(inv(X/λ)) === inv(X)
            #
            @test @inferred(inv((λ\X)')) === @inferred(inv((X/λ)'))
            @test typeof(inv((X/λ)')) <: Scaled{<:Number,typeof(inv(X'))}
            @test  first(inv((X/λ)')) ≈ conj(λ)
            @test   last(inv((X/λ)')) === inv(X')
        end

        # Left-factorization of scalar in products.
        α, β = 3//4, -2.0 + 3.0im
        X, Y = B + C*D, A - D
        @test @inferred(A*α) === @inferred(α*A)
        @test typeof(α*A) <: Scaled{typeof(α),typeof(A)}
        @test (A*B)*α === A*(B*α) === A*(α*B) === (A*α)*B === (α*A)*B === α*(A*B) === α*A*B
        @test typeof(α*A*B) <: Scaled{typeof(α),typeof(A*B)}
        @test (A*X)*α === A*(X*α) === A*(α*X) === (A*α)*X === (α*A)*X === α*(A*X) === α*A*X
        @test typeof(α*A*X) <: Scaled{typeof(α),typeof(A*X)}
        @test (A*B)/α === A*(B/α) === A*(α\B) === (A/α)*B === (α\A)*B === α\(A*B)
        @test typeof(α\(A*B)) <: Scaled{<:Number,typeof(A*B)}
        #
        @test @inferred(α*X + Y*β) === @inferred(α*X + β*Y)
        @test @inferred(X*α + β*Y) === @inferred(α*X + β*Y)
        @test @inferred(X*α + Y*β) === @inferred(α*X + β*Y)
        @test typeof(α*X + β*Y) <: Sum{Scaled{typeof(α),typeof(X)},Scaled{typeof(β),typeof(Y)}}
        #
        @test @inferred((α*X)*(β*Y)) === @inferred((α*β)*(X*Y))
        @test @inferred((α*X)*(Y*β)) === @inferred((α*β)*(X*Y))
        @test @inferred((X*α)*(β*Y)) === @inferred((α*β)*(X*Y))
        @test @inferred((X*α)*(Y*β)) === @inferred((α*β)*(X*Y))
        @test typeof((α*β)*(X*Y)) <: Scaled{<:Number,typeof(X*Y)}

        # `A\β` and `β/A` intentionally not supported.
        β = 3
        @test_throws Exception A\β
        @test_throws Exception β/A
        @test A\(β*I) === β*inv(A)
        @test (β*I)/A === β*inv(A)
        @test A\(β*Id) === β*inv(A)
        @test (β*Id)/A === β*inv(A)

        # Scaling by neutral numbers.
        @test (𝟙*Id)*A === 𝟙*A
        @test A/(𝟙*B) === 𝟙*A*inv(B)
        @test A\(𝟙*B) === 𝟙*inv(A)*B

        # Showing expressions.
        @test string(A) == "SymbolicOperator(:A)"
        @test plain(A) == "A"
        @test plain(A + B) == "A + B"
        @test plain(A * B) == "A*B"
        @test plain(A * B + C - D) == "A*B + C - D"
        @test plain(A * (B + C - D)) == "A*(B + C - D)"
        @test plain(A * (B + C * D)) == "A*(B + C*D)"
        @test plain(2A * (B + (-2 + 1im)C * D)) == "2*A*(B + (-2 + 1im)*C*D)"
        @test plain(-4A * (1*B - 3C * D)) == "-4*A*(B - 3*C*D)"

        # Unary plus and minus.
        @testset "Unary plus and minus on $X" for X in (A, A + B, A + B*C, A*(B + C*D))
            @test +X === X
            @test -X === (-𝟙)*X
            @test typeof(-X) <: Scaled{typeof(-𝟙),typeof(X)}
        end

        # Only type-stable simplifications are applied by constructors.
        @test @inferred(A + A) != @inferred(2A)
        @test typeof(A + A) <: Sum{typeof(A),typeof(A)}
        @test typeof(2A) <: Scaled{Int,typeof(A)}
        #
        @test @inferred(A - A) != @inferred(0A)
        @test @inferred(A - A) === @inferred(A + (-𝟙)*A)
        @test typeof(A - A) <: Sum{typeof(A),<:Scaled{typeof(-𝟙),typeof(A)}}
        @test typeof(0A) <: Scaled{Int,typeof(A)}
        #
        @test @inferred(Id + Id) === 2Id
        @test typeof(Id + Id) <: Scaled{Int,typeof(Id)}
        #
        @test @inferred(Id - Id) === @inferred(𝟘*Id)
        @test typeof(Id - Id) <: Scaled{typeof(𝟘),typeof(Id)}
        #
        @test @inferred(Id + 3Id - 2Id) === @inferred(2Id)
        @test typeof(2Id) <: Scaled{<:Number,typeof(Id)}

        # Neutral element for the addition.
        @test zero(A) === 𝟘*A
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
        @test isone(1*Id)
        @test !isone(-1*Id)

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
        @test typeof(@inferred(A/I))   <: Scaled{<:Number,typeof(A)}
        @test typeof(@inferred(A/I*B)) <: Scaled{<:Number,typeof(A*B)}
        @test typeof(@inferred(I\A))   <: Scaled{<:Number,typeof(A)}
        @test typeof(@inferred(A*I\B)) <: Scaled{<:Number,typeof(A\B)}
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
