using LazyAlgebra
using Test
using LinearAlgebra
using Neutrals

using LazyAlgebra: Adjoint, Conjugate, Transpose, Inverse, Prod, Sum, Scaled, divide, inverse

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

        @testset "Sums of $N terms" for N in (2, 3, 4)
            S = N == 2 ? @inferred(A + B) :
                N == 3 ? @inferred(A + B + C) :
                N == 4 ? @inferred(A + B + C + D) : nothing
            @test S isa Sum
            @test @inferred(length(S)) === 2
            @test @inferred(firstindex(S)) === 1
            @test @inferred(lastindex(S)) === 2
            S1, S2 = S
            @test @inferred(Tuple(S)) === (S1, S2)
            # NOTE `S[i]` is not inferable unless compiled with a constant `i` as in `first(S)` and
            #      `last(S)`.
            @test @inferred(first(S)) === S1
            @test @inferred( last(S)) === S2
            @test @inferred(S[1]) === S1
            @test @inferred(S[2]) === S2
            # Julia computes sums of terms by left-associativity.
            if N == 2
                @test S isa Sum{typeof(A),typeof(B)}
                @test S[1] === A
                @test S[2] === B
            elseif N == 3
                @test S isa Sum{Sum{typeof(A),typeof(B)},typeof(C)}
                @test S[1] === A + B
                @test S[1][1] === A
                @test S[1][2] === B
                @test S[2] === C
            elseif N == 4
                @test S isa Sum{Sum{Sum{typeof(A),typeof(B)},typeof(C)},typeof(D)}
                @test S[1] === A + B + C
                @test S[1][1] === A + B
                @test S[1][1][1] === A
                @test S[1][1][2] === B
                @test S[1][2] === C
                @test S[2] === D
            end
        end

        # Sums respect grouping. Left-associativity is the default for sum of terms.
        @test @inferred((A + B) + C) === @inferred(A + B + C)
        S = @inferred(A + (B + C))
        @test S[1] === A
        @test S[2] === B + C
        @test @inferred(((A + B) + C) + D) === @inferred(A + B + C + D)
        S = @inferred(A + (B + C + D))
        @test S[1] === A
        @test S[2] === @inferred(B + C + D)
        S = @inferred(A + (B + C) + D)
        @test S[1] === @inferred(A + (B + C))
        @test S[2] === D
        S = @inferred((A + B) + (C + D))
        @test S[1] === @inferred(A + B)
        @test S[2] === @inferred(C + D)

        # Sums and differences.
        @test @inferred(A - B) === @inferred(A + (-𝟙)*B)
        @test @inferred(A + B - C) === @inferred(A + B + (-𝟙)*C)
        @test @inferred(A - B - C) === @inferred(A + (-𝟙)*B + (-𝟙)*C)
        @test @inferred(A - (B + C)) === @inferred(A + (-𝟙)*(B + C))
        @test @inferred(-A + B - (C + D)) === @inferred((-𝟙)*A + B + (-𝟙)*(C + D))

        @testset "Compositions of $N terms" for N in (2, 3, 4)
            S = N == 2 ? @inferred(A * B) :
                N == 3 ? @inferred(A * B * C) :
                N == 4 ? @inferred(A * B * C * D) : nothing
            @test S isa Prod
            @test @inferred(length(S)) === 2
            @test @inferred(firstindex(S)) === 1
            @test @inferred(lastindex(S)) === 2
            S1, S2 = S
            @test @inferred(Tuple(S)) === (S1, S2)
            # NOTE `S[i]` is not inferable unless compiled with a constant `i` as in `first(S)` and
            #      `last(S)`.
            @test @inferred(first(S)) === S1
            @test @inferred( last(S)) === S2
            @test @inferred(S[1]) === S1
            @test @inferred(S[2]) === S2
            # Julia computes products of terms by left-associativity.
            if N == 2
                @test S isa Prod{typeof(A),typeof(B)}
                @test S[1] === A
                @test S[2] === B
            elseif N == 3
                @test S isa Prod{Prod{typeof(A),typeof(B)},typeof(C)}
                @test S[1] === A * B
                @test S[1][1] === A
                @test S[1][2] === B
                @test S[2] === C
            elseif N == 4
                @test S isa Prod{Prod{Prod{typeof(A),typeof(B)},typeof(C)},typeof(D)}
                @test S[1] === A * B * C
                @test S[1][1] === A * B
                @test S[1][1][1] === A
                @test S[1][1][2] === B
                @test S[1][2] === C
                @test S[2] === D
            end
        end

        # Compositions respect grouping. Left-associativity is the default for product of terms.
        @test @inferred((A * B) * C) === @inferred(A * B * C)
        S = @inferred(A * (B * C))
        @test S[1] === A
        @test S[2] === B * C
        @test @inferred(((A * B) * C) * D) === @inferred(A * B * C * D)
        S = @inferred(A * (B * C * D))
        @test S[1] === A
        @test S[2] === @inferred(B * C * D)
        S = @inferred(A * (B * C) * D)
        @test S[1] === @inferred(A * (B * C))
        @test S[2] === D
        S = @inferred((A * B) * (C * D))
        @test S[1] === @inferred(A * B)
        @test S[2] === @inferred(C * D)

        # Composition of operators.
        @test @inferred(A ∘ B) === @inferred(A * B)
        @test @inferred(A ∘ B ∘ C) === @inferred(A * B * C)

        # Division of operators. NOTE Beware that `A*B\C` is like `(A*B)\C` in Julia, not `A*(B\C)`.
        @test @inferred(A / B) === @inferred(A * inv(B))
        @test @inferred(A \ B) === @inferred(inv(A) * B)
        @test @inferred(A / B * C) === @inferred(A * inv(B) * C)
        @test @inferred(A / (B * C)) === @inferred(A * (inv(C) * inv(B)))
        @test @inferred(A \ B * C) === @inferred(inv(A) * B * C)
        @test @inferred(A * B / C * D) === @inferred(A * B * inv(C) * D)
        @test @inferred(A * B / (C * D)) === @inferred(A * B * (inv(D) * inv(C)))
        @test @inferred(A * B \ C * D) === @inferred(inv(A * B) * C * D) # cf. above note
        @test @inferred(A * B \ C * D) === @inferred((A * B) \ C * D) # cf. above note

        # Adjoint of an operator
        @test A' === @inferred(adjoint(A))
        @test A' isa Adjoint
        @test @inferred(adjoint(A')) === A
        @test @inferred(parent(A')) === A
        @test @inferred(getindex(A')) === A
        @test A'[] === A
        @test A'' === A

        # Adjoint of a sum.
        @test (A + B)' === @inferred(adjoint(A + B))
        @test (A + B)' === @inferred(A' + B')
        @test (A + B)' isa Sum
        @test Tuple((A + B)') === (A', B')
        @test (A + B + C + D)' === @inferred(adjoint(A + B + C + D))
        @test (A + B + C + D)' === @inferred(A' + B' + C' + D')
        @test (A + B + C + D)' isa Sum

        # Adjoint of a composition.
        @test (A * B)' === @inferred(adjoint(A * B))
        @test (A * B)' === @inferred(B' * A')
        @test (A * B)' isa Prod
        @test Tuple((A * B)') === (B', A')
        @test (A * B * C * D)' === @inferred(adjoint(A * B * C * D))
        @test (A * B * C * D)' === @inferred(D' * C' * B' * A')
        @test (A * B * C * D)' isa Prod

        # Transpose of an operator
        @test transpose(A) isa Transpose
        @test @inferred(transpose(transpose(A))) === A
        @test @inferred(parent(transpose(A))) === A
        @test @inferred(getindex(transpose(A))) === A
        @test transpose(A)[] === A

        # Transpose of a sum.
        @test @inferred(transpose(A + B)) isa Sum
        @test @inferred(transpose(A + B)) === @inferred(transpose(A) + transpose(B))
        @test Tuple(transpose(A + B)) === (transpose(A), transpose(B))
        @test transpose(A + B + C + D) isa Sum
        @test transpose(A + B + C + D) === @inferred(transpose(A) + transpose(B) + transpose(C) + transpose(D))
        @test transpose(A + (B + C) + D) isa Sum
        @test transpose(A + (B + C) + D) === transpose(A) + (transpose(B) + transpose(C)) + transpose(D)

        # Transpose of a composition.
        @test @inferred(transpose(A * B)) isa Prod
        @test @inferred(transpose(A * B)) === @inferred(transpose(B) * transpose(A))
        @test Tuple(transpose(A * B)) === (transpose(B), transpose(A))
        @test transpose(A * B * C * D) isa Prod
        @test transpose(A * B * C * D) === @inferred(transpose(D) * ((transpose(C) * (transpose(B) * transpose(A)))))
        @test transpose(A * (B * C) * D) isa Prod
        @test transpose(A * (B * C) * D) === transpose(D) * ((transpose(C) * transpose(B)) * transpose(A))

        # Conjugate of an operator
        @test typeof(conj(A)) <: Conjugate
        @test @inferred(conj(conj(A))) === A
        @test @inferred(parent(conj(A))) === A
        @test @inferred(getindex(conj(A))) === A
        @test conj(A)[] === A

        # Conjugate is distributive over a sum.
        @test conj(A + B) === conj(A) + conj(B)
        @test typeof(conj(A + B)) <: Sum
        @test Tuple(conj(A + B)) === (conj(A), conj(B))
        @test typeof(conj(A + B + C + D)) <: Sum
        @test conj(A + B + C + D) === conj(A) + conj(B) + conj(C) + conj(D)
        @test conj(A + B + C + D)[1] === conj(A)
        @test conj(A + B + C + D)[2][1] === conj(B)
        @test conj(A + B + C + D)[2][2][1] === conj(C)
        @test conj(A + B + C + D)[2][2][2] === conj(D)
        # Conjugate of a product.
        @test conj(A * B) === conj(A) * conj(B)
        @test typeof(conj(A * B)) <: Prod
        @test Tuple(conj(A * B)) === (conj(A), conj(B))
        @test typeof(conj(A * B * C * D)) <: Prod
        @test conj(A * B * C * D) === conj(A) * conj(B) * conj(C) * conj(D)
        @test conj(A * B * C * D)[1] === conj(A)
        @test conj(A * B * C * D)[2][1] === conj(B)
        @test conj(A * B * C * D)[2][2][1] === conj(C)
        @test conj(A * B * C * D)[2][2][2] === conj(D)

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
