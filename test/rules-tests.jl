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
        @test A == A
        @test isequal(A, A)
        @test A !== B
        @test A != B
        @test !isequal(A, B)

        @testset "Sums of $N terms" for N in (0, 1, 2, 3, 4)
            if N == 0
                @test_throws ArgumentError LazyAlgebra.Sum()
                @test_throws ArgumentError LazyAlgebra.Sum(())
                continue
            end
            if N == 1
                @test @inferred(LazyAlgebra.Sum(A)) === A
                @test @inferred(LazyAlgebra.Sum((A,))) === A
                continue
            end

            X = N == 2 ? @inferred(A + B) :
                N == 3 ? @inferred(A + B + C) :
                N == 4 ? @inferred(A + B + C + D) : continue
            @test X isa Sum{<:NTuple{N,Operator}}
            @test @inferred(length(X)) === N
            @test @inferred(firstindex(X)) === 1
            @test @inferred(lastindex(X)) === N
            # Test sum as an iterator.
            @test mapreduce(===, &, X, (A, B, C, D))
            # Conversion of sum to a tuple of terms.
            @test @inferred(Tuple(X)) === (A, B, C, D)[1:N]
            # Test indexation. NOTE `X[i]` is not inferable unless compiled with a constant
            #                       index `i` as in `first(X)` and `last(X)`.
            @test @inferred(first(X)) === X[1]
            @test @inferred( last(X)) === X[N]
            if N ≥ 1
                @test @inferred(X[1]) === A
            end
            if N ≥ 2
                @test @inferred(X[2]) === B
            end
            if N ≥ 3
                @test @inferred(X[3]) === C
            end
            if N ≥ 4
                @test @inferred(X[4]) === D
            end
        end

        @testset "Compositions of $N terms" for N in (0, 1, 2, 3, 4)
            if N == 0
                @test LazyAlgebra.Prod() === Id
                @test LazyAlgebra.Prod(()) === Id
                continue
            end
            if N == 1
                @test @inferred(LazyAlgebra.Prod(A)) === A
                @test @inferred(LazyAlgebra.Prod((A,))) === A
                continue
            end

            X = N == 2 ? @inferred(A * B) :
                N == 3 ? @inferred(A * B * C) :
                N == 4 ? @inferred(A * B * C * D) : continue
            @test X isa Prod{<:NTuple{N,Operator}}
            @test @inferred(length(X)) === N
            @test @inferred(firstindex(X)) === 1
            @test @inferred(lastindex(X)) === N
            # Test sum as an iterator.
            @test mapreduce(===, &, X, (A, B, C, D))
            # Conversion of sum to a tuple of terms.
            @test @inferred(Tuple(X)) === (A, B, C, D)[1:N]
            # Test indexation. NOTE `X[i]` is not inferable unless compiled with a constant
            #                       index `i` as in `first(X)` and `last(X)`.
            @test @inferred(first(X)) === X[1]
            @test @inferred( last(X)) === X[N]
            if N ≥ 1
                @test @inferred(X[1]) === A
            end
            if N ≥ 2
                @test @inferred(X[2]) === B
            end
            if N ≥ 3
                @test @inferred(X[3]) === C
            end
            if N ≥ 4
                @test @inferred(X[4]) === D
            end
        end

        # Sums automatically simplify parentheses but do not change order of terms.
        @test @inferred((A + B) + C) === @inferred(A + B + C)
        @test @inferred(A + (B + C)) === @inferred(A + B + C)
        @test @inferred(A + (B + C) + D) === @inferred(A + B + C + D)
        @test @inferred(A + (B + C + D)) === @inferred(A + B + C + D)
        @test @inferred((A + B) + (C + D)) === @inferred(A + B + C + D)

        # Compositions automatically simplify parentheses but do not change order of terms.
        @test @inferred((A * B) * C) === @inferred(A * B * C)
        @test @inferred(A * (B * C)) === @inferred(A * B * C)
        @test @inferred(A * (B * C) * D) === @inferred(A * B * C * D)
        @test @inferred((A * (B * C)) * D) === @inferred(A * B * C * D)
        @test @inferred(A * (B * C * D)) === @inferred(A * B * C * D)
        @test @inferred((A * B) * (C * D)) === @inferred(A * B * C * D)

        # Sums and differences.
        @test @inferred(A - B) === @inferred(A + (-𝟙)*B)
        @test @inferred(A + B - C) === @inferred(A + B + (-𝟙)*C)
        @test @inferred(A - B - C) === @inferred(A + (-𝟙)*B + (-𝟙)*C)
        @test @inferred(A - (B + C)) === @inferred(A + (-𝟙)*(B + C))
        @test @inferred(-A + B - (C + D)) === @inferred((-𝟙)*A + B + (-𝟙)*(C + D))

        # Composition of operators.
        @test @inferred(A ∘ B) === @inferred(A * B)
        @test @inferred(A ∘ B ∘ C) === @inferred(A * B * C)
        @test @inferred(A ∘ (B ∘ C) ∘ D) === @inferred(A * B * C * D)

        # Scaled operators.
        λ = 1 + 2im # complex with integer parts for exact result
        @test @inferred(λ*A) isa Scaled{typeof(λ),typeof(A)}
        @test @inferred(λ*A)[1] === λ
        @test @inferred(λ*A)[2] === A
        @test @inferred(A*λ) === @inferred(λ*A)
        @test @inferred(λ*(π*A)) isa Scaled{typeof(λ*π),typeof(A)}
        @test @inferred(λ*(π*A)) === @inferred((λ*π)*A)
        @test @inferred(λ*A*π) === @inferred((λ*π)*A)
        @test @inferred((λ*A)*π) === @inferred((λ*π)*A)
        @test @inferred(λ*(A*π)) === @inferred((λ*π)*A)

        # Automatic simplification rules for products.
        λ = 1 + 2im # complex with integer parts for exact result
        @test @inferred(λ*A*B) isa Scaled{typeof(λ),typeof(A*B)}
        @test @inferred(A*(λ*B)) === @inferred(λ*A*B)
        @test @inferred(A*(λ*B)*C) === @inferred(λ*A*B*C)
        @test @inferred((A*B)*(λ*C)) === @inferred(λ*A*B*C)
        @test @inferred((A*B)*(λ*C)*D) === @inferred(λ*A*B*C*D)
        @test @inferred(A*(B*C)*(λ*D)) === @inferred(λ*A*B*C*D)
        @test @inferred((A*B)*(λ*C)*(2*D)) === @inferred((2λ)*A*B*C*D)
        @test @inferred(A*(2*B)*(5*C)*(3*D)) === @inferred(30*A*B*C*D)
        @test @inferred(A*(-B)*(λ*C)*(2*D)) === @inferred((-2λ)*A*B*C*D)

        # Division of operators. NOTE Beware that `A*B\C` is like `(A*B)\C` in Julia, not `A*(B\C)`.
        @test @inferred(A / B) === @inferred(A * inv(B))
        @test @inferred(A \ B) === @inferred(inv(A) * B)
        @test @inferred(A / B * C) === @inferred(A * inv(B) * C)
        @test @inferred(A / (B * C)) === @inferred(A * (inv(C) * inv(B)))
        @test @inferred(A \ B * C) === @inferred(inv(A) * B * C)
        @test @inferred((A \ B) * C) === @inferred(inv(A) * B * C)
        @test @inferred(A * B / C * D) === @inferred(A * B * inv(C) * D)
        @test @inferred(A * B / (C * D)) === @inferred(A * B * (inv(D) * inv(C)))
        @test @inferred(A * B \ C * D) === @inferred(inv(A * B) * C * D) # cf. above note
        @test @inferred(A * B \ C * D) === @inferred((A * B) \ C * D) # cf. above note

        # Adjoint of an operator
        @test A' isa Adjoint{typeof(A)}
        @test A' === @inferred(adjoint(A))
        @test @inferred(adjoint(A')) === A
        @test @inferred(parent(A')) === A
        @test @inferred(getindex(A')) === A
        @test A'[] === A
        @test A'' === A

        # Adjoint of a sum.
        @test (A + B)'         isa Sum
        @test (A + B + C)'     isa Sum
        @test (A + B + C + D)' isa Sum
        @test (A + B)'         === @inferred(adjoint(A + B))
        @test (A + B + C)'     === @inferred(adjoint(A + B + C))
        @test (A + B + C + D)' === @inferred(adjoint(A + B + C + D))
        @test Tuple((A + B)')         === (A', B')
        @test Tuple((A + B + C)')     === (A', B', C')
        @test Tuple((A + B + C + D)') === (A', B', C', D')
        @test (A + B)'         === @inferred(A' + B')
        @test (A + B + C)'     === @inferred(A' + B' + C')
        @test (A + B + C + D)' === @inferred(A' + B' + C' + D')

        # Adjoint of a composition.
        @test (A * B)'         isa Prod
        @test (A * B * C)'     isa Prod
        @test (A * B * C * D)' isa Prod
        @test (A * B)'         === @inferred(adjoint(A * B))
        @test (A * B * C)'     === @inferred(adjoint(A * B * C))
        @test (A * B * C * D)' === @inferred(adjoint(A * B * C * D))
        @test Tuple((A * B)')         === (B', A')
        @test Tuple((A * B * C)')     === (C', B', A')
        @test Tuple((A * B * C * D)') === (D', C', B', A')
        @test (A * B)'         === @inferred(B' * A')
        @test (A * B * C)'     === @inferred(C' * B' * A')
        @test (A * B * C * D)' === @inferred(D' * C' * B' * A')

        # Transpose of an operator
        @test @inferred(transpose(A)) isa Transpose{typeof(A)}
        @test @inferred(transpose(transpose(A))) === A
        @test @inferred(parent(transpose(A))) === A
        @test @inferred(getindex(transpose(A))) === A
        @test transpose(A)[] === A

        # Transpose of a sum.
        @test @inferred(transpose(A + B))         isa Sum
        @test @inferred(transpose(A + B + C))     isa Sum
        @test @inferred(transpose(A + B + C + D)) isa Sum
        @test @inferred(transpose(A + B))         === @inferred(transpose(A) + transpose(B))
        @test @inferred(transpose(A + B + C))     === @inferred(transpose(A) + transpose(B) + transpose(C))
        @test @inferred(transpose(A + B + C + D)) === @inferred(transpose(A) + transpose(B) + transpose(C) + transpose(D))
        @test Tuple(transpose(A + B))         === (transpose(A), transpose(B))
        @test Tuple(transpose(A + B + C))     === (transpose(A), transpose(B), transpose(C))
        @test Tuple(transpose(A + B + C + D)) === (transpose(A), transpose(B), transpose(C), transpose(D))

        # Transpose of a composition.
        @test @inferred(transpose(A * B))         isa Prod
        @test @inferred(transpose(A * B * C))     isa Prod
        @test @inferred(transpose(A * B * C * D)) isa Prod
        @test @inferred(transpose(A * B))         === @inferred(transpose(B) * transpose(A))
        @test @inferred(transpose(A * B * C))     === @inferred(transpose(C) * transpose(B) * transpose(A))
        @test @inferred(transpose(A * B * C * D)) === @inferred(transpose(D) * transpose(C) * transpose(B) * transpose(A))
        @test Tuple(transpose(A * B))         === (transpose(B), transpose(A))
        @test Tuple(transpose(A * B * C))     === (transpose(C), transpose(B), transpose(A))
        @test Tuple(transpose(A * B * C * D)) === (transpose(D), transpose(C), transpose(B), transpose(A))

        # Transpose of products.
        λ = 1 + 2im # complex with integer parts for exact result
        @test @inferred(transpose(π*A)) isa Scaled{typeof(π),typeof(transpose(A))}
        @test @inferred(transpose(λ*A)) isa Scaled{typeof(λ),typeof(transpose(A))}
        @test @inferred(transpose(π*A)) === π*transpose(A)
        @test @inferred(transpose(λ*A)) === λ*transpose(A)

        # Conjugate of an operator
        @test @inferred(conj(A)) isa Conjugate{typeof(A)}
        @test @inferred(conj(conj(A))) === A
        @test @inferred(parent(conj(A))) === A
        @test @inferred(getindex(conj(A))) === A
        @test conj(A)[] === A

        # Conjugate of a sum.
        @test @inferred(conj(A + B))         isa Sum
        @test @inferred(conj(A + B + C))     isa Sum
        @test @inferred(conj(A + B + C + D)) isa Sum
        @test @inferred(conj(A + B))         === @inferred(conj(A) + conj(B))
        @test @inferred(conj(A + B + C))     === @inferred(conj(A) + conj(B) + conj(C))
        @test @inferred(conj(A + B + C + D)) === @inferred(conj(A) + conj(B) + conj(C) + conj(D))
        @test Tuple(conj(A + B))         === (conj(A), conj(B))
        @test Tuple(conj(A + B + C))     === (conj(A), conj(B), conj(C))
        @test Tuple(conj(A + B + C + D)) === (conj(A), conj(B), conj(C), conj(D))

        # Conjugate of a composition.
        @test @inferred(conj(A * B))         isa Prod
        @test @inferred(conj(A * B * C))     isa Prod
        @test @inferred(conj(A * B * C * D)) isa Prod
        @test @inferred(conj(A * B))         === @inferred(conj(A) * conj(B))
        @test @inferred(conj(A * B * C))     === @inferred(conj(A) * conj(B) * conj(C))
        @test @inferred(conj(A * B * C * D)) === @inferred(conj(A) * conj(B) * conj(C) * conj(D))
        @test Tuple(conj(A * B))         === (conj(A), conj(B))
        @test Tuple(conj(A * B * C))     === (conj(A), conj(B), conj(C))
        @test Tuple(conj(A * B * C * D)) === (conj(A), conj(B), conj(C), conj(D))

        # Conjugate of products.
        λ = 1 + 2im # complex with integer parts for exact result
        @test @inferred(conj(π*A)) isa Scaled{typeof(conj(π)),typeof(conj(A))}
        @test @inferred(conj(λ*A)) isa Scaled{typeof(conj(λ)),typeof(conj(A))}
        @test @inferred(conj(π*A)) === conj(π)*conj(A)
        @test @inferred(conj(λ*A)) === conj(λ)*conj(A)

        # Inverse of a number
        @test @inferred(inverse(2)) === 1//2
        @test @inferred(inverse(3.0 - 2.0im)) ≈ (3.0 + 2.0im)/13.0

        # Inverse of an operator
        @test @inferred(inv(A)) isa Inverse{typeof(A)}
        @test @inferred(inv(A)) === Inverse(A)
        @test @inferred(inv(inv(A))) === A
        @test @inferred(parent(inv(A))) === A
        @test @inferred(getindex(inv(A))) === A
        @test inv(A)[] === A

        # Inverse of a sum.
        @test @inferred(inv(A + B)) isa Inverse{<:Sum}
        @test @inferred(inv(A + B)[]) === A + B
        @test @inferred(parent(inv(A + B))) === A + B
        @test @inferred(inv(A + B + C)) isa Inverse{<:Sum}
        @test @inferred(inv(A + B + C)[]) === A + B + C
        @test @inferred(parent(inv(A + B + C))) === A + B + C
        @test @inferred(inv(A + B + C + D)) isa Inverse{<:Sum}
        @test @inferred(inv(A + B + C + D)[]) === A + B + C + D
        @test @inferred(parent(inv(A + B + C + D))) === A + B + C + D

        # Inverse of a composition.
        @test @inferred(inv(A * B)) isa Prod
        @test @inferred(inv(A * B)) === inv(B) * inv(A)
        @test @inferred(inv(A * B * C)) isa Prod
        @test @inferred(inv(A * B * C)) === inv(C) * inv(B) * inv(A)
        @test @inferred(inv(A * B * C)) isa Prod
        @test @inferred(inv(A * B * C * D)) === inv(D) * inv(C) * inv(B) * inv(A)

        # Inverse of products.
        λ = 1 + 2im # complex with integer parts for exact result
        @test @inferred(inv(π*A)) isa Scaled{typeof(inv(π)),typeof(inv(A))}
        @test @inferred(inv(λ*A)) isa Scaled{typeof(inv(λ)),typeof(inv(A))}
        @test @inferred(inv(π*A)) === inv(π)*inv(A)
        @test @inferred(inv(λ*A)) === inv(λ)*inv(A)

        # Inverse-adjoint and adjoint-inverse.
        @test inv(A)' isa Inverse{Adjoint{typeof(A)}}
        @test inv(A') isa Inverse{Adjoint{typeof(A)}}
        @test @inferred(adjoint(inv(A))) === inv(A)'
        @test @inferred(inv(adjoint(A))) === @inferred(inv(A'))
        @test @inferred(parent(inv(A'))) === A'
        @test @inferred(parent(inv(A)')) === A'
        @test @inferred(parent(parent(inv(A')))) === A
        @test @inferred(parent(parent(inv(A)'))) === A
        @test @inferred(inv(inv(A'))) === A'
        @test @inferred(inv(inv(A)')) === A'
        @test @inferred(adjoint(inv(A'))) === inv(A)
        @test @inferred(adjoint(inv(A'))) === inv(A)
        @test @inferred(adjoint(inv(adjoint(A)))) === inv(A)
        @test @inferred(inv(adjoint(inv(A)))) === adjoint(A)

        # Inverse-transpose and transpose-inverse.
        @test @inferred(transpose(inv(A))) isa Inverse{Transpose{typeof(A)}}
        @test @inferred(inv(transpose(A))) isa Inverse{Transpose{typeof(A)}}
        @test @inferred(transpose(inv(A))) === @inferred(inv(transpose(A)))
        @test @inferred(parent(transpose(inv(A)))) === transpose(A)
        @test @inferred(parent(inv(transpose(A)))) === transpose(A)
        @test @inferred(parent(parent(transpose(inv(A))))) === A
        @test @inferred(parent(parent(inv(transpose(A))))) === A
        @test @inferred(inv(transpose(inv(A)))) === transpose(A)
        @test @inferred(transpose(inv(transpose(A)))) === inv(A)

        # Scaled operators.
        @testset "Scalar (λ=$λ) times $X" for λ in (0x0, true, 𝟙, -1, 1//2, pi, 2.3f0, 2.0 - 3.0im), X in (A, A + B, A*B)
            @test @inferred(λ*X) === @inferred(X*λ)
            @test @inferred(λ*X) isa Scaled{typeof(λ),typeof(X)}
            @test  first(λ*X) === λ
            @test   last(λ*X) === X
            #
            @test @inferred(X/λ) === @inferred(λ\X)
            @test @inferred(X/λ) isa Scaled{<:Number,typeof(X)}
            @test  first(X/λ) === inverse(λ)
            @test   last(X/λ) === X
            #
            @test @inferred(adjoint((λ*X))) === (λ*X)'
            @test @inferred(adjoint((X*λ))) === (X*λ)'
            @test @inferred(conj(λ)*X') ===  (λ*X)'
            @test @inferred(adjoint(λ*X)) isa Scaled{<:Number,typeof(X')}
            @test  first((λ*X)') === conj(λ)
            @test   last((λ*X)') === X'
            #
            @test @inferred(inv(λ*X)) === @inferred(inverse(λ)*inv(X))
            @test @inferred(inv(X*λ)) === @inferred(inverse(λ)*inv(X))
            @test @inferred(inv(λ*X)) isa Scaled{<:Number,typeof(inv(X))}
            @test  first(inv(λ*X)) === inverse(λ)
            @test   last(inv(λ*X)) === inv(X)
            #
            @test (X/λ)' === @inferred(adjoint((X/λ)))
            @test (λ\X)' === @inferred(adjoint((X/λ)))
            @test @inferred(adjoint(X/λ)) isa Scaled{<:Number,typeof(X')}
            @test  first((X/λ)') === inverse(conj(λ))
            @test   last((X/λ)') === X'
            #
            @test @inferred(inv(λ\X)) === @inferred(inv(X/λ))
            @test @inferred(inv(X/λ)) isa Scaled{<:Number,typeof(inv(X))}
            @test  first(inv(X/λ)) ≈ λ
            @test   last(inv(X/λ)) === inv(X)
            #
            @test @inferred(inv((λ\X)')) === @inferred(inv((X/λ)'))
            @test @inferred(adjoint(inv((X/λ)))) isa Scaled{<:Number,typeof(inv(X'))}
            @test  first(inv((X/λ)')) ≈ conj(λ)
            @test   last(inv((X/λ)')) === inv(X')
        end
#
#        # Left-factorization of scalar in products.
#        α, β = 3//4, -2.0 + 3.0im
#        X, Y = B + C*D, A - D
#        @test @inferred(A*α) === @inferred(α*A)
#        @test typeof(α*A) <: Scaled{typeof(α),typeof(A)}
#        @test (A*B)*α === A*(B*α) === A*(α*B) === (A*α)*B === (α*A)*B === α*(A*B) === α*A*B
#        @test typeof(α*A*B) <: Scaled{typeof(α),typeof(A*B)}
#        @test (A*X)*α === A*(X*α) === A*(α*X) === (A*α)*X === (α*A)*X === α*(A*X) === α*A*X
#        @test typeof(α*A*X) <: Scaled{typeof(α),typeof(A*X)}
#        @test (A*B)/α === A*(B/α) === A*(α\B) === (A/α)*B === (α\A)*B === α\(A*B)
#        @test typeof(α\(A*B)) <: Scaled{<:Number,typeof(A*B)}
#        #
#        @test @inferred(α*X + Y*β) === @inferred(α*X + β*Y)
#        @test @inferred(X*α + β*Y) === @inferred(α*X + β*Y)
#        @test @inferred(X*α + Y*β) === @inferred(α*X + β*Y)
#        @test typeof(α*X + β*Y) <: Sum{Scaled{typeof(α),typeof(X)},Scaled{typeof(β),typeof(Y)}}
#        #
#        @test @inferred((α*X)*(β*Y)) === @inferred((α*β)*(X*Y))
#        @test @inferred((α*X)*(Y*β)) === @inferred((α*β)*(X*Y))
#        @test @inferred((X*α)*(β*Y)) === @inferred((α*β)*(X*Y))
#        @test @inferred((X*α)*(Y*β)) === @inferred((α*β)*(X*Y))
#        @test typeof((α*β)*(X*Y)) <: Scaled{<:Number,typeof(X*Y)}
#
#        # `A\β` and `β/A` intentionally not supported.
#        β = 3
#        @test_throws Exception A\β
#        @test_throws Exception β/A
#        @test A\(β*I) === β*inv(A)
#        @test (β*I)/A === β*inv(A)
#        @test A\(β*Id) === β*inv(A)
#        @test (β*Id)/A === β*inv(A)
#
        # Scaling by neutral numbers. FIXME
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
        @test @inferred(A + A) isa Sum{Tuple{typeof(A),typeof(A)}}
        @test @inferred(2A) isa Scaled{Int,typeof(A)}
        #
        @test @inferred(A - A) != @inferred(0A)
        @test @inferred(A - A) === @inferred(A + (-𝟙)*A)
        @test @inferred(A - A) isa Sum{Tuple{typeof(A),Scaled{typeof(-𝟙),typeof(A)}}}
        @test @inferred(0A) isa Scaled{Int,typeof(A)}
        #
        @test @inferred(Id + Id) === 2Id
        @test @inferred(Id + Id) isa Scaled{Int,typeof(Id)}
        #
        @test @inferred(Id - Id) === @inferred(𝟘*Id)
        @test @inferred(Id - Id) isa Scaled{typeof(𝟘),typeof(Id)}
        #
        @test @inferred(Id + 3Id - 2Id) === @inferred(2Id)
        @test @inferred(2Id) isa Scaled{<:Number,typeof(Id)}

        # Neutral element for the addition.
        @test @inferred(zero(A)) === @inferred(𝟘*A)
        @test @inferred(zero(π*A)) === @inferred(zero(A))
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
