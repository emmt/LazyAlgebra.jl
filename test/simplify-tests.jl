"""

# Tests for simplification rules

Typical usage:

    include("test/simplify-tests.jl").runtests();

"""
module LazyAlgebraSimplifyTests

using LazyAlgebra
using Test

function runtests()

    @testset "Simplification rules" begin
        @testset "General simplification rules" begin
            A, B, C, D = SymbolicOperator.((:A, :B, :C, :D))

            # Identities that hold by construction.
            @test inv(A*B) === inv(B)*inv(A)
            @test inv(A*B*C) === inv(C)*inv(B)*inv(A)
            @test (A*B)' === (B')*(A')
            @test (A*B*C)' === (C')*(B')*(A')
            @test inv(A*B*C)' === inv(A)'*inv(B)'*inv(C)'
            @test inv((A*B*C)') === inv(A)'*inv(B)'*inv(C)'
            @test inv(A*B*C)' === inv(A')*inv(B')*inv(C')
            @test ((1 + 2im)*A)' === (1 - 2im)*(A')
            @test inv(3*A) === (1//3)*inv(A)

            # Most simple simplification.
            @test simplify(A) === A

            # The composition of linear operators is non-commutative
            @test simplify(A*B) === A*B
            @test simplify(B*A) === B*A
            @test simplify(A*B*C) === A*B*C
            @test simplify(A*C*B) === A*C*B
            @test simplify(B*A*C) === B*A*C
            @test simplify(B*C*A) === B*C*A
            @test simplify(C*A*B) === C*A*B
            @test simplify(C*B*A) === C*B*A

            # Scalar multipliers commute with linear operators.
            @test simplify(2*A) === 2A
            @test simplify(A*2) === 2A
            @test simplify(1*A) === A
            @test simplify(A*1) === A
            @test simplify(A*2*B) === 2*A*B
            @test simplify(A*B*2) === 2*A*B
            @test simplify(π*A*B*C) === π*A*B*C
            @test simplify(A*π*B*C) === π*A*B*C
            @test simplify(A*B*π*C) === π*A*B*C
            @test simplify(A*B*C*π) === π*A*B*C
            let X = simplify(B + 2D)
                @test simplify(A*B*(B + 2D)*C) === A*B*X*C
                @test simplify(A*B*(D + B + D)*C) === A*B*X*C
                @test simplify(A*B*(2D + B)*C) === A*B*X*C
            end
            @test simplify(A*B + 3*A*B - C - 2*A*B) === simplify(2*A*B - C)

            # The result of simplifying a sum must not depend on the order of the operand.
            #
            # ... sum of 2 terms
            @test simplify(A + B) === simplify(B + A)
            # ... sum of 3 terms
            let X = simplify(A + B + C)
                @test simplify(A + C + B) === X
                @test simplify(B + A + C) === X
                @test simplify(B + C + A) === X
                @test simplify(C + A + B) === X
                @test simplify(C + B + A) === X
            end
            # ... sum of 4 terms
            let X = simplify(A + B + C + D)
                @test simplify(A + B + D + C) === X
                @test simplify(A + C + B + D) === X
                @test simplify(A + C + D + B) === X
                @test simplify(A + D + B + C) === X
                @test simplify(A + D + C + B) === X
                @test simplify(B + A + C + D) === X
                @test simplify(B + A + D + C) === X
                @test simplify(B + C + A + D) === X
                @test simplify(B + C + D + A) === X
                @test simplify(B + D + A + C) === X
                @test simplify(B + D + C + A) === X
                @test simplify(C + A + B + D) === X
                @test simplify(C + A + D + B) === X
                @test simplify(C + B + A + D) === X
                @test simplify(C + B + D + A) === X
                @test simplify(C + D + A + B) === X
                @test simplify(C + D + B + A) === X
                @test simplify(D + A + B + C) === X
                @test simplify(D + A + C + B) === X
                @test simplify(D + B + A + C) === X
                @test simplify(D + B + C + A) === X
                @test simplify(D + C + A + B) === X
                @test simplify(D + C + B + A) === X
            end

            @test simplify(2*(A + B) - (A + B)) === simplify(A + B)
            @test simplify(2*(A + B) - B - A) === simplify(A + B)

            # Product with inverse, etc.
            @test simplify(A/A) === Id
            @test simplify(A\A) === Id
            @test simplify(A*inv(A)) === Id
            @test simplify(inv(A)*A) === Id
            @test simplify(A'*inv(A')) === Id
            @test simplify(inv(A')*A') === Id
            @test simplify(A*inv(A)*B) === B
            @test simplify(B*A*inv(A)) === B
            @test simplify(A*B*inv(B)*C) === A*C

            @test simplify(inv(A*B)) === inv(B)*inv(A)
            @test simplify(inv(A*B)*(A*B)) === Id
            @test simplify(B*inv(A*B)*A) === Id

            @test simplify(inv((A*B)')*(A*B)') === Id
            @test simplify(inv((A*B)')*B'*A') === Id
            @test simplify(A'*inv((A*B)')*B') === Id

            @test simplify(inv(A*B)'*(A*B)') === Id
            @test simplify(inv(A*B)'*B'*A') === Id
            @test simplify(A'*inv(A*B)'*B') === Id

            @test simplify(inv(B'*A')*(A*B)') === Id
            @test simplify(inv(A'*B')*B'*A') === Id
            @test simplify(A'*inv((A*B)')*B') === Id
            #@test simplify(A*A') === Gram(A)
            #@test simplify(A'*A) === Gram(A')
            #@test simplify(Gram(A)') === Gram(A)
            #=
            @test simplify_once(Pass(2*A)*Pass(B)) === Pass(2*A*B)
            @test simplify_once(Pass(A)*Pass(2*B)) === Pass(2*A*B)
            @test simplify_once(Pass(A)*Pass(2*B)*Pass(C)) === Pass(2*A*B*C)
            @test simplify_once(Pass(A)*Pass(B)*Pass(2*C)) === Pass(2*A*B*C)

            @test simplify_once(Pass(2*A)*Pass((1//2)*B)) === Pass(A*B)
            @test simplify_once(Pass((1//2)*A)*Pass(2*B)) === Pass(A*B)
            @test simplify_once(Pass((1//2)*A)*Pass(2*B)*Pass(C)) === Pass(A*B*C)
            @test simplify_once(Pass((1//2)*A)*Pass(B)*Pass(2*C)) === Pass(A*B*C)
            =#
        end

        @testset "Simplifications of diagonal operators (T = $T)" for T in (Float64, Complex{Float32})
            A = Diag(rand(T, 3, 4) .+ one(real(T))/100)
            B = Diag(rand(real(T), 3, 4) .+ one(real(T))/100)
            let X = simplify(A + B)
                @test X isa Diag
                @test diag(X) ≈ diag(A) .+ diag(B)
            end
            let X = simplify(A*B)
                @test X isa Diag
                @test diag(X) ≈ diag(A) .* diag(B)
            end
            let X = simplify(A')
                @test X isa Diag
                @test diag(X) ≈ conj.(diag(A))
            end
            let X = simplify(inv(A))
                @test X isa Diag
                @test diag(X) ≈ inv.(diag(A))
            end
            let X = simplify(inv(A'))
                @test X isa Diag
                @test diag(X) ≈ conj.(inv.(diag(A)))
            end
            let X = simplify(A' + B)
                @test X isa Diag
                @test diag(X) ≈ conj.(diag(A)) .+ diag(B)
            end
            let X = simplify(A' + inv(B))
                @test X isa Diag
                @test diag(X) ≈ conj.(diag(A)) .+ inv.(diag(B))
            end
        end
    end
end

end # module
