#
# rules-tests.jl -
#
# Test algebraic rules and simplifications.
#

using LazyAlgebra
import LazyAlgebra: ⋅
using Test

@testset "Algebraic rules" begin
    include("common.jl")
    are_same_mappings = LazyAlgebra.are_same_mappings
    Composition = LazyAlgebra.Composition
    dims = (3,4,5)

    M = SymbolicMapping(:M)
    Q = SymbolicMapping(:Q)
    R = SymbolicMapping(:R)
    A = SymbolicLinearMapping(:A)
    B = SymbolicLinearMapping(:B)
    C = SymbolicLinearMapping(:C)
    D = SymbolicLinearMapping(:D)

    # Test properties.
    @test M !== R
    @test A !== B
    @test are_same_mappings(A, B) == false
    let E = A
        @test are_same_mappings(A, E) == true
    end
    @test is_linear(M) == false
    @test is_linear(A) == true
    @test is_linear(A + B) == true
    @test is_linear(A') == true
    @test is_linear(inv(A)) == true
    @test is_linear(inv(M)) == false
    @test is_linear(A + B + C) == true
    @test is_linear(A' + B + C) == true
    @test is_linear(A + B' + C) == true
    @test is_linear(A + B + C') == true
    @test is_linear(M + B + C) == false
    @test is_linear(A + M + C) == false
    @test is_linear(A + B + M) == false
    @test is_linear(A*B*C) == true
    @test is_linear(A'*B*C) == true
    @test is_linear(A*B'*C) == true
    @test is_linear(A*B*C') == true
    @test is_linear(M*B*C) == false
    @test is_linear(A*M*C) == false
    @test is_linear(A*B*M) == false
    @test is_endomorphism(M) == false
    @test is_endomorphism(A) == false
    @test is_selfadjoint(M) == false
    @test is_selfadjoint(A) == false
    @test_broken is_selfadjoint(A'*A) == true
    @test_broken is_selfadjoint(A*A') == true
    @test_broken is_selfadjoint(B'*A*A'*B) == true

    # Test identity.
    @test I === LazyAlgebra.I
    @test I' === I
    @test inv(I) === I
    @test 1I === I
    @test I*M*I === M
    @test I*I === I
    @test I*I*I === I
    @test I\I === I
    @test I/I === I
    @test I + I === 2I
    @test I - I === 0I
    @test I + 2I === 3I
    @test 2I + 3I === 5I
    @test -4I + (I + 2I) === -I
    @test I + I - 2I === 0I
    @test 2I - (I + I) === 0I
    @test inv(3I) === (1/3)*I
    @test SelfAdjointType(I) === SelfAdjoint()
    @test MorphismType(I) === Endomorphism()
    @test DiagonalType(I) === DiagonalMapping()
    for T in (Float32, Float64)
        atol, rtol = zero(T), sqrt(eps(T))
        x = randn(T, dims)
        y = randn(T, dims)
        @test I*x === x # test same object
        for P in (Direct, Adjoint, Inverse, InverseAdjoint)
            @test apply(P,I,x) === x # test same object
            test_api(P, I, x, y)
        end
    end

    # Neutral elements.
    @test isone(I)
    @test iszero(I) == false
    @test iszero(A - A)
    @test iszero(-M + M)
    @test one(A) === I
    @test one(M) === I
    @test zero(A) === 0*A
    @test zero(M) === 0*M

    @test B + A === A + B
    @test B + A + Q === Q + A + B
    @test B + (M + A + Q) === A + B + M + Q
    @test (-1)*A === -A
    @test -(-A) === A
    @test 2\A === (1/2)*A
    @test A + A === 2A
    @test A + B - A === B
    @test 2R + A - Q + B - 2A + Q + 1R === 3R + B - A
    # FIXME: for the following to work, we must impose the value of the
    #        first multiplier of a sum
    #@test A + B*(M - Q) + A - 2B*(Q - M) === 2*A + 3*B*(M - Q)
    @test A + B*(M - Q) + 3A - 3B*(M - Q) === 4*A - 2*B*(M - Q)
    @test A*2M === 2*(A*M)
    @test A*2M === 2*(A*M)
    @test 3A*2M === 6A*M
    @test 3R*2M !== 6R*M

    # Test adjoint.
    @test (A*A')' === A*A'
    for X in (A*A', A'*A, B'*A'*A*B, B'*A*A'*B)
        @test X' === X
    end
    @test A' === adjoint(A)
    @test (A')' === A'' === A
    @test (A + 2B)' - A' === 2*B'

    # Inverse.
    @test inv(M) === I/M
    @test inv(inv(M)) === M
    @test I/A === inv(A)
    @test I\A === A
    @test M/Q === M*inv(Q)
    @test M\Q === inv(M)*Q
    @test inv(2M) === inv(M)*(2\I)
    @test inv(2M) === inv(M)*((1/2)*I)
    @test inv(2A) === 2\inv(A)
    @test inv(2A) === (1/2)*inv(A)
    @test inv(A*B) === inv(B)*inv(A)
    @test A/3B === 3\A/B
    @test 4A/4B === A/B
    @test 4A\4B === A\B
    @test inv(A*M*B*Q) === inv(Q)*inv(B)*inv(M)*inv(A)
    @test inv(M)*M === I
    @test M*inv(M) === I
    let D = M*Q*(A - B)
        @test inv(D)*D === I
        @test D*inv(D) === I
    end
    let D = A + 2B - C
        @test inv(D)*D === I
        @test D*inv(D) === I
    end
    @test inv(M*Q*(A - B)) === inv(A - B)*inv(Q)*inv(M)
    @test inv(A*3M) === inv(M)*(3\inv(A))
    @test inv(A*3B) === 3\inv(A*B) === 3\inv(B)*inv(A)
    @test inv(A*B*3M) === inv(M)*(3\inv(B))*inv(A)
    @test inv(A*B*3C) === 3\inv(A*B*C) === 3\inv(C)*inv(B)*inv(A)

    # Inverse-adjoint.
    @test inv(A)' === inv(A')
    @test inv(A')*A' === I
    @test A'*inv(A') === I
    let E = inv(A'), F = inv(A)'
        @test inv(E) === A'
        @test inv(F) === A'
    end

    # Test aliases for composition.
    @test isa(M*R, Composition)
    @test M⋅R === M*R
    @test M∘R === M*R

    # Test associativity of sum and composition.
    @test (M + R) + Q === M + R + Q
    @test M + (R + Q) === (M + R) + Q
    @test (M*R)*Q === M*R*Q
    @test M*(R*Q) === (M*R)*Q

    # Test adjoint of sums and compositions.
    @test (A*B)' === (B')*(A') === B'*A'
    @test (A'*B)' === B'*A
    @test (A*B')' === B*A'
    @test (A*B*C)' === C'*B'*A'
    @test (A'*B*C)' === C'*B'*A
    @test (A*B'*C)' === C'*B*A'
    @test (A*B*C')' === C*B'*A'
    @test (A + B)' === A' + B'
    @test (A' + B)' === A + B'
    @test (A + B')' === A' + B
    @test (A' + B + C)' === A + B' + C'
    @test (A + B' + C)' === A' + B + C'
    @test (A + B + C')' === A' + B' + C

    # Test inverse of sums and compositions.

    # Test unary plus and negation.
    @test +M === M
    @test -(-M) === M
    @test -M === (-1)*M
    @test 2A + A === 3A
    @test 10M - 3M === 7M
    @test M + A + M === A + 2M

    @test_throws ArgumentError M' # non-linear
    @test_throws ArgumentError inv(M)' # non-linear
end
nothing
