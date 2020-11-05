#
# rules-tests.jl -
#
# Test algebraic rules and simplifications.
#
module TestingLazyAlgebraRules

using LazyAlgebra
import LazyAlgebra: ⋅,
    Adjoint, Inverse, InverseAdjoint, Jacobian,
    Scaled, Sum, Composition
using Test

function to_string(A::Mapping)
    io = IOBuffer()
    show(io, A)
    String(take!(io))
end

@testset "Algebraic rules" begin
    include("common.jl")
    identical = LazyAlgebra.identical
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
    @test identical(A, B) == false
    let E = A
        @test identical(A, E) == true
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
    @test is_selfadjoint(A'*A) == true # FIXME: was broken
    @test is_selfadjoint(A*A') == true # FIXME: was broken
    @test_broken is_selfadjoint(B'*A*A'*B) == true

    # Test identity.
    @test Id === LazyAlgebra.Id
    @test Id' === Id
    @test inv(Id) === Id
    @test 1Id === Id
    @test Id*M*Id === M
    @test Id*Id === Id
    @test Id*Id*Id === Id
    @test Id\Id === Id
    @test Id/Id === Id
    @test Id + Id === 2Id
    @test Id - Id === 0Id
    @test Id + 2Id === 3Id
    @test 2Id + 3Id === 5Id
    @test -4Id + (Id + 2Id) === -Id
    @test Id + Id - 2Id === 0Id
    @test 2Id - (Id + Id) === 0Id
    @test inv(3Id) === (1/3)*Id
    @test SelfAdjointType(Id) === SelfAdjoint()
    @test MorphismType(Id) === Endomorphism()
    @test DiagonalType(Id) === DiagonalMapping()
    for T in (Float32, Float64)
        atol, rtol = zero(T), sqrt(eps(T))
        x = randn(T, dims)
        y = randn(T, dims)
        @test Id*x === x # test same object
        for P in (Direct, Adjoint, Inverse, InverseAdjoint)
            @test apply(P,Id,x) === x # test same object
            test_api(P, Id, x, y)
        end
    end

    # Neutral elements.
    @test isone(Id)
    @test iszero(Id) == false
    @test iszero(A - A)
    @test iszero(-M + M)
    @test one(A) === Id
    @test one(M) === Id
    @test zero(A) === 0*A
    @test zero(M) === 0*M

    # Basic methods for sums and compositions of mappings.
    let A1 = A + B + M, A2 = C*B*M*Q
        @test eltype(A1) <: Mapping
        @test eltype(A2) <: Mapping
        @test Tuple(A1) === terms(A1)
        @test Tuple(A2) === terms(A2)
        @test length(A1) == 3
        @test length(A2) == 4
        @test firstindex(A1) == 1
        @test firstindex(A2) == 1
        @test lastindex(A1) == length(A1)
        @test lastindex(A2) == length(A2)
        @test first(A1) === A1[1]
        @test first(A2) === A2[1]
        @test last(A1) === A1[end]
        @test last(A2) === A2[end]
    end

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

    # Test adjoint and Jacobian.
    x = nothing
    @test (A*A')' === A*A'
    for X in (A*A', A'*A, B'*A'*A*B, B'*A*A'*B)
        @test X' === X
    end
    @test A' === Adjoint(A)
    @test A' isa Adjoint
    @test A'' === (A')' === A
    @test adjoint(A) === A'
    @test jacobian(A,x) === A
    @test ∇(A,x) === A
    @test (3A)' === 3*(A')
    @test (A + 2B)' - A' === 2*B'
    @test_throws ArgumentError Jacobian(A,x)
    @test_throws ArgumentError M'
    @test_throws ArgumentError adjoint(M)
    @test_throws ArgumentError Adjoint(M)
    @test jacobian(M,x) isa Jacobian
    @test ∇(M,x) === jacobian(M,x)
    @test ∇(3M,x) === 3*∇(M,x)
    @test ∇(M,x) + ∇(2M,x) === 3∇(M,x)

    # Inverse.
    @test inv(M) === Id/M
    @test inv(inv(M)) === M
    @test Id/A === inv(A)
    @test Id\A === A
    @test M/Q === M*inv(Q)
    @test M\Q === inv(M)*Q
    @test inv(2M) === inv(M)*(2\Id)
    @test inv(2M) === inv(M)*((1/2)*Id)
    @test inv(2A) === 2\inv(A)
    @test inv(2A) === (1/2)*inv(A)
    @test inv(A*B) === inv(B)*inv(A)
    @test A/3B === 3\A/B
    @test 4A/4B === A/B
    @test 4A\4B === A\B
    @test inv(A*M*B*Q) === inv(Q)*inv(B)*inv(M)*inv(A)
    @test inv(M)*M === Id
    @test M*inv(M) === Id
    let D = M*Q*(A - B)
        @test inv(D)*D === Id
        @test D*inv(D) === Id
    end
    let D = A + 2B - C
        @test inv(D)*D === Id
        @test D*inv(D) === Id
    end
    @test inv(M*Q*(A - B)) === inv(A - B)*inv(Q)*inv(M)
    @test inv(A*3M) === inv(M)*(3\inv(A))
    @test inv(A*3B) === 3\inv(A*B) === 3\inv(B)*inv(A)
    @test inv(A*B*3M) === inv(M)*(3\inv(B))*inv(A)
    @test inv(A*B*3C) === 3\inv(A*B*C) === 3\inv(C)*inv(B)*inv(A)

    # Inverse-adjoint.
    @test inv(A)' === inv(A')
    @test inv(A')*A' === Id
    @test A'*inv(A') === Id
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

    # Test forbidden calls to constructors because they should yield an
    # instance of a different type if simplification rules were applied.
    @test_throws ArgumentError Adjoint(Id)
    @test_throws ArgumentError Adjoint(A')
    @test_throws ArgumentError Adjoint(inv(A))
    @test_throws ArgumentError Adjoint(inv(A'))
    @test_throws ArgumentError Adjoint(3A)
    @test_throws ArgumentError Adjoint(A + B)
    @test_throws ArgumentError Adjoint(A*B)

    @test_throws ArgumentError Inverse(Id)
    @test_throws ArgumentError Inverse(A')
    @test_throws ArgumentError Inverse(inv(A))
    @test_throws ArgumentError Inverse(inv(A'))
    @test_throws ArgumentError Inverse(3A)
    @test_throws ArgumentError Inverse(A*B)

    @test_throws ArgumentError InverseAdjoint(Id)
    @test_throws ArgumentError InverseAdjoint(A')
    @test_throws ArgumentError InverseAdjoint(inv(A))
    @test_throws ArgumentError InverseAdjoint(inv(A'))
    @test_throws ArgumentError InverseAdjoint(3A)
    @test_throws ArgumentError InverseAdjoint(A*B)

    @test_throws ArgumentError Jacobian(3M,x)

    @test_throws ArgumentError Scaled(2,3M)

    # Check that sums and compositions must have at least 2 terms
    @test_throws ArgumentError Sum()
    @test_throws ArgumentError Sum(A)
    @test Sum(A,B) isa Sum
    @test_throws ArgumentError Composition()
    @test_throws ArgumentError Composition(A)
    @test Composition(A,B) isa Composition

    # Test the `show` method.
    @test to_string(A) == "A"
    @test to_string(A') == "A'"
    @test to_string(A + A) == "2⋅A"
    @test to_string(A' + A') == "2⋅A'"
    @test to_string(Id/A) == "inv(A)"
    @test to_string(Id/(A + B)) == "inv($(to_string(A + B)))"
    @test to_string(M) == "M"
    @test to_string(M + M) == "2⋅M"
    @test to_string(∇(M,x)) == "∇(M,x)"
    @test to_string(∇(M,x) + ∇(M,x)) == "2⋅∇(M,x)"
end # testset
nothing

end # module
