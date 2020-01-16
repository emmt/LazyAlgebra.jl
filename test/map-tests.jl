#
# map-tests.jl -
#
# Tests for basic mappings.
#

module LazyAlgebraMappingTests

using LazyAlgebra
using LazyAlgebra: Scaled, Sum, Composition, # not exported by default
    Endomorphism, EndomorphismType,
    is_same_mutable_object
import LazyAlgebra: are_same_mappings

using FFTW
using Test
using Printf
import Base: show, axes

const I = Identity()
@static if VERSION < v"0.7.0-DEV.3449"
    using Base.LinAlg: UniformScaling
else
    using LinearAlgebra: UniformScaling, ⋅
end

const ALPHAS = (0, 1, -1,  2.71, π)
const BETAS = (0, 1, -1, -1.33, Base.MathConstants.φ)
const OPERATIONS = (Direct, Adjoint, Inverse, InverseAdjoint)
const FLOATS = (Float32, Float64)
const COMPLEXES = (ComplexF32, ComplexF64)

function almost_equal(x::AbstractArray{Tx,Nx},
                      y::AbstractArray{Ty,Ny};
                      atol::Real=0,
                      rtol::Real=relative_tolerance(Tx,Ty)) where {Tx<:Real,Nx,
                                                                   Ty<:Real,Ny}
    @assert isfinite(atol) && atol ≥ 0
    @assert isfinite(rtol) && 0 ≤ rtol < 1
    return (axes(x) == axes(y) &&
            vnorm2(x - y) ≤ atol + rtol*max(vnorm2(x),vnorm2(y)))
end

relative_tolerance(::Type{Tx}, ::Type{Ty}) where {Tx<:Real,Ty<:Real} =
    sqrt(max(Float64(eps(Tx)), Float64(eps(Ty))))

# FIXME: The following tests result in segmentation fault with Julia ≤ 0.7
#
#     @test -M === (-1)*M
#     @test 3A === 2A + A
#     @test 7M === 10M - 3M
#     @test A + 2M === M + A + M
#
# -------------------------------------------------------------------
# Julia Options                                    Result
# --check-bounds=... --optimize=... --inline=...
# -------------------------------------------------------------------
#                yes            0            no    OK
#                yes            0            yes   Segmentation Fault
#                yes            3            no    OK
#                no             3            no    OK
#                no             3            yes   Segmentation Fault
# -------------------------------------------------------------------
# So it seems to be due to the `--inline=yes` option.

#module MyTest
#
#export
#    @mytest,
#    @mytest_report,
#    @mytest_throws

# Too many tests fail because of a segmentation fault when @test macros are
# used with --inline=yes (the default) in Julia.  The following macro provides
# a substitute.  A truth value is expected and indicate the success of a test;
# otherwise a failure is assumed.  Variables `ntests` and `nfailures` are
# updated.  Variable `verbose` is a boolean indicating whether successes should
# also be reported.
macro mytest(ex)
    msg = :(Main.Base.string($(Expr(:quote,ex))))
    quote
        $(esc(:ntests)) += 1
        if ($(esc(ex)))
            if $(esc(:verbose))
                printstyled("OK: "; color=:green, bold=true)
                printstyled($msg; color=:green)
                println()
            end
        else
            $(esc(:nfailures)) += 1
            printstyled("ERROR: "; color=:red, bold=true)
            printstyled($msg; color=:red)
            println()
        end
        nothing
    end
end

macro mytest_throws(typ, ex)
    msg = :(Main.Base.string($(Expr(:quote, ex))))
    quote
        local err = nothing
        $(esc(:ntests)) += 1
        try
            $(esc(ex))
        catch err
        end
        if isa(err, $(esc(typ)))
            if $(esc(:verbose))
                printstyled("OK: "; color=:green, bold=true)
                printstyled($msg,  " throws ", $(esc(typ)); color=:green)
            end
        else
            $(esc(:nfailures)) += 1
            printstyled("ERROR: "; color=:red, bold=true)
            printstyled($msg, "(expected ", $(esc(typ)), " got ",
                        typeof(err), ")"; color=:red)
            printstyled($msg; color=:red)
        end
        println()
        nothing
    end
end

macro mytest_report()
    ntests = esc(:ntests)
    nfailures = esc(:nfailures)
    quote
        printstyled("Number of tests: ", $ntests, " ("; color=:cyan)
        printstyled("passed: ", $ntests - $nfailures; color=:green)
        if $nfailures ≥ 1
            printstyled(", ", color=:cyan)
            printstyled("failed: ", $nfailures; color=:red)
        end
        printstyled(")", color=:cyan)
        println()
        if $nfailures ≥ 1
            error(string($nfailures, "/", $ntests, " test(s) failed"))
        end
        nothing
    end
end

#end # module MyTest
#
#import .MyTest
#using .MyTest

function test_all()
    nerrors = 0
    @testset "Mappings" begin
        test_rules()
        test_standard_uniform_scaling()
        test_rank_one_operator()
        test_non_uniform_scaling()
        test_generalized_matrices()
        test_scaling()
    end
end

function test_rules(verbose::Bool=false)
    ntests = 0
    nfailures = 0

    dims = (3,4,5)

    M = SymbolicMapping(:M)
    Q = SymbolicMapping(:Q)
    R = SymbolicMapping(:R)
    A = SymbolicLinearMapping(:A)
    B = SymbolicLinearMapping(:B)
    C = SymbolicLinearMapping(:C)
    D = SymbolicLinearMapping(:D)

    # Test properties.
    @mytest M !== R
    @mytest A !== B
    @mytest are_same_mappings(A, B) == false
    let E = A
        @mytest are_same_mappings(A, E) == true
    end
    @mytest is_linear(M) == false
    @mytest is_linear(A) == true
    @mytest is_linear(A + B) == true
    @mytest is_linear(A') == true
    @mytest is_linear(inv(A)) == true
    @mytest is_linear(inv(M)) == false
    @mytest is_linear(A + B + C) == true
    @mytest is_linear(A' + B + C) == true
    @mytest is_linear(A + B' + C) == true
    @mytest is_linear(A + B + C') == true
    @mytest is_linear(M + B + C) == false
    @mytest is_linear(A + M + C) == false
    @mytest is_linear(A + B + M) == false
    @mytest is_linear(A*B*C) == true
    @mytest is_linear(A'*B*C) == true
    @mytest is_linear(A*B'*C) == true
    @mytest is_linear(A*B*C') == true
    @mytest is_linear(M*B*C) == false
    @mytest is_linear(A*M*C) == false
    @mytest is_linear(A*B*M) == false
    @mytest is_endomorphism(M) == false
    @mytest is_endomorphism(A) == false
    @mytest is_selfadjoint(M) == false
    @mytest is_selfadjoint(A) == false
    @mytest is_selfadjoint(A'*A) == false # FIXME: should be true
    @mytest is_selfadjoint(A*A') == false # FIXME: should be true
    @mytest is_selfadjoint(B'*A*A'*B) == false # FIXME: should be true

    # Test identity.
    @mytest I === LazyAlgebra.I
    @mytest I' === I
    @mytest inv(I) === I
    @mytest 1I === I
    @mytest I*M*I === M
    @mytest I*I === I
    @mytest I*I*I === I
    @mytest I\I === I
    @mytest I/I === I
    @mytest I + I === 2I
    @mytest I - I === 0I
    @mytest I + 2I === 3I
    @mytest 2I + 3I === 5I
    @mytest -4I + (I + 2I) === -I
    @mytest I + I - 2I === 0I
    @mytest 2I - (I + I) === 0I
    @mytest inv(3I) === (1/3)*I
    @mytest SelfAdjointType(I) === SelfAdjoint()
    @mytest MorphismType(I) === Endomorphism()
    @mytest DiagonalType(I) === DiagonalMapping()
    for T in FLOATS
        atol, rtol = zero(T), sqrt(eps(T))
        x = randn(T, dims)
        y = randn(T, dims)
        @mytest is_same_mutable_object(I*x, x)
        @mytest is_same_mutable_object(I*y, y)
        for P in OPERATIONS
            @mytest is_same_mutable_object(apply(P,I,x), x)
            @static if VERSION ≥ v"0.7"
                z = vcreate(y)
                for α in ALPHAS, β in BETAS
                    @mytest almost_equal(apply!(α, P, I, x, β, vcopy!(z, y)),
                                         (α*x + β*y), rtol=rtol)
                end
            end
        end
    end

    # Neutral elements.
    @mytest isone(I)
    @mytest iszero(I) == false
    @mytest iszero(A - A)
    @mytest iszero(-M + M)
    @mytest one(A) === I
    @mytest one(M) === I
    @mytest zero(A) === 0*A
    @mytest zero(M) === 0*M

    @mytest B + A === A + B
    @mytest B + A + Q === Q + A + B
    @mytest B + (M + A + Q) === A + B + M + Q
    @mytest (-1)*A === -A
    @mytest -(-A) === A
    @mytest 2\A === (1/2)*A
    @mytest A + A === 2A
    @mytest A + B - A === B
    @mytest 2R + A - Q + B - 2A + Q + 1R === 3R + B - A
    # FIXME: for the following to work, we must impose the value of the
    #        first multiplier of a sum
    #@mytest A + B*(M - Q) + A - 2B*(Q - M) === 2*A + 3*B*(M - Q)
    @mytest A + B*(M - Q) + 3A - 3B*(M - Q) === 4*A - 2*B*(M - Q)
    @mytest A*2M === 2*(A*M)
    @mytest A*2M === 2*(A*M)
    @mytest 3A*2M === 6A*M
    @mytest 3R*2M !== 6R*M

    # Test adjoint.
    @mytest (A*A')' === A*A'
    for X in (A*A', A'*A, B'*A'*A*B, B'*A*A'*B)
        @mytest X' === X
    end
    @mytest A' === adjoint(A)
    @mytest (A')' === A'' === A
    @mytest (A + 2B)' - A' === 2*B'

    # Inverse.
    @mytest inv(M) === I/M
    @mytest inv(inv(M)) === M
    @mytest I/A === inv(A)
    @mytest I\A === A
    @mytest M/Q === M*inv(Q)
    @mytest M\Q === inv(M)*Q
    @mytest inv(2M) === inv(M)*(2\I)
    @mytest inv(2M) === inv(M)*((1/2)*I)
    @mytest inv(2A) === 2\inv(A)
    @mytest inv(2A) === (1/2)*inv(A)
    @mytest inv(A*B) === inv(B)*inv(A)
    @mytest A/3B === 3\A/B
    @mytest 4A/4B === A/B
    @mytest 4A\4B === A\B
    @mytest inv(A*M*B*Q) === inv(Q)*inv(B)*inv(M)*inv(A)
    @mytest inv(M)*M === I
    @mytest M*inv(M) === I
    let D = M*Q*(A - B)
        @mytest inv(D)*D === I
        @mytest D*inv(D) === I
    end
    let D = A + 2B - C
        @mytest inv(D)*D === I
        @mytest D*inv(D) === I
    end
    @mytest inv(M*Q*(A - B)) === inv(A - B)*inv(Q)*inv(M)
    @mytest inv(A*3M) === inv(M)*(3\inv(A))
    @mytest inv(A*3B) === 3\inv(A*B) === 3\inv(B)*inv(A)
    @mytest inv(A*B*3M) === inv(M)*(3\inv(B))*inv(A)
    @mytest inv(A*B*3C) === 3\inv(A*B*C) === 3\inv(C)*inv(B)*inv(A)

    # Inverse-adjoint.
    @mytest inv(A)' === inv(A')
    @mytest inv(A')*A' === I
    @mytest A'*inv(A') === I
    let E = inv(A'), F = inv(A)'
        @mytest inv(E) === A'
        @mytest inv(F) === A'
    end

    # Test aliases for composition.
    @mytest isa(M*R, Composition)
    @mytest M⋅R === M*R
    @mytest M∘R === M*R

    # Test associativity of sum and composition.
    @mytest (M + R) + Q === M + R + Q
    @mytest M + (R + Q) === (M + R) + Q
    @mytest (M*R)*Q === M*R*Q
    @mytest M*(R*Q) === (M*R)*Q

    # Test adjoint of sums and compositions.
    @mytest (A*B)' === (B')*(A') === B'*A'
    @mytest (A'*B)' === B'*A
    @mytest (A*B')' === B*A'
    @mytest (A*B*C)' === C'*B'*A'
    @mytest (A'*B*C)' === C'*B'*A
    @mytest (A*B'*C)' === C'*B*A'
    @mytest (A*B*C')' === C*B'*A'
    @mytest (A + B)' === A' + B'
    @mytest (A' + B)' === A + B'
    @mytest (A + B')' === A' + B
    @mytest (A' + B + C)' === A + B' + C'
    @mytest (A + B' + C)' === A' + B + C'
    @mytest (A + B + C')' === A' + B' + C

    # Test inverse of sums and compositions.

    # Test unary plus and negation.
    @mytest +M === M
    @mytest -(-M) === M
    @mytest -M === (-1)*M
    @mytest 2A + A === 3A
    @mytest 10M - 3M === 7M
    @mytest M + A + M === A + 2M

    @mytest_throws ArgumentError M' # non-linear
    @mytest_throws ArgumentError inv(M)' # non-linear

    @mytest_report
end

function test_standard_uniform_scaling()
    @testset "UniformScaling" begin
        # Check + operator.
        @test I + UniformScaling(1) === 2I
        @test I + UniformScaling(2) === 3I
        @test UniformScaling(1) + I === 2I
        @test UniformScaling(2) + I === 3I
        # Check - operator.
        @test I - UniformScaling(1) === 0I
        @test I - UniformScaling(2) === -I
        @test UniformScaling(1) - I === 0I
        @test UniformScaling(2) - I === I
        # Check * operator.
        @test I*UniformScaling(1) === I
        @test I*UniformScaling(2) === 2I
        @test UniformScaling(1)*I === I
        @test UniformScaling(2)*I === 2I
        # Check \circ operator.
        @test I∘UniformScaling(1) === I
        @test I∘UniformScaling(2) === 2I
        @test UniformScaling(1)∘I === I
        @test UniformScaling(2)∘I === 2I
        # \cdot is specific.
        @test I⋅UniformScaling(1) === I
        @test I⋅UniformScaling(2) === 2I
        @test UniformScaling(1)⋅I === I
        @test UniformScaling(2)⋅I === 2I
        # Check / operator.
        @test I/UniformScaling(1) === I
        @test I/UniformScaling(2) === (1/2)*I
        @test UniformScaling(1)/I === I
        @test UniformScaling(2)/I === 2I
        # Check \ operator.
        @test I\UniformScaling(1) === I
        @test I\UniformScaling(2) === 2I
        @test UniformScaling(1)\I === I
        @test UniformScaling(2)\I === (1/2)*I
    end
end

function test_scaling()
    dims = (3,4,5)
    @testset "Uniform scaling ($T)" for T in FLOATS
        x = randn(T, dims)
        y = randn(T, dims)
        γ = sqrt(2)
        U = γ*I
        atol, rtol = zero(T), sqrt(eps(T))
        @test almost_equal(U*x  , γ*x     , rtol=rtol)
        @test almost_equal(U'*x , γ*x     , rtol=rtol)
        @test almost_equal(U\x  , (1/γ)*x , rtol=rtol)
        @test almost_equal(U'\x , (1/γ)*x , rtol=rtol)
        for α in ALPHAS,
            β in BETAS
            for P in (Direct, Adjoint)
                @test apply!(α, P, U, x, β, vcopy(y)) ≈
                    T(α*γ)*x + T(β)*y atol=atol rtol=rtol norm=vnorm2
            end
            for P in (Inverse, InverseAdjoint)
                @test apply!(α, P, U, x, β, vcopy(y)) ≈
                    T(α/γ)*x + T(β)*y atol=atol rtol=rtol norm=vnorm2
            end
        end
    end
end

function test_rank_one_operator()
    dims = (3,4,5)
    n = prod(dims)
    @testset "Rank 1 operators ($T)" for T in FLOATS
        w = randn(T, dims)
        x = randn(T, dims)
        y = randn(T, dims)
        A = RankOneOperator(w, w)
        B = RankOneOperator(w, y)
        C = SymmetricRankOneOperator(w)
        atol, rtol = zero(T), sqrt(eps(T))
        @test LinearType(A) === Linear()
        @test LinearType(C) === Linear()
        @test MorphismType(C) === Endomorphism()
        @test A*I === A
        @test I*A === A
        @test A*x  ≈ sum(w.*x)*w atol=atol rtol=rtol norm=vnorm2
        @test A'*x ≈ sum(w.*x)*w atol=atol rtol=rtol norm=vnorm2
        @test B*x  ≈ sum(y.*x)*w atol=atol rtol=rtol norm=vnorm2
        @test B'*x ≈ sum(w.*x)*y atol=atol rtol=rtol norm=vnorm2
        @test C*x  ≈ sum(w.*x)*w atol=atol rtol=rtol norm=vnorm2
        @test C'*x ≈ sum(w.*x)*w atol=atol rtol=rtol norm=vnorm2
        for α in ALPHAS,
            β in BETAS
            for P in (Direct, Adjoint)
                @test apply!(α, P, C, x, β, vcopy(y)) ≈
                    T(α*vdot(w,x))*w + T(β)*y atol=atol rtol=rtol norm=vnorm2
            end
        end
    end
end

function test_non_uniform_scaling()

    dims = (3,4,5)
    n = prod(dims)

    @testset "Non-uniform scaling ($T)" for T in FLOATS
        w = randn(T, dims)
        for i in eachindex(w)
            while w[i] == 0
                w[i] = randn(T)
            end
        end
        x = randn(T, dims)
        y = randn(T, dims)
        z = vcreate(y)
        S = NonuniformScalingOperator(w)
        @test diag(S) === w
        @test Diag(w) === S
        atol, rtol = zero(T), sqrt(eps(T))
        @test S*x  ≈ w.*x atol=atol rtol=rtol norm=vnorm2
        @test S'*x ≈ w.*x atol=atol rtol=rtol norm=vnorm2
        @test S\x ≈ x./w atol=atol rtol=rtol norm=vnorm2
        @test S'\x ≈ x./w atol=atol rtol=rtol norm=vnorm2
        for α in ALPHAS,
            β in BETAS
            for P in (Direct, Adjoint)
                @test apply!(α, P, S, x, β, vcopy(y)) ≈
                    T(α)*w.*x + T(β)*y atol=atol rtol=rtol norm=vnorm2
            end
            for P in (Inverse, InverseAdjoint)
                @test apply!(α, P, S, x, β, vcopy(y)) ≈
                    T(α)*x./w + T(β)*y atol=atol rtol=rtol norm=vnorm2
            end
        end
    end

    @testset "Non-uniform scaling (Complex{$T})" for T in FLOATS
        w = complex.(randn(T, dims), randn(T, dims))
        for i in eachindex(w)
            while w[i] == 0
                w[i] = complex(randn(T), randn(T))
            end
        end
        x = complex.(randn(T, dims), randn(T, dims))
        y = complex.(randn(T, dims), randn(T, dims))
        wx = w.*x
        qx = x./w
        z = vcreate(y)
        S = NonuniformScalingOperator(w)
        atol, rtol = zero(T), sqrt(eps(T))
        @test S*x  ≈ w.*x atol=atol rtol=rtol norm=vnorm2
        @test S'*x ≈ conj.(w).*x atol=atol rtol=rtol norm=vnorm2
        @test S\x ≈ x./w atol=atol rtol=rtol norm=vnorm2
        @test S'\x ≈ x./conj.(w) atol=atol rtol=rtol norm=vnorm2
        for α in ALPHAS,
            β in BETAS
            @test apply!(α, Direct, S, x, β, vcopy(y)) ≈
                T(α)*w.*x + T(β)*y atol=atol rtol=rtol norm=vnorm2
            @test apply!(α, Adjoint, S, x, β, vcopy(y)) ≈
                T(α)*conj.(w).*x + T(β)*y atol=atol rtol=rtol norm=vnorm2
            @test apply!(α, Inverse, S, x, β, vcopy(y)) ≈
                T(α)*x./w + T(β)*y atol=atol rtol=rtol norm=vnorm2
            @test apply!(α, InverseAdjoint, S, x, β, vcopy(y)) ≈
                T(α)*x./conj.(w) + T(β)*y atol=atol rtol=rtol norm=vnorm2
        end
    end
end

function test_generalized_matrices()
    rows, cols = (2,3,4), (5,6)
    nrows, ncols = prod(rows), prod(cols)
    @testset "Generalized matrices ($T)" for T in FLOATS
        A = randn(T, rows..., cols...)
        x = randn(T, cols)
        y = randn(T, rows)
        G = GeneralMatrix(A)
        atol, rtol = zero(T), sqrt(eps(T))
        mA = reshape(A, nrows, ncols)
        vx = reshape(x, ncols)
        vy = reshape(y, nrows)
        Gx = G*x
        Gty = G'*y
        @test Gx  ≈ reshape(mA*vx,  rows) atol=atol rtol=rtol norm=vnorm2
        @test Gty ≈ reshape(mA'*vy, cols) atol=atol rtol=rtol norm=vnorm2
        for α in ALPHAS,
            β in BETAS
            @test apply!(α, Direct, G, x, β, vcopy(y)) ≈
                T(α)*Gx + T(β)*y atol=atol rtol=rtol norm=vnorm2
            @test apply!(α, Adjoint, G, y, β, vcopy(x)) ≈
                T(α)*Gty + T(β)*x atol=atol rtol=rtol norm=vnorm2
        end
    end
end

end # module
