#
# mappings.jl -
#
# Tests for basic mappings.
#
#-------------------------------------------------------------------------------
#
# This file is part of LazyAlgebra (https://github.com/emmt/LazyAlgebra.jl)
# released under the MIT "Expat" license.
#
# Copyright (c) 2017-2018 Éric Thiébaut.
#

#isdefined(:LazyAlgebra) || include("../src/LazyAlgebra.jl")

module LazyAlgebraMappingTests

using LazyAlgebra
using LazyAlgebra: Scaled, Sum, Composition, # not exported by default
    Endomorphism, EndomorphismType,
    is_same_mutable_object
import LazyAlgebra: is_same_mapping

import Base: show

# Deal with compatibility issues.
using Compat
using Compat.Test
using Compat: @warn
@static if isdefined(Base, :MathConstants)
    import Base.MathConstants: φ
end
@static if VERSION ≥ v"0.7.0-DEV.1776"
    using FFTW
end
@static if isdefined(Base, :axes)
    import Base: axes
else
    import Base: indices
    const axes = indices
end

const I = Identity()
@static if VERSION < v"0.7.0-DEV.3449"
    using Base.LinAlg: UniformScaling
else
    using LinearAlgebra: UniformScaling, ⋅
end

const ALPHAS = (0, 1, -1,  2.71, π)
const BETAS = (0, 1, -1, -1.33, φ)
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
# updated.  Variable `verbose` is a boolean indicating whether successes shuld
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
        test_finite_differences()
        test_fft_operator()
        if VERSION ≥ v"0.7"
            test_scaling()
            test_sparse_operator()
        else
            @warn "Some tests are broken for Julia < 0.7"
        end
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
    @mytest is_same_mapping(A, B) == false
    let E = A
        @mytest is_same_mapping(A, E) == true
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
    @mytest SelfAdjointType(I) <: SelfAdjoint
    @mytest MorphismType(I) <: Endomorphism
    @mytest DiagonalType(I) <: DiagonalMapping
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
    @mytest (A + 2B)' - A' === 2*B'


    # Test adjoint.
    @mytest (A*A')' === A*A'
    for X in (A*A', A'*A, B'*A'*A*B, B'*A*A'*B)
        @mytest X' === X
    end
    @mytest A' === adjoint(A)
    @mytest (A')' === A'' === A

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
    # FIXME: should be inv(M)*(3\A)*inv(B)
    @mytest inv(A*B*3M) === inv(M)*(3\I)*inv(A)*inv(B)
    println(inv(A*B*3M))
    @mytest inv(A*B*3C) === (1/3)*inv(C)*inv(A)*inv(B)
    println(inv(A*B*3C))

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
        @test LinearType(A) <: Linear
        @test LinearType(C) <: Linear
        @test MorphismType(C) <: Endomorphism
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

function test_sparse_operator()
    rows, cols = (2,3,4), (5,6)
    nrows, ncols = prod(rows), prod(cols)
    @testset "Sparse matrices ($T)" for T in FLOATS
        A = randn(T, rows..., cols...)
        A[rand(T, size(A)) .≤ 0.7] .= 0 # 70% of zeros
        x = randn(T, cols)
        y = randn(T, rows)
        G = GeneralMatrix(A)
        S = SparseOperator(A, length(rows))
        @test is_endomorphism(S) == (rows == cols)
        @test (EndomorphismType(S) == Endomorphism) == (rows == cols)
        @test output_size(S) == rows
        @test input_size(S) == cols
        atol, rtol = zero(T), sqrt(eps(T))
        mA = reshape(A, nrows, ncols)
        vx = reshape(x, ncols)
        vy = reshape(y, nrows)
        Sx = S*x
        @test almost_equal(Sx, reshape(mA*vx,  rows))
        Sty = S'*y
        @test almost_equal(Sty, reshape(mA'*vy, cols))
        @test almost_equal(Sx, G*x)
        @test almost_equal(Sty, G'*y)
        ## Use another constructor with integer conversion.
        R = SparseOperator(Int32.(output_size(S)),
                           Int64.(input_size(S)),
                           LazyAlgebra.coefs(S),
                           Int32.(LazyAlgebra.rows(S)),
                           Int64.(LazyAlgebra.cols(S)))
        @test almost_equal(Sx, R*x)
        @test almost_equal(Sty, R'*y)
        for α in ALPHAS,
            β in BETAS
            @test almost_equal(apply!(α, Direct, S, x, β, vcopy(y)),
                               T(α)*Sx + T(β)*y)
            @test almost_equal(apply!(α, Adjoint, S, y, β, vcopy(x)),
                               T(α)*Sty + T(β)*x)
        end
    end
end

function test_finite_differences()
    sizes = ((50,), (8, 9), (4,5,6))
    @testset "Finite differences ($T)" for T in FLOATS
        D = SimpleFiniteDifferences()
        DtD = HalfHessian(D)
        for dims in sizes
            x = randn(T, dims)
            y = randn(T, ndims(x), size(x)...)
            z = randn(T, size(x))
            # Apply direct and adjoint of D "by-hand".
            Dx_truth = Array{T}(undef, size(y))
            Dty_truth = Array{T}(undef, size(x))
            fill!(Dty_truth, 0)
            if ndims(x) == 1
                Dx_truth[1,1:end-1] = x[2:end] - x[1:end-1]
                Dx_truth[1,end] = 0
                Dty_truth[2:end]   += y[1,1:end-1]
                Dty_truth[1:end-1] -= y[1,1:end-1]
            elseif ndims(x) == 2
                Dx_truth[1,1:end-1,:] = x[2:end,:] - x[1:end-1,:]
                Dx_truth[1,end,:] .= 0
                Dx_truth[2,:,1:end-1] = x[:,2:end] - x[:,1:end-1]
                Dx_truth[2,:,end] .= 0
                Dty_truth[2:end,:]   += y[1,1:end-1,:]
                Dty_truth[1:end-1,:] -= y[1,1:end-1,:]
                Dty_truth[:,2:end]   += y[2,:,1:end-1]
                Dty_truth[:,1:end-1] -= y[2,:,1:end-1]
            elseif ndims(x) == 3
                Dx_truth[1,1:end-1,:,:] = x[2:end,:,:] - x[1:end-1,:,:]
                Dx_truth[1,end,:,:] .= 0
                Dx_truth[2,:,1:end-1,:] = x[:,2:end,:] - x[:,1:end-1,:]
                Dx_truth[2,:,end,:] .= 0
                Dx_truth[3,:,:,1:end-1] = x[:,:,2:end] - x[:,:,1:end-1]
                Dx_truth[3,:,:,end] .= 0
                Dty_truth[2:end,:,:]   += y[1,1:end-1,:,:]
                Dty_truth[1:end-1,:,:] -= y[1,1:end-1,:,:]
                Dty_truth[:,2:end,:]   += y[2,:,1:end-1,:]
                Dty_truth[:,1:end-1,:] -= y[2,:,1:end-1,:]
                Dty_truth[:,:,2:end]   += y[3,:,:,1:end-1]
                Dty_truth[:,:,1:end-1] -= y[3,:,:,1:end-1]
            end
            Dx = D*x
            Dty = D'*y
            DtDx = DtD*x
            # There should be no differences between Dx and Dx_truth because
            # they are computed in the exact same way.  For Dty and Dty_truth,
            # the comparsion must be approximative.  For testing DtD against
            # D'*D, parenthesis are needed to avoid simplifications.
            atol, rtol = zero(T), 4*eps(T)
            @test vdot(y,Dx) ≈ vdot(Dty,x) atol=atol rtol=sqrt(eps(T))
            @test vnorm2(Dx - Dx_truth) == 0
            @test Dty ≈ Dty_truth atol=atol rtol=rtol norm=vnorm2
            @test DtDx ≈ D'*(D*x) atol=atol rtol=rtol norm=vnorm2
            for α in ALPHAS,
                β in BETAS
                @test apply!(α, Direct, D, x, β, vcopy(y)) ≈
                    T(α)*Dx + T(β)*y atol=atol rtol=rtol norm=vnorm2
                @test apply!(α, Adjoint, D, y, β, vcopy(x)) ≈
                    T(α)*Dty + T(β)*x atol=atol rtol=rtol norm=vnorm2
                @test apply!(α, Direct, DtD, x, β, vcopy(z)) ≈
                    T(α)*DtDx + T(β)*z atol=atol rtol=rtol norm=vnorm2
            end
        end
    end
end

function test_fft_operator()
    @testset "FFT ($T)" for T in FLOATS
        for dims in ((45,), (20,), (33,12), (30,20), (4,5,6))
            for cmplx in (false, true)
                if cmplx
                    x = randn(T, dims) + 1im*randn(T, dims)
                else
                    x = randn(T, dims)
                end
                F = FFTOperator(x)
                if cmplx
                    y = randn(T, dims) + 1im*randn(T, dims)
                else
                    y = randn(T, output_size(F)) + 1im*randn(T, output_size(F))
                end
                ϵ = eps(T)
                atol, rtol = zero(T), eps(T)
                z = (cmplx ? fft(x) : rfft(x))
                w = (cmplx ? ifft(y) : irfft(y, dims[1]))
                @test F*x ≈ z atol=0 rtol=ϵ norm=vnorm2
                @test F\y ≈ w atol=0 rtol=ϵ norm=vnorm2
                for α in ALPHAS,
                    β in BETAS
                    @test apply!(α, Direct, F, x, β, vcopy(y)) ≈
                        T(α)*z + T(β)*y atol=0 rtol=ϵ
                    @test apply!(α, Inverse, F, y, β, vcopy(x)) ≈
                        T(α)*w + T(β)*x atol=0 rtol=ϵ
                end
            end
        end
    end
end

end
