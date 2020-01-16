#
# fft-tests.jl -
#
# Tests for FFT and circulant convolution operators.
#

using Test
using LazyAlgebra
using AbstractFFTs, FFTW

@testset "FFT methods" begin
    floats = (Float32, Float64)
    types = (Float32, #Float64, Complex{Float32},
             Complex{Float64})
    alphas = (0, 1, -1,  2.71, π)
    betas = (0, 1, -1, -1.33, Base.MathConstants.φ)

    @testset "FFT utilities" begin
        dims1 = (1, 2, 3, 4, 5, 7, 9, 287, 511)
        dims2 = (1, 2, 3, 4, 5, 8, 9, 288, 512)
        dims3 = (1, 2, 3, 4, 5, 8, 9, 288, 512)
        @test LazyAlgebra.FFTs.goodfftdims(dims1) == dims2
        @test LazyAlgebra.FFTs.goodfftdims(map(Int16, dims1)) == dims2
        @test LazyAlgebra.FFTs.goodfftdims(dims1...) == dims2
        @test LazyAlgebra.FFTs.rfftdims(1,2,3,4,5) == (1,2,3,4,5)
        @test LazyAlgebra.FFTs.rfftdims(2,3,4,5) == (2,3,4,5)
        @test LazyAlgebra.FFTs.rfftdims(3,4,5) == (2,4,5)
        @test LazyAlgebra.FFTs.rfftdims(4,5) == (3,5)
        @test LazyAlgebra.FFTs.rfftdims(5) == (3,)
        @test LazyAlgebra.FFTs.fftfreq(1) == [0]
        @test LazyAlgebra.FFTs.fftfreq(2) == [0,-1]
        @test LazyAlgebra.FFTs.fftfreq(3) == [0,1,-1]
        @test LazyAlgebra.FFTs.fftfreq(4) == [0,1,-2,-1]
        @test LazyAlgebra.FFTs.fftfreq(5) == [0,1,2,-2,-1]
    end

    @testset "FFT operator ($T)" for T in types
        R = real(T)
        ϵ = sqrt(eps(R)) # relative tolerance, can certainly be much tighter
        for dims in ((45,), (20,), (33,12), (30,20), (4,5,6))
            # Create a FFT operator given its argument.
            x = rand(T, dims)
            n = length(x)
            xsav = vcopy(x)
            F = FFTOperator(x)
            @test x == xsav # check that input has been preserved
            @test MorphismType(F) == (T<:Complex ? Endomorphism() : Morphism())
            @test input_size(F) == dims
            @test input_size(F) == ntuple(i->input_size(F,i), ndims(x))
            @test output_size(F) == (T<:Complex ? dims :
                                     Tuple(AbstractFFTs.rfft_output_size(x, 1:ndims(x))))
            @test output_size(F) == ntuple(i->output_size(F,i), ndims(x))
            @test input_eltype(F) == T
            @test output_eltype(F) == typeof(complex(zero(R)))
            @test input_ndims(F) == ndims(x)
            @test output_ndims(F) == ndims(x)
            if T<:Complex
                @test LazyAlgebra.FFTs.destroys_input(F.forward) == true
                @test LazyAlgebra.FFTs.destroys_input(F.backward) == true
            else
                @test LazyAlgebra.FFTs.preserves_input(F.forward) == true
                @test LazyAlgebra.FFTs.preserves_input(F.backward) == false
            end
            @test LazyAlgebra.are_same_mappings(F, F) == true
            io = IOBuffer()
            show(io, F)
            @test String(take!(io)) == "FFT"
            @test typeof(F') === Adjoint{typeof(F)}
            @test typeof(inv(F)) === Inverse{typeof(F)}
            @test typeof(inv(F)') === InverseAdjoint{typeof(F)}
            @test typeof(inv(F')) === InverseAdjoint{typeof(F)}
            @test F'*F == length(x)*Identity()
            @test F*F' == length(x)*Identity()
            @test inv(F)*F == Identity()
            @test F*inv(F) == Identity()
            @test inv(F')*F' == Identity()
            @test F'*inv(F') == Identity()
            @test inv(F')*inv(F) == (1//length(x))*Identity()
            @test inv(F)*inv(F') == (1//length(x))*Identity()

            # Create operators which should be considered as the same as F.
            F1 =  FFTOperator(T, dims...)
            @test LazyAlgebra.are_same_mappings(F1, F) == true
            F2 =  FFTOperator(T, map(Int16, dims))
            @test LazyAlgebra.are_same_mappings(F2, F) == true

            # Check applying operator.
            xbad = rand(T, ntuple(i -> (i == 1 ? dims[i]+1 : dims[i]), length(dims)))
            @test_throws DimensionMismatch F*xbad
            y = rand(Complex{R}, (T<:Complex ? input_size(F) :
                                  output_size(F)))
            z = (T<:Complex ? fft(x) : rfft(x))
            @test x == xsav # check that input has been preserved
            ysav = vcopy(y)
            w = (T<:Complex ? ifft(y) : irfft(y, dims[1]))
            @test F*x ≈ z atol=0 rtol=ϵ norm=vnorm2
            @test F\y ≈ w atol=0 rtol=ϵ norm=vnorm2
            @test F(x) == F*x
            @test F'(y) == F'*y
            @test inv(F)(y) == F\y
            @test inv(F')(x) == F'\x
            @test x == xsav # check that input has been preserved
            @test y == ysav # check that input has been preserved
            @test y == ysav # check that input has been preserved
            for α in alphas,
                β in betas,
                scratch in (false, true)
                @test apply!(α, Direct, F, x, scratch, β, vcopy(y)) ≈
                    R(α)*z + R(β)*y atol=0 rtol=ϵ
                if scratch
                    vcopy!(x, xsav)
                else
                    @test x == xsav # check that input has been preserved
                end
                @test apply!(α, Adjoint, F, y, scratch, β, vcopy(x)) ≈
                    R(n*α)*w + R(β)*x atol=0 rtol=ϵ
                if scratch
                    vcopy!(y, ysav)
                else
                    @test y == ysav # check that input has been preserved
                end
                @test apply!(α, Inverse, F, y, scratch, β, vcopy(x)) ≈
                    R(α)*w + R(β)*x atol=0 rtol=ϵ
                if scratch
                    vcopy!(y, ysav)
                else
                    @test y == ysav # check that input has been preserved
                end
                @test apply!(α, InverseAdjoint, F, x, scratch, β, vcopy(y)) ≈
                    R(α/n)*z + R(β)*y atol=0 rtol=ϵ
                if scratch
                    vcopy!(x, xsav)
                else
                    @test x == xsav # check that input has been preserved
                end
            end
        end
    end

    @testset "Circular convolution ($T)" for T in types
        R = real(T)
        ϵ = sqrt(eps(R)) # relative tolerance, can certainly be much tighter
        n1, n2, n3 = 18, 12, 4
        for dims in ((n1,), (n1,n2), (n1,n2,n3))
            # Basic methods.
            x = rand(T, dims)
            n = length(x)
            h = rand(T, dims)
            H = CirculantConvolution(h, shift=false, flags=FFTW.ESTIMATE)
            @test MorphismType(H) == Endomorphism()
            @test input_size(H) == dims
            @test input_size(H) == ntuple(i->input_size(H,i), ndims(x))
            @test output_size(H) == dims
            @test output_size(H) == ntuple(i->output_size(H,i), ndims(x))
            @test input_eltype(H) == T
            @test output_eltype(H) == T
            @test input_ndims(H) == ndims(x)
            @test output_ndims(H) == ndims(x)
            @test eltype(H) == T
            @test size(H) == (dims..., dims...)
            @test ndims(H) == 2*length(dims)
            @test (size(H)..., 1) == ntuple(i->size(H, i), ndims(H)+1)

            # Test apply! method.
            F = FFTOperator(x)
            G = F\Diag(F*h)*F
            y = rand(T, dims)
            xsav = vcopy(x)
            ysav = vcopy(y)
            y1 = H*x
            @test H(x) == y1
            y2 = G*x
            y3 = (T<:Real ? real(ifft(fft(h).*fft(x)))
                  :              ifft(fft(h).*fft(x)))
            @test y1 ≈ y2 atol=0 rtol=ϵ
            @test y1 ≈ y3 atol=0 rtol=ϵ
            if T<:Real
                y4 = irfft(rfft(h).*rfft(x), n1)
                @test y1 ≈ y4 atol=0 rtol=ϵ
            end
            z1 = H'*y
            z2 = G'*y
            z3 = (T<:Real ? real(ifft(conj.(fft(h)).*fft(y)))
                  :              ifft(conj.(fft(h)).*fft(y)))
            @test z1 ≈ z2 atol=0 rtol=ϵ
            @test z1 ≈ z3 atol=0 rtol=ϵ
            if T<:Real
                z4 = irfft(conj.(rfft(h)).*rfft(y), n1)
                @test z1 ≈ z4 atol=0 rtol=ϵ
            end
            for α in alphas,
                β in betas,
                scratch in (false, true)
                @test apply!(α, Direct, H, x, scratch, β, vcopy(y)) ≈
                    R(α)*y1 + R(β)*y atol=0 rtol=ϵ
                if scratch
                    vcopy!(x, xsav)
                else
                    @test x == xsav # check that input has been preserved
                end
                @test apply!(α, Adjoint, H, y, scratch, β, vcopy(x)) ≈
                    R(α)*z1 + R(β)*x atol=0 rtol=ϵ
                if scratch
                    vcopy!(y, ysav)
                else
                    @test y == ysav # check that input has been preserved
                end
            end
        end
    end
end
nothing
