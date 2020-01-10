using Test
using LazyAlgebra
using LazyAlgebra.FFT
using AbstractFFTs, FFTW

let FLOATS = (Float32, Float64),
    TYPES = (Float32, Float64, Complex{Float32}, Complex{Float64}),
    ALPHAS = (0, 1, -1,  2.71, π),
    BETAS = (0, 1, -1, -1.33, Base.MathConstants.φ)

    @testset "FFT utilities" begin
        dims1 = (1, 2, 3, 4, 5, 7, 9, 287, 511)
        dims2 = (1, 2, 3, 4, 5, 8, 9, 288, 512)
        dims3 = (1, 2, 3, 4, 5, 8, 9, 288, 512)
        @test goodfftdims(dims1) == dims2
        @test goodfftdims(dims1...) == dims2
        @test rfftdims(1,2,3,4,5) == (1,2,3,4,5)
        @test rfftdims(2,3,4,5) == (2,3,4,5)
        @test rfftdims(3,4,5) == (2,4,5)
        @test rfftdims(4,5) == (3,5)
        @test rfftdims(5) == (3,)
        @test LazyAlgebra.FFT.fftfreq(1) == [0]
        @test LazyAlgebra.FFT.fftfreq(2) == [0,-1]
        @test LazyAlgebra.FFT.fftfreq(3) == [0,1,-1]
        @test LazyAlgebra.FFT.fftfreq(4) == [0,1,-2,-1]
        @test LazyAlgebra.FFT.fftfreq(5) == [0,1,2,-2,-1]
    end

    @testset "FFT operator ($T)" for T in TYPES
        R = real(T)
        ϵ = 2*eps(R) # relative tolerance
        for dims in ((45,), (20,), (33,12), (30,20), (4,5,6))
            x = rand(T, dims)
            xsav = vcopy(x)
            F = FFTOperator(x)
            @test x == xsav # check that input has been preserved
            @test input_size(F) == dims
            y = rand(Complex{R}, (T<:Complex ? input_size(F) :
                                  output_size(F)))
            z = (T<:Complex ? fft(x) : rfft(x))
            @test x == xsav # check that input has been preserved
            ysav = vcopy(y)
            w = (T<:Complex ? ifft(y) : irfft(y, dims[1]))
            @test y == ysav # check that input has been preserved
            @test F*x ≈ z atol=0 rtol=ϵ norm=vnorm2
            @test x == xsav # check that input has been preserved
            @test F\y ≈ w atol=0 rtol=ϵ norm=vnorm2
            @test y == ysav # check that input has been preserved
            for α in ALPHAS,
                β in BETAS,
                scratch in (false, true)
                @test apply!(α, Direct, F, x, scratch, β, vcopy(y)) ≈
                    R(α)*z + R(β)*y atol=0 rtol=ϵ
                if scratch
                    vcopy!(x, xsav)
                else
                    @test x == xsav # check that input has been preserved
                end
                @test apply!(α, Inverse, F, y, scratch, β, vcopy(x)) ≈
                    R(α)*w + R(β)*x atol=0 rtol=ϵ
                if scratch
                    vcopy!(y, ysav)
                else
                    @test y == ysav # check that input has been preserved
                end
            end
        end
    end

    @testset "Circular convolution ($T)" for T in TYPES
        R = real(T)
        ϵ = 2*eps(R) # relative tolerance
        n1, n2, n3 = 18, 12, 4
        for dims in ((n1,), (n1,n2), (n1,n2,n3))
            x = rand(T, dims)
            h = rand(T, dims)
            H = CirculantConvolution(h, flags=FFTW.MEASURE)
            F = FFTOperator(x)
            G = F\Diag(F*h)*F
            y1 = H*x
            y2 = G*x
            y3 = (T<:Real ? real(ifft(fft(h).*fft(x)))
                  :              ifft(fft(h).*fft(x)))
            @test y1 ≈ y3 atol=0 rtol=ϵ
            @test y2 ≈ y3 atol=0 rtol=ϵ
            if T<:Real
                y4 = irfft(rfft(h).*rfft(x), n1)
                @test y1 ≈ y4 atol=0 rtol=ϵ
                @test y2 ≈ y4 atol=0 rtol=ϵ
            end
        end
    end

end

nothing
