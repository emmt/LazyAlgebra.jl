function test_utilities()
    is_flat_array = LazyAlgebra.is_flat_array
    convert_multiplier = LazyAlgebra.convert_multiplier
    @testset "Utilities" begin
        A = ones((2, 3, 4))
        @test is_flat_array(A) == true
        @test is_flat_array(view(A, :, :, 2:3)) == true
        @test is_flat_array(view(A, :, 2:2, :)) == false
        @test is_flat_array(view(A, :, 2, :)) == false
        @test is_flat_array(view(A, :, 2:2, 3)) == true
        @test is_flat_array(view(A, :, 2, 3)) == true
        @test is_flat_array(view(A, :, 2, 3:3)) == false
        @test is_flat_array(A, view(A, :, 2, 3), view(A, :, 2:2, 3)) == true
        @test is_flat_array(A, view(A, :, 2:2, :), view(A, :, 2:2, 3)) == false

        A = ones(Bool, (3, 7))
        B = [true, false]
        C = (false, false)
        @test allof(true) == true
        @test anyof(true) == true
        @test noneof(true) == false
        @test allof(false) == false
        @test anyof(false) == false
        @test noneof(false) == true
        @test allof(A) == true
        @test anyof(A) == true
        @test noneof(A) == false
        @test allof(B) == false
        @test anyof(B) == true
        @test noneof(B) == false
        @test allof(C) == false
        @test anyof(C) == false
        @test noneof(C) == true
        for (x, y) in ((A,A),(B,B),(C,C),(A,B),(B,C),(C,A))
            @test anyof(x, y) == (anyof(x) || anyof(y))
            @test allof(x, y) == (allof(x) && allof(y))
            @test noneof(x, y) == (noneof(x) && noneof(y))
        end
        for x in (A, B, C)
            @test allof(x) == (allof(minimum, x) && allof(maximum, x))
            @test noneof(x) == (noneof(minimum, x) && noneof(maximum, x))
            @test anyof(x) == (anyof(minimum, x) || anyof(maximum, x))
        end

        R = (Float32, Float16, BigFloat, Float64)
        C = (ComplexF32, ComplexF64)
        for k in 1:2
            i = rand(1:length(R))
            Tr = R[i]
            Trp = R[i < length(R) ? i + 1 : 1]
            Tc = C[rand(1:length(C))]
            @test convert_multiplier(1, Tr) == convert(Tr, 1)
            @test isa(convert_multiplier(π, Tc), AbstractFloat)
            @test convert_multiplier(2.0, Tr) === convert(Tr, 2)
            @test convert_multiplier(2.0, Tr, Trp) === convert(Tr, 2)
            @test convert_multiplier(2.0, Tr, Tc) === convert(Tr, 2)
            @test convert_multiplier(2.0, Tc, Tc) === convert(real(Tc), 2)
            @test convert_multiplier(1+0im, Tr) == 1
            @test convert_multiplier(1+0im, Tr, Trp) == 1
            @test_throws InexactError convert_multiplier(1+2im, Tr)
            @test convert_multiplier(1+2im, Tr, Tc) == 1+2im
        end
    end
end
