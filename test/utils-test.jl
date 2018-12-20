using Random
function test_utilities()
    is_flat_array = LazyAlgebra.is_flat_array
    convert_multiplier = LazyAlgebra.convert_multiplier
    @testset "Utilities" begin
        #
        # Tests for `is_flat_array`.
        #
        A = ones((2, 3, 4))
        L = ((nothing,             false),
             ("a",                 false),
             ((),                  false),
             (1,                   false),
             (A,                   true ),
             (view(A, :, :, 2:3),  true ),
             (view(A, :, 2:2, :),  false),
             (view(A, :, 2, :),    false),
             (view(A, :, 2:2, 3),  true ),
             (view(A, :, 2, 3),    true ),
             (view(A, :, 2, 3:3),  false))
        for i in randperm(length(L)) # prevent compilation-time optimization
            x, b = L[i]
            @test is_flat_array(x) == b
        end
        @test is_flat_array(A, view(A, :, 2, 3), view(A, :, 2:2, 3)) == true
        @test is_flat_array(A, view(A, :, 2:2, :), view(A, :, 2:2, 3)) == false
        #
        # Tests for `allof`, `anyof` and `noneof`.
        #
        A = (true, true, true)
        B = [true, false]
        C = (false, false)
        for i in randperm(length(B)) # prevent compilation-time optimization
            @test anyof(B[i]) == B[i]
        end
        @test allof(true) == true
        @test anyof(true) == true
        @test noneof(true) == false
        @test allof(false) == false
        @test anyof(false) == false
        @test noneof(false) == true
        @test allof(A) == true
        @test anyof(A) == true
        @test noneof(A) == false
        @test allof(collect(A)) == allof(A)
        @test anyof(collect(A)) == anyof(A)
        @test noneof(collect(A)) == noneof(A)
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
            @test allof(x) == (allof(minimum, x, x) && allof(maximum, x))
            @test noneof(x) == (noneof(minimum, x, x) && noneof(maximum, x))
            @test anyof(x) == (anyof(minimum, x, x) || anyof(maximum, x))
        end
        #
        # Tests for `allindices`.
        #
        dims = (3,4)
        dim = 5
        A = randn(dims)
        V = randn(dim)
        R = allindices(A)
        I1 = one(CartesianIndex{2})
        I2 = CartesianIndex(dims)
        L = ((dim,                  1:dim),
             (Int16(dim),           Base.OneTo(dim)),
             (1:dim,                Base.OneTo(dim)),
             (Int16(2):Int16(dim),  2:dim),
             (V,                    allindices(size(V))),
             (A,                    allindices(size(A))),
             ((I1,I2),              R))
        for i in randperm(length(L)) # prevent compilation-time optimization
            arg, inds = L[i]
            if isa(arg, NTuple{2,CartesianIndex})
                @test allindices(arg[1], arg[2]) == inds
            else
                @test allindices(arg) == inds
            end
        end
        #
        # Tests for `convert_multiplier`.
        #
        R = (Float32, Float16, BigFloat, Float64)
        C = (ComplexF32, ComplexF64)
        for i in randperm(length(R)) # prevent compilation-time optimization
            Tr = R[i]
            Trp = R[i < length(R) ? i + 1 : 1]
            j = rand(1:length(C))
            Tc = C[j]
            Tcp = C[length(C) + 1 - j]
            @test convert_multiplier(1, Tr) == convert(Tr, 1)
            @test isa(convert_multiplier(π, Tc), AbstractFloat)
            @test (v = convert_multiplier(2.0, Tr)) == 2 && isa(v, Tr)
            @test (v = convert_multiplier(2.0, Tr, Trp)) == 2 && isa(v, Tr)
            @test (v = convert_multiplier(2.0, Tr, Tc)) == 2 && isa(v, Tr)
            @test (v = convert_multiplier(2.0, Tc, Tc)) == 2 && isa(v, real(Tc))
            @test convert_multiplier(1+0im, Tr) == 1
            @test convert_multiplier(1+0im, Tr, Trp) == 1
            @test_throws InexactError convert_multiplier(1+2im, Tr)
            @test convert_multiplier(1+2im, Tr, Tc) == 1+2im
            @test convert_multiplier(1+2im, Tc, Tcp) == 1+2im
        end
        for T in (AbstractFloat, Complex, Number)[randperm(3)]
            @test_throws ErrorException convert_multiplier(1, T)
        end
    end
end
