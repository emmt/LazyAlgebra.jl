"""

Module `LazyAlgebraVectorBenchmarks` is to benchmark vectorized operations.
Usage:

    LazyAlgebraVectorBenchmarks.runtests(; T::Type=Float32, dims=10_123)

"""
module LazyAlgebraVectorBenchmarks

using Printf
using Statistics
using BenchmarkTools
using LazyAlgebra
using ThreadPinning
using LinearAlgebra
using LinearAlgebra: BLAS

include("benchmarking.jl")

function runtests(; T::Type = Float32, dims = 10_123)
    w = rand(T, dims)
    x = rand(T, dims)
    y = rand(T, dims)
    z = similar(x)
    n = length(x)

    pinthreads(:cores);
    BLAS.set_num_threads(1);
    opts = (; what=:min, pad=48);
    lazy = :yellow;

    title("Benchmark tests with T=$T and n=$n"; color=:blue)

    # x and y as vectors for LinearAlgebra functions
    x_flat = reshape(x, length(x))
    y_flat = reshape(y, length(y))

    println()
    @check vnorm1(x) ≈ LinearAlgebra.norm(x_flat,1)
    prt("vnorm1(x)", @benchmark(vnorm1($x_flat)); nops=2n, opts..., color=lazy)
    prt("LinearAlgebra.norm(x,1)", @benchmark(LinearAlgebra.norm($x_flat,1)); nops=2n, opts...)

    println()
    @check vnorm2(x) ≈ LinearAlgebra.norm(x_flat,2)
    prt("vnorm2(x)", @benchmark(vnorm2($x)); nops=2n, opts..., color=lazy)
    prt("LinearAlgebra.norm(x,2)", @benchmark(LinearAlgebra.norm($x_flat,2)); nops=2n, opts...)
    prt("LinearAlgebra.norm(x)", @benchmark(LinearAlgebra.norm($x_flat)); nops=2n, opts...)

    println()
    @check vnorminf(x) ≈ LinearAlgebra.norm(x_flat,Inf)
    prt("vnorminf(x)", @benchmark(vnorminf($x)); nops=2n, opts..., color=lazy)
    prt("LinearAlgebra.norm(x,Inf)", @benchmark(LinearAlgebra.norm($x_flat,Inf)); nops=2n, opts...)

    println()
    @check vdot(x, y) ≈ LinearAlgebra.dot(x_flat, y_flat)
    @check vdot(x, y) ≈ sum(conj.(x) .* y)
    prt("vdot(x, y)", @benchmark(vdot($x, $y)); nops=2n, opts..., color=lazy)
    prt("LinearAlgebra.dot(x, y)", @benchmark(LinearAlgebra.dot($x_flat, $y_flat)); nops=2n, opts...)
    @check vdot(w, x, y) ≈ sum(w .* conj.(x) .* y)
    prt("vdot(w, x, y)", @benchmark(vdot($w, $x, $y)); nops=3n, opts..., color=lazy)

    println()
    inds = eachindex(IndexLinear(), x_flat)[x_flat .< 0.3]
    @check vdot(inds, x, y) ≈ sum(conj.(x[inds]) .* y[inds])
    prt("vdot(inds, x, y)", @benchmark(vdot($inds, $x, $y)); nops=2*length(inds), opts..., color=lazy)

    println()
    vfill!(z, T <: Complex ? complex(NaN,NaN) : NaN)
    @check vcopy!(z, x) === z && z == x
    prt("vcopy!(z, x)", @benchmark(vcopy!($z, $x)); nops=n, opts..., color=lazy)
    prt("copyto!(z, x)", @benchmark(copyto!($z, $x)); nops=n, opts...)

    println()
    x_swp = copy(x)
    y_swp = copy(y)
    @check vswap!(x_swp, y_swp) === nothing && x_swp == y && y_swp == x
    prt("vswap!(x, y)", @benchmark(vswap!($x_swp, $y_swp)); nops=n, opts..., color=lazy)

    println()
    x_cpy = copy(x)
    for alpha in (-1, 0, 1, 1.3)
        @check vscale!(vcopy!(z, x), alpha) === z
        @check vscale!(vcopy!(z, x), alpha) ≈ alpha*x
        @check x == x_cpy
        @check vscale!(alpha, vcopy!(z, x)) === z
        @check vscale!(alpha, vcopy!(z, x)) ≈ alpha*x
        @check x == x_cpy
        prt("vscale!(z, $alpha, x)", @benchmark(vscale!($z, $alpha, $x)); nops=n, opts..., color=lazy)
    end

    println()
    x_cpy = copy(x)
    y_cpy = copy(y)
    @check vproduct(x, y) ≈ x .* y
    @check x == x_cpy && y == y_cpy
    @check vproduct!(z, x, y) === z
    @check x == x_cpy && y == y_cpy
    @check vproduct!(z, x, y) ≈ x .* y
    prt("vproduct!!(z, x, y)", @benchmark(vproduct!($z, $x, $y)); nops=n, opts..., color=lazy)

    println()
    u = y ./ 10_000; # to avoid overflows
    for alpha in (-1, 0, 1, 1.3)
        @check vupdate!(vcopy!(z, x), alpha, y) === z
        @check y == y_cpy
        @check vupdate!(vcopy!(z, x), alpha, y) ≈ x + alpha*y
        vcopy!(z, x)
        prt("vupdate!(x, $alpha, y)", @benchmark(vupdate!($z, $alpha, $u)); nops=2n, opts..., color=lazy)
    end

    println()
    for (α, β, nops) in ((1.5, -2.3, 3n),
                         (1,   -1,    n),
                         (1,    0,    n),
                         (0,    1,    n),
                         (1,    0,    n),
                         (0,    0,    n))
        @check vcombine!(z, α, x, β, y) === z ≈ α*x + β*y
        @check x == x_cpy && y == y_cpy
        @check vcombine!(α, x, β, vcopy!(z, y)) === z ≈ α*x + β*y
        @check x == x_cpy
        prt("vcombine!(z, $α, x, $β, y)",
            @benchmark(vcombine!($z, $α, $x, $β, $y)); nops=3n, opts..., color=lazy)
        if (α, β) == (1, 0)
            prt("copyto!(z, x)", (@benchmark copyto!($z, $x)); nops=n, opts...)
        elseif (α, β) == (0, 0)
            prt("fill!(z, 𝟘)", (@benchmark fill!($z, $(zero(eltype(z))))); nops=n, opts...)
        end
    end
end

end # module
