module TupleTests

using BenchmarkTools

@inline to_tuple1(x::AbstractVector) = Tuple(x)
@inline to_tuple2(x::AbstractVector) = (x...,)
@inline function to_tuple3(x::AbstractVector)
    n = length(x)
    @assert eachindex(x) == 1:n
    ntuple(i->x[i], n)
end

# The cutoff at n = 10 below reflects what is used by `ntuple`.  This value is
# somewhat arbitrary, on the machines where I tested the code, the explicit
# unrolled expression for n = 10 is still about 44 times faster than `(x...,)`.
# Calling `ntuple` for n ≤ 10 is about twice slower; for n > 10, `ntuple` is
# slower than `(x...,)`.

function to_tuple4(x::AbstractVector)
    n = length(x)
    @inbounds begin
        n == 0 ? () :
        n > 10 || firstindex(x) != 1 ? (x...,) :
        n == 1 ? (x[1],) :
        n == 2 ? (x[1], x[2]) :
        n == 3 ? (x[1], x[2], x[3]) :
        n == 4 ? (x[1], x[2], x[3], x[4]) :
        n == 5 ? (x[1], x[2], x[3], x[4], x[5]) :
        n == 6 ? (x[1], x[2], x[3], x[4], x[5], x[6]) :
        n == 7 ? (x[1], x[2], x[3], x[4], x[5], x[6], x[7]) :
        n == 8 ? (x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8]) :
        n == 9 ? (x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9]) :
        (x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9], x[10])
    end
end

# As to_tuple4 but inlined.
@inline function to_tuple5(x::AbstractVector)
    n = length(x)
    @inbounds begin
        n == 0 ? () :
        n > 10 || firstindex(x) != 1 ? (x...,) :
        n == 1 ? (x[1],) :
        n == 2 ? (x[1], x[2]) :
        n == 3 ? (x[1], x[2], x[3]) :
        n == 4 ? (x[1], x[2], x[3], x[4]) :
        n == 5 ? (x[1], x[2], x[3], x[4], x[5]) :
        n == 6 ? (x[1], x[2], x[3], x[4], x[5], x[6]) :
        n == 7 ? (x[1], x[2], x[3], x[4], x[5], x[6], x[7]) :
        n == 8 ? (x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8]) :
        n == 9 ? (x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9]) :
        (x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9], x[10])
    end
end

for n in ((0:20)...,30,40,50)
    println("\nmaking tuples from a vector of $n elements:")
    let x = rand(n)
        @assert to_tuple1(x) === Tuple(x)
        @assert to_tuple2(x) === Tuple(x)
        @assert to_tuple3(x) === Tuple(x)
        @assert to_tuple4(x) === Tuple(x)
        @assert to_tuple5(x) === Tuple(x)
        print("                    Tuple(x): ")
        @btime to_tuple1($x)
        print("                    (x, ...): ")
        @btime to_tuple2($x)
        print("  ntuple(i->x[i], length(x)): ")
        @btime to_tuple3($x)
        print("                to_tuple4(x): ")
        @btime to_tuple4($x)
        print("                to_tuple5(x): ")
        @btime to_tuple5($x)
    end
end

end # module

nothing
