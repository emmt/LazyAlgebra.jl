using BenchmarkTools: Trial

"""
    @check expr

Check whether `expr` holds. Print a message in color if this assertion fails.

"""
macro check(expr)
    msg = "`$expr` failed"
    esc(:($expr ? Base.nothing : Base.printstyled($msg; color=:red)))
end

"""
    prt(io=stdout, str="", trial::BenchmarkTools.Trial;
        what=:min, nops=0, pad=0, color=:normal)

Print a summary of `trial` (the output of a `@benchmarck` call) to output stream `io` with
string `str` describing the tested operations.

Keyword `what` specifies how to reduce the trial times, it is one of: `:min`, `:minumum`,
`:med`, `:median`, `:avg`, or `:mean`.

Keyword `nops` specifies the number of floating-point operations to print an estimation of
the number of floating-point operations per second. Unused if less than 1.

Keyword `pad` specifies the number of character to align results. Unused if less than 3.

Keyword `color` specifies a color to highlight the results.

"""
prt(trial::Trial; kwds...) =
    prt("", trial; kwds...)

prt(str::AbstractString, trial::Trial; kwds...) =
    prt(stdout, str, trial; kwds...)

prt(io::IO, trial::Trial; kwds...) =
    prt(io, "", trial; kwds...)

function prt(io::IO, str::AbstractString, trial::Trial;
             what::Symbol=:min, nops::Integer=0, pad::Integer=-1,
             color::Symbol=:normal)
    # members of trial:
    #     allocs::Int
    #     gctimes::Vector{Float64}
    #     memory::Int
    #     params::BenchmarkTools.Parameters
    #     times::Vector{Float64}

    # time in nanoseconds
    if what == :min || what == :minimum
        t = minimum(trial.times)
    elseif what == :med || what == :median
        t = median(trial.times)
    elseif what == :mean || what == :avg
        t = mean(trial.times)
    else
        error("illegal `what`")
    end
    t = max(t, 0.0)

    # print prefix
    len = length(str)
    if len > 0
        printstyled(io, str; color=color)
        print(io, ' ')
        pad -= len + 1
    end
    if pad ≥ 3
        for _ in 1:pad-2
            print(io, '-')
        end
        print(io, "> ")
    end

    # print time
    print(io, what, ": ")
    if t/1e9 ≥ 1.0
        s = @sprintf("%#.4g s", t/1e9)
    elseif t/1e6 ≥ 1.0
        s = @sprintf("%7.3f ms", t/1e6)
    elseif t/1e3 ≥ 1.0
        s = @sprintf("%7.3f μs", t/1e3)
    else
        s = @sprintf("%7.3f ns", t)
    end
    printstyled(io, s; color=color)

    # print number of floating-point operations per seconds
    if nops != nothing
        print(io, " ≈ ")
        printstyled(io, @sprintf(" ≈ %7.3f Gflops", nops/t); color=color)
    end

    # print allocations
    print(io, " (")
    printstyled(io, trial.allocs, " allocation",
                (trial.allocs > 0 ? "s" : ""), ", ",
                trial.memory, " byte",
                (trial.memory > 0 ? "s" : "");
                color=(trial.memory > 0 ? :red : :green))
    print(io, ")\n")
end

"""
    title(str; kwds...)

Print a title surrounded by `*` passing keywords `kwds...` to `printstyled`.

"""
function title(s::AbstractString; kwds...)
    hl = repeat('*', 4 + length(s))
    printstyled(hl, "\n* ", s, " *\n", hl, "\n"; kwds...)
end
