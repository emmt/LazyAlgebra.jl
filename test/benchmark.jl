isdefined(:MockAlgebra) || include("../src/MockAlgebra.jl")

module MockAlgebraBenchmarks

using BenchmarkTools
using MockAlgebra
import MockAlgebra: vdot

vdot(::Type{Val{:BLAS}}, x, y) =
    MockAlgebra.blas_vdot(x,y)

vdot(::Type{Val{:Julia}}, x, y) =
    dot(reshape(x, length(x)), reshape(y, length(y)))

function vdot(::Type{Val{:basic}},
              x::AbstractArray{Tx,N},
              y::AbstractArray{Ty,N}) where {Tx, Ty, N}
    @assert size(x) == size(y)
    T = promote_type(Tx, Ty)
    s = zero(T)
    for i in eachindex(x, y)
        s += convert(T, x[i])*convert(T, y[i])
    end
    return s
end

function vdot(::Type{Val{:inbounds}},
              x::AbstractArray{Tx,N},
              y::AbstractArray{Ty,N}) where {Tx, Ty, N}
    @assert size(x) == size(y)
    T = promote_type(Tx, Ty)
    s = zero(T)
    @inbounds for i in eachindex(x, y)
        s += convert(T, x[i])*convert(T, y[i])
    end
    return s
end

function vdot(::Type{Val{:simd}},
              x::AbstractArray{Tx,N},
              y::AbstractArray{Ty,N}) where {Tx, Ty, N}
    @assert size(x) == size(y)
    T = promote_type(Tx, Ty)
    s = zero(T)
    @inbounds @simd for i in eachindex(x, y)
        s += convert(T, x[i])*convert(T, y[i])
    end
    return s
end

function vdot(::Type{Val{:simdlinear}},
              x::DenseArray{Tx,N},
              y::DenseArray{Ty,N}) where {Tx, Ty, N}
    @assert size(x) == size(y)
    T = promote_type(Tx, Ty)
    s = zero(T)
    @inbounds @simd for i in 1:length(x)
        s += x[i]*y[i]
    end
    return s
end

function vdot(::Type{Val{:basic}},
              w::AbstractArray{Tw,N},
              x::AbstractArray{Tx,N},
              y::AbstractArray{Ty,N}) where {Tw, Tx, Ty, N}
    @assert size(x) == size(y)
    T = promote_type(Tw, Tx, Ty)
    s = zero(T)
    for i in eachindex(x, x, y)
        s += convert(T, w[i])*convert(T, x[i])*convert(T, y[i])
    end
    return s
end

function vdot(::Type{Val{:inbounds}},
              w::AbstractArray{Tw,N},
              x::AbstractArray{Tx,N},
              y::AbstractArray{Ty,N}) where {Tw, Tx, Ty, N}
    @assert size(w) == size(x) == size(y)
    T = promote_type(Tw, Tx, Ty)
    s = zero(T)
    @inbounds for i in eachindex(w, x, y)
        s += convert(T, w[i])*convert(T, x[i])*convert(T, y[i])
    end
    return s
end

function vdot(::Type{Val{:simd}},
              w::AbstractArray{Tw,N},
              x::AbstractArray{Tx,N},
              y::AbstractArray{Ty,N}) where {Tw, Tx, Ty, N}
    @assert size(w) == size(x) == size(y)
    T = promote_type(Tw, Tx, Ty)
    s = zero(T)
    @inbounds @simd for i in eachindex(w, x, y)
        s += convert(T, w[i])*convert(T, x[i])*convert(T, y[i])
    end
    return s
end

function vdot(::Type{Val{:simdlinear}},
              w::DenseArray{Tw,N},
              x::DenseArray{Tx,N},
              y::DenseArray{Ty,N}) where {Tw, Tx, Ty, N}
    @assert size(w) == size(x) == size(y)
    T = promote_type(Tw, Tx, Ty)
    s = zero(T)
    @inbounds @simd for i in 1:length(x)
        s += w[i]*x[i]*y[i]
    end
    return s
end

testdot() = testdot(33, 33)

function testdot(_dims::Integer...)
    global dims, w, x, y, z

    dims = Int.(_dims)

    #show(STDOUT, MIME"text/plain"(), @benchmark $p(dat, a, img, b, rois))

    println("\n\nDot products of $dims elements")


    println("\\begin{tabular}{lrr}")
    println("\\hline")
    println(" & \\multicolumn{2}{c}{Median time (ns) for ",
            dims, " elements} \\\\")
    println("Operation & \\texttt{Float32} & \\texttt{Float64}\\\\")
    println("\\hline")
    println("\\hline")
    for p in (:Julia, :BLAS, :basic, :inbounds, :simd, :simdlinear)
        @printf(" dot %-12s", string("(",p,")"))
        for T in (Float32, Float64)
            w = randn(T, dims)
            x = randn(T, dims)
            y = randn(T, dims)
            z = randn(T, dims)
            s = vdot(Val{p}, x, y)
            t = @benchmark vdot($(Val{p}), x, y)
            @printf(" & %4.0f", median(t.times))
        end
        println(" \\\\")
    end
    println("\\hline")
    for p in (:basic, :inbounds, :simd, :simdlinear)
        @printf("wdot %-12s", string("(",p,")"))
        for T in (Float32, Float64)
            w = randn(T, dims)
            x = randn(T, dims)
            y = randn(T, dims)
            z = randn(T, dims)
            s = vdot(Val{p}, x, y)
            t = @benchmark vdot($(Val{p}), w, x, y)
            @printf(" & %4.0f", median(t.times))
        end
        println(" \\\\")
    end
    println("\\hline")
    println("\\end{tabular}")
end

norminf(x::AbstractArray) = ((xmn, xmx) = extrema(x); return max(-xmn, xmx))
norm1(x::AbstractArray) = sum(abs.(x))
norm2(x::AbstractArray) = sum(abs2.(x))

end
