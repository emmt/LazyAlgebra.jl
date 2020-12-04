# Benchmarks for vectorized operations.

module LazyAlgebraVectorBenchmarks

using BenchmarkTools
using LazyAlgebra

const Vn = LazyAlgebra

#------------------------------------------------------------------------------
module V1
using ArrayTools
using LazyAlgebra:
    Floats, vscale!, vzero!,
    promote_multiplier, arguments_have_incompatible_axes

function vupdate!(y::AbstractArray{<:Floats,N},
                  α::Number,
                  x::AbstractArray{<:Floats,N}) where {N}
    I = all_indices(x, y)
    if α == 1
        @inbounds @simd for i in I
            y[i] += x[i]
        end
    elseif α == -1
        @inbounds @simd for i in I
            y[i] -= x[i]
        end
    elseif α != 0
        alpha = promote_multiplier(α, x)
        @inbounds @simd for i in I
            y[i] += alpha*x[i]
        end
    end
    return y
end

function vcombine!(dst::AbstractArray{<:Floats,N},
                   α::Number,
                   x::AbstractArray{<:Floats,N},
                   β::Number,
                   y::AbstractArray{<:Floats,N}) where {N}
    if α == 0
        axes(x) == axes(dst) || arguments_have_incompatible_axes()
        vscale!(dst, β, y)
    elseif β == 0
        axes(y) == axes(dst) || arguments_have_incompatible_axes()
        vscale!(dst, α, x)
    else
        I = all_indices(dst, x, y)
        if α == 1
            if β == 1
                @inbounds @simd for i in I
                    dst[i] = x[i] + y[i]
                end
            elseif β == -1
                @inbounds @simd for i in I
                    dst[i] = x[i] - y[i]
                end
            else
                beta = promote_multiplier(β, y)
                @inbounds @simd for i in I
                    dst[i] = x[i] + beta*y[i]
                end
            end
        elseif α == -1
            if β == 1
                @inbounds @simd for i in I
                    dst[i] = y[i] - x[i]
                end
            elseif β == -1
                @inbounds @simd for i in I
                    dst[i] = -x[i] - y[i]
                end
            else
                beta = promote_multiplier(β, y)
                @inbounds @simd for i in I
                    dst[i] = beta*y[i] - x[i]
                end
            end
        else
            alpha = promote_multiplier(α, x)
            if β == 1
                @inbounds @simd for i in I
                    dst[i] = alpha*x[i] + y[i]
                end
            elseif β == -1
                @inbounds @simd for i in I
                    dst[i] = alpha*x[i] - y[i]
                end
            else
                beta = promote_multiplier(β, y)
                @inbounds @simd for i in I
                    dst[i] = alpha*x[i] + beta*y[i]
                end
            end
        end
    end
    return dst
end

end # module V1

#------------------------------------------------------------------------------

n = 10_000
T = Float32
x = rand(T,n)
y = rand(T,n)
z1 = similar(x)
zn = similar(x)
alphas = (0,1,-1,-4.0)
betas = (0,1,-1,+2.0)
v1_vupdate!(dst, y, alpha, x) = V1.vupdate!(vcopy!(dst, y), alpha, x)
vn_vupdate!(dst, y, alpha, x) = LazyAlgebra.vupdate!(vcopy!(dst, y), alpha, x)
for a in alphas
    println("\nvupdate!(y, α=$a, x):")
    print("  v1:  ")
    @btime v1_vupdate!($z1,$y,$a,$x);
    print("  new: ")
    @btime vn_vupdate!($zn,$y,$a,$x);
    dz = vnorm2(z1 - zn)
    printstyled("  -> ‖z1 - zn‖ = $dz\n"; color=(dz==0 ? :green : :red))
end

for a in alphas, b in betas
    println("\nvcombine!(z, α=$a, x, β=$b, y):")
    print("  v1:  ")
    @btime V1.vcombine!($z1,$a,$x,$b,$y);
    print("  new: ")
    @btime LazyAlgebra.vcombine!($zn,$a,$x,$b,$y);
    dz = vnorm2(z1 - zn)
    printstyled("  -> ‖z1 - zn‖ = $dz\n"; color=(dz==0 ? :green : :red))
end

end # module
