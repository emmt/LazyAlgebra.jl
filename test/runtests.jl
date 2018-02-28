isdefined(:MockAlgebra) || include("../src/MockAlgebra.jl")

module MockAlgebraTests

using Base.Test
using MockAlgebra

check(::Type{T}, nops::Integer, a::Real, b::Real) where {T<:AbstractFloat} =
    isapprox(a, b, rtol=sqrt(nops)*eps(T), atol=0)

# Note that with low precision floating-point sucha as Float16, the level of
# requirement has to be lowered.
n1, n2 = 12, 7
for T in (Float64, Float32, Float16)
    w = randn(T, 2,n1,n2)
    i = find(v -> v > 0, w)
    x = randn(T, 2,n1,n2)
    y = randn(T, 2,n1,n2)
    zx = reinterpret(Complex{eltype(x)}, x, (n1, n2))
    zy = reinterpret(Complex{eltype(y)}, y, (n1, n2))
    n = length(x)
    m = length(i)
    #atol, rtol = zero(T), sqrt(eps(T))
    #println("T=$T, atol=$atol, rtol=$rtol")
    @assert check(T, 2n, vnorm2(x), sqrt(sum(x.*x)))
    @assert check(T,  n, vnorm1(x), sum(abs.(x)))
    @assert check(T,  1, vnorminf(x), maximum(abs.(x)))
    @assert check(T, 2n, vdot(x, y), sum(x.*y))
    @assert check(T, 2n, vdot(x, y), vdot(zx, zy))
    @assert check(T,  3, vdot([1 + 0im],[0 + 1im]), 0)
    @assert check(T, 3n, vdot(w, x, y), sum(w.*x.*y))
    @assert check(T, 2m, vdot(i, x, y), sum(x[i].*y[i]))
end

end
