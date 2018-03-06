isdefined(:LazyAlgebra) || include("../src/LazyAlgebra.jl")

module LazyAlgebraTests

using Base.Test
using LazyAlgebra

check(::Type{T}, nops::Integer, a::Real, b::Real) where {T<:AbstractFloat} =
    isapprox(a, b, rtol=2*sqrt(nops)*eps(T), atol=0)

# Note that with low precision floating-point sucha as Float16, the level of
# requirement has to be lowered.
n1, n2 = 12, 7
for T in (Float64, Float32, Float16)
    println("")
    println("Type = $T")
    println("==============")
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

    A = RankOneOperator(w,w)
    B = RankOneOperator(w,y)
    C = SymmetricRankOneOperator(w)

    println("A⋅x:  ", extrema(A*x - sum(w.*x)*w))
    println("A'⋅x: ", extrema(A'*x - sum(w.*x)*w))
    println("B⋅x:  ", extrema(B*x - sum(y.*x)*w))
    println("B'⋅x: ", extrema(B'*x - sum(w.*x)*y))
    println("C⋅x:  ", extrema(C*x - sum(w.*x)*w))
    println("C'⋅x: ", extrema(C'*x - sum(w.*x)*w))
    println()

    S = NonuniformScalingOperator(w)
    println("S⋅x:  ", extrema(S*x - w.*x))
    println("S'⋅x: ", extrema(S'*x - w.*x))

    alpha = sqrt(2)
    U = UniformScalingOperator(alpha)
    println("U⋅x:  ", extrema(U*x - alpha*x))
    println("U'⋅x: ", extrema(U'*x - alpha*x))
    println("U\\x:  ", extrema(U\x - (1/alpha)*x))
    println("U'\\x: ", extrema(U'\x - (1/alpha)*x))

end

end
