#
# common.jl -
#
# Common functions for testing.
#

"""

```julia
floatingpointtype(A, B, ...)
```

yields the floating-point type for operations between arrays `A`, `B`, ...

"""
floatingpointtype(args::AbstractArray...) =
    float(real(promote_type(map(eltype, args)...)))

"""

```julia
relativeprecision(A, B, ...)
```

yields the worst of the relative precisions of the element types of arrays `A`,
`B`, ...

"""
relativeprecision(args::AbstractArray...) = max(map(relativeprecision, args)...)
relativeprecision(A::AbstractArray{T}) where {T} = eps(float(real(T)))

"""

```julia
test_api(P, A, x, y; atol=0, rtol=sqrt(eps(relativeprecision(x,y))))
```

test LazyAlgebra API for mapping `P(A)` using variables `x` and `y`.

"""
function test_api(::Type{P}, A::Mapping, x0::AbstractArray, y0::AbstractArray;
                  rtol::Real=sqrt(relativeprecision(x0,y0)),
                  atol::Real=0) where {P<:Union{Direct,InverseAdjoint}}
    x = vcopy(x0)
    y = vcopy(y0)
    z = apply(P, A, x)
    T = floatingpointtype(x, y, z)
    @test x == x0
    for α in (0, 1, -1,  2.71, π),
        β in (0, 1, -1, -1.33, Base.MathConstants.φ),
        scratch in (false, true)
        @test apply!(α, P, A, x, scratch, β, vcopy(y)) ≈
            T(α)*z + T(β)*y  atol=atol rtol=rtol
        if scratch
            vcopy!(x, x0)
        else
            @test x == x0
        end
    end
end
function test_api(::Type{P}, A::Mapping, x0::AbstractArray, y0::AbstractArray;
                  rtol::Real=sqrt(relativeprecision(x0,y0)),
                  atol::Real=0) where {P<:Union{Adjoint,Inverse}}
    x = vcopy(x0)
    y = vcopy(y0)
    z = apply(P, A, y)
    @test y == y0
    T = floatingpointtype(x, y, z)
    for α in (0, 1, -1,  2.71, π),
        β in (0, 1, -1, -1.33, Base.MathConstants.φ),
        scratch in (false, true)
        @test apply!(α, P, A, y, scratch, β, vcopy(x)) ≈
            T(α)*z + T(β)*x  atol=atol rtol=rtol
        if scratch
            vcopy!(y, y0)
        else
            @test y == y0
        end
    end
end

"""
`relabsdif(a,b)` yields the relative difference between `a` and `b`.

See also: [`maxrelabsdif`](@ref).
"""
function relabsdif(a::Ta, b::Tb) where {Ta<:Real, Tb<:Real}
    T = float(promote_type(Ta, Tb))
    relabsdif(convert(T, a), convert(T, b))
end

function relabsdif(a::Complex{Ta}, b::Complex{Tb}) where {Ta<:Real, Tb<:Real}
    T = float(promote_type(Ta, Tb))
    relabsdif(convert(T, a), convert(T, b))
end

relabsdif(a::T, b::T) where {T<:AbstractFloat} =
    (a == b ? zero(T) : 2*abs(a - b)/(abs(a) + abs(b)))

relabsdif(a::Complex{T}, b::Complex{T}) where {T<:AbstractFloat} =
    (a == b ? zero(T) : 2*abs(a - b)/(abs(a) + abs(b)))

""" `maxrelabsdif(A,B)` yields the maximum relative difference between arrays
`A` and `B`.

See also: [`relabsdif`](@ref), [`approxsame`](@ref).
"""
maxrelabsdif(A::AbstractArray, B::AbstractArray) =
    maximum(relabsdif.(A, B))

"""

`approxsame(A,B,n)` and `approxsame(n,A,B)` check whether all elements of `A`
and `B` are the same with a relative tolerance equal to
`sqrt(n)*max(eps(eltype(A)),eps(eltype(B)))`.

See also: [`maxrelabsdif`](@ref).
"""
function approxsame(A::AbstractArray{Ta,N},
                    B::AbstractArray{Tb,N},
                    n::Integer) where {Ta<:AbstractFloat, Tb<:AbstractFloat, N}
    @assert size(A) == size(B)
    return maxrelabsdif(A, B) ≤ sqrt(n)*max(eps(Ta), eps(Tb))
end

function approxsame(n::Integer,
                    A::AbstractArray{<:AbstractFloat,N},
                    B::AbstractArray{<:AbstractFloat,N}) where {N}
    return approxsame(A, B, n)
end
