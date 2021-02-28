#
# common.jl -
#
# Common functions for testing.
#

"""

```julia
floating_point_type(A, B, ...)
```

yields the floating-point type for operations between arrays `A`, `B`, ...

"""
floating_point_type(args::AbstractArray...) =
    float(real(promote_type(map(eltype, args)...)))

"""

```julia
relative_precision(A, B, ...)
```

yields the worst of the relative precisions of the element types of arrays `A`,
`B`, ...

"""
relative_precision(args::AbstractArray...) = max(map(relative_precision, args)...)
relative_precision(A::AbstractArray{T}) where {T} = eps(float(real(T)))

"""

```julia
test_api(P, A, x, y; atol=0, rtol=sqrt(eps(relative_precision(x,y))))
```

test LazyAlgebra API for mapping `P(A)` using variables `x` and `y`.

"""
function test_api(::Type{P}, A::Mapping, x0::AbstractArray, y0::AbstractArray;
                  rtol::Real=sqrt(relative_precision(x0,y0)),
                  atol::Real=0) where {P<:Union{Direct,InverseAdjoint}}
    x = vcopy(x0)
    y = vcopy(y0)
    z = apply(P, A, x)
    @test x == x0
    T = floating_point_type(x, y, z)
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
                  rtol::Real=sqrt(relative_precision(x0,y0)),
                  atol::Real=0) where {P<:Union{Adjoint,Inverse}}
    x = vcopy(x0)
    y = vcopy(y0)
    z = apply(P, A, y)
    @test y == y0
    T = floating_point_type(x, y, z)
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
