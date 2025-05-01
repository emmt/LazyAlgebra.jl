#
# common.jl -
#
# Common functions for testing.
#

using TypeUtils
using LazyAlgebra

using LazyAlgebra: Adjoint, Inverse, InverseAdjoint

real_eltype(x::Union{Operator,AbstractArray}) = real_eltype(typeof(x))
real_eltype(::Type{x}) where {x<:Union{Operator,AbstractArray}} = real_type(eltype(x))
real_eltype(::Type{<:Identity}) = Bool

infer_type(x::Number, op::Function, y::Number) = infer_type(typeof(x), op, typeof(y))
infer_type(::Type{x}, op::Function, ::Type{y}) where {x<:Number,y<:Number} =
    typeof(op(one(x), one(y)))

# α*x
infer_multiplier_type(α::Number, x::AbstractArray) =
    infer_multiplier_type(typeof(α), typeof(x))
infer_multiplier_type(::Type{α}, ::Type{x}) where {α<:Number, x<:AbstractArray} =
    convert_real_type(float(real_eltype(x)), α)

# α*A*x
infer_multiplier_type(α::Number, A::Union{Operator,AbstractArray}, x::AbstractArray) =
    infer_multiplier_type(typeof(α), typeof(A), typeof(x))
infer_multiplier_type(::Type{α}, ::Type{A}, ::Type{x}) where {α<:Number, A<:Union{Operator,AbstractArray}, x<:AbstractArray} =
    infer_multiplier_type(α, AbstractArray{infer_output_eltype(A, x)})

# α*x
infer_output_eltype(α::Number, x::AbstractArray) =
    infer_output_eltype(typeof(α), typeof(x))
infer_output_eltype(::Type{α}, ::Type{x}) where {α<:Number, x<:AbstractArray} =
    infer_type(infer_multiplier_type(α, x), *, eltype(x))

# α*x + β*y
infer_output_eltype(α::Number, x::AbstractArray{<:Any,N}, β::Number, y::AbstractArray{<:Any,N}) where {N} =
    infer_output_eltype(typeof(α), typeof(x), typeof(β), typeof(y))
infer_output_eltype(::Type{α}, ::Type{x}, ::Type{β}, ::Type{y}) where {N, α<:Number, x<:AbstractArray{<:Any,N}, β<:Number, y<:AbstractArray{<:Any,N}} =
    infer_type(infer_output_eltype(α, x), +, infer_output_eltype(β, y))

# A*x
infer_output_eltype(A::Union{Operator,AbstractArray}, x::AbstractArray) =
    infer_output_eltype(typeof(A), typeof(x))
infer_output_eltype(::Type{A}, ::Type{x}) where {A<:Union{Operator,AbstractArray}, x<:AbstractArray} =
    infer_type(eltype(A), *, eltype(x))
infer_output_eltype(::Type{<:Identity}, ::Type{x}) where {x<:AbstractArray} =
    eltype(x)

# α*A*x
infer_output_eltype(α::Number, A::Union{Operator,AbstractArray}, x::AbstractArray) =
    infer_output_eltype(typeof(α), typeof(A), typeof(x))
infer_output_eltype(::Type{α}, ::Type{A}, ::Type{x}) where {α<:Number, A<:Union{Operator,AbstractArray}, x<:AbstractArray} =
    infer_output_eltype(α, AbstractArray{infer_output_eltype(A, x)})

# α*A*x + β*y
function infer_output_eltype(α::Number, A::Union{Operator,AbstractArray}, x::AbstractArray,
                             β::Number, y::AbstractArray)
    infer_output_eltype(typeof(α), typeof(A), typeof(x), typeof(β), typeof(y))
end
function infer_output_eltype(::Type{α}, ::Type{A}, ::Type{x}, ::Type{β},
                             ::Type{y}) where {α<:Number, A<:Union{Operator,AbstractArray},
                                               x<:AbstractArray, β<:Number, y<:AbstractArray}
    Ax = AbstractArray{infer_output_eltype(A, x), ndims(y)}
    infer_output_eltype(α, Ax, β, y)
end

#
flat(A::AbstractArray) = reshape(A, length(A))

function shift_values!(by::Real, A::AbstractArray{T}) where {T}
    R = real(T)
    B = R === T ? A : reinterpret(R, A)
    Bmin, Bmax = extrema(B)
    Badj = by*(Bmax - Bmin) - Bmin
    @. B += Badj
    return A
end

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
function test_api(::Type{P}, A::Operator, x0::AbstractArray, y0::AbstractArray;
                  rtol::Real=sqrt(relative_precision(x0,y0)),
                  atol::Real=0) where {P<:Union{Operator,InverseAdjoint}}
    x = vcopy(x0)
    y = vcopy(y0)
    z = vmul(P, A, x)
    @test x == x0
    T = floating_point_type(x, y, z)
    for α in (0, 1, -1,  2.71, π),
        β in (0, 1, -1, -1.33, Base.MathConstants.φ),
        scratch in (false, true)
        @test vmul!(α, P, A, x, scratch, β, vcopy(y)) ≈
            T(α)*z + T(β)*y  atol=atol rtol=rtol
        if scratch
            vcopy!(x, x0)

        else
            @test x == x0
        end
    end
end

function test_api(::Type{P}, A::Operator, x0::AbstractArray, y0::AbstractArray;
                  rtol::Real=sqrt(relative_precision(x0,y0)),
                  atol::Real=0) where {P<:Union{Adjoint,Inverse}}
    x = vcopy(x0)
    y = vcopy(y0)
    z = vmul(P, A, y)
    @test y == y0
    T = floating_point_type(x, y, z)
    for α in (0, 1, -1,  2.71, π),
        β in (0, 1, -1, -1.33, Base.MathConstants.φ),
        scratch in (false, true)
        @test vmul!(α, P, A, y, scratch, β, vcopy(x)) ≈
            T(α)*z + T(β)*x  atol=atol rtol=rtol
        if scratch
            vcopy!(y, y0)
        else
            @test y == y0
        end
    end
end
