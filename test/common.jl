#
# utils.jl -
#
# Utilities for testing.
#

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
