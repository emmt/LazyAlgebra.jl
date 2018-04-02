#
# utils.jl -
#
# General purpose methods.
#

"""

`promote_scalar(T1, [T2, ...] α)` yields scalar `α` converted to
`promote_type(T1, T2, ...)`.

"""
promote_scalar(::Type{T1}, alpha::Real) where {T1<:AbstractFloat} =
    convert(T1, alpha)

function promote_scalar(::Type{T1}, ::Type{T2},
                        alpha::Real) where {T1<:AbstractFloat,
                                            T2<:AbstractFloat}
    return convert(promote_type(T1, T2), alpha)
end

function promote_scalar(::Type{T1}, ::Type{T2}, ::Type{T3},
                        alpha::Real) where {T1<:AbstractFloat,
                                            T2<:AbstractFloat,
                                            T3<:AbstractFloat}
    return convert(promote_type(T1, T2, T3), alpha)
end
