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

"""
```julia
checkmapping(y, A, x) -> (v1, v2, v1 - v2)
```

yields `v1 = vdot(y, A*x)`, `v2 = vdot(A'*y, x)` and their difference for `A` a
linear mapping, `y` a "vector" of the output space of `A` and `x` a "vector"
of the input space of `A`.  In principle, the two inner products should be the
same whatever `x` and `y`; otherwise the mapping has a bug.

Simple linear mappings operating on Julia arrays can be tested on random
"vectors" with:

```julia
checkmapping([T=Float64,] outdims, A, inpdims) -> (v1, v2, v1 - v2)
```

with `outdims` and `outdims` the dimensions of the output and input "vectors"
for `A`.  Optional argument `T` is the element type.

If `A` operates on Julia arrays and methods `input_eltype`, `input_size`,
`output_eltype` and `output_size` have been specialized for `A`, then:

```julia
checkmapping(A) -> (v1, v2, v1 - v2)
```

is sufficient to check `A` against automatically generated random arrays.

See also: [`vdot`](@ref), [`vcreate`](@ref), [`apply!`](@ref),
          [`input_type`](@ref).

"""
function checkmapping(y::Ty, A::LinearMapping, x::Tx) where {Tx, Ty}
    v1 = vdot(y, A*x)
    v2 = vdot(A'*y, x)
    (v1, v2, v1 - v2)
end

function checkmapping(::Type{T},
                      outdims::Tuple{Vararg{Int}},
                      A::LinearMapping,
                      inpdims::Tuple{Vararg{Int}}) where {T<:AbstractFloat}
    checkmapping(randn(T, outdims), A, randn(T, inpdims))
end

function checkmapping(outdims::Tuple{Vararg{Int}},
                      A::LinearMapping,
                      inpdims::Tuple{Vararg{Int}})
    checkmapping(Float64, outdims, A, inpdims)
end

checkmapping(A::LinearMapping) =
    checkmapping(randn(output_eltype(A), output_size(A)), A,
                 randn(input_eltype(A), input_size(A)))


