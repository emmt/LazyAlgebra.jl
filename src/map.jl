"""
    B = LazyAlgebra.LazyMap(f, A)
    B = LazyAlgebra.LazyMap{T}(f, A)

given a function `f` and an array `A`, build a lightweight abstract array `B` such that
`B[i]` yields `as(T,f(A[i]))` for any index `i` of `A`. Optional type parameter `T` is the
guaranteed element type of `B`; if not specified, it is inferred from `f` and the element
type of `A`.

The index style of `B` is the same as that of `A`.

"""
LazyMap(f::Function, A::AbstractArray) = LazyMap{Base.promote_op(f, eltype(A))}(f, A)

Base.length(A::LazyMap) = length(A.arr)
Base.size(A::LazyMap) = size(A.arr)
Base.axes(A::LazyMap) = axes(A.arr)
for (L, S, Idecl, Icall) in ((false, :IndexCartesian, :(I::Vararg{Int,N}), :(I...)),
                             (true,  :IndexLinear,    :(i::Int),           :(i)))
    @eval begin
        Base.IndexStyle(::Type{<:LazyMap{T,N,$L}}) where {T,N} = $S()
        @inline function Base.getindex(A::LazyMap{T,N,$L}, $Idecl) where {T,N}
            @boundscheck checkbounds(A, $Icall)
            x = @inbounds getindex(A.arr, $Icall)
            return as(T, A.func(x))
        end
        @inline function Base.setindex!(A::LazyMap{T,N,$L}, x, $Idecl) where {T,N}
            @boundscheck checkbounds(A, $Icall)
            error("attempt to write read-only array")
            return A
        end
    end
end

Base.similar(A::LazyMap, ::Type{T}) where {T} = similar(A.arr, T)
Base.similar(A::LazyMap, ::Type{T}, shape::Union{Dims,ArrayAxes}) where {T} =
    similar(A.arr, T, shape)

"""
    LazyAlgebra.vmap!(dst, α, f, w, x) -> dst

overwrites `dst` with `α*f.(w, x)` and returns `dst`. An exception is thrown if `dst`,
`w`, and `x` do not have the same axes.

    LazyAlgebra.vmap!(α, f, w, x, β, y) -> y

overwrites `y` with `α*f.(w, x) + β*y` and returns `y`. An exception is thrown if `w`,
`x`, and `y` do not have the same axes.

See also [`LazyAlgebra.dispatch_vmap!`](@ref).

"""
function vmap!(dst::AbstractArray{<:Any,N}, α::Number, f::Function,
               w::AbstractArray{<:Any,N}, x::AbstractArray{<:Any,N}) where {N}
    @assert_same_axes dst w x
    T = Base.promote_op(f, eltype(w), eltype(x))
    return dispatch_vmap!(dst, convert_multiplier(α, T), f, w, x)
end

function vmap!(α::Number, f::Function,
               w::AbstractArray{<:Any,N}, x::AbstractArray{<:Any,N},
               β::Number, y::AbstractArray{<:Any,N}) where {N}
    @assert_same_axes w x y
    T = Base.promote_op(f, eltype(w), eltype(x))
    return dispatch_vmap!(convert_multiplier(α, T), f, w, x,
                          convert_multiplier(β, eltype(y)), y)
end

"""
    LazyAlgebra.dispatch_vmap!(dst, α, f, w, x) -> dst

overwrites `dst` with `α*f.(w, x)` and returns `dst`.

    LazyAlgebra.dispatch_vmap!(α, f, w, x, β, y) -> y

overwrites `y` with `α*f.(w, x) + β*y` and returns `y`.

Depending on the specific values of the multipliers, these methods choose which *unsafe*
function to call.

See also [`LazyAlgebra.vmap!`](@ref) and [`LazyAlgebra.unsafe_vmap!`](@ref).

!!! warning
    These methods assume that their array arguments have the same axes, and that multipliers
    have been converted to suitable types.


"""
function dispatch_vmap!(dst::AbstractArray, α::Number, f::Function,
                        w::AbstractArray, x::AbstractArray)
    if iszero(α)
        vzeros!(dst)
    else
        unsafe_vmap!(dst, α, f, w, x)
    end
    return dst
end

function dispatch_vmap!(α::Number, f::Function, w::AbstractArray, x::AbstractArray,
                        β::Number, y::AbstractArray)
    if iszero(α)
        dispatch_vscale!(y, β)
    elseif iszero(β)
        unsafe_vmap!(y, α, f, w, x)
    else
        unsafe_vmap!(α, f, w, x, β, y)
    end
    return y
end

"""
    LazyAlgebra.unsafe_vmap!(dst, α, f, w, x) -> dst

overwrites `dst` with `dst[i] = α*f(w[i], x[i])` and returns `dst`.

!!! warning
    This method assumes that `dst`, `w`, and `x` have the same axes, and that multiplier
    `α` has suitable type and is non-zero.

"""
function unsafe_vmap!(dst::AbstractArray, α::Number, f::Function,
                      w::AbstractArray, x::AbstractArray)
    if α == one(α)
        @inbounds @simd for i in eachindex(dst, w, x)
            dst[i] = f(w[i], x[i])
        end
    else
        @inbounds @simd for i in eachindex(dst, w, x)
            dst[i] = α*f(w[i], x[i])
        end
    end
    return dst
end

"""
    LazyAlgebra.unsafe_vmap!(α, f, w, x, β, y) -> y

overwrite `y` with `y[i] = α*f(w[i], x[i]) + β*y[i])` and returns `y`.

!!! warning
    This method assumes that `w`, `x`, and `y` have the same axes, and that multipliers
    `α` and `β` have suitable types and are both non-zero.

"""
function unsafe_vmap!(α::Number, f::Function, w::AbstractArray, x::AbstractArray,
                      β::Number, y::AbstractArray)
    if β == one(β)
        if α == one(α)
            @inbounds @simd for i in eachindex(w, x, y)
                y[i] += f(w[i], x[i])
            end
        elseif α == -one(α)
            @inbounds @simd for i in eachindex(w, x, y)
                y[i] -= f(w[i], x[i])
            end
        else
            @inbounds @simd for i in eachindex(w, x, y)
                y[i] += α*f(w[i], x[i])
            end
        end
    elseif β == -one(β)
        if α == one(α)
            @inbounds @simd for i in eachindex(w, x, y)
                y[i] = f(w[i], x[i]) - y[i]
            end
        else
            @inbounds @simd for i in eachindex(w, x, y)
                y[i] = α*f(w[i], x[i]) - y[i]
            end
        end
    else
        if α == one(α)
            @inbounds @simd for i in eachindex(w, x, y)
                y[i] = f(w[i], x[i]) + β*y[i]
            end
        else
            @inbounds @simd for i in eachindex(w, x, y)
                y[i] = α*f(w[i], x[i]) + β*y[i]
            end
        end
    end
    return y
end
