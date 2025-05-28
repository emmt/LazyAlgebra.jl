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
    LazyAlgebra.vmap!(y, α, f, w, x) -> y

overwrites `y` with `α*f.(w, x)` and returns `y`. Other possibility:

    LazyAlgebra.vmap!(α, f, w, x, β, y) -> y

to overwrite `y` with `α*f.(w, x) + β*y`. An exception is thrown if `w`, `x`, and `y` do
not have the same axes.

See also [`LazyAlgebra.unsafe_vmap!`](@ref).

"""
vmap!(y::AbstractArray, α::Number, f::Function, w::AbstractArray, x::AbstractArray) =
    vmap!(α, f, w, x, 𝟘, y)

# Stages for `vmap!(α, f, w, x, β, y)`:
#   0. Check axes.
#   1. Convert `α`.
#   2. Dispatch on `α`.
#   3. If `α` is zero, call `vscale!(y,β)` and stop; otherwise, convert `β` and proceed
#      with next stage.
#   4. Dispatch on `β` to call the unsafe method.
function vmap!(α::Number, f::Function, w::AbstractArray, x::AbstractArray,
               β::Number, y::AbstractArray)
    @assert_same_axes w x y
    return vmap!(Stage(1), α, f, w, x, β, y)
end
function vmap!(::Stage{1},
               α::Number, f::Function, w::AbstractArray{Tw,N}, x::AbstractArray{Tx,N},
               β::Number, y::AbstractArray{Ty,N}) where {Tw,Tx,Ty,N}
    α′ = convert_multiplier(α, Base.promote_op(f, eltype(w), eltype(x)))
    return vmap!(Stage(2), α′, f, w, x, β, y)
end
function vmap!(::Stage{2},
               α::Number, f::Function, w::AbstractArray{Tw,N}, x::AbstractArray{Tx,N},
               β::Number, y::AbstractArray{Ty,N}) where {Tw,Tx,Ty,N}
    @dispatch_on_multiplier α vmap!(Stage(3), α, f, w, x, β, y)
    return y
end
function vmap!(::Stage{3},
               α::Number, f::Function, w::AbstractArray{Tw,N}, x::AbstractArray{Tx,N},
               β::Number, y::AbstractArray{Ty,N}) where {Tw,Tx,Ty,N}
    if α isa StaticMultiplier{0}
        vscale!(y, β)
    else
        β′ = convert_inplace_multiplier(β, eltype(y))
        vmap!(Stage(4), α, f, w, x, β′, y)
    end
    return y
end
function vmap!(::Stage{4},
               α::Number, f::Function, w::AbstractArray{Tw,N}, x::AbstractArray{Tx,N},
               β::Number, y::AbstractArray{Ty,N}) where {Tw,Tx,Ty,N}
    @dispatch_on_multiplier β unsafe_vmap!(α, f, w, x, β, y)
    return y
end

"""
    LazyAlgebra.unsafe_vmap!(α, f, w, x, β, y) -> y

overwrite `y` with `y[i] = α*f(w[i], x[i]) + β*y[i])` and returns `y`.

!!! warning
    This method assumes that `w`, `x`, and `y` have the same axes, and that multipliers
    `α` and `β` have efficient types.

"""
function unsafe_vmap!(α::Number, f::Function, w::AbstractArray{Tw,N}, x::AbstractArray{Tx,N},
                      β::Number, y::AbstractArray{Ty,N}) where {Tw,Tx,Ty,N}
    @inbounds @simd for i in eachindex(w, x, y)
        y[i] = α*f(w[i], x[i]) + β*y[i]
    end
    return y
end
