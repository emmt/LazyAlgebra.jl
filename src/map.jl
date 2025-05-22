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

to overwrite `y` with `α*f.(w, x) + β*y` and return `y`. An exception is thrown if `w`,
`x`, and `y` do not have the same axes.

See also [`LazyAlgebra.unsafe_vmap!`](@ref).

""" vmap!

# Stage 0: Check axes.

function vmap!(y::AbstractArray, α::Number, f::Function, w::AbstractArray, x::AbstractArray,
               ::Stage{0} = _Stage(0))
    # Check axes and directly jump to stage 2 to dispatch on α.
    @assert_same_axes w x y
    return vmap!(α, f, w, x, 𝟘, y, _Stage(2))
end

function vmap!(α::Number, f::Function, w::AbstractArray, x::AbstractArray,
               β::Number, y::AbstractArray,
               ::Stage{0} = _Stage(0))
    # Check axes and jump to stage 1 to dispatch on β, before dispatching on α.
    @assert_same_axes w x y
    return vmap!(α, f, w, x, β, y, _Stage(1))
end

# Stage 1: Dispatch on multiplier `β`.

function vmap!(α::Number, f::Function, w::AbstractArray{Tw,N}, x::AbstractArray{Tx,N},
               β::Number, y::AbstractArray{Ty,N}, ::Stage{1}) where {Tw,Tx,Ty,N}
    @dispatch_on_multiplier β eltype(y) vmap!(α, f, w, x, β, y, _Stage(2))
    return y
end

# Stage 2: Dispatch on multiplier `α`. FIXME Shall we skip computations if α = 0?

function vmap!(α::Number, f::Function, w::AbstractArray{Tw,N}, x::AbstractArray{Tx,N},
               β::Number, y::AbstractArray{Ty,N}, ::Stage{2}) where {Tw,Tx,Ty,N}
    @dispatch_on_multiplier α Base.promote_op(f, eltype(w), eltype(x)) unsafe_vmap!(α, f, w, x, β, y)
    return y
end

"""
    LazyAlgebra.unsafe_vmap!(α, f, w, x, β, y) -> y

overwrite `y` with `y[i] = α*f(w[i], x[i]) + β*y[i])` and returns `y`.

!!! warning
    This method assumes that `w`, `x`, and `y` have the same axes, and that multipliers
    `α` and `β` have suitable types.

"""
function unsafe_vmap!(α::Number, f::Function, w::AbstractArray{Tw,N}, x::AbstractArray{Tx,N},
                      β::Number, y::AbstractArray{Ty,N}) where {Tw,Tx,Ty,N}
    @inbounds @simd for i in eachindex(w, x, y)
        y[i] = α*f(w[i], x[i]) + β*y[i]
    end
    return y
end
