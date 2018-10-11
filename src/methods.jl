#
# methods.jl -
#
# Implement non-specific methods for mappings.
#
#-------------------------------------------------------------------------------
#
# This file is part of LazyAlgebra (https://github.com/emmt/LazyAlgebra.jl)
# released under the MIT "Expat" license.
#
# Copyright (c) 2017-2018 Éric Thiébaut.
#

function unimplemented(::Type{P}, ::Type{T}) where {P<:Operations, T<:Mapping}
    throw(UnimplementedOperation("unimplemented operation `$P` for mapping $T"))
end

function unimplemented(func::Union{AbstractString,Symbol},
                       ::Type{T}) where {T<:Mapping}
    throw(UnimplementedMethod("unimplemented method `$func` for mapping $T"))
end

"""
# Containers and Marked Objects

Julia typing system can be exploited to "mark" some object instances so that
they are seen as another specific type.  For instance, this feature is used to
mark linear operators as being transposed (so that they behave as their
adjoint) or inverted (so that they behave as their inverse).

These marked objects have a single member: the object that is marked.  This
single member can be retrieved by the `contents` method.  The following piece
of code shows the idea:


```julia
struct MarkedType{T}
    data::T
end
MarkedType(obj::T) where {T} = MarkedType{T}(obj)
contents(obj::MarkedType) = obj.data
```

More generally, the `contents` method can be used to retrieve the contents of a
"container" object:

```julia
contents(C)
```

yields the contents of the container `C`.  By extension, a *container* is any
type which implements the `contents` method.

"""
contents(H::Union{Hessian,HalfHessian}) = H.obj
contents(A::Union{Adjoint,Inverse,InverseAdjoint}) = A.op
contents(A::Union{Sum,Composition}) = A.ops

"""
```julia
input_type([P=Direct,] A)
output_type([P=Direct,] A)
```

yield the (preferred) types of the input and output arguments of the operation
`P` with mapping `A`.  If `A` operates on Julia arrays, the element type,
list of dimensions, `i`-th dimension and number of dimensions for the input and
output are given by:

    input_eltype([P=Direct,] A)          output_eltype([P=Direct,] A)
    input_size([P=Direct,] A)            output_size([P=Direct,] A)
    input_size([P=Direct,] A, i)         output_size([P=Direct,] A, i)
    input_ndims([P=Direct,] A)           output_ndims([P=Direct,] A)

Only `input_size(A)` and `output_size(A)` have to be implemented.

Also see: [`vcreate`](@ref), [`apply!`](@ref), [`LinearMapping`](@ref),
[`Operations`](@ref).

"""
function input_type end

for sfx in (:size, :eltype, :ndims, :type),
    pfx in (:output, :input)

    fn1 = Symbol(pfx, "_", sfx)

    for P in (Direct, Adjoint, Inverse, InverseAdjoint)

        fn2 = Symbol(P == Adjoint || P == Inverse ?
                     (pfx == :output ? :input : :output) : pfx, "_", sfx)

        T = (P == Adjoint || P == InverseAdjoint ? LinearMapping : Mapping)

        # Provide basic methods for the different operations and for tagged
        # mappings.
        @eval begin

            if $(P != Direct)
                $fn1(A::$P{<:$T}) = $fn2(A.op)
            end

            $fn1(::Type{$P}, A::$T) = $fn2(A)

            if $(sfx == :size)
                if $(P != Direct)
                    $fn1(A::$P{<:$T}, dim...) = $fn2(A.op, dim...)
                end
                $fn1(::Type{$P}, A::$T, dim...) = $fn2(A, dim...)
            end
        end
    end

    # Link documentation for the basic methods.
    @eval begin
        if $(fn1 != :input_type)
            @doc @doc(:input_type) $fn1
        end
    end

end

# Provide default methods for `$(sfx)_size(A, dim...)` and `$(sfx)_ndims(A)`.
for pfx in (:input, :output)
    pfx_size = Symbol(pfx, "_size")
    pfx_ndims = Symbol(pfx, "_ndims")
    @eval begin

        $pfx_ndims(A::Mapping) = length($pfx_size(A))

        $pfx_size(A::Mapping, dim) = $pfx_size(A)[dim]

        function $pfx_size(A::Mapping, dim...)
            dims = $pfx_size(A)
            ntuple(i -> dims[dim[i]], length(dim))
        end

    end
end

for f in (:input_eltype, :output_eltype, :input_size, :output_size)
    @eval $f(::T) where {T<:Mapping} = unimplemented($(string(f)), T)
end

"""
```julia
convert_multiplier(α, T...)
```

converts the scalar `α` in a suitable type for operations involving arguments
of types `T...`.  In general, `T...` is a single type and is the element type
of the variables to be multiplied by `α`.

!!! note
    For now, complex-valued multipliers are not supported.  The type of the
    multiplier `α` must be integer or floating-point.  If `α` and the real part
    of all types `T...` are integers, the returned value is and integer;
    otherwise, the returned value is a floating-point.

See also: [`convert`](@ref) and [`promote_type`](@ref).

"""
convert_multiplier(α::Real, T::Type{<:Number}, args::Type{<:Number}...) =
    convert_multiplier(α, promote_type(T, args...))

# Sub-types of Number are: Complex and Real.
convert_multiplier(α::Real, ::Type{Complex{T}}) where {T<:Real} =
    convert_multiplier(α, T)

# Sub-types of Real are: AbstractFloat, AbstractIrrational, Integer and
# Rational.
convert_multiplier(α::Integer, T::Type{<:Integer}) = convert(T, α)
convert_multiplier(α::Real, T::Type{<:AbstractFloat}) = convert(T, α)
convert_multiplier(α::Real, T::Type{<:Real}) = convert(float(T), α)

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
function checkmapping(y::Ty, A::Mapping, x::Tx) where {Tx, Ty}
    is_linear(A) ||
        throw(ArgumentError("expecting a linear map"))
    v1 = vdot(y, A*x)
    v2 = vdot(A'*y, x)
    (v1, v2, v1 - v2)
end

function checkmapping(::Type{T},
                      outdims::Tuple{Vararg{Int}},
                      A::Mapping,
                      inpdims::Tuple{Vararg{Int}}) where {T<:AbstractFloat}
    checkmapping(randn(T, outdims), A, randn(T, inpdims))
end

function checkmapping(outdims::Tuple{Vararg{Int}},
                      A::Mapping,
                      inpdims::Tuple{Vararg{Int}})
    checkmapping(Float64, outdims, A, inpdims)
end

checkmapping(A::LinearMapping) =
    checkmapping(randn(output_eltype(A), output_size(A)), A,
                 randn(input_eltype(A), input_size(A)))

"""
```julia
densearray([T=eltype(A),] A)
```

lazyly yields a dense array based on `A`.  Optional argument `T` is to specify
the element type of the result.  Argument `A` is returned if it is already a
dense array with the requested element type; otherwise, [`convert`](@ref) is
called to produce the result.

Similarly:

```julia
densevector([T=eltype(V),] V)
densematrix([T=eltype(M),] M)
```

respectively yield a dense vector from `V` and a dense matrix from `M`.

"""
densearray(A::DenseArray) = A
densearray(::Type{T}, A::DenseArray{T,N}) where {T,N} = A
densearray(A::AbstractArray{T,N}) where {T,N} = densearray(T, A)
densearray(::Type{T}, A::AbstractArray{<:Any,N}) where {T,N} =
    convert(Array{T,N}, A)

densevector(V::AbstractVector{T}) where {T} = densearray(T, V)
densevector(::Type{T}, V::AbstractVector) where {T} = densearray(T, V)
@doc @doc(densearray) densevector

densematrix(M::AbstractMatrix{T}) where {T} = densearray(T, M)
densematrix(::Type{T}, M::AbstractMatrix) where {T} = densearray(T, M)
@doc @doc(densearray) densematrix

"""
Any of the following calls:

```julia
fastrange(A)
fastrange((n1, n2, ...))
fastrange((i1:j1, i2:j2, ...))
fastrange(CartesianIndex(i1, i2, ...), CartesianIndex(j1, j2, ...))
fastrange(R)
```

yields an instance of `CartesianIndices` or `CartesianRange` (whichever is the
most efficient depending on the version of Julia) for multi-dimensional
indexing of all the elements of array `A`, a multi-dimensional array of
dimensions `(n1,n2,...)`, a multi-dimensional region whose first and last
indices are `(i1,i2,...)` and `(j1,j2,...)` or a Cartesian region defined by
`R`, an instance of `CartesianIndices` or of `CartesianRange`.

"""
fastrange(A::AbstractArray) = fastrange(axes(A))
@static if isdefined(Base, :CartesianIndices)
    import Base: axes
    fastrange(R::CartesianIndices) = R
    fastrange(start::CartesianIndex{N}, stop::CartesianIndex{N}) where {N} =
        CartesianIndices(map((i,j) -> i:j, start, stop))
    fastrange(dims::NTuple{N,Integer}) where {N} =
        CartesianIndices(map((d) -> Base.OneTo(Int(d)), dims))
    fastrange(dims::NTuple{N,Int}) where {N} =
        CartesianIndices(map(Base.OneTo, dims))
    fastrange(inds::NTuple{N,AbstractUnitRange{<:Integer}}) where {N} =
        CartesianIndices(inds)
else
    import Base: indices
    const axes = indices
    fastrange(R::CartesianIndices) = fastrange(R.indices)
    fastrange(R::CartesianRange) = R
    fastrange(start::CartesianIndex{N}, stop::CartesianIndex{N}) where {N} =
        CartesianRange(start, stop)
    fastrange(dims::NTuple{N,Integer}) where {N} =
        CartesianRange(one(CartesianIndex{N}), CartesianIndex(dims))
    fastrange(inds::NTuple{N,AbstractUnitRange{<:Integer}}) where {N} =
        CartesianRange(CartesianIndex(map(first, inds)),
                       CartesianIndex(map(last,  inds)))
end

"""
```julia
is_same_mutable_object(a, b)
```

yields whether `a` and `b` are references to the same object.  This function
can be used to check whether [`vcreate`](@ref) returns the same object as the
input variables.

This function is very fast, it takes a few nanoseonds on my laptop.

"""
is_same_mutable_object(a, b) =
    (! isimmutable(a) && ! isimmutable(b) &&
     pointer_from_objref(a) === pointer_from_objref(b))
