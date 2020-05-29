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
# Copyright (c) 2017-2020 Éric Thiébaut.
#

function unimplemented(::Type{P}, ::Type{T}) where {P<:Operations, T<:Mapping}
    throw(UnimplementedOperation("unimplemented operation `$P` for mapping $T"))
end

function unimplemented(func::Union{AbstractString,Symbol},
                       ::Type{T}) where {T<:Mapping}
    throw(UnimplementedMethod("unimplemented method `$func` for mapping $T"))
end

"""
```julia
@callable T
```

makes concrete type `T` callable as a regular mapping that is `A(x)` yields
`apply(A,x)` for any `A` of type `T`.

"""
macro callable(T)
    quote
	(A::$(esc(T)))(x) = apply(A, x)
    end
end
@callable Adjoint
@callable Inverse
@callable InverseAdjoint
@callable Scaled
@callable Sum
@callable Composition

show(io::IO, ::MIME"text/plain", A::Mapping) = show(io, A)

show(io::IO, A::Identity) = print(io, "Id")

function show(io::IO, A::Scaled)
    λ, M = multiplier(A), unscaled(A)
    if λ == -one(λ)
        print(io, "-")
    elseif λ != one(λ)
        print(io, λ, "⋅")
    end
    show(io, M)
end

function show(io::IO, A::Scaled{<:Sum})
    λ, M = multiplier(A), unscaled(A)
    if λ == -one(λ)
        print(io, "-(")
    elseif λ != one(λ)
        print(io, λ, "⋅(")
    end
    show(io, M)
    if λ != one(λ)
        print(io, ")")
    end
end

function show(io::IO, A::Adjoint{<:Mapping})
    show(io, unveil(A))
    print(io, "'")
end

function show(io::IO, A::Adjoint{T}) where {T<:Union{Scaled,Composition,Sum}}
    print(io, "(")
    show(io, unveil(A))
    print(io, ")'")
end

function show(io::IO, A::Inverse{<:Mapping})
    print(io, "inv(")
    show(io, unveil(A))
    print(io, ")")
end

function show(io::IO, A::InverseAdjoint{<:Mapping})
    print(io, "inv(")
    show(io, unveil(A))
    print(io, ")'")
end

function show(io::IO, A::Sum{N}) where {N}
    function show_term(io::IO, A::Sum)
        print(io, "(")
        show(io, A)
        print(io, ")")
    end
    show_term(io::IO, A::Mapping) = show(io, A)

    for i in 1:N
        let B = A[i]
            if isa(B, Scaled)
                λ, M = multiplier(B), unscaled(B)
                if λ < 0
                    print(io, (i == 1 ? "-" : " - "))
                    λ = -λ
                elseif i > 1
                    print(io, " + ")
                end
                if λ != 1
                    print(io, λ, "⋅")
                end
                show_term(io, M)
            else
                if i > 1
                    print(io, " + ")
                end
                show_term(io, B)
            end
        end
    end
end

function show(io::IO, A::Composition{N}) where {N}
    for i in 1:N
        let B = A[i]
            if i > 1
                print(io, "⋅")
            end
            if isa(B, Sum) || isa(B, Scaled)
                print(io, "(")
                show(io, B)
                print(io, ")")
            else
                show(io, B)
            end
        end
    end
end

"""
    terms(A)

yields the list (as a tuple) of terms that compose mapping `A`.  If `A` is a
sum or a composition of mappings, the list of terms is returned; otherwise, the
1-tuple `(A,)` is returned.

If `A` is sum or a composition of mappings, `Tuple(A)` yields the same result
as `terms(A)`.

"""
terms(A::Union{Sum,Composition}) = getfield(A, :ops)
terms(A::Mapping) = (A,)

@deprecate operands terms

"""
    unveil(A)

unveils the mapping embedded in mapping `A` if it is a *decorated* mapping (see
[`DecoratedMapping`](@ref)); otherwise, just returns `A` if it is not a
*decorated* mapping.

As a special case, `A` may be an instance of `LinearAlgebra.UniformScaling` and
the result is the LazyAlgebra mapping corresponding to `A`.

"""
unveil(A::DecoratedMapping) = getfield(A, :op)
unveil(A::Mapping) = A
unveil(A::UniformScaling) = multiplier(A)*Id
Mapping(A::UniformScaling) = unveil(A)

@deprecate operand unveil

"""
    unscaled(A)

and

    multiplier(A)

respectively yield the mapping `M` and the multiplier `λ` if `A = λ*M` is a
scaled mapping (see [`Scaled`](@ref)); `A` and `1` otherwise.  Note that these
methods also work for intances of `LinearAlgebra.UniformScaling`.

"""
unscaled(A::Mapping) = A
unscaled(A::Scaled) = getfield(A, :M)
unscaled(A::UniformScaling) = Id
multiplier(A::Scaled) = getfield(A, :λ)
multiplier(A::Mapping) = 1
multiplier(A::UniformScaling) = getfield(A, :λ)

@doc @doc(unscaled) multiplier

"""
    identifier(A)

yields an (almost) unique identifier of the mapping `A` computed as
`objectid(unscaled(A))`.  This identifier is used for sorting terms in a sum of
mappings.

!!! warning
    For now, the sorting is not perfect as it is based on `objectid()` which
    is a hashing method.

"""
identifier(A::Mapping) = objectid(unscaled(A))

Base.isless(A::Mapping, B::Mapping) = isless(identifier(A), identifier(B))

# Extend base methods to simplify the code for reducing expressions.
first(A::Mapping) = A
last(A::Mapping) = A
first(A::Union{Sum,Composition}) = @inbounds A[1]
last(A::Union{Sum{N},Composition{N}}) where {N} = @inbounds A[N]
firstindex(A::Union{Sum,Composition}) = 1
lastindex(A::Union{Sum{N},Composition{N}}) where {N} = N
length(A::Union{Sum{N},Composition{N}}) where {N} = N
eltype(::Type{<:Sum{N,T}}) where {N,T} = eltype(T)
eltype(::Type{<:Composition{N,T}}) where {N,T} = eltype(T)
Tuple(A::Union{Sum,Composition}) = terms(A)

# FIXME: Shall we restrict the index to be an integer?
@inline @propagate_inbounds getindex(A::Union{Sum,Composition}, i) = terms(A)[i]

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
                $fn1(A::$P{<:$T}) = $fn2(unveil(A))
            end

            $fn1(::Type{$P}, A::$T) = $fn2(A)

            if $(sfx == :size)
                if $(P != Direct)
                    $fn1(A::$P{<:$T}, dim...) = $fn2(unveil(A), dim...)
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
coefficients(A)
```

yields the object backing the storage of the coefficients of the linear mapping
`A`.  Not all linear mappings extend this method.

""" coefficients

"""

```julia
check(A) -> A
```

checks integrity of mapping `A` and returns it.

"""
check(A::Mapping) = A

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
    is_linear(A) || bad_argument("expecting a linear mapping")
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

"""
```julia
are_same_mappings(A, B)
```

yields whether `A` and `B` are the same mappings in the sense that their
effects will always be the same.  This method is used to perform some
simplifications and optimizations and may have to be specialized for specific
mapping types.  The default implementation is to return `A === B`.

!!! note
    The returned result may be true although `A` and `B` are not necessarily
    the same objects.  For instance, if `A` and `B` are two sparse matrices
    whose coefficients and indices are stored in the same vectors (as can be
    tested with [`is_same_mutable_object`](@ref)) this method should return
    `true` because the two operators will behave identically (any changes in
    the coefficients or indices of `A` will be reflected in `B`).  If any of
    the vectors storing the coefficients or the indices are not the same
    objects, then `are_same_mappings(A,B)` must return `false` even though the
    stored values may be the same because it is possible, later, to change one
    operator without affecting identically the other.

"""
are_same_mappings(::Mapping, ::Mapping) = false # false if not same types
are_same_mappings(A::T, B::T) where {T<:Mapping} = (A === B)

"""

```julia
gram(A) -> A'*A
```

yields the Gram matrix of the "*columns*" of the linear mapping `A`.

See also [`Gram`](@ref).

"""
gram(A::LinearMapping) = A'*A

# Functions that will be inlined to execute an elementary operation when
# performing `α⋅x + β⋅y`.  Passing these (simple) functions to another method
# is to simplify the coding of vectorized methods and of the the `apply!`
# method by mappings.
axpby_yields_x(    α, x, β, y) = x         # α = 1, β = 0
axpby_yields_xpy(  α, x, β, y) = x + y     # α = 1, β = 1
axpby_yields_xmy(  α, x, β, y) = x - y     # α = 1, β = -1
axpby_yields_xpby( α, x, β, y) = x + β*y   # α = 1, any β
axpby_yields_ax(   α, x, β, y) = α*x       # any α, β = 0
axpby_yields_axpy( α, x, β, y) = α*x + y   # any α, β = 1
axpby_yields_axmy( α, x, β, y) = α*x - y   # any α, β = -1
axpby_yields_axpby(α, x, β, y) = α*x + β*y # any α, any β
