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

@noinline function unimplemented(::Type{P},
                                 ::Type{T}) where {P<:Operations, T<:Mapping}
    throw(UnimplementedOperation("unimplemented operation `$P` for mapping $T"))
end

@noinline function unimplemented(func::Union{AbstractString,Symbol},
                                 ::Type{T}) where {T<:Mapping}
    throw(UnimplementedMethod("unimplemented method `$func` for mapping $T"))
end

"""
    @callable T

makes concrete type `T` callable as a regular mapping, that is `A(x)` yields
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
@callable Jacobian
@callable Gram
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

function show(io::IO, A::Jacobian{<:Mapping})
    print(io, "∇(")
    show(io, primitive(A))
    print(io, ",x)")
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
    primitive(J)

and

    variables(J)

respectively yield the mapping `A` and the variables `x` embedded in Jacobian
`J = ∇(A,x)`.

"""
primitive(J::Jacobian) = getfield(J, :A)
variables(J::Jacobian) = getfield(J, :x)
@doc @doc(primitive) variables

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

@inline @propagate_inbounds getindex(A::Union{Sum,Composition}, i) =
    getindex(terms(A), i)

"""
    input_type([P=Direct,] A)
    output_type([P=Direct,] A)

yield the (preferred) types of the input and output arguments of the operation
`P` with mapping `A`.  If `A` operates on Julia arrays, the element type, list
of dimensions, `i`-th dimension and number of dimensions for the input and
output are given by:

    input_eltype([P=Direct,] A)          output_eltype([P=Direct,] A)
    input_size([P=Direct,] A)            output_size([P=Direct,] A)
    input_size([P=Direct,] A, i)         output_size([P=Direct,] A, i)
    input_ndims([P=Direct,] A)           output_ndims([P=Direct,] A)

For mappings operating on Julia arrays, only `input_size(A)` and
`output_size(A)` have to be implemented.

Also see: [`vcreate`](@ref), [`apply!`](@ref), [`LinearMapping`](@ref),
[`Operations`](@ref).

"""
function input_type end

for sfx in (:size, :eltype, :ndims, :type),
    pfx in (:output, :input)

    fn1 = Symbol(pfx, "_", sfx)

    for P in (Direct, Adjoint, Inverse, InverseAdjoint)

        fn2 = Symbol(P === Adjoint || P === Inverse ?
                     (pfx === :output ? :input : :output) : pfx, "_", sfx)

        T = (P === Adjoint || P === InverseAdjoint ? LinearMapping : Mapping)

        # Provide basic methods for the different operations and for tagged
        # mappings.
        @eval $fn1(::Type{$P}, A::$T) = $fn2(A)
        if P !== Direct
            @eval $fn1(A::$P{<:$T}) = $fn2(unveil(A))
        end
        if sfx === :size
            if P !== Direct
                @eval $fn1(A::$P{<:$T}, dim...) = $fn2(unveil(A), dim...)
            end
            @eval $fn1(::Type{$P}, A::$T, dim...) = $fn2(A, dim...)
        end
    end

    # Link documentation for the basic methods.
    if fn1 !== :input_type
        @eval @doc @doc(:input_type) $fn1
    end

end

# Provide default methods for `$(sfx)_size(A, dim...)` and `$(sfx)_ndims(A)`.
for pfx in (:input, :output)
    get_size = Symbol(pfx, "_size")
    get_ndims = Symbol(pfx, "_ndims")
    @eval begin
        $get_ndims(A::Mapping) = length($get_size(A))
        $get_size(A::Mapping, dim) = $get_size(A)[dim]
        function $get_size(A::Mapping, dim...)
            dims = $get_size(A)
            ntuple(i -> dims[dim[i]], length(dim))
        end
    end
end

for f in (:input_eltype, :output_eltype, :input_size, :output_size)
    @eval $f(::T) where {T<:Mapping} = unimplemented($(string(f)), T)
end

"""
    nrows(A)

yields the *equivalent* number of rows of the linear operator `A`.  Not all
operators extend this method.

In the implemented generalization of linear operators, the equivalent number of
rows is the number of element of the result of applying the operator be it
single- or multi-dimensional.

"""
nrows(A::LinearMapping) = prod(row_size(A))
@noinline nrows(A::Mapping) =
    throw(ArgumentError("`nrows` is only implemented for linear mappings"))

"""
    ncols(A)

yields the *equivalent* number of columns of the linear operator `A`.  Not all
operators extend this method.

In the implemented generalization of linear operators, the equivalent number of
columns is the number of element of an argument of the operator be it single-
or multi-dimensional.

"""
ncols(A::LinearMapping) = prod(col_size(A))
@noinline ncols(A::Mapping) =
    throw(ArgumentError("`ncols` is only implemented for linear mappings"))

"""
    row_size(A)

yields the dimensions of the result of applying the linear operator `A`, this
is equivalent to `output_size(A)`.  Not all operators extend this method.

"""
row_size(A::LinearMapping) = output_size(A)
@noinline row_size(A::Mapping) =
    throw(ArgumentError("`row_size` is only implemented for linear mappings"))

"""
    col_size(A)

yields the dimensions of the argument of the linear operator `A`, this
is equivalent to `input_size(A)`.  Not all
operators extend this method.

"""
col_size(A::LinearMapping) = input_size(A)
@noinline col_size(A::Mapping) =
    throw(ArgumentError("`col_size` is only implemented for linear mappings"))

"""
    coefficients(A)

yields the object backing the storage of the coefficients of the linear mapping
`A`.  Not all linear mappings extend this method.

""" coefficients

"""
    check(A) -> A

checks integrity of mapping `A` and returns it.

"""
check(A::Mapping) = A

"""
    checkmapping(y, A, x) -> (v1, v2, v1 - v2)

yields `v1 = vdot(y, A*x)`, `v2 = vdot(A'*y, x)` and their difference for `A` a
linear mapping, `y` a *vector* of the output space of `A` and `x` a *vector* of
the input space of `A`.  In principle, the two inner products should be equal
whatever `x` and `y`; otherwise the mapping has a bug.

Simple linear mappings operating on Julia arrays can be tested on random
*vectors* with:

    checkmapping([T=Float64,] outdims, A, inpdims) -> (v1, v2, v1 - v2)

with `outdims` and `outdims` the dimensions of the output and input *vectors*
for `A`.  Optional argument `T` is the element type.

If `A` operates on Julia arrays and methods `input_eltype`, `input_size`,
`output_eltype` and `output_size` have been specialized for `A`, then:

    checkmapping(A) -> (v1, v2, v1 - v2)

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
    identical(A, B)

yields whether `A` is the same mapping as `B` in the sense that their effects
will always be the same.  This method is used to perform some simplifications
and optimizations and may have to be specialized for specific mapping types.
The default implementation is to return `A === B`.

!!! note
    The returned result may be true although `A` and `B` are not necessarily
    the same objects.  For instance, if `A` and `B` are two sparse matrices
    whose coefficients and indices are stored in the same arrays (as can be
    tested with the `===` or `≡` operators, `identical(A,B)` should return
    `true` because the two operators will always behave identically (any
    changes in the coefficients or indices of `A` will be reflected in `B`).
    If any of the arrays storing the coefficients or the indices are not the
    same objects, then `identical(A,B)` must return `false` even though the
    stored values may be the same because it is possible, later, to change one
    operator without affecting identically the other.

"""
@inline identical(::Mapping, ::Mapping) = false # false if not same types
@inline identical(A::T, B::T) where {T<:Mapping} = (A === B)

"""
    gram(A) -> A'*A

yields the Gram operator built out of the linear mapping `A`.  The result is
equivalent to `A'*A` but its type depends on simplifications that may occur.

See also [`Gram`](@ref).

"""
gram(A::LinearMapping) = A'*A
gram(A::Mapping) =
    is_linear(A) ? A'*A : throw_forbidden_Gram_of_non_linear_mapping()

@noinline throw_forbidden_Gram_of_non_linear_mapping() =
    bad_argument("making a Gram operator out of a non-linear mapping is not allowed")

# Inlined functions called to perform `α*x + β*y` for specific values of the
# multipliers `α` and `β`.  Passing these (simple) functions to another method
# is to simplify the coding of vectorized methods and of the the `apply!`
# method by mappings.  NOTE: Forcing inlining may not be necessary but it does
# not hurt.
@inline axpby_yields_zero( α, x, β, y) = zero(typeof(y)) # α = 0, β = 0
@inline axpby_yields_y(    α, x, β, y) = y               # α = 0, β = 1
@inline axpby_yields_my(   α, x, β, y) = -y              # α = 0, β = -1
@inline axpby_yields_by(   α, x, β, y) = β*y             # α = 0, any β
@inline axpby_yields_x(    α, x, β, y) = x               # α = 1, β = 0
@inline axpby_yields_xpy(  α, x, β, y) = x + y           # α = 1, β = 1
@inline axpby_yields_xmy(  α, x, β, y) = x - y           # α = 1, β = -1
@inline axpby_yields_xpby( α, x, β, y) = x + β*y         # α = 1, any β
@inline axpby_yields_mx(   α, x, β, y) = -x              # α = -1, β = 0
@inline axpby_yields_ymx(  α, x, β, y) = y - x           # α = -1, β = 1
@inline axpby_yields_mxmy( α, x, β, y) = -x - y          # α = -1, β = -1
@inline axpby_yields_bymx( α, x, β, y) = β*y - x         # α = -1, any β
@inline axpby_yields_ax(   α, x, β, y) = α*x             # any α, β = 0
@inline axpby_yields_axpy( α, x, β, y) = α*x + y         # any α, β = 1
@inline axpby_yields_axmy( α, x, β, y) = α*x - y         # any α, β = -1
@inline axpby_yields_axpby(α, x, β, y) = α*x + β*y       # any α, any β
