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
# Copyright (c) 2017-2022 Éric Thiébaut.
#

"""
    @callable T

makes instances of concrete type `T` callable as a regular `LazyAlgebra`
mapping, that is `A(x)` yields `apply(A,x)` for any `A` of type `T`.

"""
macro callable(T)
    quote
	(A::$(esc(T)))(x) = apply(A, x)
    end
end
@callable Adjoint
@callable Inverse
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

function show(io::IO, A::Scaled{<:Any,<:Sum})
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

function show(io::IO, A::Adjoint)
    show(io, parent(A))
    print(io, "'")
end

function show(io::IO, A::Adjoint{<:Union{Scaled,Composition,Sum}})
    print(io, "(")
    show(io, parent(A))
    print(io, ")'")
end

function show(io::IO, A::Inverse)
    print(io, "inv(")
    show(io, parent(A))
    print(io, ")")
end

function show(io::IO, A::Jacobian)
    print(io, "∇(")
    show(io, primitive(A))
    print(io, ",x)")
end

function show(io::IO, A::Sum)
    function show_term(io::IO, A::Sum)
        print(io, "(")
        show(io, A)
        print(io, ")")
    end
    show_term(io::IO, A::Mapping) = show(io, A)

    for i in eachindex(A)
        let B = A[i]
            if isa(B, Scaled)
                λ, M = multiplier(B), unscaled(B)
                if λ < zero(λ)
                    print(io, (i == 1 ? "-" : " - "))
                    λ = -λ
                elseif i > 1
                    print(io, " + ")
                end
                if λ != one(λ)
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

function show(io::IO, A::Composition)
    for i in eachindex(A)
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

yields a tuple of the terms that compose mapping `A`. If `A` is a sum or a
composition of mappings, the list of its terms is returned; otherwise, the
1-tuple `(A,)` is returned.

If `A` is sum or a composition of mappings, `Tuple(A)` yields the same result
as `terms(A)`.

If `A` is a sum or composition type, the tuple type of the terms is returned.

"""
terms(A::Union{Sum,Composition}) = getfield(A, :terms)
terms(A::Mapping) = (A,)
terms(::Type{<:Union{Sum{L,N,T},Composition{L,N,T}}}) where {L,N,T} = T

"""
    terms(+, A)

yields a tuple of the terms whose sum are equal to the mapping `A`. If `A` is a
sum of mappings, the list of its terms is returned; otherwise, the 1-tuple
`(A,)` is returned.

"""
terms(::typeof(+), A::Sum) = terms(A)
terms(::typeof(+), A::Mapping) = (A,)

"""
    terms(*, A)

yields a tuple of the terms whose composition are equal to the mapping `A`. If
`A` is a composition of mappings, the list of its terms is returned; otherwise,
the 1-tuple `(A,)` is returned.

"""
terms(::typeof(*), A::Sum) = terms(A)
terms(::typeof(*), A::Mapping) = (A,)

"""
    nterms(A)

yields the number of terms that are returned by `terms(A)`. The returned value
is a *trait*, argument can be a mapping instance or type.

"""
nterms(A::Mapping) = nterms(typeof(A))
nterms(::Type{<:Mapping}) = 1
nterms(::Type{<:Union{Sum{L,N,T},Composition{L,N,T}}}) where {L,N,T} = N

"""
    unscaled(A)

yields the mapping `M` of the scaled mapping `A = λ*M` (see [`Scaled`](@ref));
otherwise yields `A`. This method also works for intances of
`LinearAlgebra.UniformScaling`. Call [`multiplier`](@ref) to get the multiplier
`λ`.

If `A` is a mapping type, yields the corresponding unscaled mapping type.

"""
unscaled(A::Mapping) = A
unscaled(A::Scaled) = getfield(A, :mapping)
unscaled(::Type{M}) where {M<:Mapping} = M
unscaled(::Type{<:Scaled{L,M,S}}) where {L,M,S} = M

"""
    multiplier(A)

yields the multiplier `λ` of the scaled mapping `A = λ*M` (see
[`Scaled`](@ref)); otherwise yields `1`. Note that this method also works for
intances of `LinearAlgebra.UniformScaling`. Call [`unscaled`](@ref) to get the
mapping `M`. `λ`.

If `A` is a mapping type, yields the corresponding multiplier type.

"""
multiplier(A::Scaled) = getfield(A, :multiplier)
multiplier(A::Mapping) = 1

multiplier(::Type{M}) where {M<:Mapping} = Int
multiplier(::Type{<:Scaled{L,M,S}}) where {L,M,S} = S

Base.parent(A::DecoratedMapping) = getfield(A, :parent)
Base.parent(::Type{<:DecoratedMapping{M}}) where {M} = M

# Extend a few methods for `LinearAlgebra.UniformScaling` (NOTE: see code in
# `LinearAlgebra/src/uniformscaling.jl`)
unscaled(A::UniformScaling) = Id
unscaled(::Type{<:UniformScaling}) = typeof(Id)
multiplier(A::UniformScaling) = getfield(A, :λ)
multiplier(::Type{<:UniformScaling{T}}) where {T} = T
LinearMapping(A::UniformScaling) = multiplier(A)*Id
Mapping(A::UniformScaling) = LinearMapping(A)

"""
    primitive(J)

yields the mapping `A` embedded in the Jacobian `J = ∇(A,x)`. Call
[`variables`](@ref) to get `x` instead.

"""
primitive(J::Jacobian) = getfield(J, :primitive)

"""
    variables(J)

yields the variables `x` embedded in the Jacobian `J = ∇(A,x)`. Call
[`primitive`](@ref) to get `A` instead.

"""
variables(J::Jacobian) = getfield(J, :variables)

# Extend base methods to simplify the code for reducing expressions.
# FIXME: Base.first(A::Mapping) = A
# FIXME: Base.last(A::Mapping) = A
Base.first(A::Union{Sum,Composition}) = first(terms(A))
Base.last(A::Union{Sum,Composition}) = last(terms(A))
Base.firstindex(A::Union{Sum,Composition}) = firstindex(terms(A))
Base.lastindex(A::Union{Sum,Composition}) = lastindex(terms(A))
Base.length(A::Union{Sum,Composition}) = length(terms(A))
Base.keys(A::Union{Sum,Composition}) = keys(terms(A)) # NOTE: needed by eachindex
# FIXME: Base.eltype(::Type{<:Sum{L,N,T}}) where {L,N,T} = eltype(T)
# FIXME: Base.eltype(::Type{<:Composition{L,N,T}}) where {L,N,T} = eltype(T)

@inline @propagate_inbounds getindex(A::Union{Sum,Composition}, i) = terms(A)[i]

"""
    output_eltype(A, x)

yields the output element type of the result of applying the mapping `A` to the
variables `x`.

The output element type only depends on the types of the mapping and of its
arguments. This method may be extended for `A` and `x` being specific mapping
and/or argument types.

A default implementation is provided for linear mappings `A` and arguments `x`
that implement the methods `Base.eltype(A)` and `Base.eltype(x)`.

"""
output_eltype(A::Mapping, x) = output_eltype(typeof(A), typeof(x))

output_eltype(::Type{A}, ::Type{x}) where {A<:LinearMapping,x} = float(zero(eltype(A))*zero(eltype(x)))
output_eltype(::Type{A}, ::Type{x}) where {A<:Scaled,x} = output_eltype(multiplier(A), unscaled(A), x)
output_eltype(::Type{A}, ::Type{x}) where {A<:Adjoint,x} = output_eltype(parent(A), x)
#  FIXME: output_eltype(::Type{A}, ::Type{x}) where {A<:Inverse,x} = ???

@generated output_eltype(::Type{A}, ::Type{x}) where {A<:Sum,x} =
    :($(float(typeof(+(map(a -> zero(output_eltype(a, x)), types_of_terms(A))...)))))

@generated output_eltype(::Type{A}, ::Type{x}) where {A<:Composition,x} =
    :($(float(_output_eltype(*, types_of_terms(A), 1, x))))

_output_eltype(::typeof(*), types, i, x) =
    if i < length(types)
        return output_eltype(types[i], _output_eltype(*, types, i+1, x))
    else
        return output_eltype(types[i], x)
    end

"""
    output_eltype(α::Number, A::Mapping, x)

yields the output element type of `α*A*x` for a linear mapping `A` or of
`α*A(x)` for a non-linear mapping `A`, that is the element type of the result
of `α` times `A` applied to `x`.

"""
output_eltype(α::Number, A::Mapping, x) = output_eltype(typeof(α), typeof(A), typeof(x))
@inline function output_eltype(::Type{α}, ::Type{A}, ::Type{x}) where {α<:Number,A<:Mapping,x}
    T = output_eltype(A, x)
    return convert_floating_point_type(floating_point_type(T), typeof(zero(α)*zero(T)))
end

"""
    input_ndims(A) -> n

yields the number of dimensions of the argument of mapping `A`. Argument may
also be a mapping type. Not all mappings implement this method.

"""
input_ndims(A::Mapping) = input_ndims(typeof(A))
input_ndims(::Type{T}) where {T<:Adjoint} = output_ndims(parent(T))
input_ndims(::Type{T}) where {T<:Inverse} = output_ndims(parent(T))

"""
    ouput_ndims(A) -> n

yields the number of dimensions of the result of mapping `A`. Argument may
also be a mapping type. Not all mappings implement this method.

"""
output_ndims(A::Mapping) = output_ndims(typeof(A))
output_ndims(::Type{T}) where {T<:Adjoint} = input_ndims(parent(T))
output_ndims(::Type{T}) where {T<:Inverse} = input_ndims(parent(T))

"""
    input_size(A) -> dims

yields the dimensions of the argument of mapping `A`. Not all mappings
implement this method.

"""
input_size(A::Mapping, i...) = input_size(A)[i...]
input_size(A::Adjoint) = output_size(parent(A))
input_size(A::Inverse) = output_size(parent(A))

"""
    output_size(A) -> dims

yields the dimensions of the result of mapping `A`. Not all mappings
implement this method.

    output_size(A, i) -> dim

yields the `i`-th dimension of the result of mapping `A`. Not all mappings
implement this method.

"""
output_size(A::Mapping, i...) = output_size(A)[i...]
output_size(A::Adjoint) = input_size(parent(A))
output_size(A::Inverse) = input_size(parent(A))

for f in (:input_eltype, :output_eltype, :input_size, :output_size)
    @eval $f(::T) where {T<:Mapping} = unimplemented($(string(f)), T)
end

@noinline function unimplemented(func::Union{AbstractString,Symbol},
                                 ::Type{T}) where {T<:Mapping}
    throw(UnimplementedMethod("unimplemented method `$func` for mapping $T"))
end

inv_type(::Type{T}) where {T<:Number} = error("not implemented")

# Implement a few methods of the abstract array API for the linear mappings
# seen as generalized matrices with multi-dimensional rows and columns.
Base.ndims(A::LinearMapping) = ndims(typeof(A))
Base.ndims(::Type{T}) where {T<:LinearMapping} = output_ndims(T) + input_ndims(T)
Base.size(A::LinearMapping) = (output_size(A)..., input_size(A)...)
Base.size(A::LinearMapping, i...) = size(A)[i...]
Base.eltype(A::LinearMapping) = eltype(typeof(A))
Base.eltype(::Type{Adjoint{M}}) where {M} = eltype(M)
# FIXME: Base.eltype(::Type{Inverse{true,M}}) where {M} = eltype(M)
Base.eltype(::Type{Scaled{L,M,S}}) where {L,M,S} =
    convert_floating_point_type(floating_point_type(eltype(M)),
                                promote_type(S, eltype(A)))

@generated Base.eltype(::Type{M}) where {M<:Sum} =
    :($(promote_type(map(eltype, types_of_terms(M))...)))

@generated Base.eltype(::Type{M}) where {M<:Composition{true}} =
    :($(typeof(*(map(T -> zero(eltype(T)), types_of_terms(M))...))))

# FIXME: remove this
#=
"""
    input_type(A)
    output_type(A)

yield the (preferred) types of the input and output arguments of the mapping
`A`. If `A` operates on Julia arrays, the element type, list of dimensions,
`i`-th dimension and number of dimensions for the input and output are given
by:

    input_eltype([P=Direct,] A)          output_eltype([P=Direct,] A)
    input_size([P=Direct,] A)            output_size([P=Direct,] A)
    input_size([P=Direct,] A, i)         output_size([P=Direct,] A, i)
    input_ndims([P=Direct,] A)           output_ndims([P=Direct,] A)

For mappings operating on Julia arrays, only `input_size(A)` and
`output_size(A)` have to be implemented.

Also see: [`vcreate`](@ref), [`apply!`](@ref), [`LinearMapping`](@ref).

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
            @eval $fn1(A::$P{<:$T}) = $fn2(parent(A))
        end
        if sfx === :size
            if P !== Direct
                @eval $fn1(A::$P{<:$T}, dim...) = $fn2(parent(A), dim...)
            end
            @eval $fn1(::Type{$P}, A::$T, dim...) = $fn2(A, dim...)
        end
    end

    # Link documentation for the basic methods.
    if fn1 !== :input_type
        @eval @doc @doc(:input_type) $fn1
    end

end
=#

"""
    nrows(A)

yields the *equivalent* number of rows of the linear operator `A`. Not all
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

yields the *equivalent* number of columns of the linear operator `A`. Not all
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
is equivalent to `output_size(A)`. Not all operators extend this method.

"""
row_size(A::LinearMapping) = output_size(A)
@noinline row_size(A::Mapping) =
    throw(ArgumentError("`row_size` is only implemented for linear mappings"))

"""
    col_size(A)

yields the dimensions of the argument of the linear operator `A`, this is
equivalent to `input_size(A)`. Not all operators extend this method.

"""
col_size(A::LinearMapping) = input_size(A)
@noinline col_size(A::Mapping) =
    throw(ArgumentError("`col_size` is only implemented for linear mappings"))

"""
    coefficients(A)

yields the object backing the storage of the coefficients of the linear mapping
`A`. Not all linear mappings extend this method.

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
the input space of `A`. In principle, the two inner products should be equal
whatever `x` and `y`; otherwise the mapping has a bug.

Simple linear mappings operating on Julia arrays can be tested on random
*vectors* with:

    checkmapping([T=Float64,] outdims, A, inpdims) -> (v1, v2, v1 - v2)

with `outdims` and `outdims` the dimensions of the output and input *vectors*
for `A`. Optional argument `T` is the element type.

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
    gram(A) -> A'*A

yields the Gram operator built out of the linear mapping `A`. The result is
equivalent to `A'*A` but its type depends on simplifications that may occur.

See also [`Gram`](@ref).

"""
gram(A::LinearMapping) = A'*A
@noinline gram(A::Mapping) =
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

#------------------------------------------------------------------------------
# VCREATE, APPLY AND APPLY!

"""
    vcreate(α, op(A), x, scratch=false) -> y

yields a new instance `y` suitable for storing the result of applying mapping
`op(A)` to the argument `x` and for any supported operation `op ∈
(identity,adjoint,inv,inv∘adjoint)`.

Optional argument `scratch` indicates whether input argument `x` can be
overwritten by the operation and thus used to store the result. This may be
exploited by some mappings (which are able to operate *in-place*) to avoid
allocating a new object for the result `y`.

The caller should set `scratch = true` if `x` is not needed after calling
`apply`. If `scratch = true`, then it is possible that `y` be the same object
as `x`; otherwise, `y` is a new object unless applying the operation yields the
same contents as `y` for the result `x` (this is always true for the identity
for instance). Thus, in general, it should not be assumed that the returned `y`
is different from the input `x`.

The method `vcreate(::Type{P}, A, x)` should be implemented by linear mappings
for any supported operations `P` and argument type for `x`. The result returned
by `vcreate` should be of predictible type to ensure *type-stability*. Checking
the validity (*e.g.* the size) of argument `x` in `vcreate` may be skipped
because this argument will be eventually checked by the `apply!` method.

See also: [`Mapping`](@ref), [`apply`](@ref).

"""
vcreate(A::Mapping, x, scratch::Bool=false) = vcreate(1, A, x, scratch)
vcreate(α::Number, A::Mapping, x) = vcreate(α, A, x, false)

"""
    vmul(A, x) -> y

yields `y = A*x`. The default behavior is to call `apply(A,x,false)`.
Method [`vmul!`](@ref) is the in-place version.

"""
vmul(A, x) = apply(A, x, false)

"""
    vmul!(y, A, x) -> y

overwrites `y` with the result of `A*x` and returns `y`. The default behavior
is to call `apply!(1,A,x,false,0,y)`.

!!! note
    This method is intended to be used by algorithms such as the conjugate
    gradient to apply operators. It may be specialized by the caller for its
    needs which is much easier than specializing [`apply!`](@ref) which
    requires to consider the specific values of the multipliers `α` and `β`.

"""
vmul!(y, A, x) = apply!(1, A, x, false, 0, y)

"""
    apply(A, x, scratch=false) -> y

yields the result `y` of applying mapping `A` to the argument `x`.

Optional argument `scratch` indicates whether input argument `x` can be
overwritten by the operation. This may be exploited to avoid allocating
temporary workspace(s). The caller should set `scratch = true` if `x` is not
needed after calling `apply`. If `scratch = true`, then it is possible that `y`
be the same object as `x`; otherwise, `y` is a new object unless applying the
operation yields the same contents as `y` for the result `x` (this is always
true for the identity for instance). Thus, in general, it should not be assumed
that the result of applying a mapping is different from the input.

See also: [`Mapping`](@ref), [`apply!`](@ref), [`vcreate`](@ref).

"""
apply(A::Mapping, x, scratch::Bool=false) =
    apply!(1, A, x, scratch, 0, vcreate(1, A, x, scratch))

*(A::Mapping, x) = apply(A, x)
\(A::Mapping, x) = inv(A)*x

"""
    apply!([α=1,] A::op(M), x, [scratch=false,] [β=0,] y) -> y

overwrites `y` with `α*A⋅x + β*y`. The convention is that the prior contents of
`y` is not used at all if `β = 0` so `y` can be directly used to store the
result even though it is not initialized. The `scratch` optional argument
indicates whether the input `x` is no longer needed by the caller and can thus
be used as a scratch array. Having `scratch = true` or `β = 0` may be exploited
by the specific implementation of the `apply!` method for the mapping type to
avoid allocating temporary workspace(s). `op(M)` denotes a bare mapping type
`M` modified by any of the supported operations `op ∈
(identity,adjoint,inv,inv∘adjoint)`.

The `apply!` method can be seen as a generalization of the `LinearAlgebra.mul!`
method.

The order of arguments can be changed and the same result as above is obtained
with:

    apply!([β=0,] y, [α=1,] A::op(M), x, scratch=false) -> y

The result `y` may have been allocated by:

    y = vcreate(A, x, scratch=false)

Mapping sub-types only need to extend `vcreate` and `apply!` with the specific
signatures:

    vcreate(A::T, x, scratch::Bool=false) -> y
    apply!(α::Number, A::T, x, scratch::Bool, β::Number, y) -> y

for `T` being `M` the type of the mapping, `Adjoint{M}`, `Inverse{L,M}`, and/or
`InverseAdjoint{M}` if any of these operations are supported and with `L =
is_linear(M)` is a boolean indicating whether `M` is the type of a linear
mapping. Of course, the types of arguments `x` and `y` may be specified as
well.

Optionally, the method with signature:

    apply(A::M, x, scratch::Bool=false) -> y

may also be extended to improve the default implementation which is:

    apply(A::Mapping, x, scratch::Bool=false) =
        apply!(1, A, x, scratch, 0, vcreate(A, x, scratch))

See also: [`Mapping`](@ref), [`apply`](@ref), [`vcreate`](@ref).

""" apply!

# Provide fallbacks so that bare mapping types `M` only implement the methods
# having the following signatures:
#
#     apply!(α::Number, A::op(M), x::X, scratch::Bool, β::Number, y::Y)
#
# where `op(M)` denotes the bare mapping type modified by any of the supported
# operations `op ∈ (identity,adjoint,inv,inv∘adjoint)` and, possibly, with
# restrictions on X and Y.
apply!(           A::Mapping, x,            y) = apply!(1, A, x, false, 0, y)
apply!(α::Number, A::Mapping, x,            y) = apply!(α, A, x, false, 0, y)
apply!(           A::Mapping, x, β::Number, y) = apply!(1, A, x, false, β, y)
apply!(α::Number, A::Mapping, x, β::Number, y) = apply!(α, A, x, false, β, y)

apply!(           A::Mapping, x, scratch::Bool,            y) = apply!(1, A, x, scratch, 0, y)
apply!(α::Number, A::Mapping, x, scratch::Bool,            y) = apply!(α, A, x, scratch, 0, y)
apply!(           A::Mapping, x, scratch::Bool, β::Number, y) = apply!(1, A, x, scratch, β, y)

# Change order of arguments.
apply!(           y,            A::Mapping, x, scratch::Bool=false) = apply!(1, A, x, scratch, 0, y)
apply!(           y, α::Number, A::Mapping, x, scratch::Bool=false) = apply!(α, A, x, scratch, 0, y)
apply!(β::Number, y,            A::Mapping, x, scratch::Bool=false) = apply!(1, A, x, scratch, β, y)
apply!(β::Number, y, α::Number, A::Mapping, x, scratch::Bool=false) = apply!(α, A, x, scratch, β, y)

# Extend `LinearAlgebra.mul!` so that `A'*x`, `A*B*C*x`, etc. yield the
# expected result.
mul!(y, A::LinearMapping, x) = apply!(1, A, x, false, 0, y)
mul!(y, A::Mapping, x, α::Number, β::Number) = apply!(α, A, x, false, β, y)

# Implemention of the `apply!(α,A,x,scratch,β,y)` method for a scaled mapping.

apply!(α::Number, A::Scaled, x, scratch::Bool, β::Number, y) =
    apply!(α*multiplier(A), unscaled(A), x, scratch, β, y)

#=
#FIXME:
apply!(α::Number, A::Scaled{<:Any,<:Adjoint}, x, scratch::Bool, β::Number, y) =
    apply!(α*conj(multiplier(A)), unscaled(A), x, scratch, β, y)
apply!(α::Number, A::Scaled{<:Any,<:Inverse}, x, scratch::Bool, β::Number, y) =
    apply!(α*multiplier(A), unscaled(A), x, scratch, β, y)
apply!(α::Number, A::Scaled{<:Any,<:InverseAdjoint}, x, scratch::Bool, β::Number, y) =
    apply!(α*conj(multiplier(A)), unscaled(A), x, scratch, β, y)
=#

"""
    overwritable(scratch, x, y) -> bool

yields whether the result `y` of applying a mapping to `x` with scratch flag
`scratch` can overwritten. Arguments `x` and `y` can be reversed.

"""
overwritable(scratch::Bool, x, y) = (scratch || x !== y) # FIXME: not type-stable

# Implement `apply` for scaled operators to avoid the needs of explicitly
# calling `vcreate` as done by the default implementation of `apply`.  This is
# needed for scaled compositions among others.
function apply(A::Scaled, x, scratch::Bool=false)
    # FIXME: not type-stable
    y = apply(unscaled(A), x, scratch)
    return vscale!(overwritable(scratch, x, y) ? y : vcopy(y), multiplier(A))
end

function apply(A::Adjoint{<:Scaled}, x, scratch::Bool)
    B = parent(A)
    # FIXME: not type-stable
    y = apply(unscaled(B)', x, scratch)
    return vscale!(overwritable(scratch, x, y) ? y : vcopy(y), conj(multiplier(B)))
end

function apply(A::Inverse{<:Any,<:Scaled}, x, scratch::Bool)
    B = parent(A)
    # FIXME: not type-stable
    if A isa LinearMapping
        # Linear case.
        y = apply(inv(unscaled(B)), x, scratch)
        return vscale!((overwritable(scratch, x, y) ? y : vcopy(y)), 1/multiplier(B))
    elseif scratch
        return apply(unscaled(A), vscale!(x, 1/multiplier(B)), true)
    else
        return apply(unscaled(A), vscale(x, 1/multiplier(B)), true)
    end
end

function apply(A::InverseAdjoint{<:Scaled}, x, scratch::Bool)
    B = parent(parent(A))
    # FIXME: not type-stable
    y = apply(inv(unscaled(B)'), x, scratch)
    return vscale!((overwritable(scratch, x, y) ? y : vcopy(y)), 1/conj(multiplier(B)))
end

vcreate(A::Scaled, x, scratch::Bool) = vcreate(unscaled(A), x, scratch)

# FIXME:
#=
# Implemention of the `vcreate(P,A,x,scratch)` and
# `apply!(α,P,A,x,scratch,β,y)` methods for the various decorations of a
# mapping so as to automatically reveal the embedded mapping. for (T1, T2, T3)
in ((:Direct, :Adjoint, :Adjoint), (:Adjoint, :Adjoint, :Direct), (:Inverse,
:Adjoint, :InverseAdjoint), (:InverseAdjoint, :Adjoint, :Inverse), (:Direct,
:Inverse, :Inverse), (:Adjoint, :Inverse, :InverseAdjoint), (:Inverse,
:Inverse, :Direct), (:InverseAdjoint, :Inverse, :Adjoint), (:Direct,
:InverseAdjoint, :InverseAdjoint), (:Adjoint, :InverseAdjoint, :Inverse),
(:Inverse, :InverseAdjoint, :Adjoint), (:InverseAdjoint, :InverseAdjoint,
:Direct)) @eval begin

        vcreate(::Type{$T1}, A::$T2, x, scratch::Bool) =
            vcreate($T3, parent(A), x, scratch)

        apply!(α::Number, ::Type{$T1}, A::$T2, x, scratch::Bool, β::Number, y) =
            apply!(α, $T3, parent(A), x, scratch, β, y)

    end
end
apply!(α::Number, ::Type{$T1}, A::$T2, x, scratch::Bool, β::Number, y) =
    apply!(α, $T3, parent(A), x, scratch, β, y)


# Implementation of the `vcreate(P,A,x,scratch)` and
# `apply!(α,P,A,x,scratch,β,y)` and methods for a sum of mappings.  Note that
# `Sum` instances are warranted to have at least 2 components.

function vcreate(::Type{P}, A::Sum, x,
                 scratch::Bool) where {P<:Union{Direct,Adjoint}}
    # The sum only makes sense if all mappings yields the same kind of result.
    # Hence we just call the vcreate method for the first mapping of the sum.
    vcreate(P, A[1], x, scratch)
end

function apply!(α::Number, P::Type{<:Union{Direct,Adjoint}}, A::Sum{L,N},
                x, scratch::Bool, β::Number, y) where {L,N}
    if α == 0
        # Just scale the destination.
        vscale!(y, β)
    else
        # Apply first mapping with β and then other with β=1.  Scratch flag is
        # always false until last mapping because we must preserve x as there
        # is more than one term.
        apply!(α, P, A[1], x, false, β, y)
        for i in 2:N
            apply!(α, P, A[i], x, (scratch && i == N), 1, y)
        end
    end
    return y
end
=#

const InverseOfSum{L} = Union{Inverse{L,<:Sum{L}},InverseAdjoint{<:Sum{L}}}
vcreate(α::Number, A::InverseOfSum, x, scratch::Bool) = throw_unsupported_inverse_of_sum()
apply(A::InverseOfSum, x, scratch::Bool) = throw_unsupported_inverse_of_sum()
apply!(α::Number, A::InverseOfSum, x, scratch::Bool, β::Number, y) = throw_unsupported_inverse_of_sum()
throw_unsupported_inverse_of_sum() =
    error("automatic dispatching of the inverse of a sum of mappings is not supported")

# Implementation of the `apply!(α,A,x,scratch,β,y)` method for a composition of
# mappings. There is no possible `vcreate(α,A,x,scratch)` method for a
# composition so we directly extend the `apply(A,x,scratch)` method. Note that
# `Composition` instances are warranted to have at least 2 components.
#
# The unrolled code (taking care of allowing as few temporaries as possible and
# for the Direct or InverseAdjoint operation) writes:
#
#     w1 = apply(P, A[N], x, scratch)
#     scratch = overwritable(scratch, x, w1)
#     w2 = apply!(1, P, A[N-1], w1, scratch)
#     scratch = overwritable(scratch, w1, w2)
#     w3 = apply!(1, P, A[N-2], w2, scratch)
#     scratch = overwritable(scratch, w2, w3)
#     ...
#     return apply!(α, P, A[1], wNm1, scratch, β, y)
#
# To break the type barrier, this is done by a recursion.  The recursion is
# just done in the other direction for the Adjoint or Inverse operation.

function vcreate(::Number, A::Union{Composition,
                                    Adjoint{<:Composition},
                                    Inverse{<:Any,<:Composition},
                                    InverseAdjoint{<:Composition}},
                 x, scratch::Bool)
    error("it is not possible to automatically create the output of a composition of mappings")
end

apply!(α::Number, A::Composition, x, scratch::Bool, β::Number, y) =
    apply!(α, identity, *, terms(A), x, scratch, β, y)

apply!(α::Number, A::Adjoint{<:Composition}, x, scratch::Bool, β::Number, y) =
    apply!(α, adjoint, *, terms(parent(A)), x, scratch, β, y)

apply!(α::Number, A::Inverse{<:Any,<:Composition}, x, scratch::Bool, β::Number, y) =
    apply!(α, inv, *, terms(parent(A)), x, scratch, β, y)

apply!(α::Number, A::InverseAdjoint{<:Composition}, x, scratch::Bool, β::Number, y) =
    apply!(α, inv∘adjoint, *, terms(parent(parent(A))), x, scratch, β, y)

function apply!(α::Number, op::Union{typeof(identity),typeof(inv∘adjoint)},
                ::typeof(*), A::NTuple{N,Mapping}, x, scratch::Bool,
                β::Number, y) where {N}
    if α == 0
        # Just scale the destination.
        vscale!(y, β)
    else
        w = apply(op, *, A[2:N], x, scratch)
        scratch = overwritable(scratch, w, x)
        apply!(α, op(A[1]), w, scratch, β, y)
    end
    return y
end

function apply!(α::Number, op::Union{typeof(adjoint),typeof(inv)},
                ::typeof(*), A::NTuple{N,Mapping}, x, scratch::Bool,
                β::Number, y) where {N}
    if α == 0
        # Just scale the destination.
        vscale!(y, β)
    else
        w = apply(op, *, A[1:N-1], x, scratch)
        scratch = overwritable(scratch, w, x)
        apply!(α, op(A[end]), w, scratch, β, y)
    end
    return y
end

apply(A::Composition, x, scratch::Bool) =
    apply(identity, *, terms(A), x, scratch)

apply(A::Adjoint{<:Composition}, x, scratch::Bool) =
    apply(adjoint, *, terms(parent(A)), x, scratch)

apply(A::Inverse{<:Any,<:Composition}, x, scratch::Bool) =
    apply(inv, *, terms(parent(A)), x, scratch)

apply(A::InverseAdjoint{<:Composition}, x, scratch::Bool) =
    apply(inv∘adjoint, *, terms(parent(parent(A))), x, scratch)

function apply(op::Union{typeof(identity),typeof(inv∘adjoint)},
               ::typeof(*), A::NTuple{N,Mapping}, x, scratch::Bool) where {N}
    w = apply(op(A[N]), x, scratch)
    N == 1 && return w
    scratch = overwritable(scratch, w, x)
    apply(op, *, A[1:N-1], w, scratch)
end

function apply(op::Union{typeof(adjoint),typeof(inv)},
               ::typeof(*), A::NTuple{N,Mapping}, x, scratch::Bool) where {N}
    w = apply(op(A[1]), x, scratch)
    N == 1 && return w
    scratch = overwritable(scratch, w, x)
    return apply(op, *, A[2:N], w, scratch)
end

# Default rules to apply a Gram operator, this is a bit similar to a
# composition. Gram matrices are Hermitian by construction which left only 2
# cases to deal with.

apply!(α::Number, A::Adjoint{<:Gram}, x, scratch::Bool, β::Number, y) =
    apply!(α, parent(A), x, scratch, β, y)

apply!(α::Number, A::InverseAdjoint{<:Gram}, x, scratch::Bool, β::Number, y) =
    apply!(α, inv(parent(parent(A))), x, scratch, β, y)

function apply!(α::Number, A::Gram, x, scratch::Bool, β::Number, y)
    if α == 0
        vscale!(y, β)
    else
        B = parent(A) # A ≡ B'*B
        z = apply(B, x, scratch) # z <- B⋅x
        apply!(α, B', z, (z !== x), β, y) # y <- α⋅B'⋅z + β⋅y
    end
    return y
end

function apply!(α::Number, A::Inverse{true,<:Gram}, x, scratch::Bool, β::Number, y)
    if α == 0
        vscale!(y, β)
    else
        B = parent(parent(A)) # A ≡ inv(B'⋅B) = inv(B)*inv(B')
        # Compute α⋅inv(A)⋅x + β⋅y = α⋅inv(B'⋅B)⋅x + β⋅y
        #                          = α⋅inv(B)⋅inv(B')⋅x + β⋅y
        z = apply(inv(B'), x, scratch) # z <- inv(B')⋅x
        apply!(α, inv(B), z, (z !== x), β, y) # y <- α⋅inv(B)⋅z + β⋅y
    end
    return y
end

apply(A::Adjoint{<:Gram}, x, scratch::Bool) = apply(parent(A), x, scratch)

apply(A::InverseAdjoint{<:Gram}, x, scratch::Bool) = apply(α, inv(parent(parent(A))), x, scratch)

function apply(A::Gram, x, scratch::Bool)
    B = parent(A) # A ≡ B'⋅B
    z = apply(B, x, scratch)
    return apply(B', z, (z !== x || scratch))
end

function apply(A::Inverse{true,<:Gram}, x, scratch::Bool)
    B = parent(parent(A)) # A ≡ inv(B'⋅B) = inv(B)*inv(B')
    z = apply(inv(B'), x, scratch)
    return apply(inv(B), z, (z !== x || scratch))
end
