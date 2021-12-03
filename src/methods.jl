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
# Copyright (c) 2017-2021 Éric Thiébaut.
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
    elseif !isone(λ)
        print(io, λ, "⋅")
    end
    show(io, M)
end

function show(io::IO, A::Scaled{<:Sum})
    λ, M = multiplier(A), unscaled(A)
    if λ == -one(λ)
        print(io, "-(")
    elseif !isone(λ)
        print(io, λ, "⋅(")
    end
    show(io, M)
    if !isone(λ)
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
                if !isone(λ)
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
[`LazyAlgebra.DecoratedMapping`](@ref)); otherwise, just returns `A` if it is
not a *decorated* mapping.

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

yields a hash value identifying almost uniquely the unscaled mapping `A`.  This
identifier is used for sorting terms in a sum of mappings.

!!! warning
    For now, the identifier is computed as `objectid(unscaled(A))` and is
    unique with a very high probability.

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

#------------------------------------------------------------------------------
# VCREATE, APPLY AND APPLY!

"""
```julia
vcreate([P,] A, x, scratch=false) -> y
```

yields a new instance `y` suitable for storing the result of applying mapping
`A` to the argument `x`.  Optional parameter `P ∈ Operations` is one of
`Direct` (the default), `Adjoint`, `Inverse` and/or `InverseAdjoint` and can be
used to specify how `A` is to be applied as explained in the documentation of
the [`apply`](@ref) method.

Optional argument `scratch` indicates whether input argument `x` can be
overwritten by the operation and thus used to store the result.  This may be
exploited by some mappings (which are able to operate *in-place*) to avoid
allocating a new object for the result `y`.

The caller should set `scratch = true` if `x` is not needed after calling
`apply`.  If `scratch = true`, then it is possible that `y` be the same object
as `x`; otherwise, `y` is a new object unless applying the operation yields the
same contents as `y` for the result `x` (this is always true for the identity
for instance).  Thus, in general, it should not be assumed that the returned
`y` is different from the input `x`.

The method `vcreate(::Type{P}, A, x)` should be implemented by linear mappings
for any supported operations `P` and argument type for `x`.  The result
returned by `vcreate` should be of predictible type to ensure *type-stability*.
Checking the validity (*e.g.* the size) of argument `x` in `vcreate` may be
skipped because this argument will be eventually checked by the `apply!`
method.

See also: [`Mapping`](@ref), [`apply`](@ref).

"""
vcreate(A::Mapping, x, scratch::Bool=false) = vcreate(Direct, A, x, scratch)
vcreate(::Type{P}, A::Mapping, x) where {P<:Operations} =
    vcreate(P, A, x, false)

"""
    vmul(A, x) -> y

yields `y = A*x`.  The default behavior is to call `apply(Direct,A,x,false)`.
Method [`vmul!`](@ref) is the in-place version.

"""
vmul(A, x) = apply(Direct, A, x, false)

"""
    vmul!(y, A, x) -> y

overwrites `y` with the result of `A*x` and returns `y`.  The default behavior
is to call `apply!(1,Direct,A,x,false,0,y)`.

!!! note

    This method is intended to be used by algorithms such as the conjugate
    gradient to apply operators.  It may be specialized by the caller for its
    needs which is much easier than specializing [`apply!`](@ref) which
    requires to consider the specific values of the multipliers `α` and `β`.

"""
vmul!(y, A, x) = apply!(1, Direct, A, x, false, 0, y)

"""
```julia
apply([P,] A, x, scratch=false) -> y
```

yields the result `y` of applying mapping `A` to the argument `x`.
Optional parameter `P` can be used to specify how `A` is to be applied:

* `Direct` (the default) to apply `A` and yield `y = A⋅x`;
* `Adjoint` to apply the adjoint of `A` and yield `y = A'⋅x`;
* `Inverse` to apply the inverse of `A` and yield `y = A\\x`;
* `InverseAdjoint` or `AdjointInverse` to apply the inverse of `A'` and
  yield `y = A'\\x`.

Not all operations may be implemented by the different types of mappings and
`Adjoint` and `InverseAdjoint` may only be applicable for linear mappings.

Optional argument `scratch` indicates whether input argument `x` can be
overwritten by the operation.  This may be exploited to avoid allocating
temporary workspace(s).  The caller should set `scratch = true` if `x` is not
needed after calling `apply`.  If `scratch = true`, then it is possible that
`y` be the same object as `x`; otherwise, `y` is a new object unless applying
the operation yields the same contents as `y` for the result `x` (this is
always true for the identity for instance).  Thus, in general, it should not be
assumed that the result of applying a mapping is different from the input.

Julia methods are provided so that `apply(A', x)` automatically calls
`apply(Adjoint, A, x)` so the shorter syntax may be used without impacting
performances.

See also: [`Mapping`](@ref), [`apply!`](@ref), [`vcreate`](@ref).

"""
apply(A::Mapping, x, scratch::Bool=false) = apply(Direct, A, x, scratch)
apply(P::Type{<:Operations}, A::Mapping, x, scratch::Bool=false) =
    apply!(1, P, A, x, scratch, 0, vcreate(P, A, x, scratch))

*(A::Mapping, x) = apply(Direct, A, x, false)
\(A::Mapping, x) = apply(Inverse, A, x, false)

"""
    apply!([α=1,] [P=Direct,] A::Mapping, x, [scratch=false,] [β=0,] y) -> y

overwrites `y` with `α*P(A)⋅x + β*y` where `P ∈ Operations` can be `Direct`,
`Adjoint`, `Inverse` and/or `InverseAdjoint` to indicate which variant of the
mapping `A` to apply.  The convention is that the prior contents of `y` is not
used at all if `β = 0` so `y` can be directly used to store the result even
though it is not initialized.  The `scratch` optional argument indicates
whether the input `x` is no longer needed by the caller and can thus be used as
a scratch array.  Having `scratch = true` or `β = 0` may be exploited by the
specific implementation of the `apply!` method for the mapping type to avoid
allocating temporary workspace(s).

The `apply!` method can be seen as a generalization of the `LinearAlgebra.mul!`
method.

The order of arguments can be changed and the same result as above is obtained
with:

    apply!([β=0,] y, [α=1,] [P=Direct,] A::Mapping, x, scratch=false) -> y

The result `y` may have been allocated by:

    y = vcreate([P=Direct,] A, x, scratch=false)

Mapping sub-types only need to extend `vcreate` and `apply!` with the specific
signatures:

    vcreate(::Type{P}, A::M, x, scratch::Bool=false) -> y
    apply!(α::Number, ::Type{P}, A::M, x, scratch::Bool, β::Number, y) -> y

for any supported operation `P` and where `M` is the type of the mapping.  Of
course, the types of arguments `x` and `y` may be specified as well.

Optionally, the method with signature:

    apply(::Type{P}, A::M, x, scratch::Bool=false) -> y

may also be extended to improve the default implementation which is:

    apply(P::Type{<:Operations}, A::Mapping, x, scratch::Bool=false) =
        apply!(1, P, A, x, scratch, 0, vcreate(P, A, x, scratch))

See also: [`Mapping`](@ref), [`apply`](@ref), [`vcreate`](@ref).

""" apply!

# Provide fallbacks so that `Direct` is the default operation and only the
# method with signature:
#
#     apply!(α::Number, ::Type{P}, A::MappingType, x::X, scratch::Bool,
#            β::Number, y::Y) where {P<:Operations,X,Y}
#
# has to be implemented (possibly with restrictions on X and Y) by subtypes of
# Mapping so we provide the necessary mechanism to dispatch derived methods.
apply!(A::Mapping, x, y) =
    apply!(1, Direct, A, x, false, 0, y)
apply!(α::Number, A::Mapping, x, y) =
    apply!(α, Direct, A, x, false, 0, y)
apply!(A::Mapping, x, β::Number, y) =
    apply!(1, Direct, A, x, false, β, y)
apply!(α::Number, A::Mapping, x, β::Number, y) =
    apply!(α, Direct, A, x, false, β, y)

apply!(P::Type{<:Operations}, A::Mapping, x, y) =
    apply!(1, P, A, x, false, 0, y)
apply!(α::Number, P::Type{<:Operations}, A::Mapping, x, y) =
    apply!(α, P, A, x, false, 0, y)
apply!(P::Type{<:Operations}, A::Mapping, x, β::Number, y) =
    apply!(1, P, A, x, false, β, y)
apply!(α::Number, P::Type{<:Operations}, A::Mapping, x, β::Number, y) =
    apply!(α, P, A, x, false, β, y)

apply!(A::Mapping, x, scratch::Bool, y) =
    apply!(1, Direct, A, x, scratch, 0, y)
apply!(α::Number, A::Mapping, x, scratch::Bool, y) =
    apply!(α, Direct, A, x, scratch, 0, y)
apply!(A::Mapping, x, scratch::Bool, β::Number, y) =
    apply!(1, Direct, A, x, scratch, β, y)

apply!(P::Type{<:Operations}, A::Mapping, x, scratch::Bool, y) =
    apply!(1, P, A, x, scratch, 0, y)
apply!(α::Number, P::Type{<:Operations}, A::Mapping, x, scratch::Bool, y) =
    apply!(α, P, A, x, scratch, 0, y)
apply!(P::Type{<:Operations}, A::Mapping, x, scratch::Bool, β::Number, y) =
    apply!(1, P, A, x, scratch, β, y)

# Change order of arguments.
apply!(y, A::Mapping, x, scratch::Bool=false) =
    apply!(1, Direct, A, x, scratch, 0, y)
apply!(y, P::Type{<:Operations}, A::Mapping, x, scratch::Bool=false) =
    apply!(1, P, A, x, scratch, 0, y)
apply!(y, α::Number, A::Mapping, x, scratch::Bool=false) =
    apply!(α, Direct, A, x, scratch, 0, y)
apply!(y, α::Number, P::Type{<:Operations}, A::Mapping, x, scratch::Bool=false) =
    apply!(α, P, A, x, scratch, 0, y)
apply!(β::Number, y, A::Mapping, x, scratch::Bool=false) =
    apply!(1, Direct, A, x, scratch, β, y)
apply!(β::Number, y, P::Type{<:Operations}, A::Mapping, x, scratch::Bool=false) =
    apply!(1, P, A, x, scratch, β, y)
apply!(β::Number, y, α::Number, A::Mapping, x, scratch::Bool=false) =
    apply!(α, Direct, A, x, scratch, β, y)
apply!(β::Number, y, α::Number, P::Type{<:Operations}, A::Mapping, x, scratch::Bool=false) =
    apply!(α, P, A, x, scratch, β, y)

# Extend `LinearAlgebra.mul!` so that `A'*x`, `A*B*C*x`, etc. yield the
# expected result.  FIXME: This should be restricted to linear mappings but
# this is not possible without overheads.
mul!(y, A::Mapping, x) = apply!(1, Direct, A, x, false, 0, y)
mul!(y, A::Mapping, x, α::Number, β::Number) =
    apply!(α, Direct, A, x, false, β, y)

# Implemention of the `apply!(α,P,A,x,scratch,β,y)` and
# `vcreate(P,A,x,scratch)` methods for a scaled mapping.
for (P, expr) in ((:Direct, :(α*multiplier(A))),
                  (:Adjoint, :(α*conj(multiplier(A)))),
                  (:Inverse, :(α/multiplier(A))),
                  (:InverseAdjoint, :(α/conj(multiplier(A)))))
    @eval begin

        apply!(α::Number, ::Type{$P}, A::Scaled, x, scratch::Bool, β::Number, y) =
            apply!($expr, $P, unscaled(A), x, scratch, β, y)

    end
end

"""
    overwritable(scratch, x, y) -> bool

yields whether the result `y` of applying a mapping to `x` with scratch flag
`scratch` can overwritten.  Arguments `x` and `y` can be reversed.

"""
overwritable(scratch::Bool, x, y) = (scratch || x !== y)

# Implement `apply` for scaled operators to avoid the needs of explicitly
# calling `vcreate` as done by the default implementation of `apply`.  This is
# needed for scaled compositions among others.
function apply(::Type{Direct}, A::Scaled, x, scratch::Bool)
    y = apply(Direct, unscaled(A), x, scratch)
    vscale!((overwritable(scratch, x, y) ? y : vcopy(y)), multiplier(A))
end

function apply(::Type{Adjoint}, A::Scaled, x, scratch::Bool)
    y = apply(Direct, unscaled(A), x, scratch)
    vscale!((overwritable(scratch, x, y) ? y : vcopy(y)), conj(multiplier(A)))
end

function apply(::Type{Inverse}, A::Scaled, x, scratch::Bool)
    y = apply(Direct, unscaled(A), x, scratch)
    vscale!((overwritable(scratch, x, y) ? y : vcopy(y)), 1/multiplier(A))
end

function apply(::Type{InverseAdjoint}, A::Scaled, x, scratch::Bool)
    y = apply(Direct, unscaled(A), x, scratch)
    vscale!((overwritable(scratch, x, y) ? y : vcopy(y)), 1/conj(multiplier(A)))
end

vcreate(P::Type{<:Operations}, A::Scaled, x, scratch::Bool) =
    vcreate(P, unscaled(A), x, scratch)

# Implemention of the `vcreate(P,A,x,scratch)` and
# `apply!(α,P,A,x,scratch,β,y)` methods for the various decorations of a
# mapping so as to automatically unveil the embedded mapping.
for (T1, T2, T3) in ((:Direct,         :Adjoint,        :Adjoint),
                     (:Adjoint,        :Adjoint,        :Direct),
                     (:Inverse,        :Adjoint,        :InverseAdjoint),
                     (:InverseAdjoint, :Adjoint,        :Inverse),
                     (:Direct,         :Inverse,        :Inverse),
                     (:Adjoint,        :Inverse,        :InverseAdjoint),
                     (:Inverse,        :Inverse,        :Direct),
                     (:InverseAdjoint, :Inverse,        :Adjoint),
                     (:Direct,         :InverseAdjoint, :InverseAdjoint),
                     (:Adjoint,        :InverseAdjoint, :Inverse),
                     (:Inverse,        :InverseAdjoint, :Adjoint),
                     (:InverseAdjoint, :InverseAdjoint, :Direct))
    @eval begin

        vcreate(::Type{$T1}, A::$T2, x, scratch::Bool) =
            vcreate($T3, unveil(A), x, scratch)

        apply!(α::Number, ::Type{$T1}, A::$T2, x, scratch::Bool, β::Number, y) =
            apply!(α, $T3, unveil(A), x, scratch, β, y)

    end
end

# Implementation of the `vcreate(P,A,x,scratch)` and
# `apply!(α,P,A,x,scratch,β,y)` and methods for a sum of mappings.  Note that
# `Sum` instances are warranted to have at least 2 components.

function vcreate(::Type{P}, A::Sum, x,
                 scratch::Bool) where {P<:Union{Direct,Adjoint}}
    # The sum only makes sense if all mappings yields the same kind of result.
    # Hence we just call the vcreate method for the first mapping of the sum.
    vcreate(P, A[1], x, scratch)
end

function apply!(α::Number, P::Type{<:Union{Direct,Adjoint}}, A::Sum{N},
                x, scratch::Bool, β::Number, y) where {N}
    if iszero(α)
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

vcreate(::Type{<:Union{Inverse,InverseAdjoint}}, A::Sum, x, scratch::Bool) =
    throw_unsupported_inverse_of_sum()

apply(::Type{<:Union{Inverse,InverseAdjoint}}, A::Sum, x, scratch::Bool) =
    throw_unsupported_inverse_of_sum()

function apply!(α::Number, ::Type{<:Union{Inverse,InverseAdjoint}}, A::Sum,
                x, scratch::Bool, β::Number, y)
    throw_unsupported_inverse_of_sum()
end

throw_unsupported_inverse_of_sum() =
    error("automatic dispatching of the inverse of a sum of mappings is not supported")

# Implementation of the `apply!(α,P,A,x,scratch,β,y)` method for a composition
# of mappings.  There is no possible `vcreate(P,A,x,scratch)` method for a
# composition so we directly extend the `apply(P,A,x,scratch)` method.  Note
# that `Composition` instances are warranted to have at least 2 components.
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

function vcreate(::Type{<:Operations},
                 A::Composition{N}, x, scratch::Bool) where {N}
    error("it is not possible to create the output of a composition of mappings")
end

function apply!(α::Number, ::Type{P}, A::Composition{N}, x, scratch::Bool,
                β::Number, y) where {N,P<:Union{Direct,InverseAdjoint}}
    if iszero(α)
        # Just scale the destination.
        vscale!(y, β)
    else
        ops = terms(A)
        w = apply(P, *, ops[2:N], x, scratch)
        scratch = overwritable(scratch, w, x)
        apply!(α, P, ops[1], w, scratch, β, y)
    end
    return y
end

function apply(::Type{P}, A::Composition{N}, x,
               scratch::Bool) where {N,P<:Union{Direct,InverseAdjoint}}
    apply(P, *, terms(A), x, scratch)
end

function apply(::Type{P}, ::typeof(*), ops::NTuple{N,Mapping}, x,
               scratch::Bool) where {N,P<:Union{Direct,InverseAdjoint}}
    w = apply(P, ops[N], x, scratch)
    N == 1 && return w
    scratch = overwritable(scratch, w, x)
    apply(P, *, ops[1:N-1], w, scratch)
end

function apply!(α::Number, ::Type{P}, A::Composition{N}, x, scratch::Bool,
                β::Number, y) where {N,P<:Union{Adjoint,Inverse}}
    if iszero(α)
        # Just scale the destination.
        vscale!(y, β)
    else
        ops = terms(A)
        w = apply(P, *, ops[1:N-1], x, scratch)
        scratch = overwritable(scratch, w, x)
        apply!(α, P, ops[N], w, scratch, β, y)
    end
    return y
end

function apply(::Type{P}, A::Composition{N}, x,
               scratch::Bool) where {N,P<:Union{Adjoint,Inverse}}
    apply(P, *, terms(A), x, scratch)
end

function apply(::Type{P}, ::typeof(*), ops::NTuple{N,Mapping}, x,
               scratch::Bool) where {N,P<:Union{Adjoint,Inverse}}
    w = apply(P, ops[1], x, scratch)
    N == 1 && return w
    scratch = overwritable(scratch, w, x)
    apply(P, *, ops[2:N], w, scratch)
end

# Default rules to apply a Gram operator.  Gram matrices are Hermitian by
# construction which left only 2 cases to deal with.

apply!(α::Number, ::Type{Adjoint}, A::Gram, x, scratch::Bool, β::Number, y) =
    apply!(α, Direct, A, x, scratch, β, y)

apply!(α::Number, ::Type{InverseAdjoint}, A::Gram, x, scratch::Bool, β::Number, y) =
    apply!(α, Inverse, A, x, scratch, β, y)

function apply!(α::Number, ::Type{Direct}, A::Gram, x, scratch::Bool, β::Number, y)
    if iszero(α)
        vscale!(y, β)
    else
        B = unveil(A) # A ≡ B'*B
        z = apply(Direct, B, x, scratch) # z <- B⋅x
        apply!(α, Adjoint, B, z, (z !== x), β, y) # y <- α⋅B'⋅z + β⋅y
    end
    return y
end

function apply!(α::Number, ::Type{Inverse}, A::Gram, x, scratch::Bool, β::Number, y)
    if iszero(α)
        vscale!(y, β)
    else
        B = unveil(A) # A ≡ B'⋅B
        # Compute α⋅inv(A)⋅x + β⋅y = α⋅inv(B'⋅B)⋅x + β⋅y
        #                          = α⋅inv(B)⋅inv(B')⋅x + β⋅y
        z = apply(InverseAdjoint, B, x, scratch) # z <- inv(B')⋅x
        apply!(α, Inverse, B, z, (z !== x), β, y) # y <- α⋅inv(B)⋅z + β⋅y
    end
    return y
end

# A Gram operator is self-adjoint by construction and yields result of same
# kind as input.
vcreate(::Type{<:Operations}, ::Gram, x, scratch::Bool) = vcreate(x)
