# mappings.jl -
#
# Methods for mappings.
#

for (io, param) in ((:input, :I), (:output, :O))
    # NOTE input_domain_type / output_domain_type
    func = Symbol(io,"_domain_type")
    doc = """
    $(func)(A)

yields the type of the $(io) domain of the mapping (or mapping type) `A`.

"""
    @eval begin
        @doc $doc $func
        $func(A::AbstractMapping) = $func(typeof(A))
        $func(::Type{<:AbstractMapping{I,O}}) where {I,O} = $param
    end

    # NOTE input_eltype / output_eltype
    func = Symbol(io,"_eltype")
    getter = Symbol(io,"_domain_type")
    doc = """
    $(func)(A)

yields the array element type of the elements of the $(io) domain of the
mapping (or mapping type) `A`.

"""
    @eval begin
        @doc $doc $func
        $func(A::AbstractMapping) = $func(typeof(A))
        $func(::Type{A}) where {A} = element_eltype($getter(A))
    end

    # NOTE input_ndims / output_ndims
    func = Symbol(io,"_ndims")
    getter = Symbol(io,"_domain_type")
    doc = """
    $(func)(A)

yields the number of dimensions of the elements of the $(io) domain of the
mapping (or mapping type) `A`.

"""
    @eval begin
        @doc $doc $func
        $func(A::AbstractMapping) = $func(typeof(A))
        $func(::Type{A}) where {A} = element_ndims($getter(A))
    end

    # NOTE input_length / output_length
    func = Symbol(io,"_length")
    getter = Symbol(io,"_domain")
    doc = """
    $(func)(A)

yields the number of entries in the elements of the $(io) domain of the mapping
`A`.

"""
    @eval begin
        @doc $doc $func
        $func(A::AbstractMapping) = element_length($getter(A))
    end

    # NOTE input_size / output_size
    func = Symbol(io,"_size")
    getter = Symbol(io,"_domain")
    doc = """
    $(func)(A)

yields the dimensions of the elements of the $(io) domain of the mapping `A`.

---
    $(func)(A, i)

yields the `i`-th dimension of the elements of the $(io) domain of the mapping
`A`.

"""
    @eval begin
        @doc $doc $func
        $func(A::AbstractMapping) = element_size($getter(A))
        $func(A::AbstractMapping, i::Integer) = element_size($getter(A))[i]
    end

    # NOTE input_axes / output_axes
    func = Symbol(io,"_axes")
    getter = Symbol(io,"_domain")
    doc = """
    $(func)(A)

yields the index ranges of the elements of the $(io) domain of the mapping `A`.

---
    $(func)(A, i)

yields the `i`-th index range of the elements of the $(io) domain of the
mapping `A`.

"""
    @eval begin
        @doc $doc $func
        $func(A::AbstractMapping) = element_axes($getter(A))
        $func(A::AbstractMapping, i::Integer) = element_axes($getter(A))[i]
    end
end

"""
    LazyAlgebra.Scaled(α::Number, B::AbstractMapping)

builds a mapping lazily representing `α*B`.

!!! warning
    Never directly call this constructor but use expression `α*B` instead to
    benefit from automatic simplications and minimal checks.

See also [`LazyAlgebra.multiplier`](@ref) and  [`LazyAlgebra.unscaled`](@ref).

""" Scaled
input_domain(A::Scaled) = input_domain(unscaled(A))
output_domain(A::Scaled) = getfield(A, :out)

"""
    LazyAlgebra.multiplier(A)

yields the multiplier associated with mapping `A`. For example, assuming `A`
is not a scaled mapping and `λ` is a scalar number:

    multiplier(A) -> 1
    multiplier(λ*A) -> λ

See also [`LazyAlgebra.Scaled`](@ref) and  [`LazyAlgebra.unscaled`](@ref).

"""
multiplier(A::Scaled) = getfield(A, :multiplier)
multiplier(::Type{<:Scaled{I,O,T,A}}) where {I,O,T,A} = T
multiplier(A::AbstractMapping) = 1

"""
    LazyAlgebra.unscaled(A)

yields the mapping `A` unscaled. For example, assuming `A`
is not a scaled mapping and `λ` is a scalar number:

    unscaled(A) -> A
    unscaled(λ*A) -> A

See also [`LazyAlgebra.Scaled`](@ref) and  [`LazyAlgebra.multiplier`](@ref).

"""
unscaled(A::Scaled) = getfield(A, :mapping)
unscaled(::Type{<:Scaled{I,O,T,A}}) where {I,O,T,A} = A
unscaled(A::AbstractMapping) = A

"""
    LazyAlgebra.Adjoint(A::AbstractMapping)

builds a mapping lazily representing the adjoint of `A`.

!!! warning
    Never directly call this constructor but use expressions `A'` or
    `adjoint(A)` instead to benefit from automatic simplications and minimal
    checks.

""" Adjoint
Base.parent(A::Adjoint) = getfield(A, :parent)
Base.parent(::Type{<:Adjoint{I,O,A}}) where {I,O,A}  = A
input_domain(A::Adjoint) = output_domain(parent(A))
output_domain(A::Adjoint) = input_domain(parent(A))

"""
    LazyAlgebra.Inverse(A::AbstractMapping)

builds a mapping lazily representing the inverse of `A`.

!!! warning
    Never directly call this constructor but use expression `inv(A)` instead to
    benefit from automatic simplications and minimal checks.

""" Inverse
Base.parent(A::Inverse) = getfield(A, :parent)
Base.parent(::Type{<:Inverse{I,O,A}}) where {I,O,A}  = A
input_domain(A::Inverse) = output_domain(parent(A))
output_domain(A::Inverse) = input_domain(parent(A))

"""
    LazyAlgebra.Sum(inp => out, terms)

yields a mapping lazily representing the sum of mappings in `terms` and
assuming that `inp` and `out` are the respective input and output domains of
this sum of mappings.

!!! warning
    Never directly call this constructor but use expressions like `A + B + ...`
    instead to benefit from automatic simplications and minimal checks.

""" Sum
input_domain(A::Sum) = getfield(A, :inp)
output_domain(A::Sum) = getfield(A, :out)

"""
    LazyAlgebra.Composition(terms)

yields a mapping lazily representing the composition of mappings in `terms`.

!!! warning
    Never directly call this constructor but use expressions like `A*B*...`
    instead to benefit from automatic simplications and minimal checks.

""" Composition
input_domain(A::Composition) = input_domain(last(terms(A)))
output_domain(A::Composition) = output_domain(first(terms(A)))

"""
    LazyAlgebra.terms(A)

yields the vector of terms of the sum or the composition of mappings `A`.

"""
terms(A::Sum) = getfield(A, :terms)
terms(A::Composition) = getfield(A, :terms)

"""
    Null(E)
    Null(E => F)

respectively yield the null mapping over domain `E` and the null mapping from
input domain `E` to output domain `F`. Note that `Null(E)` is equivalent to
`Null(E => E)`.

"""
Null(E::AbstractDomain) = Null(E => E)
input_domain(A::Null) = getfield(A, :inp)
output_domain(A::Null) = getfield(A, :out)
unsafe_apply!(dst, α::Number, A::Null, x, β::Number, y) =
    unsafe_scale!(dst, β, y)

"""
    Identity(io)

yields the identity mapping on domain `io`.

""" Identity
input_domain(A::Identity) = getfield(A, :io)
output_domain(A::Identity) = input_domain(A)
unsafe_apply!(dst, α::Number, A::Identity, x, β::Number, y) =
    unsafe_combine!(dst, α, x, β, y)

"""
    Diag([io::AbstractDomain,] A)

yields a diagonal mapping on domain `io` with diagonal entries specified by
`A`. If `A` is an array, `io = ArrayDomain(A)` is used by default

"""
Diag(A::AbstractArray) = Diag(ArrayDomain(A), A)
LinearAlgebra.diag(A::Diag) = getfield(A, :diag)
input_domain(A::Diag) = getfield(A, :io)
output_domain(A::Diag) = input_domain(A)

function unsafe_apply!(dst, α::Number,
                       A::Diag{<:Any,<:AbstractArray},
                       x, β::Number, y)
    # NOTE `α` is not zero, do not use `y` if `β` is zero
    d = diag(A)
    if isone(α)
        if iszero(β)
            dst .= (d .* x)
        elseif isone(β)
            dst .= (d .* x) .+ y
        else
            β = convert_multiplier(β, eltype(y))
            dst .= (d .* x) .+ β .* y
        end
    else
        α = convert_multiplier(α, eltype(d), eltype(x))
        if iszero(β)
            dst .= α .* (d .* x)
        elseif isone(β)
            dst .= α .* (d .* x) .+ y
        else
            β = convert_multiplier(β, eltype(y))
            dst .= α .* (d .* x) .+ β .* y
        end
    end
    return nothing
end

function unsafe_apply!(dst, α::Number,
                       A::Adjoint{<:Any,<:Any,<:Diag{<:Any,<:AbstractArray}},
                       x, β::Number, y)
    # NOTE `α` is not zero, do not use `y` if `β` is zero
    d = diag(A)
    if isone(α)
        if iszero(β)
            dst .= (conj.(d) .* x)
        elseif isone(β)
            dst .= (conj.(d) .* x) .+ y
        else
            β = convert_multiplier(β, eltype(y))
            dst .= (conj.(d) .* x) .+ β .* y
        end
    else
        α = convert_multiplier(α, eltype(d), eltype(x))
        if iszero(β)
            dst .= α .* (conj.(d) .* x)
        elseif isone(β)
            dst .= α .* (conj.(d) .* x) .+ y
        else
            β = convert_multiplier(β, eltype(y))
            dst .= α .* (conj.(d) .* x) .+ β .* y
        end
    end
    return nothing
end

function unsafe_apply!(dst, α::Number,
                       A::Inverse{<:Any,<:Any,<:Diag{<:Any,<:AbstractArray}},
                       x, β::Number, y)
    # NOTE `α` is not zero, do not use `y` if `β` is zero
    d = diag(A)
    if isone(α)
        if iszero(β)
            dst .= (d .\ x)
        elseif isone(β)
            dst .= (d .\ x) .+ y
        else
            β = convert_multiplier(β, eltype(y))
            dst .= (d .\ x) .+ β .* y
        end
    else
        α = convert_multiplier(α, eltype(d), eltype(x))
        if iszero(β)
            dst .= α .* (d .\ x)
        elseif isone(β)
            dst .= α .* (d .\ x) .+ y
        else
            β = convert_multiplier(β, eltype(y))
            dst .= α .* (d .\ x) .+ β .* y
        end
    end
    return nothing
end

function unsafe_apply!(dst, α::Number,
                       A::InverseAdjoint{<:Any,<:Any,<:Diag{<:Any,<:AbstractArray}},
                       x, β::Number, y)
    # NOTE `α` is not zero, do not use `y` if `β` is zero
    d = diag(A)
    if isone(α)
        if iszero(β)
            dst .= (conj.(d) .\ x)
        elseif isone(β)
            dst .= (conj.(d) .\ x) .+ y
        else
            β = convert_multiplier(β, eltype(y))
            dst .= (conj.(d) .\ x) .+ β .* y
        end
    else
        α = convert_multiplier(α, eltype(d), eltype(x))
        if iszero(β)
            dst .= α .* (conj.(d) .\ x)
        elseif isone(β)
            dst .= α .* (conj.(d) .\ x) .+ y
        else
            β = convert_multiplier(β, eltype(y))
            dst .= α .* (conj.(d) .\ x) .+ β .* y
        end
    end
    return nothing
end

Base.iszero(A::Scaled) = iszero(multiplier(A)) || iszero(unscaled(A))
Base.iszero(A::Null) = true
Base.iszero(A::AbstractMapping) = false

Base.isone(A::Scaled) = isone(multiplier(A)) && isone(unscaled(A))
Base.isone(A::Identity) = true
Base.isone(A::AbstractMapping) = false

"""
     LazyAlgebra.create_output(A::AbstractMapping, x) -> y

yields an object suitable to store the result of `A*x`, if `A` is a linear
mapping, or `A(x)`, if `A` is a non-linear mapping.

"""
function create_output(A::AbstractMapping{<:AbstractArrayDomain,
                                           <:AbstractArrayDomain},
                       x::AbstractArray)
    return similar(x, concrete_float(output_eltype(A)), output_size(A))
end

"""
     LazyAlgebra.apply(A::AbstractMapping, x) -> y

yields `A*x`, if `A` is a linear mapping, or `A(x)`, if `A` is a non-linear
mapping.

This method is not exported because it corresponds to the syntax `A*x`.

The default implementation checks that `x ∈ input_domain(A)` and calls
 [`LazyAlgebra.unsafe_apply`](@ref).

"""
function apply(A::AbstractMapping, x)
    x ∈ input_domain(A) || throw(ArgumentError(
        "argument `x` does not belong to input domain of mapping `A`"))
    return unsafe_apply(A, x)
end

"""
     LazyAlgebra.unsafe_apply(A::AbstractMapping, x) -> y

yields `A*x`, if `A` is a linear mapping, or `A(x)`, if `A` is a non-linear
mapping.

This method shall never be directly called except by
[`LazyAlgebra.apply`](@ref) after checking the arguments. Indeed, the `unsafe_`
prefix means that it is the caller's responsibility to verify the validity of
the arguments so that, the implementation may assume that it is safe to use
`@inbounds` for array arguments.

This method is intended to be extended by other packages for their mappings or
variable types. Other packages may opt to only extend
[`LazyAlgebra.unsafe_apply!`](@ref) instead.

The default implementation calls [`LazyAlgebra.create_ouput`](@ref)
to allocate the result and [`LazyAlgebra.unsafe_apply!`](@ref) to
apply the mapping.

"""
function unsafe_apply(A::AbstractMapping, x)
    dst = create_output(A, x)
    unsafe_apply!(dst, 1, A, x, 0, dst)
    return dst
end

function unsafe_apply!(A::Scaled, x)
    dst = create_output(A, x)
    unsafe_apply!(dst, multiplier(A), unscaled(A), x, 0, dst)
    return dst
end

"""
     LazyAlgebra.apply!(dst, A::AbstractMapping, x) -> dst

overwrites `dst` with `A*x`, if `A` is a linear mapping, or with `A(x)`, if
`A` is a non-linear mapping, and returns `dst`.

"""
apply!(dst, A::AbstractMapping, x) = apply!(dst, 1, A, x)

"""
     LazyAlgebra.apply!(dst, α::Number, A::AbstractMapping, x) -> dst

overwrites `dst` with `α*A*x`, if `A` is a linear mapping, or with `α*A(x)`,
if `A` is a non-linear mapping, and returns `dst`. If `iszero(α)` holds,
expression `A*x` or `A(x)` is not computed so that the contents of `x` is not
considered.

"""
apply!(dst, α::Number, A::AbstractMapping, x) = apply!(dst, α, A, x, 0, dst)

"""
     LazyAlgebra.apply!(dst, α::Number, A::AbstractMapping, x, β::Number, y) -> dst

overwrites `dst` with `α*A*x + β*y`, if `A` is a linear mapping, or with
`α*A(x) + β*y`, if `A` is a non-linear mapping, and returns `dst`. If
`iszero(α)` holds, expression `A*x` or `A(x)` is not computed so that the
contents of `x` is not considered. Similarly, if `iszero(β)` holds, the
contents of `y` is not considered.

This method checks its arguments and then do:

    if iszero(α)
        unsafe_scale!(dst, β, y)
    else
        unsafe_apply!(dst, α, A, x, β, y)
    end
    return dst

Hence, [`LazyAlgebra.unsafe_apply!`](@ref), or
[`LazyAlgebra.unsafe_scale!`](@ref) if `iszero(α)` holds, are the methods that
must be implemented for the specific types of the arguments (in particular, `A`
for `unsafe_apply!`).

"""
function apply!(dst, α::Number, A::AbstractMapping, x, β::Number, y)
    x ∈ input_domain(A) || throw(ArgumentError(
        "argument `x` does not belong to input domain of mapping `A`"))
    dst ∈ output_domain(A) || throw(ArgumentError(
        "argument `dst` does not belong to output domain of mapping `A`"))
    y === dst || y ∈ output_domain(A) || throw(ArgumentError(
        "argument `y` does not belong to output domain of mapping `A`"))
    if iszero(α)
        unsafe_scale!(dst, β, y)
    else
        unsafe_apply!(dst, α, A, x, β, y)
    end
    return dst
end

"""
    LazyAlgebra.unsafe_apply!(dst, α::Number, A::AbstractMapping, x, β::Number, y) -> nothing

overwrites the contents of `dst` with `α*A*x + β*y`, if `A` is a linear
mapping, or with `α*A(x) + β*y`, if `A` is a non-linear mapping, and returns
`nothing`. If `iszero(β)` holds, the contents of `y` is not considered. This
method is never called if `iszero(α)` holds.

This method shall never be directly called except by
[`LazyAlgebra.apply!`](@ref) after checking the arguments and that `iszero(α)`
does not hold. Indeed, the `unsafe_` prefix means that it is the caller's
responsibility to verify these assumptions. Hence, the implementation may
assume that `iszero(α)` does not hold and that it is safe to use `@inbounds`
for array arguments.

This method is intended to be extended by other packages for their mappings or
variable types. Other packages may opt to extend the out of place version
[`LazyAlgebra.unsafe_apply`](@ref). However, if only `unsafe_apply` is
implemented for the types of the arguments, the fallback implementation calls
[`LazyAlgebra.unsafe_apply`](@ref) and then
[`LazyAlgebra.unsafe_combine!`](@ref) which has some overheads.

"""
unsafe_apply!(dst, α::Number, A::AbstractMapping, x, β::Number, y) =
    unsafe_combine!(dst, α, unsafe_apply(A, x), β, y)

unsafe_apply!(dst, α::Number, A::Scaled, x, β::Number, y) =
    unsafe_apply!(dst, α*multiplier(A), unscaled(A), x, β, y)

"""
    LazyAlgebra.zerofill!(A) -> A

overwrites the contents of `A` with zeros and returns `A`.

An implementation of this method that is suitable for ordinary arrays is
provided by `LazyAlgebra`. Other packages may extend this non-exported method
for specific argument types.

"""
zerofill!(A::AbstractArray) = fill!(A, zero(eltype(A)))

"""
    LazyAlgebra.unsafe_copy!(dst, src) -> dst

overwrites the contents of `dst` with that of `src` and returns `dst`.

An implementation of this non-exported method that is suitable for ordinary
arrays is provided by `LazyAlgebra`. Other packages may extend this method for
specific argument types. The `unsafe_` prefix means that it is the caller's
responsibility to check the arguments. In particular, the implementation may
assume that it is safe to use `@inbounds` for indexing array arguments.

"""
function unsafe_copy!(dst::AbstractArray, src::AbstractArray)
    dst === src || copyto!(dst, src)
    return dst
end

"""
    LazyAlgebra.unsafe_scale!(dst, α::Number, x) -> nothing

overwrites destination `dst` with `α*x`. If `iszero(α)` holds, the destination
is zero-filled without considering the contents of `x`. Returned result is
`nothing`.

An implementation of this non-exported method that is suitable for ordinary
arrays is provided by `LazyAlgebra`. Other packages may extend this method for
specific arguments types. The `unsafe_` prefix means that it is the caller's
responsibility to check the arguments. In particular, the implementation may
assume that it is safe to use `@inbounds` for indexing array arguments.

"""
function unsafe_scale!(dst::AbstractArray, α::Number, x::AbstractArray)
    if iszero(α)
        zerofill!(A)
    elseif isone(α)
        unsafe_copy!(dst, x)
    else
        α = promote_multipler(α, eltype(x))
        @inbounds @simd for i in eachindex(dst, x)
            dst[i] = α*x[i]
        end
    end
    return nothing
end

"""
    LazyAlgebra.unsafe_combine!(dst, α::Number, x, β::Number, y) -> nothing

overwrites destination `dst` with `α*x + β*y`. If `iszero(α)` holds, the
contents of `x` is not considered. Similarly, if `iszero(β)` holds, the
contents of `y` is not considered.

An implementation of this non-exported method that is suitable for ordinary
arrays is provided by `LazyAlgebra`. Other packages may extend this method for
specific arguments types. The `unsafe_` prefix means that it is the caller's
responsibility to check the arguments. In particular, the implementation may
assume that it is safe to use `@inbounds` for indexing array arguments.

"""
function unsafe_combine!(dst::AbstractArray,
                         α::Number, x::AbstractArray,
                         β::Number, y::AbstractArray)
    if iszero(α)
        unsafe_scale!(dst, β, y)
    elseif iszero(β)
        unsafe_scale!(dst, α, x)
    else
        # FIXME optimize for other values of α and β
        α = promote_multipler(α, eltype(x))
        β = promote_multipler(β, eltype(y))
        @inbounds @simd for i in eachindex(dst, x, y)
            dst[i] = α*x[i] + β*y[i]
        end
    end
    return nothing
end
