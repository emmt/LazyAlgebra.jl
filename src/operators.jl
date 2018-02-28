#
# operators.jl -
#
# Methods for linear operators.
#
#-------------------------------------------------------------------------------
#
# This file is part of the MockAlgebra package released under the MIT "Expat"
# license.
#
# Copyright (c) 2017-2018 Éric Thiébaut.
#

# Note that we extend the meaning of * and \ only for linear operators (not for
# arrays using the generalized matrix-vector dot product).
(*)(A::LinearOperator, x) = apply(Direct, A, x)
(\)(A::LinearOperator, x) = apply(Inverse, A, x)

Inverse(A::Inverse) = A.op
Inverse(A::Adjoint{T}) where {T<:LinearOperator} = InverseAdjoint{T}(A.op)
Inverse(A::InverseAdjoint{T}) where {T<:LinearOperator} = Adjoint{T}(A.op)

Adjoint(A::Adjoint) = A.op
Adjoint(A::Inverse{T}) where {T<:LinearOperator} = InverseAdjoint{T}(A.op)
Adjoint(A::InverseAdjoint{T}) where {T<:LinearOperator} = Inverse{T}(A.op)

InverseAdjoint(A::InverseAdjoint) = A.op
InverseAdjoint(A::Adjoint{T}) where {T<:LinearOperator} = Inverse{T}(A.op)
InverseAdjoint(A::Inverse{T}) where {T<:LinearOperator} = Adjoint{T}(A.op)

# Manage to have A' and inv(A) adds the correct decoration:
Base.ctranspose(A::LinearOperator) = Adjoint(A)
inv(A::LinearOperator) = Inverse(A)

# Automatically undecorate operator for common methods.
for (T1, T2, T3) in ((:Direct,         :Adjoint,        :Adjoint),
                     (:Direct,         :Inverse,        :Inverse),
                     (:Direct,         :InverseAdjoint, :InverseAdjoint),
                     (:Adjoint,        :Adjoint,        :Direct),
                     (:Adjoint,        :Inverse,        :InverseAdjoint),
                     (:Adjoint,        :InverseAdjoint, :Inverse),
                     (:Inverse,        :Adjoint,        :InverseAdjoint),
                     (:Inverse,        :Inverse,        :Direct),
                     (:Inverse,        :InverseAdjoint, :Adjoint),
                     (:InverseAdjoint, :Adjoint,        :Inverse),
                     (:InverseAdjoint, :Inverse,        :Adjoint),
                     (:InverseAdjoint, :InverseAdjoint, :Direct))
    @eval begin

        apply!(y, ::Type{$T1}, A::$T2, x) =
            apply!(y, $T3, A.op, x)

        apply(::Type{$T1}, A::$T2, x) =
            apply($T3, A.op, x)

        vcreate(::Type{$T1}, A::$T2, x) =
            vcreate($T3, A.op, x)

        is_applicable_in_place(::Type{$T1}, A::$T2, x) =
            is_applicable_in_place($T3, A.op, x)
    end
end

# Specialize methods for self-adjoint operators so that only `Direct` and
# `Inverse` operations need to be implemented.
Adjoint(A::SelfAdjointOperator) = A
InverseAdjoint(A::SelfAdjointOperator) = Inverse(A)
for (T1, T2) in ((:Adjoint, :Direct),
                 (:InverseAdjoint, :Inverse))
    @eval begin

        apply!(y, ::Type{$T1}, A::SelfAdjointOperator, x) =
            apply!(y, $T2, A, x)

        apply(::Type{$T1}, A::SelfAdjointOperator, x) =
            apply($T2, A, x)

        vcreate(::Type{$T1}, A::SelfAdjointOperator, x) =
            vcreate($T2, A, x)

        is_applicable_in_place(::Type{$T1}, A::SelfAdjointOperator, x) =
            is_applicable_in_place($T2, A, x)

    end
end

# Basic methods:
function input_type end
function input_eltype end
function input_size end
function input_ndims end
function output_type end
function output_eltype end
function output_size end
function output_ndims end
doc"""
The calls:

    input_type(A)
    output_type(A)

yield the (preferred) types of the input and output arguments of the operator
`A`.  If `A` operates on Julia arrays, the element type, list of dimensions,
`i`-th dimension and number of dimensions for the input and output are given
by:

    input_eltype(A)                output_eltype(A)
    input_size([Op,] A)            output_size([Op,] A)
    input_size([Op,] A, i)         output_size([Op,] A, i)
    input_ndims([Op,] A)           output_ndims([Op,] A)

Only `input_size(A)` and `output_size(A)` have to be implemented.

Also see: [`vcreate`](@ref).

""" input_type
@doc @doc(input_type) input_eltype
@doc @doc(input_type) input_size
@doc @doc(input_type) input_ndims
@doc @doc(input_type) output_type
@doc @doc(input_type) output_eltype
@doc @doc(input_type) output_size
@doc @doc(input_type) output_ndims

# Provide methods for the different operations.
for sfx in (:size, :eltype, :ndims, :type),
    pfx in (:output, :input),
    Op in (Direct, Adjoint, Inverse, InverseAdjoint)

    fn1 = Symbol(pfx, "_", sfx)
    fn2 = Symbol(Op == Adjoint || Op == Inverse ?
                 (pfx == :output ? :input : :output) : pfx, "_", sfx)

    #println("$fn1($Op) -> $fn2")

    @eval begin

        if $(Op != Direct)
            $fn1(A::$Op{<:LinearOperator}) = $fn2(A.op)
        end

        $fn1(::Type{$Op}, A::LinearOperator) = $fn2(A)

        if $(sfx == :size)
            if $(Op != Direct)
                $fn1(A::$Op{<:LinearOperator}, dim...) =
                    $fn2(A.op, dim...)
            end
            $fn1(::Type{$Op}, A::LinearOperator, dim...) =
                $fn2(A, dim...)
        end
    end
end

# Provide default methods for `$(sfx)_size(A, dim...)` and `$(sfx)_ndims(A)`.
for pfx in (:input, :output)
    pfx_size = Symbol(pfx, "_size")
    pfx_ndims = Symbol(pfx, "_ndims")
    @eval begin

        $pfx_ndims(A::LinearOperator) = length($pfx_size(A))

        $pfx_size(A::LinearOperator, dim) = $pfx_size(A)[dim]

        function $pfx_size(A::LinearOperator, dim...)
            dims = $pfx_size(A)
            ntuple(i -> dims[dim[i]], length(dim))
        end

    end
end

"""
```julia
apply([Op,] A, x) -> y
```

yields the result `y` of applying operator `A` to the argument `x`.
Optional parameter `Op` can be used to specify how `A` is to be applied:

* `Direct` to apply `A`, this is the default operation;
* `Adjoint` to apply the adjoint of `A`, that is `A'`;
* `Inverse` to apply the inverse of `A`;
* `InverseAdjoint` or `AdjointInverse` to apply the inverse of `A'`.

Note that not all operations may be implemented by the different types of
linear operators.

Julia methods are provided so that `apply(A', x)` automatically calls
`apply(Adjoint, A, x)` so the shorter syntax may be used without any
performances impact.

See also: [`LinearOperator`](@ref), [`apply!`](@ref), [`vcreate`](@ref).

"""
apply(A::LinearOperator, x) = apply(Direct, A, x)

apply(::Type{Op}, A::LinearOperator, x) where {Op <: Operations} =
    apply!(vcreate(Op, A, x), Op, A, x)

"""
```julia
apply!(y, [Op,] A, x) -> y
```

stores in `y` the result of applying operator `A` to the argument `x`.  The
arguments other than `y` have the same meaning as for the [`apply`](@ref)
method.  The result may have been allocated by:

```julia
y = vcreate([Op,] A, x)
```

The method `apply!(y, Op, A, x)` should be implemented by linear operators for
any supported operations `Op`.

See also: [`LinearOperator`](@ref), [`apply`](@ref), [`vcreate`](@ref).

"""
apply!(y, A::LinearOperator, x) = apply!(y, Direct, A, x)


"""
```julia
vcreate([Op,] A, x) -> y
```

yields a new instance `y` suitable for storing the result of applying operator
`A` to the argument `x`.  Optional parameter `Op` (`Direct` by default) can be
used to specify how `A` is to be applied as explained in the documentation of
the [`apply`](@ref) method.

The method `vcreate(Op, A, x)` should be implemented by linear operators
for any supported operations `Op`.

See also: [`LinearOperator`](@ref), [`apply`](@ref).

"""
vcreate(A::LinearOperator, x) = vcreate(Direct, A, x)

"""
```julia
is_applicable_in_place([Op,] A, x)
```

yields whether operator `A` is applicable *in-place* for performing operation
`Op` with argument `x`, that is with the result stored into the argument `x`.
This can be used to spare allocating ressources.

See also: [`LinearOperator`](@ref), [`apply!`](@ref).

"""
is_applicable_in_place(::Type{<:Operations}, A::LinearOperator, x) = false
is_applicable_in_place(A::LinearOperator, x) =
    is_applicable_in_place(Direct, A, x)

#------------------------------------------------------------------------------
# IDENTITY

"""
```julia
Identity()
```

yields the identity linear operator.  Beware that the purpose of this operator
is to be as efficient as possible, hence the result of applying this operator
may be the same as the input argument.

"""
struct Identity <: SelfAdjointOperator; end

is_applicable_in_place(::Type{<:Operations}, ::Identity, x) = true

Base.inv(A::Identity) = A

apply(::Type{<:Operations}, ::Identity, x) = x

apply!(y, ::Type{<:Operations}, ::Identity, x) = vcopy!(y, x)

vcreate(::Type{<:Operations}, ::Identity, x) = similar(x)

#------------------------------------------------------------------------------
# NON-UNIFORM SCALING

"""
```julia
NonuniformScaling(A)
```

creates a nonuniform scaling linear operator whose effects is to apply
elementwsie multiplication of its argument by the scaling factors `A`.

"""
struct NonuniformScaling{T} <: LinearOperator
    scl::T
end

is_applicable_in_place(::Type{<:Operations}, ::NonuniformScaling, x) = true

Adjoint(A::NonuniformScaling{<:AbstractArray{<:AbstractFloat}}) = A
InverseAdjoint(A::NonuniformScaling{<:AbstractArray{<:AbstractFloat}}) =
    Inverse(A)

function Base.inv(A::NonuniformScaling{<:AbstractArray{T,N}}
                  ) where {T<:AbstractFloat, N}
    q = A.scl
    r = similar(q)
    @inbounds @simd for i in eachindex(q, r)
        r[i] = one(T)/q[i]
    end
    return NonuniformScaling(r)
end

function apply!(y::AbstractArray{<:AbstractFloat,N},
                ::Type{<:Union{Direct,Adjoint}},
                A::NonuniformScaling{<:AbstractArray{<:AbstractFloat,N}},
                x::AbstractArray{<:AbstractFloat,N}) where {N}
    @assert indices(y) == indices(A.scl)
    @assert indices(x) == indices(A.scl)
    @inbounds @simd for i in eachindex(x, A.scl, x)
        y[i] = A.scl[i]*x[i]
    end
    return y
end

function apply!(y::AbstractArray{<:AbstractFloat,N},
                ::Type{<:Union{Inverse,InverseAdjoint}},
                A::NonuniformScaling{<:AbstractArray{<:AbstractFloat,N}},
                x::AbstractArray{<:AbstractFloat,N}) where {N}
    @assert indices(y) == indices(A.scl)
    @assert indices(x) == indices(A.scl)
    @inbounds @simd for i in eachindex(x, A.scl, x)
        y[i] = x[i]/A.scl[i]
    end
    return y
end

function vcreate(::Type{<:Operations},
                   A::NonuniformScaling{<:AbstractArray{Ta,N}},
                   x::AbstractArray{Tx,N}
                   ) where {
                       Ta <: AbstractFloat,
                       Tx <: AbstractFloat,
                       N
                   }
    @assert indices(x) == indices(A.scl)
    T = promote_type(Ta, Tx)
    return similar(Array{T}, indices(A.scl))
end

#------------------------------------------------------------------------------
# GENERALIZED MATRIX

"""
```julia
GeneralMatrix(A)
```

creates a linear operator given a multi-dimensional array `A` whose interest is
to generalize the definition of the matrix-vector product without calling
`reshape` to change the dimensions.

For instance, assuming that `G = GeneralMatrix(A)` with `A` a regular array,
then `y = G*x` requires that the dimensions of `x` match the trailing
dimensions of `A` and yields a result `y` whose dimensions are the remaining
leading dimensions of `A`, such that `indices(A) = (indices(y)...,
indices(x)...)`.  Applying the adjoint of `G` as in `y = G'*x` requires that
the dimensions of `x` match the leading dimension of `A` and yields a result
`y` whose dimensions are the remaining trailing dimensions of `A`, such that
`indices(A) = (indices(x)..., indices(y)...)`.

See also: [`reshape`](@ref).

"""
struct GeneralMatrix{T<:AbstractArray} <: LinearOperator
    arr::T
end

# Make a GeneralMatrix behaves like an ordinary array.
Base.eltype(A::GeneralMatrix) = eltype(A.arr)
Base.length(A::GeneralMatrix) = length(A.arr)
Base.ndims(A::GeneralMatrix) = ndims(A.arr)
Base.indices(A::GeneralMatrix) = indices(A.arr)
Base.size(A::GeneralMatrix) = size(A.arr)
Base.size(A::GeneralMatrix, inds...) = size(A.arr, inds...)
Base.getindex(A::GeneralMatrix, inds...) = getindex(A.arr, inds...)
Base.setindex!(A::GeneralMatrix, x, inds...) = setindex!(A.arr, x, inds...)
Base.stride(A::GeneralMatrix, k) = stride(A.arr, k)
Base.strides(A::GeneralMatrix) = strides(A.arr)
Base.eachindex(A::GeneralMatrix) = eachindex(A.arr)

function apply!(y::AbstractArray{<:AbstractFloat},
                ::Type{Op},
                A::GeneralMatrix{<:AbstractArray{<:AbstractFloat}},
                x::AbstractArray{<:AbstractFloat}) where {Op<:Operations}
    return apply!(y, Op, A.arr, x)
end

function vcreate(::Type{Op},
                   A::GeneralMatrix{<:AbstractArray{<:AbstractFloat}},
                   x::AbstractArray{<:AbstractFloat}) where {Op<:Operations}
    return vcreate(Op, A.arr, x)
end

#------------------------------------------------------------------------------
# HALF HESSIAN

"""

`HalfHessian(A)` is a container to be interpreted as the linear operator
representing the second derivatives (times 1/2) of some objective function at
some point both represented by `A` (which can be anything).  Given `H =
HalfHessian(A)`, the contents `A` is retrieved by `contents(H)`.

For a simple quadratic objective function like:

```
f(x) = ‖D⋅x‖²
```

the half-Hessian is:

```
H = D'⋅D
```

As the half-Hessian is symmetric, a single method `apply!` has to be
implemented to apply the direct and adjoint of the operator, the signature of
the method is:

```julia
apply!(y::T, ::Type{Direct}, H::HalfHessian{typeof(A)}, x::T)
```

where `y` is overwritten by the result of applying `H` (or its adjoint) to the
argument `x`.  Here `T` is the relevant type of the variables.  Similarly, to
allocate a new object to store the result of applying the operator, it is
sufficient to implement the method:

```julia
vcreate(::Type{Direct}, H::HalfHessian{typeof(A)}, x::T)
```

See also: [`LinearOperator`][@ref).

"""
struct HalfHessian{T} <: SelfAdjointOperator
    obj::T
end

"""
```julia
contents(C)
```

yields the contents of the container `C`.  A *container* is any type which
implements the `contents` method.

"""
contents(H::HalfHessian) = H.obj
