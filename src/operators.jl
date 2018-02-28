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
Base.inv(A::LinearOperator) = Inverse(A)

# Automatically unveils operator for common methods.
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
"""
```julia
input_type([Op=Direct,] A)
output_type([Op=Direct,] A)
```

yield the (preferred) types of the input and output arguments of the operation
`Op` with operator `A`.  If `A` operates on Julia arrays, the element type,
list of dimensions, `i`-th dimension and number of dimensions for the input and
output are given by:

    input_eltype([Op=Direct,] A)          output_eltype([Op=Direct,] A)
    input_size([Op=Direct,] A)            output_size([Op=Direct,] A)
    input_size([Op=Direct,] A, i)         output_size([Op=Direct,] A, i)
    input_ndims([Op=Direct,] A)           output_ndims([Op=Direct,] A)

Only `input_size(A)` and `output_size(A)` have to be implemented.

Also see: [`vcreate`](@ref), [`apply!`](@ref), [`LinearOperator`](@ref),
[`Operations`](@ref).

"""
function input_type end

for sfx in (:size, :eltype, :ndims, :type),
    pfx in (:output, :input)

    fn1 = Symbol(pfx, "_", sfx)

    for Op in (Direct, Adjoint, Inverse, InverseAdjoint)

         fn2 = Symbol(Op == Adjoint || Op == Inverse ?
                     (pfx == :output ? :input : :output) : pfx, "_", sfx)

        #println("$fn1($Op) -> $fn2")

        # Provide basic methods for the different operations and for tagged
        # operators.
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
# UNIFORM SCALING

"""
```julia
UniformScalingOperator(α)
```

creates a uniform scaling linear operator whose effects is to multiply its
argument by the scalar `α`.

See also: [`NonuniformScalingOperator`](@ref).

"""
struct UniformScalingOperator <: SelfAdjointOperator
    α::Float64
end

is_applicable_in_place(::Type{<:Operations}, ::UniformScalingOperator, x) = true

isinvertible(A::UniformScalingOperator) = (isfinite(A.α) && A.α != 0.0)

ensureinvertible(A::UniformScalingOperator) =
    isinvertible(A) || throw(
        SingularSystem("Uniform scaling operator is singular"))

function Base.inv(A::UniformScalingOperator)
    ensureinvertible(A)
    return UniformScalingOperator(1.0/A.α)
end

apply!(y, ::Type{<:Union{Direct,Adjoint}}, A::UniformScalingOperator, x) =
    vscale!(y, A.α, x)

function apply!(y, ::Type{<:Union{Inverse,InverseAdjoint}},
                A::UniformScalingOperator, x)
    ensureinvertible(A)
    return vscale!(y, 1.0/A.α, x)
end

function vcreate(::Type{<:Operations},
                 A::UniformScalingOperator,
                 x::AbstractArray{T,N}) where {T<:Real,N}
    return similar(Array{float(T)}, indices(x))
end

vcreate(::Type{<:Operations}, A::UniformScalingOperator, x) =
    vcreate(x)

#------------------------------------------------------------------------------
# NON-UNIFORM SCALING

"""
```julia
NonuniformScalingOperator(A)
```

creates a nonuniform scaling linear operator whose effects is to apply
elementwise multiplication of its argument by the scaling factors `A`.
This operator can be thought as a *diagonal* operator.

See also: [`UniformScalingOperator`](@ref).

"""
struct NonuniformScalingOperator{T} <: SelfAdjointOperator
    scl::T
end

is_applicable_in_place(::Type{<:Operations}, ::NonuniformScalingOperator, x) = true

function Base.inv(A::NonuniformScalingOperator{<:AbstractArray{T,N}}
                  ) where {T<:AbstractFloat, N}
    q = A.scl
    r = similar(q)
    @inbounds @simd for i in eachindex(q, r)
        r[i] = one(T)/q[i]
    end
    return NonuniformScalingOperator(r)
end

function apply!(y::AbstractArray{<:AbstractFloat,N},
                ::Type{<:Union{Direct,Adjoint}},
                A::NonuniformScalingOperator{<:AbstractArray{<:AbstractFloat,N}},
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
                A::NonuniformScalingOperator{<:AbstractArray{<:AbstractFloat,N}},
                x::AbstractArray{<:AbstractFloat,N}) where {N}
    @assert indices(y) == indices(A.scl)
    @assert indices(x) == indices(A.scl)
    @inbounds @simd for i in eachindex(x, A.scl, x)
        y[i] = x[i]/A.scl[i]
    end
    return y
end

function vcreate(::Type{<:Operations},
                 A::NonuniformScalingOperator{<:AbstractArray{Ta,N}},
                 x::AbstractArray{Tx,N}) where {Ta<:AbstractFloat,
                                                Tx<:AbstractFloat, N}
    @assert indices(x) == indices(A.scl)
    T = promote_type(Ta, Tx)
    return similar(Array{T}, indices(A.scl))
end

#------------------------------------------------------------------------------
# RANK-1 OPERATORS

"""

A `RankOneOperator` is defined by two *vectors* `u` and `v` and created by:

```julia
A = RankOneOperator(u, v)
```

and behaves as if `A = u⋅v'`; that is:

```julia
A*x  = vscale(vdot(v, x)), u)
A'*x = vscale(vdot(u, x)), v)
```

See also: [`SymmetricRankOneOperator`](@ref), [`LinearOperator`](@ref),
          [`apply!`](@ref), [`vcreate`](@ref).

"""
struct RankOneOperator{U,V} <: LinearOperator
    u::U
    v::V
end

apply!(y, ::Type{Direct}, A::RankOneOperator, x) =
    vscale!(y, vdot(A.v, x), A.u)

apply!(y, ::Type{Adjoint}, A::RankOneOperator, x) =
    vscale!(y, vdot(A.u, x), A.v)

vcreate(::Type{Direct}, A::RankOneOperator, x) = vcreate(A.v)

vcreate(::Type{Adjoint}, A::RankOneOperator, x) = vcreate(A.u)

input_type(A::RankOneOperator{U,V}) where {U,V} = V
input_ndims(A::RankOneOperator) = ndims(A.v)
input_size(A::RankOneOperator) = size(A.v)
input_size(A::RankOneOperator, d...) = size(A.v, d...)
input_eltype(A::RankOneOperator) = eltype(A.v)

output_type(A::RankOneOperator{U,V}) where {U,V} = U
output_ndims(A::RankOneOperator) = ndims(A.u)
output_size(A::RankOneOperator) = size(A.u)
output_size(A::RankOneOperator, d...) = size(A.u, d...)
output_eltype(A::RankOneOperator) = eltype(A.u)

"""

A `SymmetricRankOneOperator` is defined by a *vector* `u` and created by:

```julia
A = SymmetricRankOneOperator(u)
```

and behaves as if `A = u⋅u'`; that is:

```julia
A*x = A'*x = vscale(vdot(u, x)), u)
```

See also: [`RankOneOperator`](@ref), [`LinearOperator`](@ref),
          [`SelfAdjointOperator`](@ref) [`apply!`](@ref), [`vcreate`](@ref).

"""
struct SymmetricRankOneOperator{U} <: SelfAdjointOperator
    u::U
end

is_applicable_in_place(::Type{<:Operations}, ::SymmetricRankOneOperator) = true

apply!(y, ::Type{Direct}, A::SymmetricRankOneOperator, x) =
    vscale!(y, vdot(A.u, x), A.u)

vcreate(::Type{Direct}, A::SymmetricRankOneOperator, x) =
    vcreate(A.u)

input_type(A::SymmetricRankOneOperator{U}) where {U} = U
input_ndims(A::SymmetricRankOneOperator) = ndims(A.u)
input_size(A::SymmetricRankOneOperator) = size(A.u)
input_size(A::SymmetricRankOneOperator, d...) = size(A.u, d...)
input_eltype(A::SymmetricRankOneOperator) = eltype(A.u)

# FIXME: this should be automatically done for SelfAdjointOperators?
output_type(A::SymmetricRankOneOperator{U}) where {U} = U
output_ndims(A::SymmetricRankOneOperator) = ndims(A.u)
output_size(A::SymmetricRankOneOperator) = size(A.u)
output_size(A::SymmetricRankOneOperator, d...) = size(A.u, d...)
output_eltype(A::SymmetricRankOneOperator) = eltype(A.u)

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
