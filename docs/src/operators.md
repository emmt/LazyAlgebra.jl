# Methods for operators

`LazyAlgebra` provides a number of linear operators. To create new primitive operator
types (not by combining existing operators) and benefit from the `LazyAlgebra`
infrastructure, you have to:

* Create a new type derived from `Operator`.

* In order to create the array `y` to store the result of `A*x` or of `α*A*x` or to check
  the validity of `y` when it is provided by the user, methods `LazyAlgebra.output_axes(A,
  axes(x))`, and at least one of `Base.eltype(typeof(A))` or
  `LazyAlgebra.output_eltype(typeof(A), eltype(x))` must be specialized for the operator `A`.

* In order to apply the operator `A`, the method `LazyAlgebra.unsafe_vmul!(α, A, x, β, y)`
  must be implemented to overwrite `y` with `α*A*x + β*y`. The same method may also be
  extended for `A'`, `inv(A)` and/or `inv(A')` to apply the adjoint, inverse, and/or
  inverse-adjoint of `A`.

* Optionally specialize method `Base.similar` for two arguments of the new operator type.


## The `LazyAlgebra.unsafe_vmul!` method

The signature of the `LazyAlgebra.unsafe_vmul!` method to be implemented for a specific
operator type `Ta<:Operator` is:

```julia
LazyAlgebra.unsafe_vmul!(α::Number, A::Ta, x::Tx, β::Number, y::Ty)
```

This method shall overwrite `y` with `α*A*x + β*y`. This method is called by
[`vmul`](@ref)) and [`vmul!`](@ref)) after checking that arguments`x` and `y` have correct
axes (so that `@inbounds` can be assumed to compute the result stored in `y`), with
multipliers `α` and `β` converted to suitable floating-point types, and only if
`iszero(α)` does not hold. The convention is that the prior contents of `y` is not used at
all if `iszero(β)` holds so that `y` can be directly used to store the result even though
it is not initialized. [`LazyAlgebra.unsafe_vmul!`](@ref) shall return `nothing` (any returned
value is ignored by [`vmul`](@ref)) and [`vmul!`](@ref)).

In the above signature, `Ta<:Operator` is the type of the operator to apply,
`Tx<:AbstractArray` and `Ty<:AbstractArray` are the respective types of `x` (to be
multiplied by `A`) and `y` (to store the result).

If applying the adjoint, inverse, or inverse-adjoint of an operator of type `Ta` is
supported, [`LazyAlgebra.unsafe_vmul!`](@ref) shall be implemented for
[`LazyAlgebra.Adjoint{Ta}`](@ref LazyAlgebra.Adjoint), [`LazyAlgebra.Inverse{Ta}`](@ref
LazyAlgebra.Inverse), and `LazyAlgebra.Inverse{LazyAlgebra.Adjoint{Ta}}` respectively.
Note the particular order of the latter construction: in effect, the inverse-adjoint of
`A` given by expressions `inv(A)'` and `inv(A')` is always stored as `inv(A')`. As a
facility, the alias [`LazyAlgebra.InverseAdjoint{Ta}`](@ref LazyAlgebra.InverseAdjoint)
can also be used in the signature, this alias is the union of
`LazyAlgebra.Inverse{LazyAlgebra.Adjoint{Ta}}` and
`LazyAlgebra.Adjoint{LazyAlgebra.Inverse{Ta}}`.


## Creating the output of an operator

To create the array to store `A*x` or `α*A*x`, the following method is called:

```julia
y = LazyAlgebra.create_output([α::Number,] A::Operator, x::AbstractArray)
```

where, if the multiplier `α` is supplied, it shall be assumed that `α` has been already
converted by [`LazyAlgebra.convert_multiplier`](@ref).

The default implementations are:

```julia
create_output(A::Operator, x::AbstractArray) =
    new_array(output_eltype(A, x), output_axes(A, x))

create_output(α::Number, A::Operator, x::AbstractArray) =
    new_array(output_eltype(α, A, x), output_axes(A, x))
```

where the `new_array` method is taken from the
[TypeUtils](https://github.com/emmt/TypeUtils.jl) package. As can be seen,
[`LazyAlgebra.create_output`](@ref) relies on two auxiliary methods
[`LazyAlgebra.output_eltype`](@ref) and [`LazyAlgebra.output_axes`](@ref). These two
methods must be specialized for the type of `A` (and of `A'`, `inv(A)`, or `inv(A')` if
these variants are supported) as explained next.

As a first simplification, it is assumed in `LazyAlgebra` that the element type of `A*x`
is a *trait* that only depends on the type of the operator `A` and on the element type of
the input array `x`. Following this assumption, [`LazyAlgebra.create_output`](@ref) infers
its result from that of:

```julia
LazyAlgebra.output_eltype(typeof(A), eltype(x))
```

and it is thus expected that a method with this signature be implemented for the operator `A`
to yield the element type of `A*x`. If such a method does not exists, a fallback method
is provided which amounts to calling:

```julia
Base.eltype(typeof(A))
```

to infer the type of the coefficients of `A`. The element type of `A*x` is then assumed to
be given by converting to floating-point the multiplication of two values of respective
types `eltype(A)` and `eltype(x)`.

This machinery is needed to support quantities with units in `LazyAlgebra`.

As a second simplification, it is assumed in `LazyAlgebra` that the axes of `A*x` only
depend on the operator `A` and on the axes of the input array `x`. Following this
assumption, [`LazyAlgebra.create_output`](@ref) yields an array whose axes are given by:

```julia
LazyAlgebra.output_axes(A, axes(x))
```

and it is thus expected that a method with this signature exists for the operator `A`.

!!! warning
    The method [`LazyAlgebra.output_axes`](@ref) must throw an exception if the axes of
    `x` are not valid in the expression `A*x` so that `@inbounds` can be safely assumed
    when this method returns normally.

!!! note
    Since `new_array` (from [TypeUtils](https://github.com/emmt/TypeUtils.jl)) is called
    to create the output `y`, if `LazyAlgebra.output_axes(A, x)` yields a tuple consisting
    of `Base.OneTo` instances, `y` will be an array of type `Array` with 1-based indices;
    otherwise, `y` will be an `OffsetArray`.

For more flexibility, the method [`LazyAlgebra.create_output`](@ref) may be specialized in
the operator type. In that case, implementing [`LazyAlgebra.output_eltype`](@ref) is not
necessary. Implementing [`LazyAlgebra.output_axes`](@ref) is always needed as it is used
to check the axes of an array `y` supplied by the user. If it is extended, it is critical
that [`LazyAlgebra.create_output`](@ref) throws an exception when the axes of `x` are not
valid in the expression `A*x`.


## The `identical` method

The method `identical(A,B)` yields whether `A` and `B` are the same operators in
the sense that their effects will **always** be the same.  This method is used
to perform some simplifications and optimizations and may have to be
specialized for specific operator types.  The default implementation is to
return `A === B`.

The returned result may be true although `A` and `B` are not necessarily the
same object.  In the below example, if `A` and `B` are two sparse matrices
whose coefficients and indices are stored in the same vectors (as can be tested
with the `===` operator) this method should return `true` because the two
operators will behave identically (any changes in the coefficients or indices
of `A` will be reflected in `B`).  If any of the vectors storing the
coefficients or the indices are not the same objects, then `identical(A,B)`
must return `false` even though the stored values may be the same because it is
possible, later, to change one operator without affecting identically the
other.


## Example

The following example implements a simple sparse linear operator which is able
to operate on multi-dimensional arrays (the so-called *variables*):

```julia
# Use LazyAlgebra framework and import methods that need to be extended.
using LazyAlgebra
import LazyAlgebra: vcreate, vmul!, input_size, output_size

struct SparseOperator{T<:AbstractFloat,M,N} <: Operator
    outdims::NTuple{M,Int}
    inpdims::NTuple{N,Int}
    A::Vector{T}
    I::Vector{Int}
    J::Vector{Int}
end

input_size(S::SparseOperator) = S.inpdims
output_size(S::SparseOperator) = S.outdims

function vcreate(::Type{Direct}, S::SparseOperator{Ts,M,N},
                 x::DenseArray{Tx,N},
                 scratch::Bool) where {Ts<:Real,Tx<:Real,M,N}
    @assert size(x) == input_size(S)
    Ty = promote_type(Ts, Tx)
    return Array{Ty}(undef, output_size(S))
end

function vcreate(::Type{Adjoint}, S::SparseOperator{Ts,M,N},
                 x::DenseArray{Tx,M},
                 scratch::Bool) where {Ts<:Real,Tx<:Real,M,N}
    @assert size(x) == output_size(S)
    Ty = promote_type(Ts, Tx)
    return Array{Ty}(undef, input_size(S))
end

function vmul!(α::Real,
                ::Type{Direct},
                S::SparseOperator{Ts,M,N},
                x::DenseArray{Tx,N},
                scratch::Bool,
                β::Real,
                y::DenseArray{Ty,M}) where {Ts<:Real,Tx<:Real,Ty<:Real,M,N}
    @assert size(x) == input_size(S)
    @assert size(y) == output_size(S)
    β == 1 || vscale!(y, β)
    if α != 0
        A, I, J = S.A, S.I, S.J
        alpha = convert(promote_type(Ts,Tx,Ty), α)
        @assert length(I) == length(J) == length(A)
        for k in 1:length(A)
            i, j = I[k], J[k]
            y[i] += alpha*A[k]*x[j]
        end
    end
    return y
end

function vmul!(α::Real,
                ::Type{Adjoint},
                S::SparseOperator{Ts,M,N},
                x::DenseArray{Tx,M},
                scratch::Bool,
                β::Real,
                y::DenseArray{Ty,N}) where {Ts<:Real,Tx<:Real,Ty<:Real,M,N}
    @assert size(x) == output_size(S)
    @assert size(y) == input_size(S)
    β == 1 || vscale!(y, β)
    if α != 0
        A, I, J = S.A, S.I, S.J
        alpha = convert(promote_type(Ts,Tx,Ty), α)
        @assert length(I) == length(J) == length(A)
        for k in 1:length(A)
            i, j = I[k], J[k]
            y[j] += alpha*A[k]*x[i]
        end
    end
    return y
end

identical(A::T, B::T) where {T<:SparseOperator} =
    (A.outdims == B.outdims && A.inpdims == B.inpdims &&
     A.A === B.A && A.I === B.I && A.J === B.J)
```

Remarks:

- In our example, arrays are restricted to be *dense* so that linear indexing
  is efficient.  For the sake of clarity, the above code is intended to be
  correct although there are many possible optimizations.

- If `α = 0` there is nothing to do except scale `y` by `β`.

- The call to `vscale!(β, y)` is to properly initialize `y`.  Remember the
  convention that the contents of `y` is not used at all if `β = 0` so `y`
  does not need to be properly initialized in that case, it will simply be
  zero-filled by the call to `vscale!`.  The statements

  ```julia
  β == 1 || vscale!(y, β)
  ```

  are equivalent to:

  ```julia
  if β != 1
      vscale!(y, β)
  end
  ```

  which may be simplified to just calling `vscale!` unconditionally:

  ```julia
  vscale!(y, β)
  ```

  as `vscale!(y, β)` does nothing if `β = 1`.

- `@inbounds` could be used for the loops but this would require checking that
  all indices are whithin the bounds.  In this example, only `k` is guaranteed
  to be valid, `i` and `j` have to be checked.
