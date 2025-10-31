# Methods for new primitive operators

`LazyAlgebra` provides a number of linear operators. To create new primitive operator
types (not by combining existing operators) and benefit from the `LazyAlgebra`
infrastructure, you have to:

* Create a new type derived from `Operator`.

* Specialize a few methods to assert the input shape and element type of any `x`
  acceptable to compute `A*x` and to infer the output shape and element type of `A*x`.

* In order to apply the operator `A`, the method `LazyAlgebra.unsafe_vmul!(α, A, x, β, y)`
  must be implemented to overwrite `y` with `α*A*x + β*y`. The same method may also be
  extended for `A'`, `inv(A)` and/or `inv(A')` to apply the adjoint, inverse, and/or
  inverse-adjoint of `A`.

* Optionally specialize method `Base.isequal(A, B)` to yield whether operators `A` and `B`
  of the new operator type are the same in the sense that they yield the same output when
  applied to any acceptable input. In other words, `isequal(A, B)` shall yield whether
  `isequal(A*x, B*x)` holds for any acceptable `x`. Also optionally, specialize
  [`LazyAlgebra.try_simplify`](@ref) for providing simplification rules for some
  constructions (sums, compositions, etc.) involving one or more operators of the new
  type. These methods are used by [`LazyAlgebra.simplify`](@ref).


## Input and output shapes

For any linear operator `A`, computing `A*x` requires to check whether `x` has an
acceptable shape and to infer the shape of `A*x`. The method [`LazyAlgebra.output_axes(A,
x)`](@ref LazyAlgebra.output_axes) is called to perform these two tasks. This method shall
throw a `DimensionMismatch` exception if `x` has invalid shape and shall return the axes
of `A*x` otherwise.

To have this method applicable to a given linear operator type, there are several
possibilities:

1. The method [`LazyAlgebra.output_axes(A, x)`](@ref LazyAlgebra.output_axes) may be
   directly implemented for the type of `A` an perhaps `x`.

2. If the axes of `A*x` only depend on the operator `A` and on the axes of the input array
   `x`, then it is sufficient to provide:

   ```julia
   LazyAlgebra.output_axes(A, axes(x))
   ```

3. If the shapes of the input and output of `A` are known in advance, then it is simpler
   to extend [`LazyAlgebra.output_shape(A)`](@ref LazyAlgebra.output_shape) and
   [`LazyAlgebra.input_shape(A)`](@ref LazyAlgebra.input_shape) to respectively yield the
   shapes of the input and output of `A` as tuples of dimension lengths and/or index unit
   ranges (the two may be mixed). For `LazyAlgebra` to be aware of this, the traits
   [`LazyAlgebra.InputShape`](@ref) and [`LazyAlgebra.OutputShape`](@ref) must also be
   implemented as follows:

   ```julia
   LazyAlgebra.InputShape(typeof(A)) = LazyAlgebra.HasInputShape{N}()
   LazyAlgebra.OutputShape(typeof(A)) = LazyAlgebra.HasOutputShape{M}()
   ```

   with `N` and `M` the number of dimensions of the input and output of `A`.


## Input and output element types

As for the input and output shapes, the input and output element types must be inferable
by `LazyAlgebra` in order to compute `A*x` with a linear operator `A` and some input `x`.

If the element type of any acceptable `x` to compute `A*x` is known in advance, then the
following two methods shall be specialized:

* [`LazyAlgebra.InputEltype(typeof(A))`](@ref LazyAlgebra.InputEltype) shall return
  `LazyAlgebra.HasInputEltype()`.

* [`LazyAlgebra.input_eltype(typeof(A))`](@ref LazyAlgebra.input_eltype) shall return the
  element type required for `x`.

Otherwise, if the acceptable element type of `x` to compute `A*x` is not known in advance,
then the method [`LazyAlgebra.InputEltype(typeof(A))`](@ref LazyAlgebra.InputEltype) shall
return `LazyAlgebra.InputEltypeUnknown()`. Since this is the default behavior, it is not
necessary to specialize this method for the type of `A` in that case.

If [`LazyAlgebra.InputEltype(typeof(A))`](@ref LazyAlgebra.InputEltype) returns
`LazyAlgebra.HasInputEltype()`, then any argument `x` with a different element type is
automatically converted by `LazyAlgebra` to compute `A*x`. As a consequence, consider
carefully whether this is advisable or not. In general, this is only needed if the
operator is implemented by an external library which imposes the element type.

Similarly, if the element type of `A*x` is known in advance, then the following two
methods shall be specialized:

* [`LazyAlgebra.OutputEltype(typeof(A))`](@ref LazyAlgebra.OutputEltype) shall return
  `LazyAlgebra.HasOutputEltype()`.

* [`LazyAlgebra.output_eltype(typeof(A))`](@ref LazyAlgebra.output_eltype) shall return
  the element type of `A*x`.

Otherwise, if the element type of `A*x` is not known in advance, e.g. because it depends
on both `A` and `x`, then the method [`LazyAlgebra.OutputEltype(typeof(A))`](@ref
LazyAlgebra.OutputEltype) shall return `LazyAlgebra.OutputEltypeUnknown()`. Since this is
the default behavior, it is not necessary to specialize this method for the type of `A` in
that case.

As a simplification, it is assumed that the element type of `A*x` is a *trait* that only
depends on the type of the operator `A` and on the type of the input array `x`. Following
this assumption, `LazyAlgebra` infers the element type of `A*x` from that of:

```julia
LazyAlgebra.output_eltype(typeof(A), typeof(x))
```

and it is thus expected that a method with this signature exists for the operator `A` and
that it returns the element type of `A*x`. If such a method does not exists but
[`LazyAlgebra.OutputEltype(typeof(A))`](@ref LazyAlgebra.OutputEltype) yields
`LazyAlgebra.HasOutputEltype()`, then `T` is given by:

```julia
T = float(LazyAlgebra.output_eltype(typeof(A)))
```

otherwise

```julia
Base.eltype(typeof(A))
```

is called to infer the type of the coefficients of `A` and which assumes that the element
type of `A*x` is that of the floating-point conversion of the multiplication of two values
of respective types `eltype(typeof(A))` and `eltype(x)`.


## The `LazyAlgebra.unsafe_vmul!` method

The signature of the `LazyAlgebra.unsafe_vmul!` method to be implemented for a specific
operator type `Ta<:Operator` is:

```julia
LazyAlgebra.unsafe_vmul!(α::Number, A::Ta, x::Tx, β::Number, y::Ty)
```

This method shall overwrite `y` with `α*A*x + β*y`. This method is called by
[`vmul`](@ref)) and [`vmul!`](@ref)) after checking that arguments`x` and `y` have correct
axes (so that `@inbounds` can be assumed to compute the result stored in `y`), with
multipliers `α` and `β` converted to suitable numeric types, and only if `iszero(α)` does
not hold. The convention is that the prior content of `y` is not used at all if
`iszero(β)` holds so that `y` can be directly used to store the result even though it is
not initialized. [`LazyAlgebra.unsafe_vmul!`](@ref) shall return `nothing` (any returned
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


## The `isequal` method

The method `isequal(A,B)` yields whether `A` and `B` are the same operators in the sense
that their effects will **always** be the same. This method is used to perform some
simplifications and optimizations and may have to be specialized for specific operator
types. The default implementation is to return `A === B`.

The returned result may be true although `A` and `B` are not necessarily the same object.
In the below example, if `A` and `B` are two sparse matrices whose coefficients and
indices are stored in the same vectors (as can be tested with the `===` operator) this
method should return `true` because the two operators will behave identically (any changes
in the coefficients or indices of `A` will be reflected in `B`). If any of the vectors
storing the coefficients or the indices are not the same objects, then `identical(A,B)`
must return `false` even though the stored values may be the same because it is possible,
later, to change one operator without affecting identically the other.


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

function unsafe_vmul!(α::Number,
                      S::SparseOperator{Ts,M,N},
                      x::DenseArray{Tx,N},
                      β::Number,
                      y::DenseArray{Ty,M}) where {Ts<:Real,Tx<:Real,Ty<:Real,M,N}
    isone(β) || unsafe_vscale!(y, β)
    A, I, J = S.A, S.I, S.J
    for k in 1:length(A)
        i, j = I[k], J[k]
        y[i] += α*A[k]*x[j]
    end
    return y
end

function unsafe_vmul!(α::Number,
                      S::Adjoint{<:SparseOperator{Ts,M,N}},
                      x::DenseArray{Tx,M},
                      β::Number,
                      y::DenseArray{Ty,N}) where {Ts<:Real,Tx<:Real,Ty<:Real,M,N}
    isone(β) || unsafe_vscale!(y, β)
    A, I, J = S.A, S.I, S.J
    for k in 1:length(A)
        i, j = I[k], J[k]
        y[j] += α*A[k]*x[i]
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
