# Implementation of new mappings or linear operators

`LazyAlgebra` provides a number of mappings and linear operators.  To create
new primitive mapping types (not by combining existing mappings) and benefit
from the `LazyAlgebra` infrastruture, you have to:

* Create a new type derived from `Mapping` or one of its abstract subtypes such
  as `LinearMapping`.

* Implement at least two methods `apply!` and `vcreate` specialized for the new
  mapping type.  Applying the mapping is done by the former method.  The latter
  method is called to create a new output variable suitable to store the result
  of applying the mapping (or one of its variants) to some
  input variable.


## The `vcreate` method

The signature of the `vcreate` method is:

```julia
vcreate(::Type{P}, A::Ta, x::Tx, scratch::Bool=false) -> y
```

where `A` is the mapping, `x` its argument and `P` is one of `Direct`,
`Adjoint`, `Inverse` and/or `InverseAdjoint` (or equivalently `AdjointInverse`)
and indicates how `A` is to be applied:

* `Direct` to apply `A` to `x`, *e.g.* to compute `A⋅x`;
* `Adjoint` to apply the adjoint of `A` to `x`, *e.g.* to compute `A'⋅x`;
* `Inverse` to apply the inverse of `A` to `x`, *e.g.* to compute `A\x`;
* `InverseAdjoint` or `AdjointInverse` to apply the inverse of `A'` to `x`,
  *e.g.* to compute `A'\x`.

The result returned by `vcreate` is a new output variables suitable to store
the result of applying the mapping `A` (or one of its variants as indicated by
`P`) to the input variables `x`.

The optional argument `scratch` is a boolean to let the caller indicate whether
the input variable `x` may be re-used to store the result.  If `scratch` is
`true` and if that make sense, the value returned by `vcreate` may be `x`.  The
default value of **`scratch` must be `false`**.  Calling `vcreate` with
`scratch=true` can be used to limit the allocation of resources when possible.
Having `scratch=true` is only indicative and a specific implementation of
`vcreate` may legitimately always assume `scratch=false` and return a new
variable whatever the value of this argument (e.g. because applying the
considered mapping *in-place* is not possible or because the considered mapping
is not an endomorphism).  Of course, the opposite behavior (i.e., assuming that
`scratch=true` while the method was called with `scratch=false`) is forbidden.
If `vcreate` always behaves as if `scratch=false`, then it is sufficient to
implement a version without this argument:

```julia
vcreate(::Type{P}, A::Ta, x::Tx) -> y
```



## The `apply!` method

The signature of the `apply!` method is:

```julia
apply!(α::Real, ::Type{P}, A::Ta, x::Tx, scratch::Bool=false, β::Real, y::Ty) -> y
```

This method shall overwrites the contents of output variables `y` with the
result of `α*P(A)⋅x + β*y` where `P` is one of `Direct`, `Adjoint`, `Inverse`
and/or `InverseAdjoint` (or equivalently `AdjointInverse`) and shall return
`y`.  The convention is that the prior contents of `y` is not used at all if `β
= 0` so the contents of `y` does not need to be initialized in that case.

Not all operations `P` must be implemented, only the supported ones.  For
iterative resolution of (inverse) problems, it is generally needed to implement
at least the `Direct` and `Adjoint` operations for linear operators.  However
nonlinear mappings are not supposed to implement the `Adjoint` and derived
operations.

Argument `scratch` is a boolean to let the caller indicate whether the contents
of the input variable `x` may be overwritten during the operations.  If
`scratch=false`, the `apply!` method shall not modify the contents of `x`.  If
the `apply!` method for the considered mapping never modify the contents of
`x`, then it is sufficient to implement a version without this argument:

```julia
apply!(α::Real, ::Type{P}, A::Ta, x::Tx, β::Real, y::Ty) -> y
```


## Example

The following example implements a simple sparse linear operator which is able
to operate on multi-dimensional arrays (the so-called *variables*):

```julia
using LazyAlgebra
import LazyAlgebra: vcreate, apply!, input_size, output_size

struct SparseOperator{T<:AbstractFloat,M,N} <: LinearMapping
    outdims::NTuple{M,Int}
    inpdims::NTuple{N,Int}
    A::Vector{T}
    I::Vector{Int}
    J::Vector{Int}
end

input_size(S::SparseOperator) = S.inpdims
output_size(S::SparseOperator) = S.outdims

function vcreate(::Type{Direct}, S::SparseOperator{Ts,M,N},
                 x::DenseArray{Tx,N}) where {Ts<:Real,Tx<:Real,M,N}
    @assert size(x) == input_size(S)
    Ty = promote_type(Ts, Tx)
    return Array{Ty}(undef, output_size(S))
end

function vcreate(::Type{Adjoint}, S::SparseOperator{Ts,M,N},
                 x::DenseArray{Tx,M}) where {Ts<:Real,Tx<:Real,M,N}
    @assert size(x) == output_size(S)
    Ty = promote_type(Ts, Tx)
    return Array{Ty}(undef, input_size(S))
end

function apply!(alpha::Real,
                ::Type{Direct},
                S::SparseOperator{Ts,M,N},
                x::DenseArray{Tx,N},
                beta::Real,
                y::DenseArray{Ty,M}) where {Ts<:Real,Tx<:Real,Ty<:Real,M,N}
    @assert size(x) == input_size(S)
    @assert size(y) == output_size(S)
    beta == 1 || vscale!(y, beta)
    if alpha != 0
        A, I, J = S.A, S.I, S.J
        _alpha_ = convert(promote_type(Ts,Tx,Ty), alpha)
        @assert length(I) == length(J) == length(A)
        for k in 1:length(A)
            i, j = I[k], J[k]
            y[i] += _alpha_*A[k]*x[j]
        end
    end
    return y
end

function apply!(alpha::Real, ::Type{Adjoint},
                S::SparseOperator{Ts,M,N},
                x::DenseArray{Tx,M},
                beta::Real,
                y::DenseArray{Ty,N}) where {Ts<:Real,Tx<:Real,Ty<:Real,M,N}
    @assert size(x) == output_size(S)
    @assert size(y) == input_size(S)
    beta == 1 || vscale!(y, beta)
    if alpha != 0
        A, I, J = S.A, S.I, S.J
        _alpha_ = convert(promote_type(Ts,Tx,Ty), alpha)
        @assert length(I) == length(J) == length(A)
        for k in 1:length(A)
            i, j = I[k], J[k]
            y[j] += _alpha_*A[k]*x[i]
        end
    end
    return y
end
```

Remarks:

- In our example, arrays are restricted to be *dense* so that linear indexing
  is efficient.  For the sake of clarity, the above code is intended to be
  correct although there are many possible optimizations.

- If `alpha = 0` there is nothing to do except scale `y` by `beta`.

- The call to `vscale!(beta, y)` is to properly initialize `y`.  Remember the
  convention that the contents of `y` is not used at all if `beta = 0` so `y`
  does not need to be properly initialized in that case, it will simply be
  zero-filled by the call to `vscale!`.  The statements

  ```julia
  beta == 1 || vscale!(y, beta)
  ```

  are equivalent to:

  ```julia
  if beta != 1
      vscale!(y, beta)
  end
  ```

  which may be simplified to just calling `vscale!` unconditionally:

  ```julia
  vscale!(y, beta)
  ```

  as `vscale!(y, beta)` does nothing if `beta = 1`.

- `@inbounds` could be used for the loops but this would require checking that
  all indices are whithin the bounds.  In this example, only `k` is guaranteed
  to be valid, `i` and `j` have to be checked.
