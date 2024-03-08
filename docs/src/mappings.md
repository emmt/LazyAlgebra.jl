# Methods for mappings

`LazyAlgebra` provides a number of mappings and linear operators. To create a
new primitive mapping type (not by combining existing mappings), say,
`M<:Mapping{L}` and benefit from the `LazyAlgebra` infrastructure, you have to:

* Define the new type, say `M`, as a structure derived from `LinearMapping` or
  from `NonLinearMapping`.

* Implement at least two methods `apply!` and `vcreate` specialized for the new
  mapping type `M` and any of its supported variants `Adjoint{M}`,
  `Inverse{L,M}` and/or `InverseAdjoint{M}`, with `L = is_linear(M)`. Applying
  the mapping is done by calling `vapply!` while `vcreate` is called to create
  a new output variable suitable to store the result of applying the mapping
  (or one of its variants) to some input variable.

* Optionally, specialize method `identical` for two arguments of the new
  mapping type.

* Optionally, if `M` is linear, specialize method `Base.eltype(A::M)` to yield
  the type of the coefficients of `A`.


## The `vcreate` method

The signature of the `vcreate` method to be implemented by specific mapping
types is:

```julia
vcreate(α::Number, A::Ta, x::Tx, scratch::Bool) -> y
```

where `α` is a multiplier, `A` is the mapping, and `x` its argument. The result
returned by `vcreate` is an object suitable to store the result of `α*A*x`,
that is `α` times `A` applied to `x`.

The method shall be specialized for the type of `A` being that of mapping `M`
and any supported variants `Adjoint{M}`, `Inverse{L,M}` and/or
`InverseAdjoint{M}`, with `L = is_linear(M)`.

* `Direct` to apply `A` to `x`, *e.g.* to compute `A⋅x`;
* `Adjoint` to apply the adjoint of `A` to `x`, *e.g.* to compute `A'⋅x`;
* `Inverse` to apply the inverse of `A` to `x`, *e.g.* to compute `A\x`;
* `InverseAdjoint` or `AdjointInverse` to apply the inverse of `A'` to `x`,
  *e.g.* to compute `A'\x`.

The `scratch` argument is a boolean to let the caller indicate whether the
input variable `x` may be re-used to store the result and thus spare allocation
of resources if possible. If `scratch` is `true`, the value returned by
`vcreate` may be `x`. Having `scratch=true` is only indicative and a specific
implementation of `vcreate` may legitimately always return a new variable
whatever the value of the `scratch` argument. Of course, assuming that
`scratch=true` while the method was called with `scratch=false` is forbidden.

The result returned by `vcreate` must be of predictable type to ensure
*type-stability*. In particular, `vcreate(α,A,x,true)` should only yield `x` if
it has exactly the same type as the result that `vcreate(α,A,x,false)` would
have returned. Checking the validity (*e.g.* the size) of argument `x` in
`vcreate` may be skipped because this argument will be eventually checked by
the `apply!` method.

The multiplier is needed to determine the type of the result as it may have
units. For a linear mapping:

```julia
F = floating_point_type(eltype(A), eltype(x))
Ty = convert_floating_point_type(F, promote_type(typeof(α), eltype(A), eltype(x)))
alpha = with_floating_point_type(F, α)
```

```julia
function vcreate(α::Number, A::M, x, scratch::Bool)
    T = output_type(α, A, x)
    dims = ouput_size(A, x)
    return Array{T}(undef, dims)
end

function output_type(α::Number, A::LinearMapping, x)
    # Determine numerical precision.
    F = floating_point_type(eltype(A), eltype(x))
    return convert_floating_point_type(F, promote_type(typeof(α), eltype(A), eltype(x)))
end
```

## The `apply!` method

The signature of the `apply!` method to be implemented by specific mapping
types is:

```julia
apply!(α::Number, A::Ta, x::Tx, scratch::Bool, β::Number, y::Ty) -> y
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
`scratch=false`, the `apply!` method shall not modify the contents of `x`.


## The `identical` method

The method `identical(A,B)` yields whether `A` and `B` are the same mappings in
the sense that their effects will **always** be the same.  This method is used
to perform some simplifications and optimizations and may have to be
specialized for specific mapping types.  The default implementation is to
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

function apply!(α::Real,
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

function apply!(α::Real,
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
