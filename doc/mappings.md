## Implementation of new mappings or linear operators

`LazyAlgebra` provides a number of mappings and linear operators.  To create
now primitive mapping types (not by combining existing mappings) and benefit
from the `LazyAlgebra` infrastruture, you have to:

* create a new type derived from `Mapping` or one of its abstract subtypes such
  as `LinearMapping` or `SelfAdjointOperator`;

* implement at least two methods `vcreate` and `apply!` specialized for the new
  mapping type.  The former method is to create a new output variable suitable
  to store the result of applying the mapping (or one of its variants) to some
  input variable.  Applying the mapping is done by the latter method.

The signature of the `vcreate` method is:

```julia
vcreate(::Type{P}, A::Ta, x::Tx) -> y
```

where `A` is the mapping, `x` its argument and `P` is one of `Direct`,
`Adjoint`, `Inverse` and/or `InverseAdjoint` (or equivalently `AdjointInverse`)
and indicates how `A` is to be applied:

* `Direct` to apply `A` to `x`, *e.g.* to compute `A'⋅x`;
* `Adjoint` to apply the adjoint of `A` to `x`, *e.g.* to compute `A'⋅x`;
* `Inverse` to apply the inverse of `A` to `x`, *e.g.* to compute `A\x`;
* `InverseAdjoint` or `AdjointInverse` to apply the inverse of `A'` to `x`,
  *e.g.* to compute `A'\x`.

The result returned by `vcreate` is a new output variables suitable to store
the result of applying the mapping `A` (or one of its variants as indicated by
`P`) to the input variables `x`.

The signature of the `apply!` method is:

```julia
apply!(α::Real, ::Type{P}, A::Ta, x::tx, β::Real, y::Ty) -> y
```

This method shall overwrites the output variables `y` with the result of
`α*P(A)⋅x + β*y` where `P` is one of `Direct`, `Adjoint`, `Inverse` and/or
`InverseAdjoint` (or equivalently `AdjointInverse`).  The convention is that
the prior contents of `y` is not used at all if `β = 0` so `y` does not need to
be properly initialized in that case.

Not all operations `P` must be implemented, only the supported ones.  For
iterative resolution of (inverse) problems, it is generally needed to implement
at least the `Direct` and `Adjoint` operations for linear operators.  However
nonlinear mappings are not supposed to implement the `Adjoint` and derived
operations.


### Example

The following example implements a simple sparse linear operator which is able
to operate on multi-dimensional arrays (the so-called *variables*):

```julia
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
    return Array{Ty}(output_size(S))
end
function vcreate(::Type{Adjoint}, S::SparseOperator{Ts,M,N},
                 x::DenseArray{Tx,M}) where {Ts<:Real,Tx<:Real,M,N}
    @assert size(x) == output_size(S)
    Ty = promote_type(Ts, Tx)
    return Array{T}(input_size(S))
end
function apply!(alpha::Real,
                ::Type{Direct},
                S::SparseOperator{Ts,M,N},
                x::DenseArray{Tx,N},
                beta::Real,
                y::DenseArray{Ty,M}) where {Ts<:Real,Tx<:Real,Ty<:Real,M,N}
    @assert size(x) == input_size(S)
    @assert size(y) == output_size(S)
    if alpha == 0
        vscale!(y, beta)
    else
        A, I, J = S.A, S.I, S.J
        @assert length(I) == length(J) == length(A)
        if beta != 1
            vscale!(beta, y)
        end
        for k in 1:length(A)
            i, j = I[k], J[k]
            y[i] += alpha*A[k]*x[j]
        end
    end
    return y
end
function apply!(alpha::Real, ::Type{Adjoint},
                S::SparseOperator{Ts,M,N},
                x::DenseArray{Tx,N},
                beta::Real,
                y::DenseArray{Ty,M}) where {Ts<:Real,Tx<:Real,Ty<:Real,M,N}
    @assert size(x) == output_size(S)
    @assert size(y) == input_size(S)
    if alpha == 0
        vscale!(y, beta)
    else
        A, I, J = S.A, S.I, S.J
        @assert length(I) == length(J) == length(A)
        if beta != 1
            vscale!(beta, y)
        end
        for k in 1:n
            i, j = I[k], J[k]
            y[j] += alpha*A[k]*x[i]
        end
    end
    return y
end
```

Note that, in our example, arrays are restricted to be *dense* so that linear
indexing is efficient.  For the sake of clarity, the above code is intended to
be correct although there are many possible optimizations.

Also note the call to `vscale!(beta, y)` to properly initialize `y`.  (Remember
the convention that the contents of `y` is not used at all if `β = 0` so `y`
does not need to be properly initialized in that case.)
