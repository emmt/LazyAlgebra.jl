# Methods for vectors

A *vector* is that which has the algebra of a vector space (Peano 1888, van der
Waerden 1931).  See talk by Jiahao Chen:
[*Taking Vector Transposes Seriously*](https://www.youtube.com/watch?v=C2RO34b_oPM) at JuliaCon 2017.


## Vectorized methods

Most necessary operations on the variables of interest are linear operations.
Hence variables (whatever their specific type and size) are just called
*vectors* in `LazyAlgebra`.  Numerical methods based on `LazyAlgebra`
manipulate the variables via a small number of vectorized methods:

* `vdot([T,][w,]x,y)` yields the inner product of `x` and `y`; that is, the sum
  of `conj(x[i])*y[i]` or, if `w` is specified, the sum of
  `w[i]*conj(x[i])*y[i]`, for all indices `i`.  Optional argument `T` is the
  type of the result; for real valued *vectors*, `T` is a floating-point type;
  for complex valued *vectors*, `T` can be a complex type (with floating-point
  parts) or a floating-point type to compute only the real part of the inner
  product.  `vdot([T,]sel,x,y)` yields the sum of `x[i]*y[i]` for all `i ∈ sel`
  where `sel` is a selection of indices.

* `vnorm1([T,]x)` yields the L-1 norm of `x`, that is the sum of the absolute
  values of the components of `x`.  Optional argument `T` is the floating-point
  type of the result.

* `vnorm2([T,]x)` yields the Euclidean (or L-2) norm of `x`, that is the square
  root of sum of the squared values of the components of `x`.  Optional
  argument `T` is the floating-point type of the result.

* `vnorminf([T,]x)` L-∞ norm of `x`, that is the maximal absolute values of the
  components of `x`.  Optional argument `T` is the floating-point type of the
  result

* `vcreate(x)` yields a new variable instance similar to `x`.  If `x` is an
  array, the element type of the result is a floating-point type.

* `vcopy!(dst,src)` copies the contents of `src` into `dst` and returns `dst`.

* `vcopy(x)` yields a fresh copy of the *vector* `x`.

* `vswap!(x,y)` exchanges the contents of `x` and `y` (which must have the same
  type and size if they are arrays).

* `vfill!(x,α)` sets all elements of `x` with the scalar value `α` and return
  `x`.

* `vzero!(x)`fills `x` with zeros and returns it.

* `vscale!(dst,α,src)` overwrites `dst` with `α*src` and returns `dst`.  The
  convention is that, if `α = 0`, then `dst` is filled with zeros whatever the
  contents of `src`.

* `vscale!(x,α)` and `vscale!(α,x)` overwrite `x` with `α*x` and returns `x`.
  The convention is that, if `α = 0`, then `x` is filled with zeros whatever
  its prior contents.

* `vscale(α,x)` and `vscale(x,α)` yield a new *vector* whose elements are
  those of `x` multiplied by the scalar `α`.

* `vproduct!(dst,[sel,]x,y)` overwrites `dst` with the elementwise
  multiplication of `x` by `y`.  Optional argument `sel` is a selection of
  indices to consider.

* `vproduct(x,y)` yields the elementwise multiplication of `x` by `y`.

* `vupdate!(y,[sel,]α,x)` overwrites `y` with `α*x + y` and returns `y`.
  Optional argument `sel` is a selection of indices to which apply the
  operation (if an index is repeated, the operation will be performed several
  times at this location).

* `vcombine(α,x,β,y)` yields the linear combination `α*x` or `α*x + β*y`.

* `vcombine!(dst,α,x,β,y)` overwrites `dst` with the linear combination `dst =
  α*x` or `dst = α*x + β*y` and returns `dst`.

Note that the names of these methods all start with a `v` (for **v**ector) as
the conventions used by these methods may be particular.  For instance,
compared to `copy!` and when applied to arrays, `vcopy!` imposes that the two
arguments have exactly the same dimensions.  Another example is the `vdot`
method which has a slightly different semantics than Julia `dot` method.

`LazyAlgebra` already provides implementations of these methods for Julia
arrays with floating-point type elements.  This implementation assumes that an
array is a valid *vector* providing it has suitable type and dimensions.


## Implementing a new vector type

To have a numerical method based on `LazyAlgebra` be applicable to a new given
type of variables, it is sufficient to implement a subset of these basic
methods specialized for this kind of variables.

The various operations that should be implemented for a *vector* are:

* compute the inner product of two vectors of the same kind (`vdot(x,y)`
  method);
* create a vector of a given kind (`vcreate(x)` method);
* copy a vector (`vcopy!(dst,src)`);
* fill a vector with a given value (`vfill!(x,α)` method);
* exchange the contents of two vectors (`vswap!(x,y)` method);
* linearly combine several vectors (`vcombine!(dst,α,x,β,y)` method).

Derived methods are:
* compute the Euclidean norm of a vector (`vnorm2` method, based on `vdot` by
  default);
* multiply a vector by a scalar: `vscale!(dst,α,src)` and/or `vscale!(x,α)`
  methods (based on `vcombine!` by default);
* update a vector by a scaled step: `vupdate!(y,α,x)` method (based on
  `vcombine!` by default) and, for some constrained optimization methods,
  `vupdate!(y,sel,α,x)` method;
* erase a vector: `vzero!(x)` method (based on `vfill!` by default);
* `vscale` and `vcopy` methods are implemented with `vcreate` and
  respectively`vscale!` and `vcopy!`.

Other methods which may be required by some packages:
* compute the L-1 norm of a vector: `vnorm1(x)` method;
* compute the L-∞ norm of a vector: `vnorminf(x)` method;


Methods that must be implemented (`V` represent the vector type):

```julia
vdot(::Type{T}, x::Tx, y::Ty) :: T where {T<:AbstractFloat,Tx,Ty}
```

```julia
vscale!(dst::V, alpha::Real, src::V) -> dst
```

methods that may be implemented:

```julia
vscale!(alpha::Real, x::V) -> x
```

For mappings and linear operators (see
[Implementation of new mappings](mappings.md) for details), implement:

```julia
apply!(α::Scalar, P::Type{<:Operations}, A::Ta, x::Tx, β::Scalar, y::Ty) -> y
```

and

```julia
vcreate(P::Type{P}, A::Ta, x::Tx) -> y
```

for `Ta<:Mapping` and the supported operations `P<:Operations`.
