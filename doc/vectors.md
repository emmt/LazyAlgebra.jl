## Methods for vectors

### Vectorized methods

Most necessary operations on the variables of interest are linear operations.
Hence variables (whatever their specific type and size) are just called
*vectors* in `LazyAlgebra`.  Numerical methods based on `LazyAlgebra`
manipulate the variables via a small number of vectorized methods:

* `vdot(x,y)`  `vdot(w,x,y)`
* `vnorm1(x)` yields the L-1 norm of `x`, that is the sum of the absolute
  values of the components of `x`.
* `vnorm2(x)` yields the Euclidean (or L-2) norm of `x`, that is the square
  root of sum of the squared values of the components of `x`.
* `vnorminf(x)` L-∞ norm of `x`, that is the maximal absolute values of the
  components of `x`.
* `vcreate(x)`
* `vcopy!(x,y)`
* `vswap!(x,y)`
* `vscale`, `vscale!`
* `vupdate!`
* `vcombine!`
* `vproduct!`, `vproduct`
* `vfill!`
* `vzero!(x)`

Note that the names of these methods all start with a `v` (for **v**ector) as
the conventions used by these methods may be specific.  For instance, compared
to `copy!` and when applied to arrays, `vcopy!` imposes that the two arguments
have exactly the same dimensions.  Another example is the `vdot` method which
has a slightly different semantics than Julia `dot` method.

`LazyAlgebra` already provides implementations of these methods for Julia
arrays with floating-point type elements.  This implementation assumes that an
array is a valid *vector* providing it has suitable type and dimensions.


### Implementing a new vector type

To have a numerical method based on `LazyAlgebra` be applicable to a new given
type of variables, it is sufficient to implement a subset of these basic
methods specialized for this kind of variables.

The various operations that should be implemented for a *vector* are:

* compute the inner product of two vectors of the same kind (`vdot` method);
* create a vector of a given kind (`vcreate` method);
* copy a vector (`vcopy!` and `vcopy` methods);
* fill a vector with a given value (`vfill!` method);
* exchange the contents of two vectors (`vswap!` method);
* multiply a vector by a scalar (`vscale` and `vscale!` methods);
* linearly combine several vectors (`vcombine!` and `vcombine` methods).

Derived methods are:
* compute the Euclidean norm of a vector (`vnorm2` method, based on `vdot` by
  default);
* update a vector by a scaled step (`vupdate!` method, based on `vcombine!` by
  default);
* erase a vector (`vzero!` method based on `vfill!` by default);

Other methods which may be required by some packages:
* compute the L-1 norm of a vector (`vnorm1` method);
* compute the L-∞ norm of a vector (`vnorminf` method);


methods that must be implemented (`V` represent the vector type):

```julia
vscale!(dst::V, alpha::Real, x::V) -> dst
```

methods that may be implemented:

```julia
vscale!(alpha::Real, x::V) -> x
```

For linear operators:

implement:
```julia
apply!(β::Real, y, α::Real, P::Type{<:Operations}, A::T, x) -> y
```
or at least:
```julia
apply!(y, ::Type{P}, A::T, x) -> y
```
for `T<:Operator` and the supported operations `P<:Operations`.

and
```julia
vcreate(P::Type{P}, A::T, x) -> y
```
