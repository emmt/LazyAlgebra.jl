# Methods for vectors

A *vector* is that which has the algebra of a vector space (Peano 1888, van der Waerden
1931). See talk by Jiahao Chen: [*Taking Vector Transposes
Seriously*](https://www.youtube.com/watch?v=C2RO34b_oPM) at JuliaCon 2017. in
`LazyAlgebra` any (abstract) array object can be considered as a *vector*.


## Vectorized methods

Most necessary operations on the variables of interest are linear operations. Hence
variables (whatever their specific type and size) are just called *vectors* in
`LazyAlgebra`. Numerical methods based on `LazyAlgebra` manipulate the variables via a
small number of vectorized methods:

* `vdot([T,][w,]x,y)` yields the inner product of `x` and `y`; that is, the sum of
  `conj(x[i])*y[i]` or, if `w` is specified, the sum of `w[i]*conj(x[i])*y[i]`, for all
  indices `i`. Optional argument `T` is the type of the result; for real valued *vectors*,
  `T` is a floating-point type; for complex valued *vectors*, `T` can be a complex type
  (with floating-point parts) or a floating-point type to compute only the real part of
  the inner product. `vdot([T,]sel,x,y)` yields the sum of `x[i]*y[i]` for all `i ∈ sel`
  where `sel` is a selection of indices. A `DimensionMismatch` exception is thrown if the
  axes of `w` (if specified), `x`, and `y` are not the same.

* `vnorm1([T,]x)` yields the L-1 norm of `x`, that is the sum of the absolute values of
  the components of `x`. Optional argument `T` is the floating-point type of the result.

* `vnorm2([T,]x)` yields the Euclidean (or L-2) norm of `x`, that is the square root of
  sum of the squared values of the components of `x`. Optional argument `T` is the
  floating-point type of the result.

* `vnorminf([T,]x)` L-∞ norm of `x`, that is the maximal absolute values of the components
  of `x`. Optional argument `T` is the floating-point type of the result

* `vcreate(x)` yields a new array similar to `x` but with floating-point elements.

* `vcopy!(dst,src)` copies the contents of `src` into `dst` and returns `dst`. A
  `DimensionMismatch` exception is thrown if the axes of `dst` and `src` are not the same.

* `vcopy(x)` yields a fresh copy of the *vector* `x` with floating-point elements.

* `vswap!(x,y)` exchanges the contents of `x` and `y`. A `DimensionMismatch` exception is
  thrown if the axes of `x` and `y` are not the same.

* `vfill!(x,α)` sets all elements of `x` with the scalar value `α` and returns `x`.

* `vzero!(x)`fills `x` with zeros and returns it.

* `vscale!(dst,α,src)` overwrites `dst` with `α*src` and returns `dst`. The convention is
  that, if `α = 0`, then `dst` is filled with zeros whatever the contents of `src`. A
  `DimensionMismatch` exception is thrown if the axes of `dst` and `src` are not the same.

* `vscale!(x,α)` and `vscale!(α,x)` overwrite `x` with `α*x` and returns `x`. The
  convention is that, if `α = 0`, then `x` is filled with zeros whatever its prior
  contents.

* `vscale(α,x)` and `vscale(x,α)` yield a new *vector* with floating-point elements equal
  to those of `x` multiplied by the scalar `α`.

* `vproduct!(dst,[sel,]x,y)` overwrites `dst` with the elementwise multiplication of `x`
  by `y`. Optional argument `sel` is a selection of indices to consider. A
  `DimensionMismatch` exception is thrown if the axes of `dst`, `x`, and `y` are not the
  same. If `sel` is specified, a `BoundsError` exception is thrown if the values in `sel`
  are not all valid for indexing `dst`, `x`, and `y`.

* `vproduct(x,y)` yields the elementwise multiplication of `x` by `y`. A
  `DimensionMismatch` exception is thrown if the axes of `x` and `y` are not the same.

* `vupdate!(y,[sel,]α,x)` overwrites `y` with `α*x + y` and returns `y`. Optional argument
  `sel` is a selection of indices to which apply the operation (if an index is repeated,
  the operation will be performed several times at this location). A `DimensionMismatch`
  exception is thrown if the axes of `x` and `y` are not the same. If `sel` is specified,
  a `BoundsError` exception is thrown if the values in `sel` are not all valid for
  indexing `x` and `y`.

* `vcombine(α,x,β,y)` yields the linear combination `α*x + β*y`. A `DimensionMismatch`
  exception is thrown if the axes of `x` and `y` are not the same.

* `vcombine!(dst=y,α,x,β,y)` overwrites `dst` with the linear combination `dst = α*x` or
  `dst = α*x + β*y` and returns `dst`. If `dst` is not specified, it is assumed to be `y`.
  A `DimensionMismatch` exception is thrown if the axes of `dst`, `x`, and `y` are not the
  same.

Note that the names of these methods all start with a `v` (for **v**ector) as the
conventions used by these methods may be particular. For instance, compared to `copy!` and
when applied to arrays, `vcopy!` imposes that the two arguments have the same axes.
Another example is the `vdot` method which has a slightly different semantics than Julia
`dot` method.

`LazyAlgebra` already provides implementations of these methods for Julia arrays of most
types. This implementation assumes that an array is a valid *vector* providing it has
suitable type and dimensions.

## Extending methods for other array types

Methods involving multipliers (scalar factors) like [`vscale!`](@ref), [`vupdate!`](@ref),
and [`vcombine!`](@ref) split their work in the following 3 stages:

1. The top-level method checks whether array arguments have compatible indices (throwing a
   `DimensionMismatch` or a `BoundsError` exception if this is not the case) and call the
   corresponding *dispatch* method ( [`LazyAlgebra.dispatch_vscale!`](@ref),
   [`LazyAlgebra.dispatch_vupdate!`](@ref), or [`LazyAlgebra.dispatch_vcombine!`](@ref))
   with the multipliers converted to suitable floating-point types (using
   [`LazyAlgebra.convert_multiplier`](@ref) or [`LazyAlgebra.multiplier_type`](@ref)).

2. Depending on the specific values of the multipliers, the *dispatch* method calls one of
   the *unsafe* method ([`LazyAlgebra.unsafe_vscale!`](@ref),
   [`LazyAlgebra.unsafe_vupdate!`](@ref), [`LazyAlgebra.unsafe_vcombine!`](@ref),
   [`LazyAlgebra.unsafe_vcopy!`](@ref) or [`vzero!`](ref)).

3. The *unsafe* method computes the result assuming that `@inbounds` can be applied, that
   multipliers do not need to be converted to more suitable types, and that specific
   conditions hold for the values of the multipliers.

To summarize, methods with the `dispatch_` prefix receive arguments whose indices are
guaranteed to be compatible and multipliers converted to suitable type and are in charge
of calling the most appropriate unsafe method depending ob the values of the multipliers,
methods with the `unsafe_` prefix are called when all assumptions hold for their
arguments.

This splitting is intended to avoid repeating checks (for the compatibility of the indices
of the arguments) or conversions (of the multipliers) and yet apply the most optimized
operation depending on the multipliers values. Another motivation is to make easier to
extend the methods to other types of *vectors*. Indeed, this only requires to specialize
the *unsafe* version of the methods listed below:

* [`unsafe_vcombine!`](@ref)
* [`unsafe_vcopy!`](@ref)
* [`unsafe_vdot`](@ref)
* [`unsafe_vmul!`](@ref)
* [`unsafe_vproduct!`](@ref)
* [`unsafe_vproduct`](@ref)
* [`unsafe_vscale!`](@ref)
* [`unsafe_vswap!`](@ref)
* [`unsafe_vupdate!`](@ref)
