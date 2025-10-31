* Fix sparse constructors and tests sparse operators.

* `DiagonalShape`, `TridiagonalShape` traits and `is_diagonal`, `is_tridiagonal`, and
  `is_triangular` methods.

* Implement a `Null` operator.

* Implement [Kronecker](https://en.wikipedia.org/wiki/Kronecker_product) product.

* Use BLAS for pseudo-matrices and flexible matrices.

* Rationalize names of structures for sparse operators.

* By default, store internal sparse structure as fast non-resizable vectors `Memory{T}`.

* Checking types of sparse operator members should be done at build time.

* Define `const RealComplex{T<:Real} = Union{T,Complex{T}}` and use better
  names for `Reals`, `Floats` and `Complexes`.

* In `using LinearAlgebra`, `norm(z,1)` is defined as the sum of the absolute
  values of the elements of the complex-valued array `z` while `norm(z,Inf)` is
  defined as the maximum absolute value of the elements of the complex-valued
  array `z` .

* Fix doc. about the type argument for `vnorm2(x)`, etc.

* Remove file `test/common.jl`.

* Rationalize exceptions and error messages.

* Optimize composition of cropping and zero-padding operators.  The adjoint of
  a cropping or zero-padding operator is the pseudo-inverse of the operator,
  hence extend the `pinv` method.  If input and output dimensions are the same
  (and offsets are all zeros), a cropping/zero-padding operator is the
  identity.

* `vscale!` can call `rmul!`?

* Implement *preconditioned* conjugate gradient.

* Provide means to convert a sparse operator to a `SparseMatrixCSC`.

* Write an implementation of the L-BFGS operator and of the SR1 operator and
  perhaps of other low-rank operators.

* Use more extensively BLAS subroutines.  Fix usage of BLAS `dot` and `axpy`
  routines for dense arrays (use flat arrays).

* Make a demo like:

  ```julia
  using LazyAlgebra
  psf = read_image("psf.dat")
  dat = read_image("data.dat")
  wgt = read_image("weights.dat")
  µ = 1e-3 # choose regularization level
  .... # deal with sizes, zero-padding, or cropping etc.
  F = FFT(dat)         # make a FFT operator to work with arrays similar to dat
  # Build instrumental model H (convolution by the PSF)
  H = F\Diag(F*ifftshift(psf))*F
  W = Diag(wgt)        # W is the precision matrix for independent noise
  D = Diff()           # D will be used for the regularization
  # Build (and simplify) the left hand-side matrix of the normal equations
  A = simplify(H'*W*H + µ*D'*D)
  # Compute the right hand-side vector of the normal equations
  b = H'*W*y
  # Solve the normal equations using linear conjugate gradients
  img = conjgrad(A, b)
  save_image(img, "result.dat")
  ```
