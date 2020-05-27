#
# conjgrad.jl -
#
# Linear conjugate-gradient.
#
#-------------------------------------------------------------------------------
#
# This file is part of LazyAlgebra (https://github.com/emmt/LazyAlgebra.jl)
# released under the MIT "Expat" license.
#
# Copyright (c) 2017-2020 Éric Thiébaut.
#

struct WrappedLeftHandSideMatrix{T}
    op::T
end
(obj::WrappedLeftHandSideMatrix)(dst, src) = apply!(dst, obj.op, src)

"""
```julia
conjgrad(A, b, x0=vzeros(b)) -> x
```

solves the symmetric linear system `A⋅x = b` starting at `x0` by means of the
iterative conjugate gradient method.  The returned solution `x` is a new object
similar to `b` and to `x0`.

Argument `A` implements the symmetric positive definite linear mapping `A`, it
can be provided as a Julia array (interpreted as a general matrix, see
[`GeneralMatrix`](@ref)), as an instance of [`LinearMapping`](@ref) or as a
callable object (like a function) which is used as:

```julia
A(dst, src)
```

to overwrite `dst` with `A⋅src`.  If `A` has been implemented as a callable
object, such that `A(x)` yields `A⋅x`, then call `conjgrad` with an
inline function:

```julia
conjgrad((dst,src) -> (dst .= A(src); return dst), b, ...)
```

See [`conjgrad!`](@ref) for accepted keywords and more details.

"""
conjgrad(A, b, x0; kwds...) =
    conjgrad!(vcreate(b), A, b, x0; kwds...)

function conjgrad(A, b; kwds...)
    x = vzeros(b)
    return conjgrad!(x, A, b, x; kwds...)
end

"""
# Linear conjugate gradient

```julia
conjgrad!(x, A, b, [x0=vfill!(x,0), p, q, r]) -> x
```

finds an approximate solution to the symmetric linear system `A⋅x = b` starting
at `x0` by means of the iterative conjugate gradient method.  The result is
stored in `x` which is returned.

Argument `A` implements the symmetric positive definite linear mapping `A`, it
can be provided as a Julia array (interpreted as a general matrix, see
[`GeneralMatrix`](@ref)), as an instance of [`LinearMapping`](@ref) or as a
callable object (like a function) which is used as:

```julia
A(dst, src)
```

to overwrite `dst` with `A⋅src`.  If `A` has been implemented as a callable
object, such that `A(x)` yields `A⋅x`, then call `conjgrad!` with an inline
function:

```julia
conjgrad!(x, (dst,src) -> (dst .= A(src); return dst), b, ...)
```

If no initial variables are specified, the default is to start with all
variables set to zero.

Optional arguments `p`, `q` and `r` are writable workspace *vectors*.  On
return, `p` is the last search direction, `q = A⋅p` and `r = b - A⋅xp` with
`xp` the previous or last solution.  If provided, these workspaces must be
distinct.  All *vectors* must have the same sizes.  If all workspace vectors
are provided, no other memory allocation is necessary (unless `A` needs
to allocate some temporaries).

Provided `A` be positive definite, the solution `x` of the equations `A⋅x = b`
is also the minimum of the quadratic function:

    f(x) = (1/2) x'⋅A⋅x - b'⋅x + ϵ

where `ϵ` is an arbitrary constant.  The variations of `f(x)` between
successive iterations, the norm of the gradient of `f(x)` or the variations of
`x` may be used to decide the convergence of the algorithm (see keywords
`ftol`, `gtol` and `xtol` below).


## Saving memory

To save memory, `x` and `x0` can be the same object.  Otherwise, if no
restarting occurs (see keyword `restart` below), `b` can also be the same as
`r` but this is not recommended.


## Keywords

There are several keywords to control the algorithm:

* Keyword `ftol` specifies the function tolerance for convergence.  The
  convergence is assumed as soon as the variation of the objective function
  `f(x)` between two successive iterations is less or equal `ftol` times the
  largest variation so far.  By default, `ftol = 1e-8`.

* Keyword `gtol` specifies the gradient tolerances for convergence, it is a
  tuple of two values `(gatol, grtol)` which are the absolute and relative
  tolerances.  Convergence occurs when the Euclidean norm of the residuals
  (which is that of the gradient of the associated objective function) is less
  or equal the largest of `gatol` and `grtol` times the Euclidean norm of the
  initial residuals.  By default, `gtol = (0.0, 0.0)`.

* Keyword `xtol` specifies the variables tolerance for convergence.  The
  convergence is assumed as soon as the Euclidean norm of the change of
  variables is less or equal `xtol` times the Euclidean norm of the variables
  `x`.  By default, `xtol = 0`.

* Keyword `maxiter` specifies the maximum number of iterations which is
  practically unlimited by default.

* Keyword `restart` may be set with the maximum number of iterations before
  restarting the algorithm.  By default, `restart` is set with the smallest of
  `50` and the number of variables.  Set `restart` to at least `maxiter` if you
  do not want that any restarts ever occur.

* Keyword `strict` can be set to a boolean value (default is `true`) to specify
  whether non-positive definite operator `A` throws a `NonPositiveDefinite`
  exception or just returns the best solution found so far (with a warning if
  `quiet` is false).

* Keyword `quiet` can be set to a boolean value (default is `false`) to specify
  whether or not to print warning messages.

See also: [`conjgrad`][@ref).

"""
conjgrad!(x, A::Union{LinearMapping,AbstractArray}, b, args...; kwds...) =
    conjgrad!(x, WrappedLeftHandSideMatrix(A), b, args...; kwds...)

function conjgrad!(x, A::Mapping, b, args...; kwds...)
    is_linear(A) || bad_argument("`A` must be a linear map")
    conjgrad!(x, WrappedLeftHandSideMatrix(A), b, args...; kwds...)
end

function conjgrad!(x, A, b, x0 = vfill!(x, 0),
                   p = vcreate(x), q = vcreate(x), r = vcreate(x);
                   ftol::Real = 1e-8,
                   gtol::NTuple{2,Real} = (0.0,0.0),
                   xtol::Real = 0.0,
                   maxiter::Integer = typemax(Int),
                   restart::Integer = min(50, length(b)),
                   verb::Bool = false,
                   io::IO = stdout,
                   quiet::Bool = false,
                   strict::Bool = true)
    # Initialization.
    0 ≤ ftol < 1 ||
        bad_argument("bad function tolerance (ftol = ", ftol, ")")
    gtol[1] ≥ 0 ||
        bad_argument("bad gradient absolute tolerance (gtol[1] = ",
                     gtol[1], ")")
    0 ≤ gtol[2] < 1 ||
        bad_argument("bad gradient relative tolerance (gtol[2] = ",
                     gtol[2], ")")
    0 ≤ xtol < 1 ||
        bad_argument("bad variables tolerance (xtol = ", xtol, ")")
    restart ≥ 1 ||
        bad_argument("bad number of iterations for restarting (restart = ",
                     restart,")")
    vcopy!(x, x0)
    if maxiter < 1 && quiet && !verb
        return x
    end
    if vnorm2(x) > 0 # cheap trick to check whether x is non-zero
        # Compute r = b - A⋅x.
        A(r, x)
        vcombine!(r, 1, b, -1, r)
    else
        # Save applying A since x = 0.
        vcopy!(r, b)
    end
    local rho    :: Float64 = vdot(r, r)
    local ftest  :: Float64 = ftol
    local gtest  :: Float64 = max(gtol[1], gtol[2]*sqrt(rho))
    local xtest  :: Float64 = xtol
    local psimax :: Float64 = 0
    local psi    :: Float64 = 0
    local oldrho :: Float64
    local gamma  :: Float64

    # Conjugate gradient iterations.
    k = 0
    while true
        if verb
            if k == 0
                @printf(io, "# %s\n# %s\n",
                        "Iter.    Δf(x)       ||∇f(x)||",
                        "-------------------------------")
            end
            @printf(io, "%6d %12.4e %12.4e\n", k, psi, sqrt(rho))
        end
        k += 1
        if sqrt(rho) ≤ gtest
            # Normal convergence.
            if verb
                @printf(io, "# %s\n", "Convergence (gtest statisfied).")
            end
            break
        elseif k > maxiter
            verb && @printf(io, "# %s\n", "Too many iteration(s).")
            quiet || warn("too many (", k, " conjugate gradient iteration(s)")
            break
        end
        if rem(k, restart) == 1
            # Restart or first iteration.
            if k > 1
                # Restart.
                A(r, x)
                vcombine!(r, 1, b, -1, r)
            end
            vcopy!(p, r)
        else
            beta = rho/oldrho
            vcombine!(p, beta, p, +1, r)
        end

        # Compute optimal step size.
        A(q, p)
        gamma = vdot(p, q)
        if gamma ≤ 0
            verb && @printf(io, "# %s\n", "Operator is not positive definite.")
            strict && throw(NonPositiveDefinite("in conjugate gradient"))
            quiet || warn("operator is not positive definite")
            break
        end
        alpha = rho/gamma

        # Update variables and check for convergence.
        vupdate!(x, +alpha, p)
        psi = alpha*rho/2      # psi = f(x_{k}) - f(x_{k+1})
        psimax = max(psi, psimax)
        if psi ≤ ftest*psimax
            # Normal convergence.
            verb && @printf(io, "# %s\n", "Convergence (ftest statisfied).")
            break
        end
        if xtest > 0 && alpha*vnorm2(p) ≤ xtest*vnorm2(x)
            # Normal convergence.
            verb && @printf(io, "# %s\n", "Convergence (xtest statisfied).")
            break
        end

        # Update residuals and related quantities.
        vupdate!(r, -alpha, q)
        oldrho = rho
        rho = vdot(r, r)
    end
    return x
end
