#
# conjgrad.jl -
#
# Linear conjugate-gradient.
#
#-------------------------------------------------------------------------------
#
# This file is part of the LazyAlgebra package released under the MIT "Expat"
# license.
#
# Copyright (c) 2017-2018 Éric Thiébaut.
#

"""
```julia
conjgrad(A, b, x0=vzeros(b)) -> x
```

solves the linear system `A⋅x = b` starting at `x0` by means of the iterative
conjugate gradient method.  Argument `A` implements a symmetric positive
definite linear map, `A` can be a Julia array (interpreted as a general matrix,
see [`GeneralMatrix`](@ref)), an instance of [`LinearMapping`](@ref) or a
callable object which is used as:

```julia
A(q, p)
```

to overwrite `q` with `q = A⋅p`.  This method can be extended to be specialized
for the specific type of `A`.

See [`conjgrad!`][@ref) for accepted keywords and more details.

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
conjgrad!(x, A, b, [x0=vzeros(b), p, q, r]) -> x
```

solves the linear system `A⋅x = b` starting at `x0` by means of the iterative
conjugate gradient method.  The result is stored in `x` which is returned.
Argument `A` implements a symmetric positive definite linear map, `A` can be a
Julia array (interpreted as a general matrix, see [`GeneralMatrix`](@ref)), an
instance of [`LinearMapping`](@ref) or a callable object which is used as:

```julia
A(q, p)
```

to overwrite `q` with `q = A⋅p`.  This method can be extended to be specialized
for the specific type of `A`.

If no initial variables are specified, the default is to start with all
variables set to zero.

Optional arguments `p`, `q` and `r` are writable workspace *vectors*.  On
return, `p` is the last search direction, `q = A⋅p` and `r = b - A.xp` with
`xp` the previous or last solution.  If provided, these workspaces must be
distinct.  All *vectors* must have the same sizes.  If all workspace vectors
are provided, no other memory allocation is necessary.

Providing `A` is positive definite, the solution `x` of the equations
`A⋅x = b` is also the minimum of the quadratic function:

    f(x) = (1/2) x'⋅A⋅x - b'⋅x + ϵ

where `ϵ` is an arbitrary constant.  The variations of `f(x)` between
successive iterations or the norm of the gradient `f(x)` of may be used to
decide the convergence of the algorithm (see keywords `ftol` and `gtol` below).


## Saving memory

To save memory, `x` and `x0` can be the same object.  Otherwise, if no
restarting occurs (see keyword `restart` below), `b` can also be the same as
`x`.


## Keywords

There are several keywords to control the algorithm:

* Keyword `ftol` specifies the function tolerance for convergence.  The
  convergence is assumed as soon as the variation of the objective function
  `f(x)` between two successive iterations is less or equal `ftol` times the
  largest variation so far.  By default, `ftol = 1e-7`.

* Keyword `gtol` specifies the gradient tolerances for convergence, it is a
  tuple of two values `(gatol, grtol)` where `gatol` and `grtol` are the
  absolute and relative tolerances.  Convergence occurs when the Euclidean norm
  of the residuals (which is that of the gradient of the associated objective
  function) is less or equal the largest of `gatol` and `grtol` times the
  Euclidean norm of the initial residuals.  By default, `gtol = (0.0, 0.0)`.

* Keyword `xtol` specifies the variables tolerance for convergence.  The
  convergence is assumed as soon as the Euclidean norm of the change of
  variables is less or equal `xtol` times the Euclidean norm of the variables
  `x`.  By default, `xtol = 0`.

* Keyword `maxiter` specifies the maximum number of iterations which is
  practically unlimited by default.

* Keyword `restart` may be set with the mwimum number of iterations before
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
    conjgrad!(x, (dst, src) -> apply!(dst, A, src), b, args...; kwds...)

function conjgrad!(x, A::Mapping, b, args...; kwds...)
    is_linear(A) || throws(ArgumentError("`A` must be a linear map"))
    conjgrad!(x, (dst, src) -> apply!(dst, A, src), b, args...; kwds...)
end

function conjgrad!(x, A, b, x0 = vzeros(b),
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
        throw(ArgumentError("bad function tolerance (ftol = $ftol)"))
    gtol[1] ≥ 0 ||
        throw(ArgumentError("bad gradient absolute tolerance (gtol[1] = ",
                            gtol[1], ")"))
    0 ≤ gtol[2] < 1 ||
        throw(ArgumentError("bad gradient relative tolerance (gtol[2] = ",
                            gtol[2], ")"))
    0 ≤ xtol < 1 ||
        throw(ArgumentError("bad variables tolerance (xtol = $xtol)"))
    restart ≥ 1 ||
        throw(ArgumentError("bad number of iterations for restarting ",
                            "(restart = $restart)"))
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
    rho = Float64(vdot(r, r))
    ftest = Float64(ftol)
    gtest = Float64(max(gtol[1], gtol[2]*sqrt(rho)))
    xtest = Float64(xtol)
    psimax = Float64(0)
    psi = Float64(0)
    oldrho = Float64(0)

    # Conjugate gradient iterations.
    k = 0
    while true
        if verb
            if k == 0
                @printf(io, "# %s\n# %s\n",
                        "Iter.    Δf(x)       ||∇f(x)||",
                        "-------------------------------")
            end
            @printf(io, "%6d %12.4e %12.4e\n",
                    k, Float64(psi), Float64(sqrt(rho)))
        end
        k += 1
        if sqrt(rho) ≤ gtest
            # Normal convergence.
            if verb
                @printf(io, "# %s\n", "Convergence (gtest statisfied).")
            end
            break
        elseif k > maxiter
            if verb
                @printf(io, "# %s\n", "Too many iteration(s).")
            end
            if !quiet
                @warn("too many ($k) conjugate gradient iteration(s)")
            end
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
        A(q, p)
        gamma = Float64(vdot(p, q))
        if gamma ≤ 0
            if verb
                @printf(io, "# %s\n", "Operator is not positive definite.")
            end
            strict && throw(NonPositiveDefinite("in conjugate gradient"))
            if !quiet
                @warn("operator is not positive definite")
            end
            break
        end
        alpha = rho/gamma
        vupdate!(x, +alpha, p)
        vupdate!(r, -alpha, q) # FIXME: spare this update if next tests succeed
        psi = alpha*rho/2      # psi = f(x_{k}) - f(x_{k+1})
        psimax = max(psi, psimax)
        if psi ≤ ftest*psimax
            # Normal convergence.
            if verb
                @printf(io, "# %s\n", "Convergence (ftest statisfied).")
            end
            break
        end
        if xtest > 0 && alpha*vnorm2(p) ≤ xtest*vnorm2(x)
            # Normal convergence.
            if verb
                @printf(io, "# %s\n", "Convergence (xtest statisfied).")
            end
            break
        end
        oldrho = rho
        rho = Float64(vdot(r, r))
    end
    return x
end
