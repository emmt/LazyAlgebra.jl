"""

The `LazyAlgebra.ConjugateGradient` module implements the linear conjugate-gradient
algorithm to iteratively solve:

    A⋅x = b    ⇐=⇒    x = argmin { f(x) = 1/2 x'⋅A⋅x - b'⋅x  + ϵ }

where `A` denotes the *left hand-side matrix* of the linear equations implemented by any
positive definite linear operator, `x` denotes the solution of the problem, `b` denotes
the *right hand-side vector* of the linear equations, and `ϵ` is an arbitrary constant.

See [`conjgrad`](@ref) and [`conjgrad!`](@ref) for simple methods to apply the algorithm.

To avoid any further allocations, you may first create a context by calling
[`ConjugateGradient.Context`](@ref) and then solve the problem with
[`ConjugateGradient.solve!`](@ref) as many times as needed, perhaps with different `A`,
`b`, or initial solution `x₀` and/or different settings.

"""
module ConjugateGradient

export issuccess, conjgrad, conjgrad!

using TypeUtils: @public
@public Context, Status, solve!, configure!

using ..LazyAlgebra
using ..LazyAlgebra: prod_type, sample
using Printf
using CEnum
using LinearAlgebra
using TypeUtils
using Neutrals

struct NonPositiveDefinite <: Exception
    arg::String
    NonPositiveDefinite(arg::AbstractString = "linear operator") = new(msg)
end
Base.showerror(io::IO, err::NonPositiveDefinite) =
    print(io, err.arg, "is non-positive definite")

"""
    ConjugateGradient.Status

Type of result returned by [`ConjugateGradient.solve!`](@ref). This is an enumeration type
whose instances can be compared as integers against another status or an integer because
status is negative for errors, zero if maximum number of iterations has been exceeded, or
positive on convergence.

The status can be converted or compared to an integer or to a symbolic value. For example:

```julia
julia> s = ConjugateGradient.TOO_MANY_ITERATIONS
TOO_MANY_ITERATIONS::Status = 0

julia> Symbol(s)
:TOO_MANY_ITERATIONS

julia> s == :TOO_MANY_ITERATIONS
true
```

Call `summary(io, status)` to print a textual description of a status value or
`summary(status)` to retrieve this summary as a string. For example:

```julia
julia> s = ConjugateGradient.CONVERGENCE_IN_G
CONVERGENCE_IN_G::Status = 2

julia> summary(s)
"convergence in the objective function gradient"
```

`LinearAlgebra.issuccess(status)` yields whether algorithm has converged.

"""
@cenum Status begin
    NOT_POSITIVE_DEFINITE = -1
    WORK_IN_PROGRESS      =  0
    TOO_MANY_ITERATIONS   =  1
    CONVERGENCE_IN_F      =  2
    CONVERGENCE_IN_G      =  3
    CONVERGENCE_IN_X      =  4
end

# Comparison of a status with an integer or a symbolic name.
for f in (:isless, :(<), :(<=), :(>), :(>=), :isequal, :(==))
    @eval begin
        Base.$f(x::Status, y::Integer) = $f(Integer(x), y)
        Base.$f(x::Integer, y::Status) = $f(x, Integer(y))
        Base.$f(x::Status, y::Symbol) = $f(x, Status(y))
        Base.$f(x::Symbol, y::Status) = $f(Status(x), y)
    end
end

let expr1 = :(throw(ArgumentError("unknown status"))),
    expr2 = :(throw(ArgumentError("unknown symbolic status")))
    for val in sort!(collect(instances(Status)), rev=true)
        expr1 = :(x === $(QuoteNode(Symbol(val))) ? $val : $expr1)
        expr2 = :(x === $val ? $(QuoteNode(Symbol(val))) : $expr2)
    end
    @eval begin
        Status(x::Symbol) = $expr1
        Base.Symbol(x::Status) = $expr2
    end
end

LinearAlgebra.issuccess(x::Status) = x > TOO_MANY_ITERATIONS

color(x::Status) =
    x ≤ NOT_POSITIVE_DEFINITE ? :red :
    x ≤ TOO_MANY_ITERATIONS ? :yellow : :green

"""
    summary(status::ConjugateGradient.Status) -> str

Give a textual description of the status returned by the conjugate gradient method.

"""
Base.summary(status::Status) =
    status === NOT_POSITIVE_DEFINITE ? "LHS operator is not positive definite" :
    status === WORK_IN_PROGRESS      ? "algorithm is running or has never been started" :
    status === TOO_MANY_ITERATIONS   ? "too many iterations" :
    status === CONVERGENCE_IN_F      ? "convergence in the objective function" :
    status === CONVERGENCE_IN_G      ? "convergence in the objective function gradient" :
    status === CONVERGENCE_IN_X      ? "convergence in the variables" :
    "unknown conjugate gradient result"

"""
    summary(io::IO, status::ConjugateGradient.Status)

Print to `io` a textual description of the status returned by the conjugate gradient method.

"""
Base.summary(io::IO, status::Status) = print(io, summary(status))

const default_norm = vnorm2
const default_fatol(::Type{T}) where {T<:Number} = zero(T)
const default_frtol(::Type{T}) where {T<:AbstractFloat} = T(1e-8)::T
const default_gatol(::Type{T}) where {T<:Number} = zero(T)
const default_grtol(::Type{T}) where {T<:AbstractFloat} = T(1e-5)::T
const default_xatol(::Type{T}) where {T<:Number} = zero(T)
const default_xrtol(::Type{T}) where {T<:AbstractFloat} = T(1e-6)::T

mutable struct Context{T<:AbstractFloat #= numerical precision for scalars =#,
                       Xv #= "vector" type for variables =#,
                       Gv #= "vector" type for dual (gradient) variables =#,
                       Xa #= element type of variables at precision `T` =#,
                       Ga #= element type of gradient at precision `T` =#,
                       Fa #= type of function value at precision `T` =#,
                       Fn #= type of the object implementing the norm =#,
                       Fo #= type of the observer object =#,
                       }
    d::Xv # current search direction
    q::Gv # q = A⋅d
    r::Gv # current residuals: r = A⋅x - b
    z::Xv # preconditioned residuals: z = M*r

    # Flags.
    preconditioning::Bool # algorithm is using a preconditioner?
    restarting::Bool      # algorithm is starting or restarting the conjugate gradient recurrence?
    stopping::Bool        # algorithm is about to stop?

    # Stopping criteria.
    status::Status
    maxiter::Int
    restart::Int
    fatol::Fa  # absolute tolerance for function reduction
    frtol::T   # relative tolerance for function reduction
    gatol::Ga  # absolute tolerance for Mahalanobis norm of the gradient
    grtol::T   # relative tolerance for gradient norm
    xatol::Xa  # absolute tolerance for norm of variables change
    xrtol::T   # absolute tolerance for norm of variables change

    # Counters.
    iterations::Int # number of iterations
    restarts::Int   # number of restarts

    # Factors.
    alpha::T   # optimal step length
    beta::T    # fraction of previous direction
    gamma::Fa  # ⟨d, q⟩
    rho::Fa    # ⟨r, z⟩, squared Mahalanobis norm of the residuals
    psi::Fa    # function reduction
    psimax::Fa # maximal function reduction so far

    # Functions.
    norm::Fn
    observer::Fo

    function Context{T,Xv,Gv,Xa,Ga,Fa,Fn,Fo}(
        ::UndefInitializer) where {T<:AbstractFloat,Xv,Gv,Xa,Ga,Fa,Fn,Fo}
        return new{T,Xv,Gv,Xa,Ga,Fa,Fn,Fo}()
    end
end

"""
    ConjugateGradient.Context{T}(x, b; kwds...) -> ctx
    ConjugateGradient.Context(x, b; kwds...) -> ctx

Return a structure with all parameters and storage for temporary variables needed for
running the linear conjugate gradient algorithm. Arguments `x` and `b` specify the
variables of the problem and the left hand-side vector of the linear system of equations
to solve. Arguments `x` and `b` are used to allocate temporary variables by calling
`similar(x)` and `similar`b`. The returned context holds no references neither on `x` nor
on `b`. Optional parameter `T <: AbstractFloat` is the floating-point precision for the
scalars of the algorithm. By default, it is given by the numerical precision of `x` and
`b` using at least `Float64` precision.

Algorithm parameters are specified the keywords accepted by
[`ConjugateGradient.configure!`](@ref) plus the following ones:

* `preconditioning` is to specify whether to allocate temporary variables to store the
  preconditioned residuals. By default, `preconditioning = false`. If false, only the
  un-preconditioned version of the algorithm can be run.

* `norm` is the norm to use for the convergence in the variables. The Euclidean norm is used
  by default. This setting cannot be changed after creation of the context.

* `observer` is the function to call every `observing` iterations as `observer(ctx, t, x)`
  with `ctx` the context, `t` the elapsed time (in seconds), and `x` the current solution.
  This function may change `ctx.status` with a value other than `WORK_IN_PROGRESS` to
  force stopping the algorithm at this iteration. Conversely, this function may set
  `ctx.status` to `WORK_IN_PROGRESS` to force the algorithm to keep iterating.

---
    ConjugateGradient.Context{T}(p, q, r, z = r; kwds...) -> ctx
    ConjugateGradient.Context(p, q, r, z = r; kwds...) -> ctx

Return a context for running the linear conjugate gradient algorithm and using the objects
`d`, `q`, `r`, and `z` for the storage. Assuming `x` and `b` denote the variables of the
problem and the left hand-side vector of the linear system of equations to solve, these
arguments should be allocated by:

    d = similar(x)
    q = similar(b)
    r = similar(b)
    z = similar(x)

If `z` is not supplied and `typeof(d) == typeof(q)`, `z = r` is assumed which also amounts
to assuming that no preconditioner will be used.

All keywords but `preconditioning` are available when at least `d`, `q`, and `r` are specified.

""" Context

# Supply default parameter `T` for scalars precision and using at least double precision.
function Context(x::Xv, b::Gv; kwds...) where {Xv,Gv}
    T = get_precision(Xv, Gv, Float64)
    return Context{T}(x, b; kwds...)
end
function Context(d::Xv, q::Gv, r::Gv; kwds...) where {Xv,Gv}
    T = get_precision(Xv, Gv, Float64)
    return Context{T}(d, q, r; kwds...)
end
function Context(d::Xv, q::Gv, r::Gv, z::Xv; kwds...) where {Xv,Gv}
    T = get_precision(Xv, Gv, Float64)
    return Context{T}(d, q, r, z; kwds...)
end

# Supply all storage.
function Context{T}(x::Xv, b::Gv;
                    preconditioning::Bool = (Xv != Gv),
                    kwds...) where {T<:AbstractFloat,Xv,Gv}
    Xv === Gv || preconditioning || throw(ArgumentError(
        "variables and residuals with different types implies to use a preconditioner"))
    d = similar(x)
    q = similar(b)
    r = similar(b)
    z = (preconditioning ? similar(x) : r)
    return Context{T}(d, q, r, z; kwds...)
end

# Supply storage for `z` assuming no-preconditioner will be used. This is only
# possible if `Xv` and `Gv` are the same "vector" types.
function Context{T}(d::Xv, q::Xv, r::Xv; kwds...) where {T<:AbstractFloat,Xv}
    return Context{T}(d, q, r, r; kwds...)
end

function Context{T}(d::Xv, q::Gv, r::Gv, z::Xv;
                    norm::Fn = default_norm,
                    observer::Fo = default_observer,
                    kwds...) where {T<:AbstractFloat,Xv,Gv,Fn,Fo}
    Xa = convert_floating_point_type(T, eltype(Xv))
    Fa = convert_floating_point_type(T, prod_type(eltype(Xv), eltype(Gv)))
    Ga = typeof(sqrt(sample(Fa)))
    ctx = Context{T,Xv,Gv,Xa,Ga,Fa,Fn,Fo}(undef)
    ctx.d = d
    ctx.q = q
    ctx.r = r
    ctx.z = z
    ctx.preconditioning = (z !== r)
    ctx.restarting = false
    ctx.stopping = false
    ctx.status = WORK_IN_PROGRESS
    ctx.maxiter = typemax(Int)
    ctx.restart = typemax(Int)
    ctx.fatol = default_fatol(Fa)
    ctx.frtol = default_frtol(T)
    ctx.gatol = default_gatol(Ga)
    ctx.grtol = default_grtol(T)
    ctx.xatol = default_xatol(Xa)
    ctx.xrtol = default_xrtol(T)
    ctx.iterations = 0
    ctx.restarts = 0
    ctx.alpha = zero(ctx.alpha)
    ctx.beta = zero(ctx.beta)
    ctx.gamma = zero(ctx.gamma)
    ctx.rho = zero(ctx.rho)
    ctx.psi = zero(ctx.psi)
    ctx.psimax = typemin(ctx.psimax)
    ctx.norm = norm
    ctx.observer = observer
    return configure!(ctx; kwds...)
end

function Base.show(io::IO, ctx::Context{T,Xv,Gv}) where {T,Xv,Gv}
    print(io, "ConjugateGradient.Context{", T, "}(\n")
    print(io, "    x #= $Xv, of size $(size(ctx.d)) =#,\n")
    print(io, "    b #= $Gv of size $(size(ctx.r)) =#;\n")
    print(io, "    preconditioning = $(ctx.z !== ctx.r),\n")
    print(io, "    maxiter = $(ctx.maxiter),\n")
    print(io, "    restart = $(ctx.restart),\n")
    print(io, "    ftol = ($(ctx.fatol), $(ctx.frtol)),\n")
    print(io, "    gtol = ($(ctx.gatol), $(ctx.grtol)),\n")
    print(io, "    xtol = ($(ctx.xatol), $(ctx.xrtol)))")
end

"""
    ConjugateGradient.configure!(ctx; kwds...) -> ctx

Configure settings in context `ctx` for the linear conjugate gradient algorithm to solve:

    A⋅x = b    ⇐=⇒    x = argmin { f(x) = 1/2 x'⋅A⋅x - b'⋅x }

All settings are configurable by keywords whose default values are given by `ctx` itself
initialized with default settings. Available keywords are:

* `maxiter` specifies the maximum number of iterations to perform which is initially
  practically unlimited.

* `restart` specifies the number of consecutive iterations before restarting the conjugate
  gradient recurrence. Restarting the algorithm is to cope with the accumulation of
  rounding errors. Initially, `restart = min(50,length(x)+1)`. Set `restart` to a value
  less or equal zero or greater than `maxiter` if you do not want that any restarts ever
  occur.

* `ftol = (fatol,frtol)` specifies the absolute and relative tolerances for stopping the
  algorithm based on the reduction of the objective fucntion `f(x)`. Initially, `ftol =
  (zero(typeof(f(x))), $(default_frtol(Float64)))`.

* `gtol = (gatol,grtol)` specifies the absolute and relative tolerances for stopping the
  algorithm based on the norm of the gradient `∇f(x) = A⋅x - b` of the objective function.
  Initially, `gtol = (zero(eltype(b)), $(default_grtol(Float64)))`.

* `xtol = (xatol,xrtol)` specifies the absolute and relative tolerances for the change in
  variables `x`. Initially, `xtol = (zero(eltype(x)), $(default_xrtol(Float64)))`.

The exact use of the convergence tolerances are explained in the documentation of
[`ConjugateGradient.solve`](@ref) which to see.

"""
function configure!(ctx::Context;
                    maxiter::Integer = ctx.maxiter,
                    restart::Integer = ctx.restart,
                    ftol::Tuple{Number,Real} = (ctx.fatol, ctx.frtol),
                    gtol::Tuple{Number,Real} = (ctx.gatol, ctx.grtol),
                    xtol::Tuple{Number,Real} = (ctx.xatol, ctx.xrtol))
    fatol = as(typeof(ctx.fatol), ftol[1])
    frtol = as(typeof(ctx.frtol), ftol[2])
    gatol = as(typeof(ctx.gatol), gtol[1])
    grtol = as(typeof(ctx.grtol), gtol[2])
    xatol = as(typeof(ctx.xatol), xtol[1])
    xrtol = as(typeof(ctx.xrtol), xtol[2])
    maxiter ≥ 0 || throw(bad_argument(
        "bad maximum number of iterations (maxiter = ", maxiter, ")"))
    fatol ≥ zero(fatol) || throw(bad_argument(
        "bad function reduction absolute tolerance (ftol[1] = ", fatol, ")"))
    𝟘 ≤ frtol < 𝟙 || throw(bad_argument(
        "bad function reduction relative tolerance (ftol[2] = ", frtol, ")"))
    gatol ≥ zero(gatol) || throw(bad_argument(
        "bad gradient absolute tolerance (gtol[1] = ", gatol, ")"))
    𝟘 ≤ grtol < 𝟙 || throw(bad_argument(
        "bad gradient relative tolerance (gtol[2] = ", grtol, ")"))
    xatol ≥ zero(xatol) || throw(bad_argument(
        "bad variables change absolute tolerance (xtol[1] = ", xatol, ")"))
    𝟘 ≤ xrtol < 𝟙 || throw(bad_argument(
        "bad variables change relative tolerance (xtol[2] = ", xrtol, ")"))
    ctx.maxiter = maxiter
    ctx.restart = restart
    ctx.fatol = fatol
    ctx.frtol = frtol
    ctx.gatol = gatol
    ctx.grtol = grtol
    ctx.xatol = xatol
    ctx.xrtol = xrtol
    return ctx
end

"""
    ConjugateGradient.solve!(ctx, A, b, x, [M=Id,] observing=𝟘) -> status

Run the (preconditioned) linear conjugate gradient algorithm to solve the system of
equations `A⋅x = b` in `x` and according to the settings in context `ctx`. Both `x` and
`ctx` may be modified on return.

Argument `ctx` is a [`ConjugateGradient.Context`](@ref) structure storing all temporary
variables and parameters of the algorithm. This argument is reusable and is required to
avoid any additional allocations. On return, `ctx.iterations` and `ctx.restarts` give the
number of iterations and restarts performed by the algorithm. The context shall not be
shared between parallel computations.

Argument `A` implements the *left hand-side (LHS) matrix* of the equations. It is used as
`LazyAlgebra.vmul!(dst,A,src)` to store in `dst` the result of applying `A` to `src` and
where `src` and `dst` are similar to arguments `x` and `b`. If none of these is suitable,
the method `OptimBase.apply!` can be extended. Note that, as `A` and `M` must be symmetric,
it may be faster to apply their adjoint.

Argument `b` is the *right hand-side (RHS) vector* of the equations. It is left unchanged.

Argument `x` stores the initial solution on entry and the estimated solution on return.

Optional argument `M` is a preconditioner. If `M` is unspecified or if `M` is the
identity, `Id`, the un-preconditioned version of the algorithm is run. The preconditioner
can be specified in various forms (as for the LHS operator `A`).

Optional argument `observing` specifies whether and how frequently to call the observer
set in `ctx`. If `observing` is less or equal zero (the default), the observer is never
called. Otherwise, the observer is called every `observing` iterations (which includes the
0-th one) and at the last iteration (with `ctx.stopping` set to `true`).


## Convergence criteria

Provided `A` be positive definite, the solution `x` of the equations `A⋅x = b` is unique and
is also the minimum of the following convex quadratic objective function:

    f(x) = (1/2) x'⋅A⋅x - b'⋅x + ϵ

where `ϵ` is an arbitrary constant. The gradient of this objective function is:

    ∇f(x) = A⋅x - b

Hence, solving `A⋅x = b` for `x` yields the minimum of `f(x)`. The variations of `f(x)`
between successive iterations, the norm of the gradient `∇f(x)`, or the norm of the
variation of variables `x` may be used to decide the convergence of the algorithm.

Let `xₖ`, `fₖ = f(xₖ)` and `∇fₖ = ∇f(xₖ)` denote the variables, the objective function and
its gradient at iteration `k`. The argument `x` gives the initial variables `x₀`. Starting
with `k = 0`, the different possibilities for the convergence of the algorithm are listed
below.

* The convergence in the function reduction between successive iterations occurs at
  iteration `k ≥ 1` if:

  ```
  fₖ₋₁ - fₖ ≤ max(fatol, frtol*max_{j ≤ k}(fⱼ₋₁ - fⱼ))
  ```

  Call `ConjugateGradient.configure!(ctx; ftol=(fatol,frtol))` to change the tolerances for
  this criterion.

* The convergence in the gradient norm occurs at iteration `k ≥ 0` if:

  ```
  ‖∇fₖ‖_M ≤ max(gatol, grtol*‖∇f_{0}‖_M)
  ```

  where `‖u‖_M = sqrt(u'⋅M⋅u)` is the Mahalanobis norm of `u` with precision matrix `M`
  which is equal to the usual Euclidean norm of `u` if no preconditioner is used or if `M`
  is the identity.

  Call `ConjugateGradient.configure!(ctx; gtol=(gatol,grtol))` to change the tolerances for
  this criterion.

* The convergence in the variables occurs at iteration `k ≥ 1` if:

  ```
  ‖xₖ - xₖ₋₁‖ ≤ max(xatol, xrtol*‖xₖ‖)
  ```

  Call `ConjugateGradient.configure!(ctx; xtol=(xatol,xrtol))` to change the tolerances for
  this criterion.

In the conjugate gradient algorithm, the objective function is always reduced at each
iteration, but be aware that the gradient and the change of variables norms are not always
decreasing.


## Returned Status

The returned value `status` may be one of (in increasing order of their integer value):

- `ConjugateGradient.NOT_POSITIVE_DEFINITE` if the left-hand-side matrix `A` is found to
  be not positive definite;

- `ConjugateGradient.TOO_MANY_ITERATIONS` if the maximum number of iterations have been
  reached;

- `ConjugateGradient.CONVERGENCE_IN_F` if convergence occurred because the function
  reduction satisfies the criterion specified by `ftol`;

- `ConjugateGradient.CONVERGENCE_IN_G` if convergence occurred because the gradient norm
  satisfies the criterion specified by `gtol`;

- `ConjugateGradient.CONVERGENCE_IN_X` if convergence occurred because the norm of the
  variation of variables satisfies the criterion specified by `xtol`.

Method `summary` may be called to get a textual explanation about the returned status.
Method `LinearAlgebra.issuccess(status)` may be called to check whether algorithm has
converged.

"""
solve!(ctx::Context{T,Xv,Gv}, A, b::Gv, x::Xv, observing::Integer) where {T,Xv,Gv} =
    solve!(ctx, x, A, b, Id, observing)

function solve!(ctx::Context{T,Xv,Gv,Xa,Ga,Fa}, A, b::Gv, x::Xv, M = Id,
                observing::Integer = 𝟘) where {T,Xv,Gv,Xa,Ga,Fa}
    # Get workspace variables.
    d, q, r, z = ctx.d, ctx.q, ctx.r, ctx.z
    ctx.preconditioning = !(z === r)
    M === Id || ctx.preconditioning || error(
        "`M` must be the identity when work-spaces `z` and `r` are the same")

    # Initialize factors and local variables.
    ctx.alpha = zero(ctx.alpha)
    ctx.rho = zero(ctx.rho)
    rhoprev = ctx.rho
    ctx.psi = zero(ctx.psi)    # phi = f(xₖ₋₁) - f(xₖ) ≥ zero(Fa)
    ctx.psimax = typemin(ctx.psimax)
    xtest = (ctx.xatol > zero(ctx.xatol) || ctx.xrtol > 𝟘) # test for convergence in x?
    gtest = zero(ctx.gatol) # this value initialized with 1st residuals
    t0 = time()

    # Conjugate gradient iterations.
    ctx.iterations = 0
    ctx.restarts = 0
    ctx.restarting = true # starting or restarting conjugate gradient recurrence?
    ctx.stopping = false # algorithm is about to stop?
    ctx.status = WORK_IN_PROGRESS # will be set when convergence detected
    while true
        # If status is not WORK_IN_PROGRESS, then convergence in x or in f holds or an
        # error has occurred, hence updating r, z, rho, etc. can be skipped to save
        # computations unless there are no errors and the observer may be called that may
        # require these values.
        if ctx.status == WORK_IN_PROGRESS || (ctx.status > WORK_IN_PROGRESS && observing > 𝟘)
            # Compute or update the residuals.
            if ctx.restarting
                # Compute residuals.
                if ctx.iterations > 0 || !iszero(vnorm2(x))
                    # Compute r = b - A⋅x using r to temporarily store A⋅x.
                    vcombine!(r, 𝟙, b, -𝟙, vmul!(r, A, x))
                else
                    # Spare applying A since x = 0.
                    vcopy!(r, b)
                end
                if ctx.iterations > 0
                    ctx.restarts += 1
                end
            else
                # Update residuals.
                vupdate!(r, -ctx.alpha, q) # r -= α⋅q
            end
            if ctx.preconditioning
                # Apply preconditioner.
                vmul!(z, M, r) # z = M*r
            end
            rhoprev = ctx.rho
            ctx.rho = vdot(r, z) # rho = ⟨r, z⟩ = ‖r‖_M^2
            if ctx.iterations == 0
                gtest = tolerance(ctx.gatol, ctx.grtol, sqrt(ctx.rho))
            end
            if sqrt(ctx.rho) ≤ gtest
                # Normal convergence in the gradient norm.
                ctx.status = CONVERGENCE_IN_G
            elseif ctx.iterations ≥ ctx.maxiter
                # Maximum number of iterations exhausted.
                ctx.status = TOO_MANY_ITERATIONS
            end
        end
        ctx.stopping = (ctx.status != WORK_IN_PROGRESS)
        if observing > 𝟘 && (ctx.stopping || ctx.iterations % observing == 0)
            # Call observer which may change the status.
            ctx.observer(ctx, time() - t0, x)
            ctx.stopping = (ctx.status != WORK_IN_PROGRESS)
        end
        if ctx.stopping
            return ctx.status
        end

        # Compute search direction d.
        if ctx.restarting
            # Restarting or first iteration.
            ctx.beta = zero(ctx.beta)
            vcopy!(d, z)
        else
            # Apply recurrence.
            ctx.beta = ctx.rho/rhoprev
            vcombine!(d, 𝟙, z, ctx.beta, d)
        end

        # Compute q = A⋅d and gamma = ⟨d, q⟩
        vmul!(q, A, d)
        ctx.gamma = vdot(d, q)

        if ctx.gamma > zero(ctx.gamma)
            # Compute optimal step size alpha = rho/gamma and update variables x.
            ctx.alpha = ctx.rho/ctx.gamma
            vupdate!(x, ctx.alpha, d) # x += α⋅d

            # Check for convergence in f or in x.
            ctx.psi = ctx.alpha*ctx.rho/2 # psi = f(xₖ) - f(xₖ₊₁) ≥ 0
            ctx.psimax = max(ctx.psimax, ctx.psi)
            if ctx.psi ≤ tolerance(ctx.fatol, ctx.frtol, ctx.psimax)
                # Normal convergence in the function reduction.
                ctx.status = CONVERGENCE_IN_F
            elseif xtest && ctx.alpha*ctx.norm(d) ≤ tolerance(ctx.xatol, ctx.xrtol, x; norm=ctx.norm)
                # Normal convergence in the variables.
                ctx.status = CONVERGENCE_IN_X
            end

            # Increment number of iterations and decide whether restarting or not the conjugate
            # gradient recurrence.
            ctx.iterations += 1
            ctx.restarting = ctx.restart > 0 && ctx.iterations % ctx.restart == 0
        else
            # LHS matrix A is not positive definite.
            ctx.status = NOT_POSITIVE_DEFINITE
            ctx.alpha = NaN
        end
    end
end

# The stealth observer does nothing.
function stealth_observer(ctx, args...; kwds...)
    return nothing
end

# TODO put units in the head line like (ms)
function default_observer(ctx, t, x)
    io = stdout
    t *= 1E3 # elapsed time in ms
    if ctx.preconditioning
        if ctx.iterations == 0
            print(io,
                  "# Iter.   Time (ms)       Δf(x)         ‖∇f(x)‖     ‖∇f(x)‖_M\n",
                  "# -------------------------------------------------------------\n")
        end
        @printf(io, "%7d %11.3f %16.8e %12.4e %12.4e\n",
                ctx.iterations, t, ctx.psi, vnorm2(ctx.r), sqrt(ctx.rho))
    else
        if ctx.iterations == 0
            print(io,
                  "# Iter.   Time (ms)       Δf(x)         ‖∇f(x)‖\n",
                  "# ------------------------------------------------\n")
        end
        @printf(io, "%7d %11.3f %16.8e %12.4e\n",
                ctx.iterations, t, ctx.psi, sqrt(ctx.rho))
    end
    if ctx.stopping
        # Algorithm is about to stop.
        print(io, "# ")
        printstyled(io, summary(ctx.status); color = color(ctx.status))
        println(io)
    end
    return nothing
end

"""
    ConjugateGradient.tolerance(atol::Number, rtol::Real, var) -> tol

Return the tolerance given absolute and relative tolerances `atol` and `rtol` and
a variation `var` which may be a number or an array. The result is computed as:

    tol = max(zero(atol), atol, rtol*norm(var))

taking care of a number of details:

- type-stability and numerical precision of the result;

- if tolerances `atol` or `rtol` are negative or NaNs, they are assumed to be zero;

- to save computations, the computation of `norm(var)` is avoided if `rtol > 0` does not
  hold.

Keyword `norm` can be used to specify another norm than the default Euclidean norm. Beware
that the provided norm must be applicable to numbers and must consider arrays as simple
vectors. These conditions hold for the norms [`LazyAlgebra.vnorm1`], [`LazyAlgebra.vnorm2`],
and [`LazyAlgebra.vnorminf`].

"""
function tolerance(atol::Number, rtol::Real, var::Union{Number,AbstractArray{<:Number}};
                   norm = vnorm2)
    # Infer numerical precision.
    T = get_precision(typeof(atol), typeof(rtol), eltype(var))
    # Compute `tol = max(zero(atol), atol)` at the numerical precision and taking care of
    # NaNs.
    tol = positive_part(convert_floating_point_type(T, atol))
    # Return max(tol, rtol*norm(var)) avoiding computing the norm if possible.
    (isnan(rtol) || rtol ≤ zero(rtol)) && return tol
    len = var isa Number ? abs(var) : norm(var)
    return max(tol, oftype(tol, rtol*len))
end

# Other possibility relying on IEEE rules for NaNs: x ≥ 0 ? x : 0
positive_part(x::Number) = ifelse(isnan(x) | (x < zero(x)), zero(x), x)

"""
    bad_argument(args...)

yields an `ArgumentError` exception with error message given by `args...` converted into a
string.

"""
@noinline bad_argument(msg::AbstractString) = ArgumentError(msg)
@noinline bad_argument(args...) = bad_argument(string(args...))

"""
    conjgrad(A, b, x₀ = vzeros(b)) -> x

Approximately solve the symmetric linear system `A⋅x = b` starting at `x₀` by means of the
iterative conjugate gradient method. The returned solution `x` is a new object similar to
`x₀`.

See [`conjgrad!`](@ref) for accepted keywords and more details.

"""
conjgrad(A, b; kwds...) = conjgrad!(A, b, vzeros(b); kwds...)
conjgrad(A, b, x₀; kwds...) = conjgrad!(A, b, vcopy(x₀); kwds...)

"""
    conjgrad!(A, b, x; preconditioner=Id) -> x

Approximately solve the symmetric linear system `A⋅x = b` starting at `x` by means of the
iterative conjugate gradient method. The result is stored in `x` which is returned.

Argument `A` implements a symmetric positive definite linear operator, it is used as:

```julia
vmul!(dst, A, src) -> dst
```

to overwrite `dst` with `A⋅rc` and yield `dst`. Linear operator `A` can be provided as a
Julia array (interpreted as a *flexible matrix*, see [`FlexibleMatrix`](@ref)), as an
instance of [`Operator`](@ref), or as any object for which [`vmul!`](@ref) is extended as
shown above.

If no initial variables `x₀` are specified, the default is to start with all variables set
to zero.

Optional arguments `p`, `q`, and `r` are writable workspace *vectors*. On return, `p` is
the last search direction, `q = A⋅p`, and `r = b - A⋅xp` with `xp` the previous or last
solution. If provided, these workspaces must be distinct. All *vectors* must have the same
axes. If all workspace vectors are provided, no other memory allocation is necessary
(unless `A` needs to allocate some temporaries).

Provided `A` be positive definite, the solution `x` of the equations `A⋅x = b` is also the
minimum of the quadratic function:

    f(x) = (1/2) x'⋅A⋅x - b'⋅x + ϵ

where `ϵ` is an arbitrary constant. The variations of `f(x)` between successive
iterations, the norm of the gradient of `f(x)` or the variations of `x` may be used to
decide the convergence of the algorithm (see keywords `ftol`, `gtol` and `xtol` below).

## Saving memory

To save memory, `x` and `x₀` can be the same object. Otherwise, if no restarting occurs
(see keyword `restart` below), `b` can also be the same as `r` but this is not
recommended.


## Keywords

There are several keywords to control the algorithm:

* FIXME `preconditioner`

See also: [`conjgrad`][@ref).

"""
function conjgrad!(A, b, x;
                   preconditioner = Id,
                   observing::Integer = 𝟘,
                   kwds...)
    ctx = Context(x, b; preconditioning = (preconditioner !== Id), kwds...)
    status = solve!(ctx, x, A, b, preconditioner, observing)
    status == NOT_POSITIVE_DEFINITE && throw(NonPositiveDefinite("LHS matrix `A`"))
    return status, x
end

end # module
