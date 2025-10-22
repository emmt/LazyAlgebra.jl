# Notes about implementation and development

## Operations on vectors

Computing linear combination of vectors like `z <- α*x + β*y` involves several steps:

1. Check arguments for compatibility of indices, types, and units. After these checks,
   bound checking for indices is not needed and *unsafe* methods can be called.

2. Convert multipliers `α` and `β` so that the precision of computations is driven by that
   of `x` and `y`. This done by the `convert_multiplier` function.

3. Dispatch on the value of the multipliers `α` and `β` to convert them to fast *neutral*
   numbers if possible. For example, `α` is converted to `𝟘*unit(α)` if `α == zero(α)`
   holds, to `𝟙*unit(α)` if `α == oneunit(α)` holds, and to to `-𝟙*unit(α)` if `α ==
   -oneunit(α)` holds. This is required to speedup computations in these special cases
   without requiring to write different versions of the code to handle each possible case.

The units of the multipliers, if any, must be preserved by the possible conversions and
the type of their result must be inferable.

A first version was:

```julia
next(::Val{:alpha}) = Val(:none)
next(::Val{:alpha_beta}) = Val(:beta)
next(::Val{:beta}) = Val(:none)
next(::Val{:beta_alpha}) = Val(:alpha)

function vcombine!(z::AbstractArray,
                   α::Number, x::AbstractArray,
                   β::Number, y::AbstractArray)
    # Check arguments indices, types, and units.
    @assert_same_axes x y z
    _ = convert(eltype(z), zero(α)*zero(eltype(x)) + zero(β)*zero(eltype(y)))::eltype(z)
    # Deal with multipliers.
    unsafe_vcombine!(Val(:alpha_beta), z, α, x, β, y)
    return z
end

function unsafe_vcombine!(stage::Val,
                          z::AbstractArray,
                          α::Number, x::AbstractArray,
                          β::Number, y::AbstractArray)
    if stage isa Union{Val{:alpha}, Val{:alpha_beta}}
        α = convert_multiplier(α, eltype(x))
        @dispatch_on_multiplier α unsafe_vcombine!(next(stage), z, α, x, β, y)
    elseif stage isa Union{Val{:beta}, Val{:beta_alpha}}
        β = convert_multiplier(β, eltype(y))
        @dispatch_on_multiplier β unsafe_vcombine!(next(stage), z, α, x, β, y)
    elseif stage isa Val{:none}
        unsafe_vcombine!(z, α, x, β, y)
    else
        throw_unexpected_stage(stage)
    end
end

function unsafe_vcombine!(z::AbstractArray,
                          α::Number, x::AbstractArray,
                          β::Number, y::AbstractArray)
    @inbounds @fastmath @simd for i in eachindex(x, y, z)
        z[i] = α*x[i] + β*y[i]
    end
end
```

Compatibility of indices is asserted by the macro `assert_same_axes` from the
[`ArrayTools`](https://github.com/emmt/ArrayTools.jl) package.

Compatibility of types and units is checked by:

```julia
_ = convert(eltype(z), zero(α)*zero(eltype(x)) + zero(β)*zero(eltype(y)))::eltype(z)
```

which just ensures that expression `α*x[i] + β*y[i]` can be computed and stored in `z[i]`
for any index `i`.

Then conversion of multipliers and dispatching on their values is done by an auxiliary
method that call itself with an updated `stage` argument recording the tasks that remain
to be performed. Since `stage` is a singleton, the code in this auxiliary method should be
trimmed by the optimizer.

The macro `@dispatch_on_multiplier sym expr` expands to code where expression `expr` may
be evaluated with special values of the the multiplier bound to symbol `sym`. For example:

```julia
@dispatch_on_multiplier α unsafe_vcombine!(z, α, x, β, y)
```

expands to (with comments and module prefixes removed for clarity):

```julia
  if isequal(α, iszero(α))
      unsafe_vcombine!(z, 𝟘*unit(α), x, β, y)
  elseif isequal(α, oneunit(α))
      unsafe_vcombine!(z, 𝟙*unit(α), x, β, y)
  elseif isequal(α, -oneunit(α))
      unsafe_vcombine!(z, -𝟙*unit(α), x, β, y)
  else
      unsafe_vcombine!(α, x, β, y)
  end
```

where `𝟘` and `𝟙` are *neutral* numbers provided by the
[`Neutrals`](https://github.com/emmt/Neutrals.jl) package while the `unit` method is from
the [`Unitful`](https://github.com/JuliaPhysics/Unitful.jl) package. Using this macro
improve clarity and avoid errors in typing similar expressions.

In spite of the complexity of the machinery (there are 4×4 = 16 different possibilities
depending on the values of `α` and `β`), the above code is both simple and understandable.
Benchmarking the code with Julia 1.12.1 on an Intel Xeon Silver 4215R CPU at 3.20GHz
yields:

```julia
julia> using BenchmarkTools, ThreadPinning, LazyAlgebra;

julia> T = Float32; n = 10_123; x = rand(T,n); y = rand(T,n); z = similar(x);

julia> pinthreads(:cores);

julia> α, β = 1.5, -2.3; @btime vcombine!($z, $α, $x, $β, $y);
  1.060 μs (2 allocations: 32 bytes)

julia> α, β = 1, -1; @btime vcombine!($z, $α, $x, $β, $y);
  1.047 μs (0 allocations: 0 bytes)

julia> α, β = 1, 0; @btime vcombine!($z, $α, $x, $β, $y);
  762.088 ns (0 allocations: 0 bytes)

julia> α, β = 0, 1; @btime vcombine!($z, $α, $x, $β, $y);
  789.939 ns (0 allocations: 0 bytes)

julia> @btime copyto!(z, x);
  779.747 ns (0 allocations: 0 bytes)

julia> α, β = 0, 0; @btime vcombine!($z, $α, $x, $β, $y);
  558.123 ns (0 allocations: 0 bytes)

julia> @btime fill!(z, zero(eltype(z)));
  813.830 ns (0 allocations: 0 bytes)

```

The performances are very good: nearly 30 Gflops, that is about 9 operations per cycle.
Also note that this generic method is as fast as the equivalent `copyto!` when `α = 1` and
`β = 0` (or `α = 0` and `β = 1`) and faster than `fill!` when `α = 0` and `β = 0`. These
performances may be increased by using the `@turbo` macro of the
[LoopVectorization](https://github.com/JuliaSIMD/LoopVectorization.jl) package.

There is however an issue: the code may allocate memory, not much (at most 32 bytes in 2
allocations) but still. This must be avoided for real-time applications.

The solution is to split the auxiliary methods in 3 distinct pieces:

```julia
function unsafe_vcombine!(::Val{:alpha_beta},
                          z::AbstractArray,
                          α::Number, x::AbstractArray,
                          β::Number, y::AbstractArray)
    α = convert_multiplier(α, eltype(x))
    @dispatch_on_multiplier α unsafe_vcombine!(Val(:beta), z, α, x, β, y)
end
function unsafe_vcombine!(::Val{:alpha},
                          z::AbstractArray,
                          α::Number, x::AbstractArray,
                          β::Number, y::AbstractArray)
    α = convert_multiplier(α, eltype(x))
    @dispatch_on_multiplier α unsafe_vcombine!(z, α, x, β, y)
end
function unsafe_vcombine!(::Val{:beta},
                          z::AbstractArray,
                          α::Number, x::AbstractArray,
                          β::Number, y::AbstractArray)
    β = convert_multiplier(β, eltype(y))
    @dispatch_on_multiplier β unsafe_vcombine!(z, α, x, β, y)
end
```

The `Val{:alpha}` one is only needed if some other method want to call `unsafe_vcombine!`
at this specific stage, i.e., after having changing the value of `α` but not that of `β`
which requires to convert `α` and dispatch on its value.

This change solves the issue:

```julia
julia> α, β = 1.5f0, -2.3f0; @btime vcombine!($z, $α, $x, $β, $y);
  1.034 μs (0 allocations: 0 bytes)

```
