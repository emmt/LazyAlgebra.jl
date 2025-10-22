"""
    LazyAlgebra.@callable T
    LazyAlgebra.@callable struct T ... end

make instances of concrete type `T` callable as a regular `LazyAlgebra` operator, that is
`A(x)` behaves as `A*x` and calls [`vmul(A, x)`](@ref vmul) for any operator `A` of type
`T`. The `LazyAlgebra.@callable` macro can also prefix the definition of the structure
`T`.

!!! note
    Since Julia 1.3, methods can be added to an abstract type and it is not necessary to
    use this macro for user-defined operators unless retro-compatibility is needed.

"""
macro callable(ex)
    if ex isa Symbol
        quote
            ((A::$(esc(ex)))(x) = vmul(A, x))
        end
    else
        name = try_get_struct_name_from_definition(ex)
        name === nothing && throw(ArgumentError(
            "expecting a structure name or a structure definition"))
        quote
            $(esc(ex))
	    (A::$(esc(name)))(x) = vmul(A, x)
        end
    end
end

try_get_struct_name_from_definition(ex::Any) = nothing

function try_get_struct_name_from_definition(ex::Expr)
    # In a structure definition, args[1] = true if mutable, false otherwise, ars[2] is
    # the structure name + type parameters (if any) + parent type (if any).
    ex.head === :struct || return nothing
    a = ex.args[2]
    a isa Symbol && return a # no type parameters, no parent type
    a isa Expr || return nothing
    if a.head === :curly
        # Have type parameters but no parent type.
        return a.args[1] isa Symbol ? a.args[1] : nothing
    end
    if a.head === :(<:)
        # Have parent type.
        b = a.args[1]
        b isa Symbol && return b
        b isa Expr && b.head === :curly && b.args[1] isa Symbol && return b.args[1]
    end
    return nothing
end

"""
     LazyAlgebra.@dispatch_on_multiplier sym expr

Expand to code dispatching expression `expr` depending on the value and type of the
multiplier bound to symbol `sym`.

For example:

```julia
@dispatch_on_multiplier β unsafe_vcombine!(α, x, β, y)
```

expands to (with comments removed):

```julia
if !(β isa LazyAlgebra.StaticMultiplier) && Base.iszero(β)
    unsafe_vcombine!(α, x, Neutrals.Neutral{0}()*Unitful.unit(β), y)
elseif !(β isa LazyAlgebra.StaticMultiplier) && β == Base.oneunit(β)
    unsafe_vcombine!(α, x, Neutrals.Neutral{1}()*Unitful.unit(β), y)
elseif !(β isa LazyAlgebra.StaticMultiplier) && TypeUtils.is_signed(β) && β == -Base.oneunit(β)
    unsafe_vcombine!(α, x, Neutrals.Neutral{-1}()*Unitful.unit(β), y)
else
    unsafe_vcombine!(α, x, β, y)
end
```

This can be checked thanks to `@macroexpand`:

```julia
@macroexpand LazyAlgebra.@dispatch_on_multiplier β unsafe_vcombine!(α, x, β, y)
```

"""
macro dispatch_on_multiplier(sym::Union{Symbol,QuoteNode}, expr::Expr)
    esc(:(if !($sym isa LazyAlgebra.StaticMultiplier) && Base.isequal($sym, Base.zero($sym))
              $(substitute(expr, sym => :(Neutrals.Neutral{0}()*Unitful.unit($sym))))
          elseif !($sym isa LazyAlgebra.StaticMultiplier) && Base.isequal($sym, Base.oneunit($sym))
              $(substitute(expr, sym => :(Neutrals.Neutral{1}()*Unitful.unit($sym))))
          elseif !($sym isa LazyAlgebra.StaticMultiplier) && TypeUtils.is_signed($sym) && Base.isequal($sym, -Base.oneunit($sym))
              $(substitute(expr, sym => :(Neutrals.Neutral{-1}()*Unitful.unit($sym))))
          else
              $expr
          end))
end

substitute(ex::Expr, old_new::Pair{Symbol}) = substitute!(deepcopy(ex), old_new)

function substitute!(ex::Expr, old_new::Pair{Symbol})
    for i in eachindex(ex.args)
        if ex.args[i] isa Expr
            substitute!(ex.args[i], old_new)
        elseif ex.args[i] === first(old_new)
            ex.args[i] = last(old_new)
        end
    end
    return ex
end
