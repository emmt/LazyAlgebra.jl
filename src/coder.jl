#
# coder.jl -
#
# Implement coder to automatically generate Julia code.
#
#-------------------------------------------------------------------------------
#
# This file is part of LazyAlgebra (https://github.com/emmt/LazyAlgebra.jl)
# released under the MIT "Expat" license.
#
# Copyright (c) 2017-2019 Éric Thiébaut.
#

module Coder

export
    encode,
    encode_sum_of_terms,
    generate_symbols

_macrocall(name::AbstractString, expr::Expr) =
    Expr(:macrocall, Symbol(name), (), expr)

# A token is an expression or a symbol.
const Token = Union{Expr,Symbol}

"""
```julia
generate_symbols(pfx, n)
```

yields a list of symbols of the form `pfx#` where `#` runs from 1 to `n`, if
`n` is an integer, or for all the values of `n`, if it is a range.

See also: [`encode`](@ref), [`encode_sum_of_terms`](@ref).

"""
generate_symbols(prefix::Union{AbstractString,Symbol}, number::Integer) =
    generate_symbols(prefix, 1:number)

function generate_symbols(prefix::Union{AbstractString,Symbol},
                         range::AbstractRange{<:Integer})
    map(i -> Symbol(prefix, i), range)
end

"""
```julia
encode_sum_of_terms(args)
```

yields an expression which represents the sum of the terms in `args`.

See also: [`encode`](@ref), [`generate_symbols`](@ref).

"""
encode_sum_of_terms(arg::Token) = arg

function encode_sum_of_terms(args::Union{Tuple{Vararg{Token}},Vector{<:Token}})
    argc = length(args)
    (argc == 1 ? args[1] :
     argc > 1 ? Expr(:call, :+, args...) :
     error("too few terms in `encode_sum_of_terms`"))
end


# Dictionary of syntaxes, for error reporting.
const SYNTAX = Dict(:inbounds => ":inbounds body",
                    :for      => ":for ctrl body",
                    :simd_for => ":simd_for ctrl body",
                    :if       => ":if test body ...",
                    :elseif   => "... :elseif test body ...",
                    :else     => "... :else body")

missing_arguments(head::Symbol) =
    error(string("missing argument(s) after `:", head, "` in `",
                 SYNTAX[head], "`"))

expecting_expression(token::Union{AbstractString,Symbol}, head::Symbol) =
    error(string("expecting expression for `", token, "` in `",
                 SYNTAX[head], "`"))

function invalid_argument(token::Union{AbstractString,Symbol},
                          head::Symbol, err::ErrorException)
    invalid_argument(token, head, err.msg)
end

function invalid_argument(token::Union{AbstractString,Symbol},
                          head::Symbol, err::Exception)
    invalid_argument(token, head, string(err))
end

function invalid_argument(token::Union{AbstractString,Symbol},
                          head::Symbol, reason::AbstractString)
    error(string("invalid `", token, "` in `", SYNTAX[head],
                 "` (", reason, ")"))
end

parameter(::Val{T}) where {T} = T

"""
```julia
encode(args...)
```

yields an expression corresponding to the semi-meta code in `args...`.
The result can be used in [`@generated`](@ref) functions.

The following example generate a method `sumofterms` which returns the sum of
the elements of its argument:

```julia
using LazyAlgebra.Coder
genratesumofterms(x) = genratesumofterms(typeof(x))
genratesumofterms(::Type{T}) where {T} = error("unsupported argument type \$T")
genratesumofterms(::Type{<:DenseArray{T,N}}) where {T,N} =
    encode(
        :(s = zero(T)),
        :(n = length(x)),
        :inbounds,
        (
            :simd_for, :(i in 1:n),
            (
                :(s += x[i]),
            )
        ),
        :(return s)
    )
@generated sumofterms(x::AbstractArray{T,N}) where {T,N} = genratesumofterms(x)
```

This example is purposely too simple but illustrates the kind of syntax
understood by the coder and how to separate the generated function, here
`sumofterms`, and the method in charge of generating the code, here
`generatesumofterms`, so that this latter can be directly called to check the
resulting code.  Also note that the code remains quite readable (even though
direct Julia code would have been more readable in that specific case).

More elaborated code can be generated with a minimum of readability and with
computed expressions.

The meta-code understood by the coder consists in tokens of type `Symbol` or
`Expr`.  Tokens can be grouped in tuples or in vectors to represent a block of
instructions.  Symbols are interpreted as the *keywords* of the meta-language.
The syntax understood by the coder are:

* `:for, ctrl, body` to encode a `for` loop with control part `ctrl` and
  block of instructions given by `body`;

* `:simd_for, ctrl, body` to encode a
  [SIMD](https://en.wikipedia.org/wiki/SIMD) `for` loop with control part
  `ctrl` and block of instructions given by `body`;

* `:inbounds, body` to execute the instructions in `body` (a single
  expression, or a block of meta-code) without bounds checking;

* `:if, test1, body1, [:elseif, test2, body2, ...] [:else, bodyN]` to generate
  code corresponding to an `if ... elseif ... else ...` statement in Julia.
  The number of `:elseif` clauses if arbitrary and the `:else` clause is
  optional.

"""
function encode end

# Fast parse into an expression.  Returned value is an empty expression, a
# single expression of a quoted expression depending whether arguments consist
# in 0, 1, or more expressions.
encode(::Tuple{}) = Expr(:block, ())
encode(expr::Expr) = expr
encode(args::NTuple{1,Expr}) = args[1]
encode(args::Tuple{Vararg{Expr}}) = Expr(:block, args...)
encode(args::AbstractVector) :: Expr = encode(args...)
encode(args...) :: Expr = encode(args)

# Parse a non-empty tuple not solely consisting in expressions.  Parsing is
# done in a non-recursive way (avoiding statck overflow) tkanks to the
# auxiliary function _encode.
function encode(args::Tuple) :: Expr
    code = Vector{Expr}(undef, 0)
    argc = length(args)
    k = 1 # index to "head" token
    while k ≤ argc
        expr, n = _encode(args[k], args[k+1:end])
        push!(code, expr)
        k += n + 1
    end
    if length(code) == 1
        return code[1]
    else
        # Zero or more expressions.
        return Expr(:block, code...)
    end
end


"""
The call:

```julia
_encode(head, args) -> expr, n
```

parses next token `head` in meta-code and returns the resulting expression
`expr` and the number `n` of consumed arguments in `args`, a tuple with the
remaing part of the meta-code.  An exception is thrown if the meta-code cannot
be parsed.

"""
function _encode end

# Default method which matches on syntax error.
_encode(head::T, args::Tuple) where {T} =
    error("expecting expression(s) of symbol, got $T")

# Next token is a Symbol.
_encode(head::Symbol, args::Tuple) = _encode(Val(head), args)

# Next token is an expression.
_encode(head::Expr, args::Tuple) = (head, 0)

# Next token is a tuple (or a vector) of expressions.  Make these expression in
# an expression block, append the resulting expression to the code and proceee
# with the remaing arguments.
_encode(head::Union{Tuple{Vararg{Expr}},AbstractVector{Expr}}, args::Tuple) =
    ((length(head) == 1 ? head[1] : Expr(:block, head...)), 0)

# Next token is :inbounds.
function _encode(kwd::Val{:inbounds}, args::Tuple)
    head = parameter(kwd)
    argc = length(args)
    argc ≥ 1 || missing_arguments(head)
    local body
    try
        body = encode(args[1])
    catch err
        invalid_argument(:body, head, err)
    end
    expr = _macrocall("@inbounds", body)
    return (expr, 1)
end

# Next token is :for or :simd_for.
function _encode(kwd::Union{Val{:for},Val{:simd_for}}, args::Tuple)
    head = parameter(kwd)
    argc = length(args)
    argc ≥ 2 || missing_arguments(head)
    # Get the "control" part of the loop.  This expression may have to be
    # rewritten as an assignation if the "in" keyword was used.
    isa(args[1], Expr) || expecting_expression(:ctrl, head)
    ctrl = args[1] :: Expr
    if ctrl.head == :call && length(ctrl.args) == 3 && ctrl.args[1] == :in
        # Rewrite expression.
        ctrl = Expr(:(=), ctrl.args[2], ctrl.args[3])
    end
    # Get the "boby" part of the loop.
    local body
    try
        body = encode(args[2])
    catch err
        invalid_argument(:body, head, err)
    end
    expr = (head == :for ?      Expr(:for, ctrl, body) :
            _macrocall("@simd", Expr(:for, ctrl, body)))
    return (expr, 2)
end

function _encode(kwd::Val{:if}, args::Tuple)
    # The parsing of a `if ... elseif ... else ...` construction is done in two
    # passes.  The first is to count the number of arguments involved in this
    # piece of code.  The second proceeds from the end to build up the
    # resulting expression.
    head = parameter(kwd)
    argc = length(args)
    k = 0 # index to token :if, :elseif or :else
    n = 0 # number of consumed arguments
    while true
        # After a :if or a :elseif token, we expect an expression (the "test")
        # and a "body" at args[k+1] and args[k+2].  Below we only check the test
        # part, the body is extracted (and checked) in the second pass.
        argc ≥ k + (head == :else ? 1 : 2) || missing_arguments(head)
        if head == :else
            n = k + 1
            break
        end
        isa(args[k+1], Expr) || expecting_expression(:test, head)
        # Check whether next token is :else of :elseif.
        k += 3
        if k > argc || ! isa(args[k], Symbol) || (args[k] != :elseif &&
                                                  args[k] != :else)
            n = k - 1
            break
        end
        head = args[k]
    end
    local expr::Expr
    noexpr = true # expr is undefined?
    k = n # index to last "body", start with last consumed argument
    while true
        # Encode the "body" of the :if/:elseif/:else clause.
        local body::Expr
        try
            body = encode(args[k])
        catch err
            invalid_argument(:body, head, err)
        end
        if head == :else
            expr = body
            k -= 2
        else
            expr = (noexpr ?
                    Expr(head, args[k-1], body) :
                    Expr(head, args[k-1], body, expr))
            k -= 3
        end
        if k > 2
            head = args[k-2]
        elseif k == 2
            head = :if
        else
            break
        end
        noexpr = false
    end
    return (expr, n)
end

"""
```julia
_quote(expr)
```

quotes expression `expr`.

"""
_quote(expr::Expr) = (expr.head === :block ? expr : Expr(:block, expr))

end # module
