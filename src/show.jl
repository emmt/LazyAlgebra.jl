# Context for showing a representation of operators.
struct ShowContext{M<:Union{MIME,Nothing},S<:IO}
    io::S
    mime::M
end
ShowContext(io::IO) = ShowContext(io, nothing)

function Base.show(ctx::ShowContext, args...)
    if ctx isa ShowContext{Nothing}
        show(ctx.io, args...)
    else
        show(ctx.io, ctx.mime, args...)
    end
    nothing
end
Base.write(ctx::ShowContext, args...) = write(ctx.io, args...)
Base.print(ctx::ShowContext, args...) = print(ctx.io, args...)
Base.println(ctx::ShowContext, args...) = println(ctx.io, args...)

# MIME"text/plain" is for the REPL.
Base.show(io::IO, A::Operator) = show(io, typeof(A))
Base.show(io::IO, mime::MIME"text/plain", A::Operator) =
    print(io, parameterless(typeof(A)))

for T in (:Sum, :Prod, :Adjoint, :Inverse)
    @eval begin
        Base.show(io::IO, mime::MIME, A::$T) = show(ShowContext(io, mime), A)
        Base.show(io::IO, mime::MIME"text/plain", A::$T) = show(ShowContext(io, mime), A)
        Base.show(io::IO, A::$T) = show(ShowContext(io), A)
    end
end

function Base.show(ctx::ShowContext, A::Adjoint)
    B = parent(A)
    show_paren(ctx, B, B isa Union{Sum,Prod,Adjoint})
    write(ctx, '\'')
end

function Base.show(ctx::ShowContext, A::Inverse)
    write(ctx, "inv(")
    show(ctx, parent(A))
    write(ctx, ')')
end

function Base.show(ctx::ShowContext, A::Prod)
    protect = A[2] isa Sum
    if A[1] isa Number
        λ = A[1]
        if λ == -1
            write(ctx, '-')
        elseif λ == 1
            protect = false
        else
            show_multiplier(ctx, λ)
            write(ctx, '*')
        end
    else
        show_in_prod(ctx, A[1])
        write(ctx, '*')
    end
    show_paren(ctx, A[2], protect)
end

function Base.show(ctx::ShowContext, A::Sum)
    show(ctx, A[1])
    show_next_in_sum(ctx, A[2])
end

# Show a multiplier (surrounded by parentheses if not a real, i.e. if a complex).
show_multiplier(ctx::ShowContext, λ::Number) = show_paren(ctx, λ, !isreal(λ))

# Show a term in a product.
show_in_prod(ctx::ShowContext, A::Operator) = show_paren(ctx, A, A isa Sum)

# Show a term optionally enclosed by parentheses.
function show_paren(ctx::ShowContext, x, paren::Bool)
    paren && print(ctx, '(')
    show(ctx, x)
    paren && print(ctx, ')')
    nothing
end

# `show_next_in_sum` shows a term in a sum (not the first one).
function show_next_in_sum(ctx::ShowContext, A::Operator)
    write(ctx, " + ")
    show(ctx, A)
end

function show_next_in_sum(ctx::ShowContext, A::Sum)
    show_next_in_sum(ctx, A[1])
    show_next_in_sum(ctx, A[2])
end

function show_next_in_sum(ctx::ShowContext, A::Prod)
    if A[1] isa Number
        λ = A[1]
        if isreal(λ) && λ < zero(λ)
            λ = -λ
            write(ctx, " - ")
        else
            write(ctx, " + ")
        end
        if λ != one(λ)
            show_multiplier(ctx, λ)
            write(ctx, '*')
        end
    else
        write(ctx, " + ")
        show_in_prod(ctx, A[1])
        write(ctx, '*')
    end
    show_in_prod(ctx, A[2])
end

print_axis(ctx::ShowContext, args...) = print_axis(ctx.io, args...)
print_axes(ctx::ShowContext, args...) = print_axes(ctx.io, args...)
print_shape(ctx::ShowContext, args...) = print_shape(ctx.io, args...)

print_axis(io::IO, dim::Integer) =
    print(io, "1:", max(0, Int(dim)))

print_axis(io::IO, rng::AbstractUnitRange{<:Integer}) =
    print(io, first(rng), ':', last(rng))

function print_axes(io::IO, rngs::Tuple{Vararg{AbstractUnitRange{<:Integer}}})
    write(io, '(')
    for (i, rng) in enumerate(rngs)
        i > 1 && write(io, ", ")
        print_axis(io, rng)
    end
    length(rngs) == 1 && write(io, ',')
    write(io, ')')
    nothing
end

function print_shape(io::IO, shape::ArrayShape)
    if shape isa Tuple{Vararg{Union{Integer,Base.OneTo}}}
        show(io, as_array_size(shape))
    else
        print_axes(io, shape)
    end
    nothing
end
