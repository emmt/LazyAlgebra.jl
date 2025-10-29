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

for S in (:Sum, :Prod, :Scaled, :Adjoint, :Inverse)
    @eval begin
        Base.show(io::IO, mime::MIME, A::$S) = show(ShowContext(io, mime), A)
        Base.show(io::IO, mime::MIME"text/plain", A::$S) = show(ShowContext(io, mime), A)
        Base.show(io::IO, A::$S) = show(ShowContext(io), A)
    end
end

function Base.show(ctx::ShowContext, A::Adjoint)
    B = parent(A)
    show_paren(ctx, B, B isa Union{Sum,Prod,Scaled,Adjoint})
    write(ctx, '\'')
    return nothing
end

for (f, W) in (:transpose => :Transpose,
               :conj      => :Conjugate,
               :inv       => :Inverse)
    @eval begin
        function Base.show(ctx::ShowContext, A::$W)
            write(ctx, $(string(f,"(")))
            show(ctx, parent(A))
            write(ctx, ')')
            return nothing
        end
    end
end

function Base.show(ctx::ShowContext, A::Scaled)
    λ = A[1]
    if λ == -1
        write(ctx, '-')
    elseif λ == 1
        protect = false
    else
        show_multiplier(ctx, λ)
        write(ctx, '*')
    end
    protect = A[2] isa Sum
    show_paren(ctx, A[2], protect)
    return nothing
end

function Base.show(ctx::ShowContext, A::Prod)
    show_in_prod(ctx, A[1])
    write(ctx, '*')
    protect = A[2] isa Sum
    show_paren(ctx, A[2], protect)
    return nothing
end

function Base.show(ctx::ShowContext, A::Sum)
    show(ctx, A[1])
    show_next_in_sum(ctx, A[2])
    return nothing
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
    return nothing
end

# `show_next_in_sum` shows a term in a sum (not the first one).
function show_next_in_sum(ctx::ShowContext, A::Operator)
    write(ctx, " + ")
    show(ctx, A)
    return nothing
end

function show_next_in_sum(ctx::ShowContext, A::Sum)
    show_next_in_sum(ctx, A[1])
    show_next_in_sum(ctx, A[2])
    return nothing
end

function show_next_in_sum(ctx::ShowContext, A::Scaled)
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
    show_in_prod(ctx, A[2])
    return nothing
end

function show_next_in_sum(ctx::ShowContext, A::Prod)
    write(ctx, " + ")
    show_in_prod(ctx, A[1])
    write(ctx, '*')
    show_in_prod(ctx, A[2])
    return nothing
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
    return nothing
end

function print_shape(io::IO, shape::ArrayShape)
    if shape isa Tuple{Vararg{Union{Integer,Base.OneTo}}}
        show(io, as_array_size(shape))
    else
        print_axes(io, shape)
    end
    return nothing
end
