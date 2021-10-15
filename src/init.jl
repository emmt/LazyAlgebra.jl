using Requires

function __init__()
    @require Unitful="1986cc42-f94f-5a68-af5c-568840ba703d" begin
        multiplier_type(::Unitful.AbstractQuantity{T}) where {T} = T
    end
end
