TypeUtils.get_precision(x, y, z...) = get_precision(get_precision(x, y), z...)
TypeUtils.get_precision(x, y) =
    get_precision(get_precision(x)::Type{<:AbstractFloat},
                  get_precision(y)::Type{<:AbstractFloat})::Type{<:AbstractFloat}
TypeUtils.get_precision(::Type{AbstractFloat}, ::Type{AbstractFloat}) = AbstractFloat
TypeUtils.get_precision(::Type{T}, ::Type{AbstractFloat}) where {T<:AbstractFloat} = T
TypeUtils.get_precision(::Type{AbstractFloat}, ::Type{T}) where {T<:AbstractFloat} = T
TypeUtils.get_precision(::Type{S}, ::Type{T}) where {S<:AbstractFloat,T<:AbstractFloat} =
    promote_type(S, T)::Type{<:AbstractFloat}

# These may be unnecessary. At least we need to check in some @test...
TypeUtils.adapt_precision(::Type{<:TypeUtils.Precision}, x::Neutral) = x
TypeUtils.adapt_precision(::Type{<:TypeUtils.Precision}, x::Unitful.AbstractQuantity{<:Neutral}) = x
