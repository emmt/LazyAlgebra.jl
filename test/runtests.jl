using LazyAlgebra:
    Identity, Null, Diag,
    ArrayDomain, OffsetArrayDomain,
    element_axes, element_eltype, element_length, element_ndims, element_size,
    input_domain_type, input_domain, input_eltype, input_ndims,
    input_length, input_size, input_axes,
    output_domain_type, output_domain, output_eltype, output_ndims,
    output_length, output_size, output_axes,
    is_linear
using Test

@testset "LazyAlgebra (T = $T)" for T in (Float32, Float64)
    dims = (3,4,5)
    dims_ = (UInt8(dims[1]), dims[2:end-1]..., Int16(dims[end]))
    println(stderr, "dims_ = ", dims_)
    inds = (0:5, Base.OneTo(7))
    inds_ = (Int16(first(inds[1])):Int16(last(inds[1])), inds[2:end-1]..., convert(Base.OneTo{Int16}, inds[end]))
    println(stderr, "inds_ = ", inds_)
    @testset "Domains" begin
        E = @inferred ArrayDomain{T}(dims)
        @test @inferred(ArrayDomain{T}(dims...)) === E
        @test @inferred(ArrayDomain{T}(dims_)) === E
        @test @inferred(ArrayDomain{T}(dims_...)) === E
        @test @inferred(element_eltype(E)) === T
        @test @inferred(element_ndims(E)) === length(dims)
        @test @inferred(element_length(E)) === prod(dims)
        @test @inferred(element_size(E)) === dims
        @test @inferred(element_axes(E)) === map(Base.OneTo, dims)

        F = @inferred OffsetArrayDomain{T}(inds)
        @test @inferred(OffsetArrayDomain{T}(inds...)) === F
        @test @inferred(OffsetArrayDomain{T}(inds_)) === F
        @test @inferred(OffsetArrayDomain{T}(inds_...)) === F
        @test @inferred(element_eltype(F)) === T
        @test @inferred(element_ndims(F)) === length(inds)
        @test @inferred(element_length(F)) === prod(map(length, inds))
        @test @inferred(element_size(F)) === map(length, inds)
        @test @inferred(element_axes(F)) === inds
    end

    @testset "Identity" begin
        E = @inferred ArrayDomain{T}(dims)
        Id = @inferred Identity(E)
        @test @inferred(isone(Id)) == true
        @test @inferred(iszero(Id)) == false
        @test @inferred(is_linear(Id)) == true
        @test @inferred(is_linear(typeof(Id))) == true
        @test @inferred(input_domain(Id)) === E
        @test @inferred(input_domain_type(Id)) === typeof(E)
        @test @inferred(input_eltype(Id)) === @inferred(element_eltype(E))
        @test @inferred(input_ndims(Id)) === @inferred(element_ndims(E))
        @test @inferred(input_length(Id)) === @inferred(element_length(E))
        @test @inferred(input_size(Id)) === @inferred(element_size(E))
        @test @inferred(input_axes(Id)) === @inferred(element_axes(E))
        @test @inferred(input_size(Id, 3)) === @inferred(element_size(E, 3))
        @test @inferred(input_axes(Id, 2)) === @inferred(element_axes(E, 2))
        @test @inferred(output_domain_type(Id)) === typeof(E)
        @test @inferred(output_domain(Id)) === E
        @test @inferred(output_eltype(Id)) === @inferred(element_eltype(E))
        @test @inferred(output_ndims(Id)) === @inferred(element_ndims(E))
        @test @inferred(output_length(Id)) === @inferred(element_length(E))
        @test @inferred(output_size(Id)) === @inferred(element_size(E))
        @test @inferred(output_axes(Id)) === @inferred(element_axes(E))
        @test @inferred(output_size(Id, 3)) === @inferred(element_size(E, 3))
        @test @inferred(output_axes(Id, 2)) === @inferred(element_axes(E, 2))
    end

    @testset "Zero" begin
        E = @inferred ArrayDomain{T}(dims)
        F = @inferred OffsetArrayDomain{T}(inds)
        Zero = @inferred Null(E)
        @test @inferred(isone(Zero)) == false
        @test @inferred(iszero(Zero)) == true
        @test @inferred(is_linear(Zero)) == true
        @test @inferred(is_linear(typeof(Zero))) == true
        @test @inferred(input_domain_type(Zero)) === typeof(E)
        @test @inferred(input_domain(Zero)) === E
        @test @inferred(input_eltype(Zero)) === @inferred(element_eltype(E))
        @test @inferred(input_ndims(Zero)) === @inferred(element_ndims(E))
        @test @inferred(input_length(Zero)) === @inferred(element_length(E))
        @test @inferred(input_size(Zero)) === @inferred(element_size(E))
        @test @inferred(input_axes(Zero)) === @inferred(element_axes(E))
        @test @inferred(input_size(Zero, 3)) === @inferred(element_size(E, 3))
        @test @inferred(input_axes(Zero, 2)) === @inferred(element_axes(E, 2))
        @test @inferred(output_domain_type(Zero)) === typeof(E)
        @test @inferred(output_domain(Zero)) === E
        @test @inferred(output_eltype(Zero)) === @inferred(element_eltype(E))
        @test @inferred(output_ndims(Zero)) === @inferred(element_ndims(E))
        @test @inferred(output_length(Zero)) === @inferred(element_length(E))
        @test @inferred(output_size(Zero)) === @inferred(element_size(E))
        @test @inferred(output_axes(Zero)) === @inferred(element_axes(E))
        @test @inferred(output_size(Zero, 3)) === @inferred(element_size(E, 3))
        @test @inferred(output_axes(Zero, 2)) === @inferred(element_axes(E, 2))

        Zero = @inferred Null(E => F)
        @test @inferred(isone(Zero)) == false
        @test @inferred(iszero(Zero)) == true
        @test @inferred(is_linear(Zero)) == true
        @test @inferred(is_linear(typeof(Zero))) == true
        @test @inferred(input_domain_type(Zero)) === typeof(E)
        @test @inferred(input_domain(Zero)) === E
        @test @inferred(input_eltype(Zero)) === @inferred(element_eltype(E))
        @test @inferred(input_ndims(Zero)) === @inferred(element_ndims(E))
        @test @inferred(input_length(Zero)) === @inferred(element_length(E))
        @test @inferred(input_size(Zero)) === @inferred(element_size(E))
        @test @inferred(input_axes(Zero)) === @inferred(element_axes(E))
        @test @inferred(input_size(Zero, 3)) === @inferred(element_size(E, 3))
        @test @inferred(input_axes(Zero, 2)) === @inferred(element_axes(E, 2))
        @test @inferred(output_domain_type(Zero)) === typeof(F)
        @test @inferred(output_domain(Zero)) === F
        @test @inferred(output_eltype(Zero)) === @inferred(element_eltype(F))
        @test @inferred(output_ndims(Zero)) === @inferred(element_ndims(F))
        @test @inferred(output_length(Zero)) === @inferred(element_length(F))
        @test @inferred(output_size(Zero)) === @inferred(element_size(F))
        @test @inferred(output_axes(Zero)) === @inferred(element_axes(F))
        @test output_size(Zero, 1) === element_size(F, 1)
        @test output_axes(Zero, 2) === element_axes(F, 2)
    end
end
nothing
