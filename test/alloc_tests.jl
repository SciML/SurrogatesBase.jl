using SurrogatesBase
using AllocCheck
using LinearAlgebra
using Test

# This file demonstrates allocation testing for surrogate implementations.
# Since SurrogatesBase is an interface package with no concrete implementations,
# these tests verify that example implementations can be allocation-free.

# Simple allocation-free surrogate for testing
struct AllocTestSurrogate{X, Y} <: AbstractDeterministicSurrogate
    xs::Vector{X}
    ys::Vector{Y}
end

# Allocation-free call - returns the closest y value using pre-computed indices
function (s::AllocTestSurrogate)(x::Vector{Float64})
    min_idx = 1
    min_dist = Inf
    @inbounds for i in eachindex(s.xs)
        dist = zero(Float64)
        xi = s.xs[i]
        for j in eachindex(x)
            diff = x[j] - xi[j]
            dist += diff * diff
        end
        if dist < min_dist
            min_dist = dist
            min_idx = i
        end
    end
    return @inbounds s.ys[min_idx]
end

@testset "AllocCheck - Surrogate Operations" begin
    # Create surrogate with test data
    xs = [Float64[1.0, 2.0], Float64[3.0, 4.0], Float64[5.0, 6.0]]
    ys = [1, 2, 3]
    s = AllocTestSurrogate(xs, ys)

    # Test that calling the surrogate is allocation-free
    x_test = Float64[2.0, 3.0]

    # Verify correctness first
    @test s(x_test) == 1  # Closest to [1.0, 2.0]

    # Check that the call is allocation-free after warmup
    s(x_test)  # warmup
    allocs = @allocated s(x_test)
    @test allocs == 0

    # Test with @check_allocs macro
    @check_allocs function test_surrogate_call(surr::AllocTestSurrogate{Vector{Float64}, Int}, x::Vector{Float64})
        return surr(x)
    end
    @test test_surrogate_call(s, x_test) == 1
end

@testset "AllocCheck - Type Stability" begin
    # Verify that abstract type checks are allocation-free
    xs = [Float64[1.0, 2.0]]
    ys = [1]
    s = AllocTestSurrogate(xs, ys)

    @check_allocs function check_type(surr::AllocTestSurrogate)
        return surr isa AbstractDeterministicSurrogate
    end

    @test check_type(s) == true
end
