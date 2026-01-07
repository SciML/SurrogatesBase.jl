using SurrogatesBase
using AllocCheck
using BenchmarkTools
using Test

# This file tests that implementations of the SurrogatesBase interface
# can be made allocation-free when properly implemented.
# Since SurrogatesBase is an interface-only package, we test with
# simple concrete implementations.

# Simple deterministic surrogate that returns a constant value
struct ConstantSurrogate <: AbstractDeterministicSurrogate
    value::Float64
end

# Allocation-free call operator
(s::ConstantSurrogate)(::Float64) = s.value

# Simple linear surrogate: f(x) = a*x + b
struct LinearSurrogate <: AbstractDeterministicSurrogate
    a::Float64
    b::Float64
end

# Allocation-free call operator
(s::LinearSurrogate)(x::Float64) = s.a * x + s.b

@testset "Allocation Tests" begin
    @testset "ConstantSurrogate allocation-free" begin
        s = ConstantSurrogate(42.0)
        x = 1.0

        # Warmup
        s(x)

        # Use BenchmarkTools for accurate allocation measurement
        result = @benchmark $s($x)
        @test result.memory == 0
        @test result.allocs == 0
    end

    @testset "LinearSurrogate allocation-free" begin
        s = LinearSurrogate(2.0, 1.0)
        x = 3.0

        # Warmup
        s(x)

        # Use BenchmarkTools for accurate allocation measurement
        result = @benchmark $s($x)
        @test result.memory == 0
        @test result.allocs == 0
    end
end
