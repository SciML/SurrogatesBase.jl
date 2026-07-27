using SurrogatesBase

using Statistics
using Test

mutable struct DeterministicMockSurrogate <: AbstractDeterministicSurrogate
    slope::Float64
    offset::Float64
end

(surrogate::DeterministicMockSurrogate)(x::Real) = surrogate.slope * x + surrogate.offset

function SurrogatesBase.update!(
        surrogate::DeterministicMockSurrogate, new_x, new_y
    )
    surrogate.offset = last(new_y) - surrogate.slope * last(new_x)
    return nothing
end

SurrogatesBase.parameters(surrogate::DeterministicMockSurrogate) =
    (; slope = surrogate.slope, offset = surrogate.offset)
SurrogatesBase.hyperparameters(surrogate::DeterministicMockSurrogate) = (; slope = surrogate.slope)

function SurrogatesBase.update_hyperparameters!(
        surrogate::DeterministicMockSurrogate, prior
    )
    surrogate.slope = prior.slope
    return nothing
end

mutable struct StochasticMockSurrogate <: AbstractStochasticSurrogate
    mean_value::Float64
end

struct MockPosterior
    means::Vector{Float64}
end

Statistics.mean(posterior::MockPosterior) = posterior.means

function SurrogatesBase.update!(surrogate::StochasticMockSurrogate, new_x, new_y)
    surrogate.mean_value = sum(new_y) / length(new_y)
    return nothing
end

SurrogatesBase.parameters(surrogate::StochasticMockSurrogate) = (; mean = surrogate.mean_value)

function SurrogatesBase.finite_posterior(surrogate::StochasticMockSurrogate, xs)
    return MockPosterior(fill(surrogate.mean_value, length(xs)))
end

@testset "Public surrogate extension contracts" begin
    @testset "Deterministic surrogate" begin
        surrogate = DeterministicMockSurrogate(2.0, 1.0)

        @test surrogate(3.0) == 7.0
        @test isnothing(update!(surrogate, [1.0, 2.0], [5.0, 7.0]))
        @test surrogate(2.0) == 7.0
        @test parameters(surrogate) == (; slope = 2.0, offset = 3.0)
        @test hyperparameters(surrogate) == (; slope = 2.0)
        @test isnothing(update_hyperparameters!(surrogate, (; slope = 4.0)))
        @test surrogate(2.0) == 11.0
    end

    @testset "Stochastic surrogate" begin
        surrogate = StochasticMockSurrogate(0.0)

        @test isnothing(update!(surrogate, [1.0, 2.0], [2.0, 6.0]))
        @test parameters(surrogate) == (; mean = 4.0)
        @test mean(finite_posterior(surrogate, [0.0, 1.0, 2.0])) == fill(4.0, 3)
    end
end
