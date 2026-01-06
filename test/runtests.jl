using SurrogatesBase

using Test
using SafeTestsets
@safetestset "Quality Assurance" include("qa.jl")
@safetestset "JET Static Analysis" include("jet.jl")

using LinearAlgebra
import Statistics

struct DummySurrogate{X, Y} <: AbstractDeterministicSurrogate
    xs::Vector{X}
    ys::Vector{Y}
end
# return y value of the closest ξ in xs to x
(s::DummySurrogate)(x) = s.ys[argmin(norm(x - ξ) for ξ in s.xs)]
function SurrogatesBase.update!(s::DummySurrogate, new_xs, new_ys)
    append!(s.xs, new_xs)
    return append!(s.ys, new_ys)
end

mutable struct HyperparameterDummySurrogate{X, Y} <: AbstractDeterministicSurrogate
    xs::Vector{X}
    ys::Vector{Y}
    θ::NamedTuple
end
# return y value of the closest ξ in xs to x, in p-norm where p is a hyperparameter
(s::HyperparameterDummySurrogate)(x) = s.ys[argmin(norm(x - ξ, s.θ.p) for ξ in s.xs)]
function SurrogatesBase.update!(s::HyperparameterDummySurrogate, new_xs, new_ys)
    append!(s.xs, new_xs)
    return append!(s.ys, new_ys)
end

SurrogatesBase.hyperparameters(s::HyperparameterDummySurrogate) = s.θ

function SurrogatesBase.update_hyperparameters!(s::HyperparameterDummySurrogate, prior)
    # "hyperparmeter optimization"
    return s.θ = (; p = (s.θ.p + prior.p) / 2)
end

@testset "update!" begin
    # use DummySurrogate
    d = DummySurrogate(Vector{Vector{Float64}}(), Vector{Int}())
    update!(d, [[10.3, 0.1], [1.9, 2.1]], [5, 6])
    update!(d, [[-0.3, 9.9], [-0.1, -10.0]], [1, 3])
    @test length(d.xs) == 4
    @test d([0.0, -9.9]) == 3
end

@testset "default implementations" begin
    # use DummySurrogate
    d = DummySurrogate(Vector{Vector{Float64}}(), Vector{Float64}())
    update!(d, [[1.9, 2.1]], [5.0])
    update!(d, [[10.3, 0.1]], [9.0])

    @test d([2.0, 2.0]) == 5.0
    @test_throws MethodError hyperparameters(d)
    @test_throws MethodError update_hyperparameters!(d, 5)
end

@testset "hyperparameter interface" begin
    # use HyperparameterDummySurrogate
    hd = HyperparameterDummySurrogate(
        Vector{Vector{Float64}}(),
        Vector{Float64}(),
        (; p = 2)
    )
    update!(hd, [[1.9, 2.1], [10.3, 0.1]], [5.0, 9.0])

    @test hyperparameters(hd).p == 2
    update_hyperparameters!(hd, (; p = 4))
    @test hyperparameters(hd).p == 3
end

mutable struct DummyStochasticSurrogate{X, Y} <: AbstractStochasticSurrogate
    xs::Vector{X}
    ys::Vector{Y}
    ys_mean::Y
end
function SurrogatesBase.update!(s::DummyStochasticSurrogate, new_xs, new_ys)
    append!(s.xs, new_xs)
    append!(s.ys, new_ys)
    # update mean
    return s.ys_mean = (s.ys_mean * (length(s.xs) - length(new_xs)) + sum(new_ys)) / length(s.xs)
end

SurrogatesBase.parameters(s::DummyStochasticSurrogate) = s.ys_mean

struct FiniteDummyStochasticSurrogate{X}
    means::Vector{X}
end
Statistics.mean(s::FiniteDummyStochasticSurrogate) = s.means

# xs are arbitrary points where we wish to get a joint posterior
function FiniteDummyStochasticSurrogate(s, xs)
    return FiniteDummyStochasticSurrogate(s.ys_mean .* ones(length(xs)))
end

function SurrogatesBase.finite_posterior(s::DummyStochasticSurrogate, xs)
    return FiniteDummyStochasticSurrogate(s, xs)
end

@testset "finite_posterior, parameters" begin
    # use HyperparameterDummySurrogate
    ss = DummyStochasticSurrogate(
        Vector{Vector{Float64}}(),
        Vector{Float64}(), 0.0
    )

    update!(ss, [[1.9, 2.1], [10.3, 0.1]], [5.0, 9.0])
    # test parameters
    @test parameters(ss) ≈ 7.0
    update!(ss, [[2.0, 4.0]], [3.0])
    @test parameters(ss) ≈ 17 / 3

    m = Statistics.mean(finite_posterior(ss, [[3.5, 2.0], [4.0, 5.0], [1.0, 67.0]]))
    @test length(m) == 3
    @test m[1] ≈ parameters(ss)
end
