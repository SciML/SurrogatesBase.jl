using SurrogatesBase

using Test
using LinearAlgebra
import Statistics

struct DummySurrogate{X, Y} <: AbstractDeterministicSurrogate
    xs::Vector{X}
    ys::Vector{Y}
end
# return y value of the closest ξ in xs to x
(s::DummySurrogate)(x) = s.ys[argmin(norm(x - ξ) for ξ in s.xs)]
function SurrogatesBase.add_points!(s::DummySurrogate, new_xs, new_ys)
    append!(s.xs, new_xs)
    append!(s.ys, new_ys)
end

mutable struct HyperparameterDummySurrogate{X, Y} <: AbstractDeterministicSurrogate
    xs::Vector{X}
    ys::Vector{Y}
    θ::NamedTuple
end
# return y value of the closest ξ in xs to x, in p-norm where p is a hyperparameter
(s::HyperparameterDummySurrogate)(x) = s.ys[argmin(norm(x - ξ, s.θ.p) for ξ in s.xs)]
function SurrogatesBase.add_points!(s::HyperparameterDummySurrogate, new_xs, new_ys)
    append!(s.xs, new_xs)
    append!(s.ys, new_ys)
end

SurrogatesBase.hyperparameters(s::HyperparameterDummySurrogate) = s.θ

function SurrogatesBase.update_hyperparameters!(s::HyperparameterDummySurrogate, prior)
    # "hyperparmeter optimization"
    s.θ = (; p = (s.θ.p + prior.p) / 2)
end

@testset "add_points!" begin
    # use DummySurrogate
    d = DummySurrogate(Vector{Vector{Float64}}(), Vector{Int}())
    add_points!(d, [[10.3, 0.1],[1.9, 2.1]], [5,6])
    add_points!(d, [[-0.3, 9.9], [-0.1, -10.0]], [1, 3])
    @test length(d.xs) == 4
    @test d([0.0, -9.9]) == 3
end

@testset "default implementations" begin
    # use DummySurrogate
    d = DummySurrogate(Vector{Vector{Float64}}(), Vector{Float64}())
    add_points!(d, [[1.9, 2.1]], [5.0])
    add_points!(d, [[10.3, 0.1]], [9.0])

    @test d([2.0, 2.0]) == 5.0
    @test_throws MethodError hyperparameters(d)
    @test_throws MethodError update_hyperparameters!(d, 5)
end

@testset "hyperparameter interface" begin
    # use HyperparameterDummySurrogate
    hd = HyperparameterDummySurrogate(Vector{Vector{Float64}}(),
        Vector{Float64}(),
        (; p = 2))
    add_points!(hd, [[1.9, 2.1],[10.3, 0.1]], [5.0, 9.])

    @test hyperparameters(hd).p == 2
    update_hyperparameters!(hd, (; p = 4))
    @test hyperparameters(hd).p == 3
end

struct DummyStochasticSurrogate{X, Y} <: AbstractStochasticSurrogate
    xs::Vector{X}
    ys::Vector{Y}
end
function SurrogatesBase.add_points!(s::DummyStochasticSurrogate, new_xs, new_ys)
    append!(s.xs, new_xs)
    append!(s.ys, new_ys)
end

struct FiniteDummyStochasticSurrogate{X}
    means::Vector{X}
end
Statistics.mean(s::FiniteDummyStochasticSurrogate) = s.means

function FiniteDummyStochasticSurrogate(s, xs)
    return FiniteDummyStochasticSurrogate(zeros(length(xs)) .+ sum(s.ys))
end

SurrogatesBase.finite_posterior(s::DummyStochasticSurrogate, xs) = FiniteDummyStochasticSurrogate(s, xs)

@testset "finite_posterior" begin
    # use HyperparameterDummySurrogate
    ss = DummyStochasticSurrogate(Vector{Vector{Float64}}(),
        Vector{Float64}())

    add_points!(ss, [[1.9, 2.1],[10.3, 0.1]], [5.0, 9.])
    m = Statistics.mean(finite_posterior(ss, [[3.5,2.], [4.,5.],[1.,67.]]))
    @test length(m) == 3
    @test m[1] ≈ 14.
end
