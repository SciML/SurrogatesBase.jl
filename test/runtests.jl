using SurrogatesBase
import SurrogatesBase: add_point!,
    update_hyperparameters!, hyperparameters
using Test
using LinearAlgebra

mutable struct DummySurrogate{X, Y} <: AbstractSurrogate
    xs::Vector{X}
    ys::Vector{Y}
end
# return y value of the closest ξ in xs to x
(s::DummySurrogate)(x) = s.ys[argmin(norm(x - ξ) for ξ in s.xs)]
function add_point!(s::DummySurrogate, new_x, new_y)
    push!(s.xs, new_x)
    push!(s.ys, new_y)
end

mutable struct HyperparameterDummySurrogate{X, Y} <: AbstractSurrogate
    xs::Vector{X}
    ys::Vector{Y}
    θ::NamedTuple
end
# return y value of the closest ξ in xs to x, in p-norm where p is a hyperparameter
(s::HyperparameterDummySurrogate)(x) = s.ys[argmin(norm(x - ξ, s.θ.p) for ξ in s.xs)]
function add_point!(s::DummySurrogate, new_x, new_y)
    push!(s.xs, new_x)
    push!(s.ys, new_y)
end
supports_hyperparameters(s::HyperparameterDummySurrogate) = true
hyperparameters(s::HyperparameterDummySurrogate) = s.θ
function update_hyperparameters!(s::HyperparameterDummySurrogate, prior)
    # "hyperparmeter optimization"
    s.θ = (; p = (s.θ.p + prior.p) / 2)
end

@testset "add_points!" begin
    # use DummySurrogate
    d = DummySurrogate(Vector{Vector{Float64}}(), Vector{Int}())
    add_point!(d, [1.9, 2.1], 5)
    add_point!(d, [10.3, 0.1], 9)
    add_points!(d, [[-0.3, 9.9], [-0.1, -10.0]], [1, 3])
    # check if add_points! added points correctly
    @test length(d.xs) == 4
    @test d([0.0, -9.9]) == 3
end

@testset "default implementations" begin
    # use DummySurrogate
    d = DummySurrogate(Vector{Vector{Float64}}(), Vector{Float64}())
    add_point!(d, [1.9, 2.1], 5.0)
    add_point!(d, [10.3, 0.1], 9.0)

    @test d([2.0, 2.0]) == 5.0
    @test_throws ErrorException hyperparameters(d)
    @test_throws ErrorException update_hyperparameters!(d, 5)
    @test_throws ErrorException posterior(d, 5)
end

@testset "hyperparameter interface" begin
    # use HyperparameterDummySurrogate
    hd = HyperparameterDummySurrogate(Vector{Vector{Float64}}(),
        Vector{Float64}(),
        (; p = 2))
    add_point!(hd, [1.9, 2.1], 5.0)
    add_point!(hd, [10.3, 0.1], 9.0)

    @test hyperparameters(hd).p == 2
    update_hyperparameters!(hd, (; p = 4))
    @test hyperparameters(hd).p == 3
end
