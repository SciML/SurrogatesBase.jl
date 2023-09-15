using SurrogatesBase
import SurrogatesBase:
    add_point!,
    update_hyperparameters!, hyperparameters,
    posterior, mean, var, rand, logpdf

using Test
using LinearAlgebra

struct DummySurrogate{D, R} <: AbstractSurrogate{D, R}
    xs::Vector{D}
    ys::Vector{R}
end

# return y value of the closest ξ in xs to x
(s::DummySurrogate{D, R})(x::D) where {D, R} = s.ys[argmin(norm(x - ξ) for ξ in s.xs)]
function add_point!(s::DummySurrogate{D, R}, new_x::D, new_y::R) where {D, R}
    push!(s.xs, new_x)
    push!(s.ys, new_y)
end

# dummy mean at point
function mean(s::DummySurrogate{D, R},
    x::D) where {D <: Union{<:Number, <:AbstractVector}, R}
    norm(x)
end
# dummy variance at point
var(s::DummySurrogate{D, R}, x::D) where {D, R} = 5
# dummy variance at more points
var(s::DummySurrogate{D, R}, xs::Vector{D}) where {D, R} = ones(Base.length(xs))
# dummy logpdf
logpdf(s::DummySurrogate) = 0.5
# dummy rand from joint posterior
function rand(s::AbstractSurrogate{D, R},
    xs::Vector{D}) where {D <: Union{<:Number, <:Vector}, R <: Union{Float32, Float64}}
    rand(R, length(xs))
end

@testset "add_point! with broadcasting implementation" begin
    d = DummySurrogate(Vector{Vector{Float64}}(), Vector{Int}())
    add_point!(d, [1.9, 2.1], 5)
    add_point!(d, [10.3, 0.1], 9)
    add_point!(d, [[-0.3, 9.9], [-0.1, -10.0]], [1, 3])
    # check if add_points! added points correctly
    @test length(d.xs) == 4
    @test d([0.0, -9.9]) == 3
end

@testset "var" begin
    # use DummySurrogate
    d = DummySurrogate(Vector{Vector{Float64}}(), Vector{Int}())
    add_point!(d, [1.9, 2.1], 5)
    @test var(d, [1.0, 4.0]) == 5
    @test var(d, [[1.0, 4.0], [5.0, 6.0]]) == ones(2)
end

@testset "mean" begin
    # use DummySurrogate
    d = DummySurrogate(Vector{Vector{Float64}}(), Vector{Int}())
    add_point!(d, [1.9, 2.1], 5)
    @test isapprox(mean(d, [1.0, 4.0]), norm([1.0, 4.0]))
    @test isapprox(mean(d, [[1.0, 4.0], [5.0, 6.0]]), [norm([1.0, 4.0]), norm([5.0, 6.0])])
end

@testset "logpdf" begin
    d = DummySurrogate(Vector{Vector{Float64}}(), Vector{Float64}())
    add_point!(d, [1.9, 2.1], 5.9)
    @test logpdf(d) == 0.5
end

@testset "not implemented methods that are not imported in SurrogatsBase throw an error" begin
    d = DummySurrogate(Vector{Vector{Float64}}(), Vector{Float64}())
    add_point!(d, [1.9, 2.1], 5.9)
    @test_throws MethodError hyperparameters(d)
    @test_throws MethodError update_hyperparameters!(d, 5)
    @test_throws MethodError posterior(d, 5)
end

@testset "rand, test default pointwise implementation" begin
    d = DummySurrogate(Vector{Vector{Float64}}(), Vector{Float32}())
    add_point!(d, [1.9, 2.1], 5.9f0)
    @test length(rand(d, [[1.3, 4.], [5.,6.]])) == 2
    @test isa(rand(d, [1., 4.]), Float32)
end

@testset "add_point! in 1D surrogate" begin
    d = DummySurrogate(Vector{Float64}(), Vector{Float64}())
    add_point!(d, 1., 3.)
    add_point!(d, [2., 3., 4.], [4., 5., 6.])
    @test length(d.xs) == 4
end

# mutable b/c of hyperparameters in θ that change
mutable struct HyperparameterDummySurrogate{D, R} <: AbstractSurrogate{D, R}
    xs::Vector{D}
    ys::Vector{R}
    θ::NamedTuple
end
# return y value of the closest ξ in xs to x, in p-norm where p is a hyperparameter
(s::HyperparameterDummySurrogate)(x) = s.ys[argmin(norm(x - ξ, s.θ.p) for ξ in s.xs)]
function add_point!(s::HyperparameterDummySurrogate{D, R}, new_x::D, new_y::R) where {D, R}
    push!(s.xs, new_x)
    push!(s.ys, new_y)
end
hyperparameters(s::HyperparameterDummySurrogate) = s.θ
function update_hyperparameters!(s::HyperparameterDummySurrogate, prior)
    # "hyperparmeter optimization"
    s.θ = (; p = (s.θ.p + prior.p) / 2)
end

@testset "hyperparameter interface" begin
    hd = HyperparameterDummySurrogate(Vector{Vector{Float64}}(),
        Vector{Float64}(),
        (; p = 2))
    add_point!(hd, [1.9, 2.1], 5.0)
    add_point!(hd, [10.3, 0.1], 9.0)

    @test hyperparameters(hd).p == 2
    update_hyperparameters!(hd, (; p = 4))
    @test hyperparameters(hd).p == 3
end
