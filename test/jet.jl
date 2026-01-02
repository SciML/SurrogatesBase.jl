using SurrogatesBase
using JET
using LinearAlgebra
import Statistics

@testset "JET static analysis" begin
    @testset "Package analysis" begin
        result = JET.report_package("SurrogatesBase")
        @test length(JET.get_reports(result)) == 0
    end

    @testset "DummySurrogate type stability" begin
        # Test implementation from runtests.jl
        struct JETDummySurrogate{X, Y} <: AbstractDeterministicSurrogate
            xs::Vector{X}
            ys::Vector{Y}
        end
        (s::JETDummySurrogate)(x) = s.ys[argmin(norm(x - ξ) for ξ in s.xs)]
        function SurrogatesBase.update!(s::JETDummySurrogate, new_xs, new_ys)
            append!(s.xs, new_xs)
            append!(s.ys, new_ys)
        end

        d = JETDummySurrogate(Vector{Vector{Float64}}(), Vector{Int}())
        SurrogatesBase.update!(d, [[10.3, 0.1], [1.9, 2.1]], [5, 6])

        # Test call method
        result = JET.report_call(d, Tuple{Vector{Float64}})
        @test length(JET.get_reports(result)) == 0

        # Test update! method
        result = JET.report_call(
            SurrogatesBase.update!,
            Tuple{JETDummySurrogate{Vector{Float64}, Int}, Vector{Vector{Float64}},
                Vector{Int}})
        @test length(JET.get_reports(result)) == 0
    end

    @testset "Stochastic surrogate type stability" begin
        mutable struct JETDummyStochasticSurrogate{X, Y} <: AbstractStochasticSurrogate
            xs::Vector{X}
            ys::Vector{Y}
            ys_mean::Y
        end
        function SurrogatesBase.update!(s::JETDummyStochasticSurrogate, new_xs, new_ys)
            append!(s.xs, new_xs)
            append!(s.ys, new_ys)
            s.ys_mean = (s.ys_mean * (length(s.xs) - length(new_xs)) + sum(new_ys)) /
                        length(s.xs)
        end
        SurrogatesBase.parameters(s::JETDummyStochasticSurrogate) = s.ys_mean

        struct JETFiniteDummyStochasticSurrogate{X}
            means::Vector{X}
        end
        Statistics.mean(s::JETFiniteDummyStochasticSurrogate) = s.means
        function JETFiniteDummyStochasticSurrogate(s, xs)
            return JETFiniteDummyStochasticSurrogate(s.ys_mean .* ones(length(xs)))
        end
        function SurrogatesBase.finite_posterior(s::JETDummyStochasticSurrogate, xs)
            JETFiniteDummyStochasticSurrogate(s, xs)
        end

        # Test update! method
        result = JET.report_call(
            SurrogatesBase.update!,
            Tuple{JETDummyStochasticSurrogate{Vector{Float64}, Float64},
                Vector{Vector{Float64}}, Vector{Float64}})
        @test length(JET.get_reports(result)) == 0

        # Test finite_posterior
        result = JET.report_call(
            SurrogatesBase.finite_posterior,
            Tuple{JETDummyStochasticSurrogate{Vector{Float64}, Float64},
                Vector{Vector{Float64}}})
        @test length(JET.get_reports(result)) == 0
    end
end
