using SciMLTesting
using SurrogatesBase
using JET
using LinearAlgebra
using Test
import Statistics

run_qa(SurrogatesBase; explicit_imports = true)

# JET.report_call type-stability analysis of concrete user-defined surrogates.
# This goes beyond run_qa's package-level JET.test_package: it checks that the
# interface contract (update!/finite_posterior/parameters and the call method)
# stays inferable for downstream subtypes.
#
# On Julia 1.12+, LinearAlgebra.norm_recursive_check has a type inference issue
# that surfaces as JET false positives through `norm`. We ignore LinearAlgebra
# and Base frames to filter those stdlib issues while still checking our own code.
const JET_CONFIG = (
    ignored_modules = (
        JET.AnyFrameModule(LinearAlgebra),
        JET.AnyFrameModule(Base),
    ),
)

@testset "JET report_call type stability" begin
    @testset "DummySurrogate type stability" begin
        struct JETDummySurrogate{X, Y} <: AbstractDeterministicSurrogate
            xs::Vector{X}
            ys::Vector{Y}
        end
        (s::JETDummySurrogate)(x) = s.ys[argmin([norm(x - ξ) for ξ in s.xs])]
        function SurrogatesBase.update!(s::JETDummySurrogate, new_xs, new_ys)
            append!(s.xs, new_xs)
            append!(s.ys, new_ys)
        end

        d = JETDummySurrogate(Vector{Vector{Float64}}(), Vector{Int}())
        SurrogatesBase.update!(d, [[10.3, 0.1], [1.9, 2.1]], [5, 6])

        result = JET.report_call(d, Tuple{Vector{Float64}}; JET_CONFIG...)
        @test length(JET.get_reports(result)) == 0

        result = JET.report_call(
            SurrogatesBase.update!,
            Tuple{
                JETDummySurrogate{Vector{Float64}, Int}, Vector{Vector{Float64}},
                Vector{Int},
            }; JET_CONFIG...
        )
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

        result = JET.report_call(
            SurrogatesBase.update!,
            Tuple{
                JETDummyStochasticSurrogate{Vector{Float64}, Float64},
                Vector{Vector{Float64}}, Vector{Float64},
            }; JET_CONFIG...
        )
        @test length(JET.get_reports(result)) == 0

        result = JET.report_call(
            SurrogatesBase.finite_posterior,
            Tuple{
                JETDummyStochasticSurrogate{Vector{Float64}, Float64},
                Vector{Vector{Float64}},
            }; JET_CONFIG...
        )
        @test length(JET.get_reports(result)) == 0
    end
end
