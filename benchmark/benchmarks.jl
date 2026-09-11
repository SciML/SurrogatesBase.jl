using SurrogatesBase, BenchmarkTools
using Statistics

const SUITE = BenchmarkGroup()

# Minimal deterministic surrogate implementing the interface
mutable struct LinearSurrogateModel <: SurrogatesBase.AbstractDeterministicSurrogate
    a::Float64
    b::Float64
    xs::Vector{Float64}
    ys::Vector{Float64}
end

function LinearSurrogateModel(xs, ys)
    n = length(xs)
    a = cov(xs, ys) / var(xs)
    b = mean(ys) - a * mean(xs)
    return LinearSurrogateModel(a, b, copy(xs), copy(ys))
end

(s::LinearSurrogateModel)(x) = s.a * x + s.b

function SurrogatesBase.update!(s::LinearSurrogateModel, new_xs, new_ys)
    append!(s.xs, new_xs)
    append!(s.ys, new_ys)
    s.a = cov(s.xs, s.ys) / var(s.xs)
    s.b = mean(s.ys) - s.a * mean(s.xs)
    return s
end

SurrogatesBase.parameters(s::LinearSurrogateModel) = (a = s.a, b = s.b)

xs = collect(0.0:0.1:10.0)
ys = 2.0 .* xs .+ 1.0
s = LinearSurrogateModel(xs, ys)

# =============================================================================
# Interface operations
# =============================================================================

SUITE["interface"] = BenchmarkGroup()

SUITE["interface"]["construct"] = @benchmarkable LinearSurrogateModel($xs, $ys)
SUITE["interface"]["call"] = @benchmarkable $s(3.5)
SUITE["interface"]["parameters"] = @benchmarkable SurrogatesBase.parameters($s)
SUITE["interface"]["update!"] = @benchmarkable SurrogatesBase.update!(
    s, [11.0, 12.0], [23.0, 25.0]
) setup = (s = LinearSurrogateModel($xs, $ys))
