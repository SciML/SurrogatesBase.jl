module SurrogatesBase

export AbstractDeterministicSurrogate
export AbstractStochasticSurrogate

export update!, parameters
export update_hyperparameters!, hyperparameters
export finite_posterior

"""
    abstract type AbstractDeterministicSurrogate <: Function end
    (s::AbstractDeterministicSurrogate)(xs)

Interface tag for deterministic surrogate models.

Subtypes approximate a deterministic function, or a deterministic statistic of a
conditional distribution, from observed data. A deterministic surrogate is callable on a
collection of input points `xs` and should return one surrogate value for each point.

# Required Methods

  - `(s)(xs)`: evaluate the surrogate at the points in `xs`.
  - [`update!(s, new_xs, new_ys)`](@ref): incorporate new observations.

# Optional Methods

  - [`parameters(s)`](@ref): return learned parameter values.
  - [`hyperparameters(s)`](@ref): return tunable hyperparameter values.
  - [`update_hyperparameters!(s, prior)`](@ref): update tunable hyperparameters.

# Examples

```jldoctest
julia> struct ConstantSurrogate{T} <: AbstractDeterministicSurrogate
           value::T
       end

julia> (s::ConstantSurrogate)(xs) = fill(s.value, length(xs));

julia> surrogate = ConstantSurrogate(1.5);

julia> surrogate([[0.0, 1.0], [1.0, 2.0]])
2-element Vector{Float64}:
 1.5
 1.5
```
"""
abstract type AbstractDeterministicSurrogate <: Function end

"""
    abstract type AbstractStochasticSurrogate end

Interface tag for stochastic surrogate models.

Subtypes approximate a conditional distribution, stochastic process, or uncertainty-aware
surrogate from observed data.

# Required Methods

  - [`update!(s, new_xs, new_ys)`](@ref): incorporate new observations.
  - [`finite_posterior(s, xs)`](@ref): return a finite-dimensional posterior object at
    the points in `xs`.

# Optional Methods

  - [`parameters(s)`](@ref): return learned parameter values.
  - [`hyperparameters(s)`](@ref): return tunable hyperparameter values.
  - [`update_hyperparameters!(s, prior)`](@ref): update tunable hyperparameters.

See also [`finite_posterior`](@ref).
"""
abstract type AbstractStochasticSurrogate end

"""
    update!(s, new_xs::AbstractVector, new_ys::AbstractVector)

Incorporate observations `new_ys` at points `new_xs` into the surrogate `s`.

Implementations usually mutate and return `s`. For deterministic surrogates, `new_ys`
contains function evaluations or deterministic statistics. For stochastic surrogates,
`new_ys` contains observed samples from the modeled conditional distribution.

# Arguments

  - `s`: surrogate to refit or update.
  - `new_xs`: input points to add to `s`.
  - `new_ys`: observed values corresponding to `new_xs`.

Use `update!(s, eachslice(X; dims = 2), new_ys)` when columns of a matrix `X` are the input
points.

# Examples

```jldoctest
julia> mutable struct UpdateExampleSurrogate <: AbstractDeterministicSurrogate
           xs::Vector{Float64}
           ys::Vector{Float64}
       end

julia> (s::UpdateExampleSurrogate)(xs) = fill(last(s.ys), length(xs));

julia> function SurrogatesBase.update!(s::UpdateExampleSurrogate, new_xs, new_ys)
           append!(s.xs, new_xs)
           append!(s.ys, new_ys)
           return s
       end;

julia> surrogate = UpdateExampleSurrogate(Float64[], Float64[]);

julia> update!(surrogate, [1.0, 2.0], [3.0, 4.0]) === surrogate
true

julia> surrogate.ys
2-element Vector{Float64}:
 3.0
 4.0
```
"""
function update! end

"""
    parameters(s)

Return the current learned parameter values of the surrogate `s`.

This is an optional interface method for surrogate implementations that expose fitted
parameters separately from tunable hyperparameters.

# Examples

```jldoctest
julia> struct ParameterExampleSurrogate <: AbstractDeterministicSurrogate
           weights::Vector{Float64}
       end

julia> (s::ParameterExampleSurrogate)(xs) = fill(sum(s.weights), length(xs));

julia> SurrogatesBase.parameters(s::ParameterExampleSurrogate) = s.weights;

julia> parameters(ParameterExampleSurrogate([1.0, 2.0]))
2-element Vector{Float64}:
 1.0
 2.0
```
"""
function parameters end

"""
    update_hyperparameters!(s, prior)

Update tunable hyperparameters of the surrogate `s` using information in `prior`.

Implementations usually mutate and return `s`. After changing hyperparameters, the
surrogate should be refit to its existing observations when the hyperparameters affect the
fitted representation.

# Arguments

  - `s`: surrogate whose hyperparameters are updated.
  - `prior`: implementation-defined prior, bounds, or configuration used by the update.

# Examples

```jldoctest
julia> mutable struct HyperparameterUpdateExample <: AbstractDeterministicSurrogate
           scale::Float64
       end

julia> (s::HyperparameterUpdateExample)(xs) = fill(s.scale, length(xs));

julia> function SurrogatesBase.update_hyperparameters!(s::HyperparameterUpdateExample, prior)
           s.scale = (s.scale + prior.scale) / 2
           return s
       end;

julia> surrogate = HyperparameterUpdateExample(2.0);

julia> update_hyperparameters!(surrogate, (; scale = 4.0)) === surrogate
true

julia> surrogate.scale
3.0
```

See also [`hyperparameters`](@ref).
"""
function update_hyperparameters! end

"""
    hyperparameters(s)

Return the current tunable hyperparameter values of the surrogate `s`.

This is an optional interface method for surrogate implementations with configuration
values that control fitting or posterior construction.

# Examples

```jldoctest
julia> struct HyperparameterReadExample <: AbstractDeterministicSurrogate
           settings::NamedTuple
       end

julia> (s::HyperparameterReadExample)(xs) = fill(s.settings.scale, length(xs));

julia> SurrogatesBase.hyperparameters(s::HyperparameterReadExample) = s.settings;

julia> hyperparameters(HyperparameterReadExample((; scale = 2.0)))
(scale = 2.0,)
```

See also [`update_hyperparameters!`](@ref).
"""
function hyperparameters end

"""
    finite_posterior(s::AbstractStochasticSurrogate, xs::AbstractVector)

Return a finite-dimensional posterior object at points `xs`.

The returned object represents the joint posterior over the requested points. An
`AbstractStochasticSurrogate` implementation may support some or all of the following
methods on that object:

  - `mean(finite_posterior(s, xs))`: posterior means at `xs`.
  - `var(finite_posterior(s, xs))`: posterior variances at `xs`.
  - `mean_and_var(finite_posterior(s, xs))`: posterior means and variances at `xs`.
  - `rand(finite_posterior(s, xs))`: a sample from the joint posterior at `xs`.

Use `mean(finite_posterior(s, eachslice(X; dims = 2)))` when columns of a matrix `X` are
the input points.

# Examples

```jldoctest
julia> using Statistics

julia> struct PosteriorExampleSurrogate <: AbstractStochasticSurrogate
           value::Float64
       end

julia> struct PosteriorExample
           means::Vector{Float64}
       end

julia> Statistics.mean(p::PosteriorExample) = p.means;

julia> function SurrogatesBase.finite_posterior(s::PosteriorExampleSurrogate, xs)
           return PosteriorExample(fill(s.value, length(xs)))
       end;

julia> posterior = finite_posterior(PosteriorExampleSurrogate(1.25), [0.0, 1.0]);

julia> mean(posterior)
2-element Vector{Float64}:
 1.25
 1.25
```
"""
function finite_posterior end

end
