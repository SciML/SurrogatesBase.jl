"""
    SurrogatesBase

Common public interfaces for deterministic and stochastic surrogate implementations.

See [`AbstractDeterministicSurrogate`](@ref), [`AbstractStochasticSurrogate`](@ref), and the
developer interface guide for extension rules.
"""
module SurrogatesBase

using PrecompileTools: @compile_workload, @setup_workload

export AbstractDeterministicSurrogate
export AbstractStochasticSurrogate
export update!, parameters
export update_hyperparameters!, hyperparameters
export finite_posterior

"""
    abstract type AbstractDeterministicSurrogate <: Function end

Abstract interface for a fitted deterministic surrogate.

`AbstractDeterministicSurrogate` has no fields. Concrete subtypes own their training data,
fitted state, and domain representation. A subtype is a function and represents a deterministic
approximation such as a regression model or an interpolant.

# Interface

To extend this interface, define a concrete subtype and implement the following methods for that
subtype:

  - `(surrogate)(x)`: evaluate the fitted approximation at an input `x` supported by the
    implementation.
  - [`update!(surrogate, new_x, new_y)`](@ref): incorporate paired observations.

The base interface does not prescribe whether `x` is scalar, a point container, or a batch of
points. A consuming package may require one of those forms, so implementations must document the
input forms they support. When an implementation supports batched evaluation, its result must
preserve the correspondence between requested inputs and returned predictions.

`update!` is an in-place interface: it must leave `surrogate` representing the updated fit. Its
return value is intentionally unspecified, because existing implementations return `nothing`, the
surrogate, or an implementation-specific value. Generic callers must use the mutated surrogate and
must not depend on the return value.

# Optional Methods

  - [`parameters(surrogate)`](@ref): expose learned parameters or fitted state.
  - [`hyperparameters(surrogate)`](@ref): expose tunable fitting configuration.
  - [`update_hyperparameters!(surrogate, prior)`](@ref): update that configuration in place.

# Examples

```jldoctest
julia> mutable struct ConstantSurrogate{T} <: AbstractDeterministicSurrogate
           value::T
       end

julia> (surrogate::ConstantSurrogate)(x) = surrogate.value;

julia> function SurrogatesBase.update!(surrogate::ConstantSurrogate, new_x, new_y)
           surrogate.value = last(new_y)
           return nothing
       end;

julia> surrogate = ConstantSurrogate(1.5);

julia> update!(surrogate, [0.0, 1.0], [2.0, 3.0]); surrogate(0.25)
3.0
```
"""
abstract type AbstractDeterministicSurrogate <: Function end

"""
    abstract type AbstractStochasticSurrogate end

Abstract interface for an uncertainty-aware surrogate.

`AbstractStochasticSurrogate` has no fields. Concrete subtypes own their observations, fitted
state, and posterior representation. A subtype represents a conditional distribution or stochastic
process approximation rather than a deterministic callable approximation.

# Interface

To extend this interface, define a concrete subtype and implement the following methods for that
subtype:

  - [`update!(surrogate, new_x, new_y)`](@ref): incorporate paired observations in place.
  - [`finite_posterior(surrogate, xs)`](@ref): construct a posterior object for query inputs `xs`.

`xs` is normally a collection of query inputs, but its precise representation is owned by the
concrete surrogate. The returned posterior object is also implementation-defined. It should expose
the statistical operations promised by the concrete implementation, such as `Statistics.mean`,
`Statistics.var`, or `rand`. Its values must correspond to the supplied query inputs.

As for [`AbstractDeterministicSurrogate`](@ref), the `update!` return value is unspecified. Generic
callers must use the mutated surrogate rather than its return value.

# Optional Methods

  - [`parameters(surrogate)`](@ref): expose learned parameters or fitted state.
  - [`hyperparameters(surrogate)`](@ref): expose tunable fitting configuration.
  - [`update_hyperparameters!(surrogate, prior)`](@ref): update that configuration in place.
"""
abstract type AbstractStochasticSurrogate end

"""
    update!(surrogate, new_x, new_y)

Incorporate paired observations into `surrogate` in place.

# Arguments

  - `surrogate`: an [`AbstractDeterministicSurrogate`](@ref) or
    [`AbstractStochasticSurrogate`](@ref) concrete implementation to update.
  - `new_x`: one input or a batch of input locations accepted by the concrete implementation.
  - `new_y`: observed value or values paired with `new_x`.

# Interface Contract

Concrete surrogate implementations must extend `update!` for their own subtype. For batched
updates, `new_x` and `new_y` must encode the same number of observations in the same order. For
single-observation updates, they must encode one paired input and value. The method must update the
surrogate state; no particular return value is part of this interface.

# Examples

```jldoctest
julia> mutable struct UpdateExampleSurrogate <: AbstractDeterministicSurrogate
           values::Vector{Float64}
       end

julia> (surrogate::UpdateExampleSurrogate)(x) = last(surrogate.values);

julia> function SurrogatesBase.update!(surrogate::UpdateExampleSurrogate, new_x, new_y)
           append!(surrogate.values, new_y)
           return nothing
       end;

julia> surrogate = UpdateExampleSurrogate(Float64[]);

julia> update!(surrogate, [1.0, 2.0], [3.0, 4.0]); surrogate(0.0)
4.0
```
"""
function update! end

"""
    parameters(surrogate)

Return learned parameters or fitted state exposed by `surrogate`.

# Arguments

  - `surrogate`: a concrete surrogate implementation that documents this optional method.

# Returns

An implementation-defined representation of learned parameters or fitted state. The returned object
may be a scalar, tuple, named tuple, array, or model object.

# Interface Contract

`parameters` is optional. Extend it only for concrete surrogate subtypes that expose learned state;
calling it for a subtype without a method raises a `MethodError`.

# Examples

```jldoctest
julia> struct ParameterExampleSurrogate <: AbstractDeterministicSurrogate
           weight::Float64
       end

julia> (surrogate::ParameterExampleSurrogate)(x) = surrogate.weight * x;

julia> SurrogatesBase.parameters(surrogate::ParameterExampleSurrogate) = (; weight = surrogate.weight);

julia> parameters(ParameterExampleSurrogate(2.0))
(weight = 2.0,)
```
"""
function parameters end

"""
    update_hyperparameters!(surrogate, prior)

Update the tunable fitting configuration of `surrogate` in place.

# Arguments

  - `surrogate`: a concrete surrogate implementation with tunable hyperparameters.
  - `prior`: implementation-defined prior, bounds, or optimization configuration.

# Interface Contract

This optional method must leave `surrogate` consistent with its updated hyperparameters. If those
hyperparameters affect the fitted representation, the implementation must refit or invalidate that
representation before subsequent evaluation. The return value is not part of the interface.

# Examples

```jldoctest
julia> mutable struct HyperparameterExample <: AbstractDeterministicSurrogate
           scale::Float64
       end

julia> (surrogate::HyperparameterExample)(x) = surrogate.scale * x;

julia> function SurrogatesBase.update_hyperparameters!(surrogate::HyperparameterExample, prior)
           surrogate.scale = prior.scale
           return nothing
       end;

julia> surrogate = HyperparameterExample(2.0);

julia> update_hyperparameters!(surrogate, (; scale = 4.0)); surrogate(0.5)
2.0
```
"""
function update_hyperparameters! end

"""
    hyperparameters(surrogate)

Return the tunable fitting configuration exposed by `surrogate`.

# Arguments

  - `surrogate`: a concrete surrogate implementation that documents this optional method.

# Returns

An implementation-defined representation of tunable configuration, commonly a named tuple or a
small immutable configuration object.

# Interface Contract

`hyperparameters` is optional. When both this method and [`update_hyperparameters!`](@ref) are
implemented, the returned configuration must describe the setting used for subsequent evaluations.

# Examples

```jldoctest
julia> struct HyperparameterReadExample <: AbstractDeterministicSurrogate
           scale::Float64
       end

julia> (surrogate::HyperparameterReadExample)(x) = surrogate.scale * x;

julia> SurrogatesBase.hyperparameters(surrogate::HyperparameterReadExample) =
           (; scale = surrogate.scale);

julia> hyperparameters(HyperparameterReadExample(2.0))
(scale = 2.0,)
```
"""
function hyperparameters end

"""
    finite_posterior(surrogate, xs)

Construct the finite-dimensional posterior represented by `surrogate` at query inputs `xs`.

# Arguments

  - `surrogate`: an [`AbstractStochasticSurrogate`](@ref) concrete implementation.
  - `xs`: query inputs in a representation accepted by that implementation.

# Returns

An implementation-defined posterior object for the requested query inputs. Concrete implementations
must document the statistical operations they support, for example `Statistics.mean`,
`Statistics.var`, or `rand`.

# Interface Contract

Concrete stochastic surrogate implementations must extend this function for their own subtype. The
posterior must preserve correspondence with `xs`; for a batch, the returned statistics and samples
must use the same query ordering. This function has no fallback implementation.

# Examples

```jldoctest
julia> using Statistics

julia> struct PosteriorExampleSurrogate <: AbstractStochasticSurrogate
           value::Float64
       end

julia> struct PosteriorExample
           means::Vector{Float64}
       end

julia> Statistics.mean(posterior::PosteriorExample) = posterior.means;

julia> function SurrogatesBase.finite_posterior(surrogate::PosteriorExampleSurrogate, xs)
           PosteriorExample(fill(surrogate.value, length(xs)))
       end;

julia> posterior = finite_posterior(PosteriorExampleSurrogate(1.25), [0.0, 1.0]);

julia> mean(posterior)
2-element Vector{Float64}:
 1.25
 1.25
```
"""
function finite_posterior end

@setup_workload begin
    @compile_workload begin
        typeof(AbstractDeterministicSurrogate)
        typeof(AbstractStochasticSurrogate)
        typeof(update!)
        typeof(parameters)
        typeof(finite_posterior)
    end
end

end
