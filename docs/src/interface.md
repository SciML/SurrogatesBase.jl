# Extending SurrogatesBase

SurrogatesBase defines the small public contract shared by deterministic and stochastic surrogate
packages. It supplies interface tags and generic functions only; concrete packages own fitting,
evaluation, posterior representations, and data storage.

## Deterministic Surrogates

A deterministic implementation subtypes [`AbstractDeterministicSurrogate`](@ref) and is callable.
The minimum interface is:

```julia
(surrogate)(x)
update!(surrogate, new_x, new_y)
```

`x` can be a scalar, one point, or a batch, according to the concrete package's documented domain.
The base interface does not impose a container representation. A batch call must preserve the
correspondence between its inputs and predictions.

`update!` is in place. `new_x` and `new_y` represent matching observations: a batch has the same
number of locations and values in the same order, and a scalar update represents exactly one pair.
Implementations may return `nothing`, the updated surrogate, or another implementation-specific
value. Generic code must inspect the mutated surrogate instead of relying on that return value.

An implementation may additionally expose [`parameters`](@ref), [`hyperparameters`](@ref), and
[`update_hyperparameters!`](@ref). These are optional, so consumers should only call them when a
concrete surrogate documents support.

```julia
using SurrogatesBase

mutable struct LinearMock <: AbstractDeterministicSurrogate
    slope::Float64
end

(surrogate::LinearMock)(x) = surrogate.slope * x

function SurrogatesBase.update!(surrogate::LinearMock, new_x, new_y)
    surrogate.slope = last(new_y) / last(new_x)
    return nothing
end
```

## Stochastic Surrogates

A stochastic implementation subtypes [`AbstractStochasticSurrogate`](@ref) and implements:

```julia
update!(surrogate, new_x, new_y)
finite_posterior(surrogate, xs)
```

The update rules are the same as for deterministic surrogates. `finite_posterior` receives query
inputs in a concrete implementation's documented representation and returns that implementation's
finite-dimensional posterior object. That object must preserve the query ordering and document the
operations it supports, such as `Statistics.mean`, `Statistics.var`, or `rand`.

```julia
using Statistics
using SurrogatesBase

mutable struct ConstantPosteriorSurrogate <: AbstractStochasticSurrogate
    mean_value::Float64
end

struct ConstantPosterior
    means::Vector{Float64}
end

Statistics.mean(posterior::ConstantPosterior) = posterior.means

function SurrogatesBase.finite_posterior(surrogate::ConstantPosteriorSurrogate, xs)
    return ConstantPosterior(fill(surrogate.mean_value, length(xs)))
end
```

The package test suite validates these rules using mock subtypes that are driven only through the
public SurrogatesBase functions. This keeps the extension contract independent of implementation
details from any concrete surrogate package.
