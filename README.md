# SurrogatesBase.jl

API for deterministic and stochastic surrogates.

Given data $((x_1, y_1), \ldots, (x_N, y_N))$ obtained by evaluating a function $y_i =
f(x_i)$ or sampling from a conditional probability density $p_{Y|X}(Y = y_i|X = x_i)$,
a **deterministic surrogate** is a function $s(x)$ (e.g. a [radial basis function
interpolator](https://en.wikipedia.org/wiki/Radial_basis_function_interpolation)) that
uses the data to approximate $f$ or some statistic of $p_{Y|X}$ (e.g. the mean),
whereas a **stochastic surrogate** is a stochastic process (e.g. a [Gaussian process
approximation](https://en.wikipedia.org/wiki/Gaussian_process_approximations)) that uses
the data to approximate $f$ or $p_{Y|X}$ *and* quantify the uncertainty of the
approximation.

## Deterministic Surrogates

Deterministic surrogates `s` are subtypes of `SurrogatesBase.AbstractDeterministicSurrogate`, 
which is a subtype of `Function`.
The method `add_points!(s, xs, ys)` **must** be implemented and the surrogate **must** be
[callable](https://docs.julialang.org/en/v1/manual/methods/#Function-like-objects)
`s(xs)`, where `xs` is a `Vector` of inputs and `ys` is a `Vector` of corresponding outputs.
For single points `x` and `y`, call these methods via `add_points!(s, [x], [y])`
and `s([x])`.

If the surrogate `s` has tunable hyper-parameters, the methods
`update_hyperparameters!(s, prior)` and `hyperparameters(s)` have to be implemented.

### Example

```julia
using SurrogatesBase

struct RBF{T} <: AbstractDeterministicSurrogate
    scale::T
    centers::Vector{T}
    weights::Vector{T}
end

(rbf::RBF)(xs) = [rbf.weights' * exp.(-rbf.scale * (x .- rbf.centers).^2)
                  for x in xs]

function SurrogatesBase.add_points!(rbf::RBF, xs, ys)
    # update rbf.weights
    return rbf
end

SurrogatesBase.hyperparameters(rbf::RBF) = (rbf.scale,)

function SurrogatesBase.update_hyperparameters!(rbf::RBF, prior)
    # change rbf.scale and adapt rbf.weights, if necessary
    return rbf
end
```

## Stochastic Surrogates

Stochastic surrogates `s` are subtypes of `SurrogatesBase.AbstractStochasticSurrogate`.
The method `add_points!(s, xs, ys)` **must** be implemented, where `xs` is a `Vector` of
inputs and `ys` is a `Vector` of corresponding outputs. For single points `x` and `y`, call 
`add_points!(s, [x], [y])` for adding `x`, `y` into the surrogate `s`.

A stochastic surrogate `s` **must** implement a method `finite_posterior(s, xs)`,  where `xs` is 
a `Vector` of points. The returned object provides methods for  working with the finite 
dimensional posterior distribution at points `xs`.
The following methods might be supported:

- `mean(finite_posterior(s,xs))` returns a `Vector` of posterior means at `xs`
- `var(finite_posterior(s,xs))` returns a `Vector` of posterior variances at `xs`
- `mean_and_var(finite_posterior(s,xs))` returns a `Tuple` consisting of a `Vector`
of posterior means and a `Vector` of posterior variances at `xs`
- `rand(finite_posterior(s,xs))` returns a `Vector`, which is a sample from the joint posterior at points `xs`

If the surrogate `s` has tunable hyper-parameters, the methods
`update_hyperparameters!(s, prior)` and `hyperparameters(s)` have to be implemented.


### Example

```julia
using SurrogatesBase

mutable struct GaussianProcessSurrogate{D, R, GP, H <: NamedTuple} <: AbstractStochasticSurrogate
    xs::Vector{D}
    ys::Vector{R}
    gp_process::GP
    hyperparameters::H
end

function SurrogatesBase.add_points!(g::GaussianProcessSurrogate, new_xs, new_ys)
    append!(g.xs, new_xs)
    append!(g.ys, new_ys)
    # condition the prior `g.gp_process` on new data to obtain a posterior
    # update g.gp_process to the posterior process
    return g
end

function SurrogatesBase.finite_posterior(g::GaussianProcessSurrogate, xs)
    # Return a finite dimensional projection of g.gp_process at points xs.
    # The returned object GP_finite supports methods mean(GP_finite) and
    # var(GP_finite) for obtaining the vector of means and variances at points xs.
end

SurrogatesBase.hyperparameters(g::GaussianProcessSurrogate) = g.hyperparameters

function SurrogatesBase.update_hyperparameters!(g::GaussianProcessSurrogate, prior)
    # Use the passed prior on hyperparameters, e.g., some parameter could be uniformly 
    # distributed between an upper and lower bound, to find better hyperparameters.
    # Update g.gp_process, g.hyperparameters to the improved hyperparameters.
    return g
end
```