module SurrogatesBase

export AbstractDeterministicSurrogate
export AbstractStochasticSurrogate

export include_data!
export update_hyperparameters!, hyperparameters
export finite_posterior

"""
    abstract type AbstractDeterministicSurrogate <: Function end

An abstract type for deterministic surrogates.

    (s::AbstractDeterministicSurrogate)(xs)

Subtypes of `AbstractDeterministicSurrogate` are callable with a `Vector` of points `xs`.
The result is a `Vector` of evaluations of the surrogate at points `xs`, corresponding to
approximations of the underlying function at points `xs` respectively.

 # Examples
 ```jldoctest
 julia> struct ZeroSurrogate <: AbstractDeterministicSurrogate end

 julia> (::ZeroSurrogate)(xs) = 0

 julia> s = ZeroSurrogate()
 ZeroSurrogate()

 julia> s([4]) == 0
 true
 ```
"""
abstract type AbstractDeterministicSurrogate <: Function end

"""
    abstract type AbstractStochasticSurrogate end

An abstract type for stochastic surrogates.

See also [`finite_posterior`](@ref).
"""
abstract type AbstractStochasticSurrogate end

"""
    include_data!(s, new_xs::AbstractVector, new_ys::AbstractVector)

Include data `new_ys` at points `new_xs` into the surrogate `s`, i.e., refit the surrogate `s`
to incorporate new data points.

If the surrogate `s` is deterministic, the `new_ys` correspond to function evaluations, if
`s` is a stochastic surrogate, the `new_ys` are samples from a conditional probability
distribution.

Use `include_data!(s, eachslice(X, dims = 2), new_ys)` if `X` is a matrix.
"""
function include_data! end

"""
    update_hyperparameters!(s, prior)

Update the hyperparameters of the surrogate `s` by performing hyperparameter optimization
using the information in `prior`. After changing hyperparameters of `s`, fit `s` to past
data.

See also [`hyperparameters`](@ref).
"""
function update_hyperparameters! end

"""
    hyperparameters(s)

Return a `NamedTuple`, in which names are hyperparameters and values are currently used
values of hyperparameters in `s`.

See also [`update_hyperparameters!`](@ref).
"""
function hyperparameters end

"""
    finite_posterior(s::AbstractStochasticSurrogate, xs::AbstractVector)

Return a posterior distribution at points `xs`.

An `AbstractStochasticSurrogate` might implement some or all of the following methods on
the returned object:

- `mean(finite_posterior(s,xs))` returns a `Vector` of posterior means at `xs`
- `var(finite_posterior(s,xs))` returns a `Vector` of posterior variances at `xs`
- `mean_and_var(finite_posterior(s,xs))` returns a `Tuple` consisting of a `Vector`
of posterior means and a `Vector` of posterior variances at `xs`
- `rand(finite_posterior(s,xs))` returns a `Vector`, which is a sample from the joint
posterior at points `xs`

Use `mean(finite_posterior(s, eachslice(X, dims = 2)))` if `X` is a matrix.
"""
function finite_posterior end

end
