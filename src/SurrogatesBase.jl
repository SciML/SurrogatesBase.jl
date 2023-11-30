module SurrogatesBase

export AbstractDeterministicSurrogate
export AbstractStochasticSurrogate

export add_points!
export update_hyperparameters!, hyperparameters
export finite_posterior

"""
    abstract type AbstractDeterministicSurrogate <: Function end

An abstract type for deterministic surrogates.

    (s::AbstractDeterministicSurrogate)(x)

Subtypes of `AbstractDeterministicSurrogate` are callable with an input point `x`. The result
is an evaluation of the surrogate at `x`, corresponding to an approximation of the underlying
function at `x`.

 # Examples
 ```jldoctest
 julia> struct ZeroSurrogate <: AbstractDeterministicSurrogate end

 julia> (::ZeroSurrogate)(x) = 0

 julia> s = ZeroSurrogate()
 ZeroSurrogate()

 julia> s(4) == 0
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
    add_points!(s, new_xs::AbstractVector, new_ys::AbstractVector)

Add evaluations `new_ys` at points `new_xs` to a surrogate `s`.

Use `add_points!(s, eachslice(X, dims = 2), new_ys)` if `X` is a matrix.
"""
function add_points! end

"""
    update_hyperparameters!(s, prior)

Use information passed in `prior` to perform a hyperparameter update.

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

- `mean(finite_posterior(s,xs))` returns a vector of posterior means at `xs`
- `var(finite_posterior(s,xs))` returns a vector of posterior variances at `xs`
- `mean_and_var(finite_posterior(s,xs))` returns a `Tuple` consisting of a vector
of posterior means and a vector of posterior variances at `xs`
- `rand(finite_posterior(s,xs))` returns a sample from the joint posterior at points `xs`

Use `mean(finite_posterior(s, eachslice(X, dims = 2)))` if `X` is a matrix.
"""
function finite_posterior end

end
