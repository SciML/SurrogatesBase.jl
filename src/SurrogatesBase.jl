module SurrogatesBase

import Statistics: mean, var
import Base: rand

# AbstractSurrogate interface
export AbstractSurrogate,
    add_point!, add_points!,
    update_hyperparameters!, hyperparameters,
    posterior, posterior_at_point,
    mean, mean_at_point,
    var, var_at_point,
    rand, rand_at_point,
    logpdf

abstract type AbstractSurrogate <: Function end

# setup broadcasting used in, e.g., default implementation of add_points!
Base.broadcastable(s::AbstractSurrogate) = Ref(s)

"""
    (s::AbstractSurrogate)(x::AbstractVector)

Compute an approximation at point `x`.
"""
function (s::AbstractSurrogate)(x::AbstractVector) end

"""
    add_point!(s::AbstractSurrogate, new_x::AbstractVector, new_y)

Add an evaluation `new_y` at point `new_x` to the surrogate.
"""
function add_point! end
"""
    add_points!(s::AbstractSurrogate, new_xs::AbstractVector{<:AbstractVector}, new_ys::AbstractVector)

Add evaluations `new_ys` at points `new_xs` to the surrogate.

Use `add_points!(s, eachslice(X, dims = 2), new_ys)` if `X` is a matrix.
"""
function add_points!(s::AbstractSurrogate,
    new_xs::AbstractVector{<:AbstractVector},
    new_ys::AbstractVector)
    add_point!.(s, new_xs, new_ys)
end

"""
    update_hyperparameters!(s::AbstractSurrogate, prior)

Use prior on hyperparameters passed in `prior` to perform an update.

See also [`hyperparameters`](@ref).
"""
function update_hyperparameters! end

"""
    hyperparameters(s::AbstractSurrogate)

Return a `NamedTuple`, in which names are hyperparameters and values are currently used
values of hyperparameters in `s`.

See also [`update_hyperparameters!`](@ref).
"""
function hyperparameters end

"""
    posterior(s::AbstractSurrogate, xs::AbstractVector{<:AbstractVector})

Return a joint posterior at points `xs`.

Use `posterior(s, eachslice(X, dims = 2))` if `X` is a matrix.
"""
function posterior end
"""
    posterior_at_point(s::AbstractSurrogate, x::AbstractVector)

Return posterior at point `x`.
"""
posterior_at_point(s::AbstractSurrogate, x::AbstractVector) = posterior(s, [x])

"""
    mean_at_point(s::AbstractSurrogate, x::AbstractVector)

Return mean at point `x`.
"""
function mean_at_point end
"""
    mean(s::AbstractSurrogate, xs::AbstractVector{<:AbstractVector})

Return a vector of means at points `xs`.

Use `mean(s, eachslice(X, dims = 2))` if `X` is a matrix.
"""
function mean(s::AbstractSurrogate, xs::AbstractVector{<:AbstractVector})
    mean_at_point.(s, xs)
end

"""
    var_at_point(s::AbstractSurrogate, x::AbstractVector)

Return variance at point `x`.
"""
function var_at_point end

"""
    var(s::AbstractSurrogate, xs::AbstractVector{<:AbstractVector})

Return a vector of variances at points `xs`.
"""
function var(s::AbstractSurrogate, xs::AbstractVector{<:AbstractVector})
    var_at_point.(s, xs)
end

"""
    rand(s::AbstractSurrogate, xs::AbstractVector{<:AbstractVector})

Return a sample from the joint posterior at points `xs`.
"""
function rand end

"""
    rand_at_point(s::AbstractSurrogate, x::AbstractVector)

Return a sample from the posterior distribution at a point `x`.
"""
function rand_at_point(s::AbstractSurrogate, x::AbstractVector)
    only(rand(s::AbstractSurrogate, [x]))
end

"""
    logpdf(s::AbstractSurrogate)

Return a log marginal posterior predictive probability.
"""
function logpdf end

end
