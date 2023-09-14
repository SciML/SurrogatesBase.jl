module SurrogatesBase

# AbstractSurrogate interface
export AbstractSurrogate,
    add_point!, add_points!,
    update_hyperparameters!, hyperparameters,
    posterior, posterior_at_point,
    mean, mean_at_point
    var, var_at_point,
    rand, rand_at_point,
    logpdf

abstract type AbstractSurrogate <: Function end

"""
    (s::AbstractSurrogate)(x)

Compute an approximation at `x`.
"""
function (s::AbstractSurrogate)(x) end

"""
    add_point!(s::AbstractSurrogate, new_x, new_y)

Add an evaluation `new_y` at point `new_x` to the surrogate.

See also [`add_points!`](@ref).
"""
function add_point!(s::AbstractSurrogate, new_x, new_y) end
"""
    add_points!(s::AbstractSurrogate, new_xs, new_ys)

Add evaluations `new_ys` at points `new_xs` to the surrogate.

See also [`add_point!`](@ref).
"""
function add_points!(s::AbstractSurrogate, new_xs, new_ys)
    length(new_xs) == length(new_ys) ||
        throw(ArgumentError("new_xs, new_ys have different lengths"))
    for (x, y) in zip(new_xs, new_ys)
        add_point!(s, x, y)
    end
end

"""
    update_hyperparameters!(s::AbstractSurrogate, prior)

If `s` supports hyperparameters, use `prior` to perform an update.

See also [`hyperparameters`](@ref).
"""
function update_hyperparameters!(s::AbstractSurrogate, prior)
    error("update_hyperparameters! is not implemented")
end

"""
    hyperparameters(s::AbstractSurrogate)

Return a `NamedTuple`, where names are hyperparameters and values are currently used
values of hyperparameters by `s`.

See also [`update_hyperparameters!`](@ref).
"""
function hyperparameters(s::AbstractSurrogate)
    error("access to hyperparameters is not implemented")
end

"""
    posterior(s::AbstractSurrogate, x)

Return a joint posterior at `m` points in `x = [x_1, ..., x_m]`.
"""
posterior(s::AbstractSurrogate, x) = error("posterior is not implemented")

"""
    posterior_at_point(s::AbstractSurrogate, x)

Return a posterior at point `x`.
"""
posterior_at_point(s::AbstractSurrogate, x) = posterior(s, [x])

"""
    mean(s::AbstractSurrogate, x)

For `m` points in `x = [x_1, ..., x_m]`, return a vector of means `[μ_at_x_1, ..., μ_at_x_m]`.
"""
function mean(s::AbstractSurrogate, x) end

"""
    mean_at_point(s::AbstractSurrogate, x)

For a point `x`, return a mean at `x`.
"""
mean_at_point(s::AbstractSurrogate, x) = only(mean(s, [x]))
"""
    var(s::AbstractSurrogate, x)

For `m` points in `x = [x_1, ..., x_m]`, return a vector of variances `[σ²_at_x_1, ..., σ²_at_x_m]`
"""
function var(s::AbstractSurrogate, x) end
"""
    var_at_point(s::AbstractSurrogate, x)

For a point `x`, return a variance at `x`.
"""
var_at_point(s::AbstractSurrogate, x) = only(var(s, [x]))
"""
    rand(s::AbstractSurrogate, x)

Sample from a joint posterior distribution at `m` points in `x = [x_1, ..., x_m]`.
"""
function rand(s::AbstractSurrogate, x) = end
"""
    rand_at_point(s::AbstractSurrogate, x)

Sample from a posterior distribution at a point `x`.
"""
rand_at_point(s::AbstractSurrogate, x) = rand(s::AbstractSurrogate, [x])
"""
    logpdf(s::AbstractSurrogate)

Log marginal posterior predictive probability.
"""
function logpdf(s::AbstractSurrogate) end

end
