module SurrogatesBase

# AbstractSurrogate interface
export AbstractSurrogate,
    add_point!, add_points!,
    supports_hyperparameters, update_hyperparameters!, hyperparameters,
    supports_posterior, posterior

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
    supports_hyperparameters(s::AbstractSurrogate)

Return `true` if `s` supports hyperparameters, otherwise return `false`.

See also [`update_hyperparameters!`](@ref), [`hyperparameters`](@ref).
"""
supports_hyperparameters(s::AbstractSurrogate) = false
"""
    update_hyperparameters!(s::AbstractSurrogate, prior)

If `s` supports hyperparameters, use `prior` to perform an update.

See also [`supports_hyperparameters`](@ref),  [`hyperparameters`](@ref).
"""
function update_hyperparameters!(s::AbstractSurrogate, prior)
    error("update_hyperparameters! is not implemented")
end

"""
    hyperparameters(s::AbstractSurrogate)

Return a `NamedTuple`, where names are hyperparameters and values are currently used
values of hyperparameters by `s`.

See also [`supports_hyperparameters`](@ref),  [`update_hyperparameters!`](@ref).
"""
function hyperparameters(s::AbstractSurrogate)
    error("access to hyperparameters is not implemented")
end

"""
    supports_posterior(s::AbstractSurrogate)

Return `true`, if `s` can provide joint posterior distribution, otherwise return `false`.

See also [`posterior`](@ref).
"""
supports_posterior(s::AbstractSurrogate) = false
"""
    posterior(s::AbstractSurrogate, x)

Return a joint posterior at `x = [x_1, ..., x_m]`.
"""
posterior(s::AbstractSurrogate, x) = error("posterior is not implemented")

end
