module SurrogatesBase

import Statistics: mean, var
import Base: rand
import StatsBase: mean_and_var

# AbstractSurrogate interface
export AbstractSurrogate
export add_point!, add_points!
export update_hyperparameters!, hyperparameters
export mean, mean_at_point
export var, var_at_point
export mean_and_var, mean_and_var_at_point
export rand, rand_at_point

"""
    abstract type AbstractSurrogate end

An abstract type for formalizing surrogates.

    (s::AbstractSurrogate)(x)

Subtypes of `AbstractSurrogate` can be callable with input points `x` such that the result
is an evaluation of the surrogate at `x`, corresponding to an approximation of the underlying
function at `x`.

 # Examples
 ```jldoctest
 julia> struct ZeroSurrogate <: AbstractSurrogate end

 julia> (::ZeroSurrogate)(x) = 0

 julia> s = ZeroSurrogate()
 ZeroSurrogate()

 julia> s(4) == 0
 true
 ```
"""
abstract type AbstractSurrogate <: Function end

"""
    add_point!(s::AbstractSurrogate, new_x, new_y)

Add an evaluation `new_y` at point `new_x` to the surrogate.
"""
function add_point! end

"""
    add_points!(s::AbstractSurrogate, new_xs::AbstractVector, new_ys::AbstractVector)

Add evaluations `new_ys` at points `new_xs` to the surrogate.

Use `add_points!(s, eachslice(X, dims = 2), new_ys)` if `X` is a matrix.
"""
function add_points!(s::AbstractSurrogate,
    new_xs::AbstractVector,
    new_ys::AbstractVector)
    add_point!.(Ref(s), new_xs, new_ys)
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
    mean_at_point(s::AbstractSurrogate, x)

Return mean at point `x`.
"""
function mean_at_point end

"""
    mean(s::AbstractSurrogate, xs::AbstractVector)

Return a vector of means at points `xs`.

Use `mean(s, eachslice(X, dims = 2))` if `X` is a matrix.
"""
function mean(s::AbstractSurrogate, xs::AbstractVector)
    mean_at_point.(Ref(s), xs)
end

"""
    var_at_point(s::AbstractSurrogate, x::AbstractVector)

Return variance at point `x`.
"""
function var_at_point end

"""
    var(s::AbstractSurrogate, xs::AbstractVector)

Return a vector of variances at points `xs`.
"""
function var(s::AbstractSurrogate, xs::AbstractVector)
    var_at_point.(Ref(s), xs)
end

"""
mean_and_var_at_point(s::AbstractSurrogate, x)

Return a Tuple of mean and variance at point `x`.
"""
function mean_and_var_at_point(s::AbstractSurrogate, x)
    mean_at_point(s, x), var_at_point(s, x)
end

"""
    mean_and_var(s::AbstractSurrogate, xs)

Return a Tuple of vector of means and vector of variances at points `xs`.
"""
function mean_and_var(s::AbstractSurrogate, xs::AbstractVector)
    mean(s, xs), var(s, xs)
end

"""
    rand(s::AbstractSurrogate, xs::AbstractVector)

Return a sample from the joint posterior at points `xs`.
"""
function rand end

"""
    rand_at_point(s::AbstractSurrogate, x)

Return a sample from the posterior distribution at a point `x`.
"""
function rand_at_point(s::AbstractSurrogate, x)
    only(rand(s::AbstractSurrogate, [x]))
end

end
