module SurrogatesBase

import Statistics: mean, var
import Base: rand

# AbstractSurrogate interface
export AbstractSurrogate,
    add_point!,
    update_hyperparameters!, hyperparameters,
    posterior, mean, var, rand, logpdf

"""
    abstract type AbstractSurrogate end

An abstract type for defining surrogate interface.

    (s::AbstractSurrogate)(x::AbstractVector)

Subtypes of `AbstractSurrogate` need to be callable with input points `x` that returns
an approximation at `x`, i.e., evaluates the surrogate at `x`.

 # Examples
 ```jldoctest
 julia> struct ZeroSurrogate <: AbstractSurrogate end

 julia> (::ZeroSurrogate)(x::AbstractVector) = 0

 julia> s = ZeroSurrogate()
 ZeroSurrogate()

 julia> s(rand(5)) == 0
 true
 ```
 """
abstract type AbstractSurrogate <: Function end

"""
    add_point!(s::AbstractSurrogate, new_x::AbstractVector, new_y::Number)
    add_point!(s::AbstractSurrogate, new_x::Union{AbstractVector, Number}, new_y::Number)

Add an evaluation `new_y` at point `new_x` to the surrogate.
"""
function add_point! end
"""
    add_point!(s::AbstractSurrogate, new_xs::AbstractVector{<:AbstractVector}, new_ys::AbstractVector)

Add evaluations `new_ys` at points `new_xs` to the surrogate.

Use `add_point!(s, eachslice(X, dims = 2), new_ys)` if `X` is a matrix.
"""
function add_point!(s::AbstractSurrogate,
    new_xs::AbstractVector{<:AbstractVector},
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
    posterior(s::AbstractSurrogate, xs::AbstractVector{<:AbstractVector})

Return a joint posterior at points `xs`.

Use `posterior(s, eachslice(X, dims = 2))` if `X` is a matrix.
"""
function posterior end
"""
    posterior(s::AbstractSurrogate, x::AbstractVector)

Return posterior at point `x`.
"""
posterior(s::AbstractSurrogate, x::AbstractVector) = posterior(s, [x])

"""
    mean(s::AbstractSurrogate, x::AbstractVector)

Return mean at point `x`.
"""
function mean end
"""
    mean(s::AbstractSurrogate, xs::AbstractVector{<:AbstractVector})

Return a vector of means at points `xs`.

Use `mean(s, eachslice(X, dims = 2))` if `X` is a matrix.
"""
function mean(s::AbstractSurrogate, xs::AbstractVector{<:AbstractVector})
    mean.(Ref(s), xs)
end

"""
    var(s::AbstractSurrogate, x::AbstractVector)

Return variance at point `x`.
"""
function var end

"""
    var(s::AbstractSurrogate, xs::AbstractVector{<:AbstractVector})

Return a vector of variances at points `xs`.
"""
function var(s::AbstractSurrogate, xs::AbstractVector{<:AbstractVector})
    var.(Ref(s), xs)
end

"""
    rand(s::AbstractSurrogate, xs::AbstractVector{<:AbstractVector})

Return a sample from the joint posterior at points `xs`.
"""
function rand end

"""
    rand(s::AbstractSurrogate, x::AbstractVector)

Return a sample from the posterior distribution at a point `x`.
"""
rand(s::AbstractSurrogate, x::AbstractVector) = only(rand(s::AbstractSurrogate, [x]))

"""
    logpdf(s::AbstractSurrogate)

Return a log marginal posterior predictive probability.
"""
function logpdf end

end
