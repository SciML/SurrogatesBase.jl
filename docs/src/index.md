# SurrogatesBase.jl: A Common Interface for Surrogate Libraries

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

## Installation

To install SurrogatesBase.jl, use the Julia package manager:

```julia
using Pkg
Pkg.add("SurrogatesBase")
```

## Contributing

  - Please refer to the
    [SciML ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://github.com/SciML/ColPrac/blob/master/README.md)
    for guidance on PRs, issues, and other matters relating to contributing to SciML.

  - See the [SciML Style Guide](https://github.com/SciML/SciMLStyle) for common coding practices and other style decisions.
  - There are a few community forums:
    
      + The #diffeq-bridged and #sciml-bridged channels in the
        [Julia Slack](https://julialang.org/slack/)
      + The #diffeq-bridged and #sciml-bridged channels in the
        [Julia Zulip](https://julialang.zulipchat.com/#narrow/stream/279055-sciml-bridged)
      + On the [Julia Discourse forums](https://discourse.julialang.org)
      + See also [SciML Community page](https://sciml.ai/community/)

## Reproducibility

```@raw html
<details><summary>The documentation of this SciML package was built using these direct dependencies,</summary>
```

```@example
using Pkg # hide
Pkg.status() # hide
```

```@raw html
</details>
```

```@raw html
<details><summary>and using this machine and Julia version.</summary>
```

```@example
using InteractiveUtils # hide
versioninfo() # hide
```

```@raw html
</details>
```

```@raw html
<details><summary>A more complete overview of all dependencies and their versions is also provided.</summary>
```

```@example
using Pkg # hide
Pkg.status(; mode = PKGMODE_MANIFEST) # hide
```

```@raw html
</details>
```

```@eval
using TOML
using Markdown
version = TOML.parse(read("../../Project.toml", String))["version"]
name = TOML.parse(read("../../Project.toml", String))["name"]
link_manifest = "https://github.com/SciML/" * name * ".jl/tree/gh-pages/v" * version *
                "/assets/Manifest.toml"
link_project = "https://github.com/SciML/" * name * ".jl/tree/gh-pages/v" * version *
               "/assets/Project.toml"
Markdown.parse("""You can also download the
[manifest]($link_manifest)
file and the
[project]($link_project)
file.
""")
```
