using Documenter, SurrogatesBase

DocMeta.setdocmeta!(
    SurrogatesBase,
    :DocTestSetup,
    :(using SurrogatesBase)
)

pages = [
    "Home" => "index.md",
    "Developer Interface" => "interface.md",
    "Public API" => "api.md",
]

ENV["GKSwstype"] = "100"

makedocs(
    modules = [SurrogatesBase],
    sitename = "SurrogatesBase.jl",
    clean = true,
    checkdocs = :exports,
    doctest = true,
    linkcheck = true,
    format = Documenter.HTML(
        assets = ["assets/favicon.ico"],
        canonical = "https://docs.sciml.ai/SurrogatesBase/stable/"
    ),
    pages = pages
)

deploydocs(repo = "github.com/SciML/SurrogatesBase.jl"; push_preview = true)
