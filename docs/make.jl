using Documenter, SurrogatesBase

DocMeta.setdocmeta!(SurrogatesBase,
    :DocTestSetup,
    :(using SurrogatesBase))

cp("./docs/Manifest.toml", "./docs/src/assets/Manifest.toml", force = true)
cp("./docs/Project.toml", "./docs/src/assets/Project.toml", force = true)

pages = [
    "Home" => "index.md",
    "interface.md",
    "api.md"
]

ENV["GKSwstype"] = "100"

makedocs(modules = [SurrogatesBase],
    sitename = "SurrogatesBase.jl",
    clean = true,
    doctest = true,
    linkcheck = true,
    format = Documenter.HTML(assets = ["assets/favicon.ico"],
        canonical = "https://docs.sciml.ai/SurrogatesBase/stable/"),
    pages = pages)

deploydocs(repo = "github.com/SciML/SurrogatesBase.jl"; push_preview = true)
