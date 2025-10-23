using StreamingSampling
using Documenter

DocMeta.setdocmeta!(StreamingSampling, :DocTestSetup, :(using StreamingSampling); recursive=true)

makedocs(;
    modules=[StreamingSampling],
    authors="Emmanuel Lujan",
    sitename="StreamingSampling.jl",
    format=Documenter.HTML(;
        canonical="https://emmanuellujan.github.io/StreamingSampling.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/emmanuellujan/StreamingSampling.jl",
    devbranch="main",
)
