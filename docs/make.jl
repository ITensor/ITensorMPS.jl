using ITensorMPS
using Documenter

DocMeta.setdocmeta!(ITensorMPS, :DocTestSetup, :(using ITensorMPS); recursive=true)

makedocs(;
    modules=[ITensorMPS],
    authors="ITensor developers",
    sitename="ITensorMPS.jl",
    format=Documenter.HTML(;
        canonical="https://ITensor.github.io/ITensorMPS.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/ITensor/ITensorMPS.jl",
    devbranch="main",
)
