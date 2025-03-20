using ITensorMPS: ITensorMPS
using Documenter: Documenter, DocMeta, deploydocs, makedocs

DocMeta.setdocmeta!(ITensorMPS, :DocTestSetup, :(using ITensorMPS); recursive=true)

include("make_index.jl")

makedocs(;
  modules=[ITensorMPS],
  authors="ITensor developers <support@itensor.org> and contributors",
  sitename="ITensorMPS.jl",
  format=Documenter.HTML(;
    canonical="https://ITensor.github.io/ITensorMPS.jl", edit_link="main", assets=String[]
  ),
  pages=["Home" => "index.md"],
  warnonly=true,
)

deploydocs(; repo="github.com/ITensor/ITensorMPS.jl", devbranch="main", push_preview=true)
