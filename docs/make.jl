using ITensorMPS
using ITensors
using Documenter: Documenter, DocMeta, deploydocs, makedocs

DocMeta.setdocmeta!(ITensorMPS, :DocTestSetup, :(using ITensorMPS); recursive=true)
DocMeta.setdocmeta!(ITensors, :DocTestSetup, :(using ITensors); recursive=true)

include("make_index.jl")

makedocs(;
  # Allows using ITensors.jl docstrings in ITensorMPS.jl documentation:
  # https://github.com/JuliaDocs/Documenter.jl/issues/1734
  modules=[ITensorMPS, ITensors],
  authors="ITensor developers <support@itensor.org> and contributors",
  sitename="ITensorMPS.jl",
  format=Documenter.HTML(;
    canonical="https://itensor.github.io/ITensorMPS.jl",
    edit_link="main",
    assets=["assets/favicon.ico", "assets/extras.css"],
    prettyurls=false,
  ),
  pages=[
    "Home" => "index.md",
    "Tutorials" => [
      "DMRG" => "tutorials/DMRG.md",
      "Quantum Number Conserving DMRG" => "tutorials/QN_DMRG.md",
      "MPS Time Evolution" => "tutorials/MPSTimeEvolution.md",
    ],
    "Code Examples" => [
      "MPS and MPO Examples" => "examples/MPSandMPO.md",
      "DMRG Examples" => "examples/DMRG.md",
      "Physics (SiteType) System Examples" => "examples/Physics.md",
    ],
    "Documentation" => [
      "MPS and MPO" => "MPSandMPO.md",
      "SiteType and op, state, val functions" => "SiteType.md",
      "SiteTypes Included with ITensor" => "IncludedSiteTypes.md",
      "DMRG" => [
        "DMRG.md",
        "Sweeps.md",
        "ProjMPO.md",
        "ProjMPOSum.md",
        "Observer.md",
        "DMRGObserver.md",
      ],
      "OpSum" => "OpSum.md",
    ],
    "Frequently Asked Questions" =>
      ["DMRG FAQs" => "faq/DMRG.md", "Quantum Number (QN) FAQs" => "faq/QN.md"],
    "HDF5 File Formats" => "HDF5FileFormats.md",
  ],
  warnonly=true,
)

deploydocs(; repo="github.com/ITensor/ITensorMPS.jl", devbranch="main", push_preview=true)
