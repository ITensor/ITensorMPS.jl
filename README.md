# ITensorMPS.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://itensor.github.io/ITensorMPS.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://itensor.github.io/ITensorMPS.jl/dev/)
[![Build Status](https://github.com/ITensor/ITensorMPS.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/ITensor/ITensorMPS.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/ITensor/ITensorMPS.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/ITensor/ITensorMPS.jl)
[![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/invenia/BlueStyle)

Finite MPS and MPO methods based on ITensor (ITensors.jl).

This package currently re-exports the MPS and MPO functionality of the [ITensors.jl](https://github.com/ITensor/ITensors.jl) and [ITensorTDVP.jl](https://github.com/ITensor/ITensorTDVP.jl) packages, including applications like DMRG, TDVP, applying MPO to MPS, applying gates to MPS and MPO, etc.

See the [ITensor documentation](https://itensor.github.io/ITensors.jl/dev) and the [ITensorTDVP.jl examples](https://github.com/ITensor/ITensorTDVP.jl/tree/main/examples) for guides and examples on using this package.

## Upgrade guide

The goal will be to move the MPS and MPO code from the ITensors.jl package, along with all of the code from the ITensorTDVP.jl package, into this repository. If you are using any MPS/MPO functionality of ITensors.jl, such as `OpSum`, `siteinds`, `MPO`, `MPS, `randomMPS`, `dmrg`, `apply`, etc. you should install the ITensorMPS.jl package with `import Pkg; Pkg.add("ITensorMPS")` and add `using ITensorMPS` to your code. Additionally, if you are currently using ITensorTDVP, you should replace `using ITensorTDVP` with `using ITensorMPS` in your codes.
