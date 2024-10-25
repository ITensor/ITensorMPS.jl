# ITensorMPS.jl

[![Docs](https://img.shields.io/badge/docs-latest-blue.svg)](https://itensor.github.io/ITensors.jl/dev/)
[![Build Status](https://github.com/ITensor/ITensorMPS.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/ITensor/ITensorMPS.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/invenia/BlueStyle)

Finite MPS and MPO methods based on the Julia version of [ITensor](https://www.itensor.org) ([ITensors.jl](https://github.com/ITensor/ITensors.jl)). See the [ITensors.jl documentation](https://itensor.github.io/ITensors.jl/dev/) for more details.

## News

### ITensorMPS.jl v0.3 release notes

All MPS/MPO code from [ITensors.jl](https://github.com/ITensor/ITensors.jl) and [ITensorTDVP.jl](https://github.com/ITensor/ITensorTDVP.jl) has been moved into this repository and this repository now relies on ITensors.jl v0.7 and above. All of the MPS/MPO functionality that was previously in ITensors.jl and ITensorTDVP.jl will be developed here from now on. For users of this repository, this change should not break any code, though please let us know if you have any issues.

#### Upgrade guide

If you are using any MPS/MPO functionality of ITensors.jl, such as the `MPS` and `MPO` types or constructors thereof (like `random_mps`), `OpSum`, `siteinds`, `dmrg`, `apply`, etc. you should install the ITensorMPS.jl package with `import Pkg; Pkg.add("ITensorMPS")` and add `using ITensorMPS` to your code. Additionally, if you are currently using [ITensorTDVP.jl](https://github.com/ITensor/ITensorTDVP.jl), you should replace `using ITensorTDVP` with `using ITensorMPS` in your code.

### ITensorMPS.jl v0.2.1 release notes

#### New features

This release introduces a new (experimental) function `expand` for performing global Krylov expansion based on [arXiv:2005.06104](https://arxiv.org/abs/2005.06104). It is a re-export of the `expand` function introduced in ITensorTDVP.jl v0.4.1, see the [ITensorTDVP.jl v0.4.1 release notes](https://github.com/ITensor/ITensorTDVP.jl/tree/main?tab=readme-ov-file#itensortdvpjl-v041-release-notes) for more details.

### ITensorMPS.jl v0.2 release notes

#### Breaking changes

ITensorMPS.jl v0.2 has been released, which is a breaking release. It updates to using ITensorTDVP.jl v0.4, which has a number of breaking changes to the `tdvp`, `linsolve`, and `dmrg_x` functions. See the [ITensorTDVP.jl v0.4 release notes](https://github.com/ITensor/ITensorTDVP.jl/blob/main/README.md#itensortdvpjl-v04-release-notes) for details.
