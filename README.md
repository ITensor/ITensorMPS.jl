# ITensorMPS.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://itensor.github.io/ITensorMPS.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://itensor.github.io/ITensorMPS.jl/dev/)
[![Build Status](https://github.com/ITensor/ITensorMPS.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/ITensor/ITensorMPS.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/ITensor/ITensorMPS.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/ITensor/ITensorMPS.jl)
[![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/invenia/BlueStyle)

MPS and MPO methods based on ITensor (ITensors.jl).

This package currently re-exports the MPS and MPO functionality of the [ITensors.jl](https://github.com/ITensor/ITensors.jl) and [ITensorTDVP.jl](https://github.com/ITensor/ITensorTDVP.jl) packages, including applications like DMRG, TDVP, applying MPO to MPS, applying gates, etc.

The goal will be to move the MPS and MPO code from those packages into this repository, and for the `ITensorMPS.jl` package to be the primary entry point for the finite MPS and MPO functionality of ITensor.
