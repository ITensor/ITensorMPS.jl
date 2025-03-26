# ITensorMPS.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://docs.itensor.org/ITensorMPS/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://docs.itensor.org/ITensorMPS/dev/)
[![Build Status](https://github.com/ITensor/ITensorMPS.jl/actions/workflows/Tests.yml/badge.svg?branch=main)](https://github.com/ITensor/ITensorMPS.jl/actions/workflows/Tests.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/ITensor/ITensorMPS.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/ITensor/ITensorMPS.jl)
[![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/invenia/BlueStyle)
[![Aqua](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

Finite MPS and MPO methods based on the Julia version of [ITensor](https://www.itensor.org) ([ITensors.jl](https://github.com/ITensor/ITensors.jl)). See the [ITensorMPS.jl documentation](https://docs.itensor.org/ITensorMPS) for more details.

## Support

<picture>
  <source media="(prefers-color-scheme: dark)" width="20%" srcset="docs/src/assets/CCQ-dark.png">
  <img alt="Flatiron Center for Computational Quantum Physics logo." width="20%" src="docs/src/assets/CCQ.png">
</picture>


ITensorMPS.jl is supported by the Flatiron Institute, a division of the Simons Foundation.

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

## Example DMRG Calculation

DMRG is an iterative algorithm for finding the dominant
eigenvector of an exponentially large, Hermitian matrix.
It originates in physics with the purpose of finding
eigenvectors of Hamiltonian (energy) matrices which model
the behavior of quantum systems.

````julia
using ITensors, ITensorMPS
let
  # Create 100 spin-one indices
  N = 100
  sites = siteinds("S=1", N)

  # Input operator terms which define
  # a Hamiltonian matrix, and convert
  # these terms to an MPO tensor network
  # (here we make the 1D Heisenberg model)
  os = OpSum()
  for j in 1:(N - 1)
    os += "Sz", j, "Sz", j + 1
    os += 0.5, "S+", j, "S-", j + 1
    os += 0.5, "S-", j, "S+", j + 1
  end
  H = MPO(os, sites)

  # Create an initial random matrix product state
  psi0 = random_mps(sites)

  # Plan to do 5 passes or 'sweeps' of DMRG,
  # setting maximum MPS internal dimensions
  # for each sweep and maximum truncation cutoff
  # used when adapting internal dimensions:
  nsweeps = 5
  maxdim = [10, 20, 100, 100, 200]
  cutoff = 1E-10

  # Run the DMRG algorithm, returning energy
  # (dominant eigenvalue) and optimized MPS
  energy, psi = dmrg(H, psi0; nsweeps, maxdim, cutoff)
  println("Final energy = $energy")

  nothing
end

# output

# After sweep 1 energy=-137.954199761732 maxlinkdim=9 maxerr=2.43E-16 time=9.356
# After sweep 2 energy=-138.935058943878 maxlinkdim=20 maxerr=4.97E-06 time=0.671
# After sweep 3 energy=-138.940080155429 maxlinkdim=92 maxerr=1.00E-10 time=4.522
# After sweep 4 energy=-138.940086009318 maxlinkdim=100 maxerr=1.05E-10 time=11.644
# After sweep 5 energy=-138.940086058840 maxlinkdim=96 maxerr=1.00E-10 time=12.771
# Final energy = -138.94008605883985
````

You can find more examples of running `dmrg` and related algorithms [here](https://github.com/ITensor/ITensorMPS.jl/tree/main/examples).

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

