using ITensors, ITensorMPS
using Printf
using Random

function heisenberg(N)
  os = OpSum()
  for j in 1:(N - 1)
    os += "Sz", j, "Sz", j + 1
    os += 0.5, "S+", j, "S-", j + 1
    os += 0.5, "S-", j, "S+", j + 1
  end
  return os
end

Random.seed!(1234)

N = 10

# Create N spin-one degrees of freedom
sites = siteinds("S=1", N)
# Alternatively can make spin-half sites instead
#sites = siteinds("S=1/2", N)

os = heisenberg(N)

# Input operator terms which define a Hamiltonian
# Convert these terms to an MPO tensor network
H = MPO(os, sites)

# Create an initial random matrix product state
# psi0 = random_mps(sites; linkdims=10)
psi0 = MPS(sites, j -> isodd(j) ? "↑" : "↓")

# Plan to do 5 DMRG sweeps:
nsweeps = 5
# Set maximum MPS bond dimensions for each sweep
maxdim = [10]
# Set maximum truncation error allowed when adapting bond dimensions
cutoff = [1E-11]

# Run the DMRG algorithm, returning energy and optimized MPS
energy, psi = dmrg(H, psi0; nsweeps, maxdim, cutoff)
@show inner(psi', H, psi)
@show inner(psi, psi)
@printf("Final energy = %.12f\n", energy)
