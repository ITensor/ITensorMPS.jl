using ITensors: @disable_warn_order
using ITensorMPS: MPO, MPS, contract, inner, random_mps, siteinds, tdvp
using LinearAlgebra: norm
using Random: Random

include("03_models.jl")
include("03_updaters.jl")

function main()
  Random.seed!(1234)

  # Time dependent Hamiltonian is:
  # H(t) = H₁(t) + H₂(t) + …
  #      = f₁(t) H₁(0) + f₂(t) H₂(0) + …
  #      = cos(ω₁t) H₁(0) + cos(ω₂t) H₂(0) + …

  # Number of sites
  n = 6

  # How much information to output from TDVP
  # Set to 2 to get information about each bond/site
  # evolution, and 3 to get information about the
  # updater.
  outputlevel = 3

  # Frequency of time dependent terms
  ω₁ = 0.1
  ω₂ = 0.2

  # Nearest and next-nearest neighbor
  # Heisenberg couplings.
  J₁ = 1.0
  J₂ = 1.0

  time_step = 0.1
  time_stop = 1.0

  # nsite-update TDVP
  nsite = 2

  # Starting state bond/link dimension.
  # A product state starting state can
  # cause issues for TDVP without
  # subspace expansion.
  start_linkdim = 4

  # TDVP truncation parameters
  maxdim = 100
  cutoff = 1e-8

  tol = 1e-15

  @show n
  @show ω₁, ω₂
  @show J₁, J₂
  @show maxdim, cutoff, nsite
  @show start_linkdim
  @show time_step, time_stop

  ω⃗ = (ω₁, ω₂)
  f⃗ = map(ω -> (t -> cos(ω * t)), ω⃗)

  # H₀ = H(0) = H₁(0) + H₂(0) + …
  ℋ₁₀ = heisenberg(n; J=J₁, J2=0.0)
  ℋ₂₀ = heisenberg(n; J=0.0, J2=J₂)
  ℋ⃗₀ = (ℋ₁₀, ℋ₂₀)

  s = siteinds("S=1/2", n)

  H⃗₀ = map(ℋ₀ -> MPO(ℋ₀, s), ℋ⃗₀)

  # Initial state, ψ₀ = ψ(0)
  # Initialize as complex since that is what OrdinaryDiffEq.jl/DifferentialEquations.jl
  # expects.
  ψ₀ = complex.(random_mps(s, j -> isodd(j) ? "↑" : "↓"; linkdims=start_linkdim))

  @show norm(ψ₀)

  println()
  println("#"^100)
  println("Running TDVP with ODE updater")
  println("#"^100)
  println()

  ψₜ_ode = tdvp(
    -im * TimeDependentSum(f⃗, H⃗₀),
    time_stop,
    ψ₀;
    updater=ode_updater,
    updater_kwargs=(; reltol=tol, abstol=tol),
    time_step,
    maxdim,
    cutoff,
    nsite,
    outputlevel,
  )

  println()
  println("Finished running TDVP with ODE updater")
  println()

  println()
  println("#"^100)
  println("Running TDVP with Krylov updater")
  println("#"^100)
  println()

  ψₜ_krylov = tdvp(
    -im * TimeDependentSum(f⃗, H⃗₀),
    time_stop,
    ψ₀;
    updater=krylov_updater,
    updater_kwargs=(; tol, eager=true),
    time_step,
    cutoff,
    nsite,
    outputlevel,
  )

  println()
  println("Finished running TDVP with Krylov updater")
  println()

  println()
  println("#"^100)
  println("Running full state evolution with ODE updater")
  println("#"^100)
  println()

  @disable_warn_order begin
    ψₜ_full, _ = ode_updater(
      -im * TimeDependentSum(f⃗, contract.(H⃗₀)),
      contract(ψ₀);
      internal_kwargs=(; time_step=time_stop, outputlevel),
      reltol=tol,
      abstol=tol,
    )
  end

  println()
  println("Finished full state evolution with ODE updater")
  println()

  @show norm(ψₜ_ode)
  @show norm(ψₜ_krylov)
  @show norm(ψₜ_full)

  @show 1 - abs(inner(contract(ψₜ_ode), ψₜ_full))
  @show 1 - abs(inner(contract(ψₜ_krylov), ψₜ_full))
  return nothing
end

main()
