@eval module $(gensym())
using ITensors: ITensors, dag, delta, denseblocks
using ITensors: MPO, OpSum, apply, contract, inner, random_mps, siteinds, truncate!
using ITensorTDVP: ITensorTDVP
using StableRNGs: StableRNG
using Test: @test, @test_throws, @testset
@testset "Contract MPO (eltype=$elt, conserve_qns=$conserve_qns)" for elt in (
    Float32, Float64, Complex{Float32}, Complex{Float64}
  ),
  conserve_qns in [false, true]

  N = 20
  s = siteinds("S=1/2", N; conserve_qns)
  rng = StableRNG(1234)
  psi = random_mps(rng, elt, s, j -> isodd(j) ? "↑" : "↓"; linkdims=8)
  os = OpSum()
  for j in 1:(N - 1)
    os += 0.5, "S+", j, "S-", j + 1
    os += 0.5, "S-", j, "S+", j + 1
    os += "Sz", j, "Sz", j + 1
  end
  for j in 1:(N - 2)
    os += 0.5, "S+", j, "S-", j + 2
    os += 0.5, "S-", j, "S+", j + 2
    os += "Sz", j, "Sz", j + 2
  end
  H = MPO(elt, os, s)
  @testset "apply (standard indices, nsite=2)" begin
    Hpsi = apply(H, psi; alg="fit", nsweeps=2)
    @test_throws ErrorException apply(H, psi; alg="fit")
    @test ITensors.scalartype(Hpsi) == elt
    @test inner(psi, Hpsi) ≈ inner(psi', H, psi) rtol = 10 * √eps(real(elt))
  end
  @testset "contract (non-standard indices)" begin
    # Change "top" indices of MPO to be a different set
    t = siteinds("S=1/2", N; conserve_qns)
    Ht = deepcopy(H)
    psit = deepcopy(psi)
    for j in 1:N
      Ht[j] *= delta(elt, dag(s[j])', t[j])
      psit[j] *= delta(elt, dag(s[j]), t[j])
    end

    # Test with nsweeps=2
    Hpsit = contract(Ht, psi; alg="fit", nsweeps=2)
    @test ITensors.scalartype(Hpsit) == elt
    @test inner(psit, Hpsit) ≈ inner(psit, Ht, psi) rtol = 10 * √eps(real(elt))

    # Test with less good initial guess MPS not equal to psi
    psit_guess = copy(psit)
    truncate!(psit_guess; maxdim=2)
    Hpsit = contract(Ht, psi; alg="fit", nsweeps=4, init=psit_guess)
    @test ITensors.scalartype(Hpsit) == elt
    @test inner(psit, Hpsit) ≈ inner(psit, Ht, psi) rtol = 20 * √eps(real(elt))
  end
  @testset "apply (standard indices, nsite=1)" begin
    # Test with nsite=1
    Hpsi_guess = apply(H, psi; alg="naive", cutoff=1e-4)
    Hpsi = apply(H, psi; alg="fit", init=Hpsi_guess, nsite=1, nsweeps=2)
    @test ITensors.scalartype(Hpsi) == elt
    scale(::Type{Float32}) = 10^2
    scale(::Type{Float64}) = 10^6
    @test inner(psi, Hpsi) ≈ inner(psi', H, psi) rtol = √eps(real(elt)) * scale(real(elt))
  end
end
end
