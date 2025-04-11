@eval module $(gensym())
using ITensorMPS:
  AbstractObserver, MPO, MPS, OpSum, apply, expect, inner, random_mps, siteinds, tdvp
using ITensors: ITensors, ITensor, dag, noprime, op, prime, scalar
using KrylovKit: exponentiate
using LinearAlgebra: norm
using Observers: observer
using StableRNGs: StableRNG
using Test: @test, @test_throws, @testset
const elts = (Float32, Float64, Complex{Float32}, Complex{Float64})
@testset "Basic TDVP (eltype=$elt)" for elt in elts
  N = 10
  cutoff = eps(real(elt)) * 10^4
  s = siteinds("S=1/2", N)
  os = OpSum()
  for j in 1:(N - 1)
    os += 0.5, "S+", j, "S-", j + 1
    os += 0.5, "S-", j, "S+", j + 1
    os += "Sz", j, "Sz", j + 1
  end
  H = MPO(elt, os, s)
  rng = StableRNG(1234)
  ψ0 = random_mps(rng, elt, s; linkdims=10)
  time_step = elt(0.1) * im
  # Time evolve forward:
  ψ1 = tdvp(H, -time_step, ψ0; cutoff, nsite=1)
  @test ITensors.scalartype(ψ1) == complex(elt)
  #Different backend updaters, default updater_backend = "exponentiate"
  @test ψ1 ≈ tdvp(H, -time_step, ψ0; cutoff, nsite=1, updater_backend="applyexp")
  @test norm(ψ1) ≈ 1 rtol = √eps(real(elt)) * 10
  ## Should lose fidelity:
  #@test abs(inner(ψ0,ψ1)) < 0.9
  # Average energy should be conserved:
  @test real(inner(ψ1', H, ψ1)) ≈ inner(ψ0', H, ψ0) rtol = √eps(real(elt)) * 10
  # Time evolve backwards:
  ψ2 = tdvp(H, time_step, ψ1; cutoff)
  @test ITensors.scalartype(ψ2) == complex(elt)
  @test norm(ψ2) ≈ 1 rtol = √eps(real(elt)) * 10
  # Should rotate back to original state:
  @test abs(inner(ψ0, ψ2)) > 0.99

  @testset "Failure mode tests" begin
    total_time = elt(0.1) * im
    # matching total time and time step directions
    @test_throws "signs agree" tdvp(H, -total_time, ψ0; time_step)
    @test_throws "signs agree" tdvp(H, total_time, ψ0; time_step=(-time_step))
    # total time is an integer of time steps
    @test_throws "integer" tdvp(H, total_time * 1.5, ψ0; time_step)
  end
end

@testset "TDVP: Sum of Hamiltonians (eltype=$elt)" for elt in elts
  N = 10
  cutoff = 1e-10

  s = siteinds("S=1/2", N)

  os1 = OpSum()
  for j in 1:(N - 1)
    os1 += 0.5, "S+", j, "S-", j + 1
    os1 += 0.5, "S-", j, "S+", j + 1
  end
  os2 = OpSum()
  for j in 1:(N - 1)
    os2 += "Sz", j, "Sz", j + 1
  end
  H1 = MPO(elt, os1, s)
  H2 = MPO(elt, os2, s)
  Hs = [H1, H2]
  rng = StableRNG(1234)
  ψ0 = random_mps(rng, elt, s; linkdims=10)
  ψ1 = tdvp(Hs, -elt(0.1) * im, ψ0; cutoff, nsite=1)
  @test ITensors.scalartype(ψ1) === complex(elt)
  @test norm(ψ1) ≈ 1 rtol = √eps(real(elt))
  ## Should lose fidelity:
  #@test abs(inner(ψ0,ψ1)) < 0.9
  # Average energy should be conserved:
  @test real(sum(H -> inner(ψ1', H, ψ1), Hs)) ≈ sum(H -> inner(ψ0', H, ψ0), Hs) rtol =
    4 * √eps(real(elt))
  # Time evolve backwards:
  ψ2 = tdvp(Hs, elt(0.1) * im, ψ1; cutoff)
  @test ITensors.scalartype(ψ2) === complex(elt)
  @test norm(ψ2) ≈ 1 rtol = √eps(real(elt))
  # Should rotate back to original state:
  @test abs(inner(ψ0, ψ2)) > 0.99
end
@testset "Custom updater in TDVP" begin
  N = 10
  cutoff = 1e-12
  s = siteinds("S=1/2", N)
  os = OpSum()
  for j in 1:(N - 1)
    os += 0.5, "S+", j, "S-", j + 1
    os += 0.5, "S-", j, "S+", j + 1
    os += "Sz", j, "Sz", j + 1
  end
  H = MPO(os, s)
  rng = StableRNG(1234)
  ψ0 = random_mps(rng, s; linkdims=10)
  function updater(PH, state0; internal_kwargs, kwargs...)
    return exponentiate(PH, internal_kwargs.time_step, state0; kwargs...)
  end
  updater_kwargs = (;
    ishermitian=true, tol=1e-12, krylovdim=30, maxiter=100, verbosity=0, eager=true
  )
  t = -0.1im
  ψ1 = tdvp(H, t, ψ0; updater, updater_kwargs, cutoff, nsite=1)
  @test norm(ψ1) ≈ 1
  ## Should lose fidelity:
  #@test abs(inner(ψ0,ψ1)) < 0.9
  # Average energy should be conserved:
  @test real(inner(ψ1', H, ψ1)) ≈ inner(ψ0', H, ψ0)
  # Time evolve backwards:
  ψ2 = tdvp(H, +0.1im, ψ1; cutoff)
  @test norm(ψ2) ≈ 1
  # Should rotate back to original state:
  @test abs(inner(ψ0, ψ2)) > 0.99
end
@testset "Accuracy Test" begin
  N = 4
  tau = 0.1
  ttotal = 1.0
  cutoff = 1e-12
  s = siteinds("S=1/2", N; conserve_qns=false)
  os = OpSum()
  for j in 1:(N - 1)
    os += 0.5, "S+", j, "S-", j + 1
    os += 0.5, "S-", j, "S+", j + 1
    os += "Sz", j, "Sz", j + 1
  end
  H = MPO(os, s)
  HM = prod(H)
  Ut = exp(-im * tau * HM)
  state = MPS(s, n -> isodd(n) ? "Up" : "Dn")
  state2 = deepcopy(state)
  statex = prod(state)
  Sz_tdvp = Float64[]
  Sz_tdvp2 = Float64[]
  Sz_exact = Float64[]
  c = div(N, 2)
  Szc = op("Sz", s[c])
  Nsteps = Int(ttotal / tau)
  for step in 1:Nsteps
    statex = noprime(Ut * statex)
    statex /= norm(statex)

    state = tdvp(
      H,
      -im * tau,
      state;
      cutoff,
      normalize=false,
      updater_kwargs=(; tol=1e-12, maxiter=500, krylovdim=25),
    )
    push!(Sz_tdvp, real(expect(state, "Sz"; sites=c:c)[1]))
    state2 = tdvp(
      H,
      -im * tau,
      state2;
      cutoff,
      normalize=false,
      updater_kwargs=(; tol=1e-12, maxiter=500, krylovdim=25),
    )
    push!(Sz_tdvp2, real(expect(state2, "Sz"; sites=c:c)[1]))
    push!(Sz_exact, real(scalar(dag(prime(statex, s[c])) * Szc * statex)))
    F = abs(scalar(dag(statex) * prod(state)))
  end
  @test norm(Sz_tdvp - Sz_exact) < 1e-5
  @test norm(Sz_tdvp2 - Sz_exact) < 1e-5
end
@testset "TEBD Comparison" begin
  N = 10
  cutoff = 1e-12
  tau = 0.1
  ttotal = 1.0
  s = siteinds("S=1/2", N; conserve_qns=true)
  os = OpSum()
  for j in 1:(N - 1)
    os += 0.5, "S+", j, "S-", j + 1
    os += 0.5, "S-", j, "S+", j + 1
    os += "Sz", j, "Sz", j + 1
  end
  H = MPO(os, s)
  gates = ITensor[]
  for j in 1:(N - 1)
    s1 = s[j]
    s2 = s[j + 1]
    hj =
      op("Sz", s1) * op("Sz", s2) +
      1 / 2 * op("S+", s1) * op("S-", s2) +
      1 / 2 * op("S-", s1) * op("S+", s2)
    Gj = exp(-1.0im * tau / 2 * hj)
    push!(gates, Gj)
  end
  append!(gates, reverse(gates))
  state = MPS(s, n -> isodd(n) ? "Up" : "Dn")
  phi = deepcopy(state)
  c = div(N, 2)
  # Evolve using TEBD
  Nsteps = convert(Int, ceil(abs(ttotal / tau)))
  Sz1 = zeros(Nsteps)
  En1 = zeros(Nsteps)
  Sz2 = zeros(Nsteps)
  En2 = zeros(Nsteps)
  for step in 1:Nsteps
    state = apply(gates, state; cutoff)
    nsite = (step <= 3 ? 2 : 1)
    phi = tdvp(
      H, -tau * im, phi; cutoff, nsite, normalize=true, updater_kwargs=(; krylovdim=15)
    )
    Sz1[step] = expect(state, "Sz"; sites=c:c)[1]
    Sz2[step] = expect(phi, "Sz"; sites=c:c)[1]
    En1[step] = real(inner(state', H, state))
    En2[step] = real(inner(phi', H, phi))
  end
  # Evolve using TDVP
  function measure_sz(; state, bond, half_sweep)
    if bond == 1 && half_sweep == 2
      return expect(state, "Sz"; sites=c)
    end
    return nothing
  end
  function measure_en(; state, bond, half_sweep)
    if bond == 1 && half_sweep == 2
      return real(inner(state', H, state))
    end
    return nothing
  end
  obs = observer("Sz" => measure_sz, "En" => measure_en)

  phi = MPS(s, n -> isodd(n) ? "Up" : "Dn")
  phi = tdvp(
    H, -im * ttotal, phi; time_step=-im * tau, cutoff, normalize=false, (observer!)=obs
  )
  Sz2 = obs.Sz
  En2 = obs.En
  @test norm(Sz1 - Sz2) < 1e-3
  @test norm(En1 - En2) < 1e-3
end
@testset "Imaginary Time Evolution" for reverse_step in [true, false]
  N = 10
  cutoff = 1e-12
  tau = 1.0
  ttotal = 50.0
  s = siteinds("S=1/2", N)
  os = OpSum()
  for j in 1:(N - 1)
    os += 0.5, "S+", j, "S-", j + 1
    os += 0.5, "S-", j, "S+", j + 1
    os += "Sz", j, "Sz", j + 1
  end
  H = MPO(os, s)
  rng = StableRNG(1234)
  state = random_mps(rng, s; linkdims=2)
  state2 = deepcopy(state)
  trange = 0.0:tau:ttotal
  for (step, t) in enumerate(trange)
    nsite = (step <= 10 ? 2 : 1)
    state = tdvp(
      H,
      -tau,
      state;
      cutoff,
      nsite,
      reverse_step,
      normalize=true,
      updater_kwargs=(; krylovdim=15),
    )
    state2 = tdvp(
      H,
      -tau,
      state2;
      cutoff,
      nsite,
      reverse_step,
      normalize=true,
      updater_kwargs=(; krylovdim=15),
    )
  end
  @test state ≈ state2 rtol = 1e-6
  en1 = inner(state', H, state)
  en2 = inner(state2', H, state2)
  @test en1 < -4.25
  @test en1 ≈ en2
end
@testset "Observers" begin
  N = 10
  cutoff = 1e-12
  tau = 0.1
  ttotal = 1.0
  s = siteinds("S=1/2", N; conserve_qns=true)
  os = OpSum()
  for j in 1:(N - 1)
    os += 0.5, "S+", j, "S-", j + 1
    os += 0.5, "S-", j, "S+", j + 1
    os += "Sz", j, "Sz", j + 1
  end
  H = MPO(os, s)
  c = div(N, 2)
  # Using Observers.jl
  function measure_sz(; state, bond, half_sweep)
    if bond == 1 && half_sweep == 2
      return expect(state, "Sz"; sites=c)
    end
    return nothing
  end
  function measure_en(; state, bond, half_sweep)
    if bond == 1 && half_sweep == 2
      return real(inner(state', H, state))
    end
    return nothing
  end
  function identity_info(; info)
    return info
  end
  obs = observer("Sz" => measure_sz, "En" => measure_en, "info" => identity_info)
  step_measure_sz(; state) = expect(state, "Sz"; sites=c)
  step_measure_en(; state) = real(inner(state', H, state))
  step_obs = observer("Sz" => step_measure_sz, "En" => step_measure_en)
  state = MPS(s, n -> isodd(n) ? "Up" : "Dn")
  tdvp(
    H,
    -im * ttotal,
    state;
    time_step=-im * tau,
    cutoff,
    normalize=false,
    (observer!)=obs,
    (step_observer!)=step_obs,
  )
  Sz = filter(!isnothing, obs.Sz)
  En = filter(!isnothing, obs.En)
  infos = obs.info
  Sz_step = step_obs.Sz
  En_step = step_obs.En
  @test length(Sz) == 10
  @test length(En) == 10
  @test length(Sz_step) == 10
  @test length(En_step) == 10
  @test Sz ≈ Sz_step
  @test En ≈ En_step
  @test all(x -> x.info.converged == 1, infos)
  @test length(values(infos)) == 180
end
end
