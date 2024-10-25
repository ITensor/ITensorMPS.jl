using ITensorMPS: MPO, MPS, OpSum, expect, inner, siteinds, tdvp
using Observers: observer

function main()
  function heisenberg(N)
    os = OpSum()
    for j in 1:(N - 1)
      os += 0.5, "S+", j, "S-", j + 1
      os += 0.5, "S-", j, "S+", j + 1
      os += "Sz", j, "Sz", j + 1
    end
    return os
  end

  N = 10
  s = siteinds("S=1/2", N; conserve_qns=true)
  H = MPO(heisenberg(N), s)

  step(; sweep) = sweep
  current_time(; current_time) = current_time
  return_state(; state) = state
  measure_sz(; state) = expect(state, "Sz"; sites=length(state) ÷ 2)
  obs = observer(
    "steps" => step, "times" => current_time, "states" => return_state, "sz" => measure_sz
  )

  init = MPS(s, n -> isodd(n) ? "Up" : "Dn")
  state = tdvp(
    H, -1.0im, init; time_step=-0.1im, cutoff=1e-12, (step_observer!)=obs, outputlevel=1
  )

  println("\nResults")
  println("=======")
  for n in 1:length(obs.steps)
    print("step = ", obs.steps[n])
    print(", time = ", round(obs.times[n]; digits=3))
    print(", |⟨ψⁿ|ψⁱ⟩| = ", round(abs(inner(obs.states[n], init)); digits=3))
    print(", |⟨ψⁿ|ψᶠ⟩| = ", round(abs(inner(obs.states[n], state)); digits=3))
    print(", ⟨Sᶻ⟩ = ", round(obs.sz[n]; digits=3))
    println()
  end
  return nothing
end

main()
