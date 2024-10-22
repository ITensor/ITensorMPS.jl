using Compat: @compat
using ITensors: ITensor, array, inds, itensor
using ITensorMPS: TimeDependentSum, to_vec
using KrylovKit: exponentiate
using OrdinaryDiffEqTsit5: ODEProblem, Tsit5, solve

function ode_updater(operator, init; internal_kwargs, alg=Tsit5(), kwargs...)
  @compat (; current_time, time_step) = (; current_time=zero(Bool), internal_kwargs...)
  time_span = typeof(time_step).((current_time, current_time + time_step))
  init_vec, to_itensor = to_vec(init)
  f(init::ITensor, p, t) = operator(t)(init)
  f(init_vec::AbstractArray, p, t) = to_vec(f(to_itensor(init_vec), p, t))[1]
  prob = ODEProblem(f, init_vec, time_span)
  sol = solve(prob, alg; kwargs...)
  state_vec = sol.u[end]
  return to_itensor(state_vec), (;)
end

function krylov_updater(operator, init; internal_kwargs, kwargs...)
  @compat (; current_time, time_step) = (; current_time=zero(Bool), internal_kwargs...)
  state, info = exponentiate(operator(current_time), time_step, init; kwargs...)
  return state, (; info)
end
