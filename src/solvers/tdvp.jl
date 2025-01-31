using ITensors: Algorithm, @Algorithm_str
using KrylovKit: exponentiate

function exponentiate_updater(operator, init; internal_kwargs, kwargs...)
  state, info = exponentiate(operator, internal_kwargs.time_step, init; kwargs...)
  return state, (; info)
end

function applyexp_updater(operator, init; internal_kwargs, kwargs...)
  state, info = applyexp(operator, internal_kwargs.time_step, init; kwargs...)
  return state, (; info)
end

tdvp_updater(updater_backend::String) = tdvp_updater(Algorithm(updater_backend))
tdvp_updater(::Algorithm"exponentiate") = exponentiate_updater
tdvp_updater(::Algorithm"applyexp") = applyexp_updater
function tdvp_updater(updater_backend::Algorithm)
  return error("`updater_backend=$(String(updater_backend))` not recognized.")
end

function time_step_and_nsteps(t, time_step::Nothing, nsteps::Nothing)
  # Default to 1 step.
  nsteps = 1
  return time_step_and_nsteps(t, time_step, nsteps)
end

function time_step_and_nsteps(t, time_step::Nothing, nsteps)
  return t / nsteps, nsteps
end

function time_step_and_nsteps(t, time_step, nsteps::Nothing)
  nsteps_float = t / time_step
  nsteps_rounded = round(nsteps_float)
  if nsteps_float ≉ nsteps_rounded
    return error("`t / time_step = $t / $time_step = $(t / time_step)` must be an integer.")
  end
  if real(nsteps_rounded) < 0
    return error(
      "computed number of steps is negative ($nsteps_rounded), check that total time ($t) and time step ($time_step) signs agree",
    )
  end
  return time_step, Int(nsteps_rounded)
end

function time_step_and_nsteps(t, time_step, nsteps)
  if time_step * nsteps ≠ t
    return error(
      "Calling `tdvp(operator, t, state; time_step, nsteps, kwargs...)` with `t = $t`, `time_step = $time_step`, and `nsteps = $nsteps` must satisfy `time_step * nsteps == t`, while `time_step * nsteps = $time_step * $nsteps = $(time_step * nsteps)`.",
    )
  end
  return time_step, nsteps
end

"""
    tdvp(operator, t::Number, init::MPS; time_step, nsteps, kwargs...)

Use the time dependent variational principle (TDVP) algorithm
to compute `exp(t * operator) * init` using an efficient algorithm based
on alternating optimization of the MPS tensors and local Krylov
exponentiation of `operator`.

Specify one of `time_step` or `nsteps`. If they are both specified, they
must satisfy `time_step * nsteps == t`. If neither are specified, the
default is `nsteps=1`, which means that `time_step == t`.

Returns:
* `state::MPS` - time-evolved MPS

"""
function tdvp(
  operator,
  t::Number,
  init::MPS;
  updater_backend="exponentiate",
  updater=tdvp_updater(updater_backend),
  reverse_step=true,
  time_step=nothing,
  time_start=zero(t),
  nsweeps=nothing,
  nsteps=nsweeps,
  (step_observer!)=default_sweep_observer(),
  (sweep_observer!)=step_observer!,
  kwargs...,
)
  time_step, nsteps = time_step_and_nsteps(t, time_step, nsteps)
  return alternating_update(
    operator,
    init;
    updater,
    reverse_step,
    nsweeps=nsteps,
    time_start,
    time_step,
    sweep_observer!,
    kwargs...,
  )
end
