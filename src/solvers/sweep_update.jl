using ITensors: ITensors, uniqueinds
using LinearAlgebra: norm, normalize!, svd
using Printf: @printf

function sweep_update(
  order::TDVPOrder,
  reduced_operator,
  state::MPS;
  current_time=nothing,
  time_step=nothing,
  kwargs...,
)
  order_orderings = orderings(order)
  order_sub_time_steps = sub_time_steps(order)
  if !isnothing(time_step)
    order_sub_time_steps = eltype(time_step).(order_sub_time_steps)
    order_sub_time_steps *= time_step
  end
  info = nothing
  sub_time_step = nothing
  for substep in 1:length(order_sub_time_steps)
    if !isnothing(time_step)
      sub_time_step = order_sub_time_steps[substep]
    end
    state, reduced_operator, info = sub_sweep_update(
      order_orderings[substep],
      reduced_operator,
      state;
      current_time,
      time_step=sub_time_step,
      kwargs...,
    )
    if !isnothing(time_step)
      current_time += sub_time_step
    end
  end
  return state, reduced_operator, info
end

isforward(direction::Base.ForwardOrdering) = true
isforward(direction::Base.ReverseOrdering) = false
isreverse(direction) = !isforward(direction)

function sweep_bonds(direction::Base.ForwardOrdering, n::Int; ncenter::Int)
  return 1:(n - ncenter + 1)
end

function sweep_bonds(direction::Base.ReverseOrdering, n::Int; ncenter::Int)
  return reverse(sweep_bonds(Base.Forward, n; ncenter))
end

is_forward_done(direction::Base.ForwardOrdering, b, n; ncenter) = (b + ncenter - 1 == n)
is_forward_done(direction::Base.ReverseOrdering, b, n; ncenter) = false
is_reverse_done(direction::Base.ForwardOrdering, b, n; ncenter) = false
is_reverse_done(direction::Base.ReverseOrdering, b, n; ncenter) = (b == 1)
function is_half_sweep_done(direction, b, n; ncenter)
  return is_forward_done(direction, b, n; ncenter) ||
         is_reverse_done(direction, b, n; ncenter)
end

function sub_sweep_update(
  direction::Base.Ordering,
  reduced_operator,
  state::MPS;
  updater,
  updater_kwargs,
  which_decomp=nothing,
  svd_alg=nothing,
  sweep=default_sweep(),
  current_time=nothing,
  time_step=nothing,
  nsite=default_nsite(),
  reverse_step=default_reverse_step(),
  normalize=default_normalize(),
  (observer!)=default_observer(),
  outputlevel=default_outputlevel(),
  maxdim=default_maxdim(),
  mindim=default_mindim(),
  cutoff=default_cutoff(ITensors.scalartype(state)),
  noise=default_noise(),
)
  reduced_operator = copy(reduced_operator)
  state = copy(state)
  if length(state) == 1
    error(
      "`tdvp`, `dmrg`, `linsolve`, etc. currently does not support system sizes of 1. You can diagonalize the MPO tensor directly with tools like `LinearAlgebra.eigen`, `KrylovKit.exponentiate`, etc.",
    )
  end
  N = length(state)
  set_nsite!(reduced_operator, nsite)
  if isforward(direction)
    if !isortho(state) || orthocenter(state) != 1
      orthogonalize!(state, 1)
    end
    @assert isortho(state) && orthocenter(state) == 1
    position!(reduced_operator, state, 1)
  elseif isreverse(direction)
    if !isortho(state) || orthocenter(state) != N - nsite + 1
      orthogonalize!(state, N - nsite + 1)
    end
    @assert(isortho(state) && (orthocenter(state) == N - nsite + 1))
    position!(reduced_operator, state, N - nsite + 1)
  end
  maxtruncerr = 0.0
  info = nothing
  for b in sweep_bonds(direction, N; ncenter=nsite)
    current_time, maxtruncerr, spec, info = region_update!(
      reduced_operator,
      state,
      b;
      updater,
      updater_kwargs,
      nsite,
      reverse_step,
      current_time,
      outputlevel,
      time_step,
      normalize,
      direction,
      noise,
      which_decomp,
      svd_alg,
      cutoff,
      maxdim,
      mindim,
      maxtruncerr,
    )
    if outputlevel >= 2
      if nsite == 1
        @printf("Sweep %d, direction %s, bond (%d,) \n", sweep, direction, b)
      elseif nsite == 2
        @printf("Sweep %d, direction %s, bond (%d,%d) \n", sweep, direction, b, b + 1)
      end
      print("  Truncated using")
      @printf(" cutoff=%.1E", cutoff)
      @printf(" maxdim=%.1E", maxdim)
      print(" mindim=", mindim)
      print(" current_time=", round(current_time; digits=3))
      println()
      if spec != nothing
        @printf(
          "  Trunc. err=%.2E, bond dimension %d\n", spec.truncerr, dim(linkind(state, b))
        )
      end
      flush(stdout)
    end
    update_observer!(
      observer!;
      state,
      reduced_operator,
      bond=b,
      sweep,
      half_sweep=isforward(direction) ? 1 : 2,
      spec,
      outputlevel,
      half_sweep_is_done=is_half_sweep_done(direction, b, N; ncenter=nsite),
      current_time,
      info,
    )
  end
  # Just to be sure:
  normalize && normalize!(state)
  return state, reduced_operator, (; maxtruncerr)
end

function region_update!(
  reduced_operator,
  state,
  b;
  updater,
  updater_kwargs,
  nsite,
  reverse_step,
  current_time,
  outputlevel,
  time_step,
  normalize,
  direction,
  noise,
  which_decomp,
  svd_alg,
  cutoff,
  maxdim,
  mindim,
  maxtruncerr,
)
  return region_update!(
    Val(nsite),
    Val(reverse_step),
    reduced_operator,
    state,
    b;
    updater,
    updater_kwargs,
    current_time,
    outputlevel,
    time_step,
    normalize,
    direction,
    noise,
    which_decomp,
    svd_alg,
    cutoff,
    maxdim,
    mindim,
    maxtruncerr,
  )
end

function region_update!(
  nsite_val::Val{1},
  reverse_step_val::Val{false},
  reduced_operator,
  state,
  b;
  updater,
  updater_kwargs,
  current_time,
  outputlevel,
  time_step,
  normalize,
  direction,
  noise,
  which_decomp,
  svd_alg,
  cutoff,
  maxdim,
  mindim,
  maxtruncerr,
)
  N = length(state)
  nsite = 1
  # Do 'forwards' evolution step
  set_nsite!(reduced_operator, nsite)
  position!(reduced_operator, state, b)
  reduced_state = state[b]
  internal_kwargs = (; current_time, time_step, outputlevel)
  reduced_state, info = updater(
    reduced_operator, reduced_state; internal_kwargs, updater_kwargs...
  )
  if !isnothing(time_step)
    current_time += time_step
  end
  normalize && (reduced_state /= norm(reduced_state))
  spec = nothing
  state[b] = reduced_state
  if !is_half_sweep_done(direction, b, N; ncenter=nsite)
    # Move ortho center
    Δ = (isforward(direction) ? +1 : -1)
    orthogonalize!(state, b + Δ)
  end
  return current_time, maxtruncerr, spec, info
end

function region_update!(
  nsite_val::Val{1},
  reverse_step_val::Val{true},
  reduced_operator,
  state,
  b;
  updater,
  updater_kwargs,
  current_time,
  outputlevel,
  time_step,
  normalize,
  direction,
  noise,
  which_decomp,
  svd_alg,
  cutoff,
  maxdim,
  mindim,
  maxtruncerr,
)
  N = length(state)
  nsite = 1
  # Do 'forwards' evolution step
  set_nsite!(reduced_operator, nsite)
  position!(reduced_operator, state, b)
  reduced_state = state[b]
  internal_kwargs = (; current_time, time_step, outputlevel)
  reduced_state, info = updater(
    reduced_operator, reduced_state; internal_kwargs, updater_kwargs...
  )
  current_time += time_step
  normalize && (reduced_state /= norm(reduced_state))
  spec = nothing
  state[b] = reduced_state
  if !is_half_sweep_done(direction, b, N; ncenter=nsite)
    # Do backwards evolution step
    b1 = (isforward(direction) ? b + 1 : b)
    Δ = (isforward(direction) ? +1 : -1)
    uinds = uniqueinds(reduced_state, state[b + Δ])
    U, S, V = svd(reduced_state, uinds)
    state[b] = U
    bond_reduced_state = S * V
    if isforward(direction)
      ITensorMPS.setleftlim!(state, b)
    elseif isreverse(direction)
      ITensorMPS.setrightlim!(state, b)
    end
    set_nsite!(reduced_operator, nsite - 1)
    position!(reduced_operator, state, b1)
    internal_kwargs = (; current_time, time_step=(-time_step), outputlevel)
    bond_reduced_state, info = updater(
      reduced_operator, bond_reduced_state; internal_kwargs, updater_kwargs...
    )
    current_time -= time_step
    normalize && (bond_reduced_state ./= norm(bond_reduced_state))
    state[b + Δ] = bond_reduced_state * state[b + Δ]
    if isforward(direction)
      ITensorMPS.setrightlim!(state, b + Δ + 1)
    elseif isreverse(direction)
      ITensorMPS.setleftlim!(state, b + Δ - 1)
    end
    set_nsite!(reduced_operator, nsite)
  end
  return current_time, maxtruncerr, spec, info
end

function region_update!(
  nsite_val::Val{2},
  reverse_step_val::Val{false},
  reduced_operator,
  state,
  b;
  updater,
  updater_kwargs,
  current_time,
  time_step,
  outputlevel,
  normalize,
  direction,
  noise,
  which_decomp,
  svd_alg,
  cutoff,
  maxdim,
  mindim,
  maxtruncerr,
)
  N = length(state)
  nsite = 2
  # Do 'forwards' evolution step
  set_nsite!(reduced_operator, nsite)
  position!(reduced_operator, state, b)
  reduced_state = state[b] * state[b + 1]
  internal_kwargs = (; current_time, time_step, outputlevel)
  reduced_state, info = updater(
    reduced_operator, reduced_state; internal_kwargs, updater_kwargs...
  )
  if !isnothing(time_step)
    current_time += time_step
  end
  normalize && (reduced_state /= norm(reduced_state))
  spec = nothing
  ortho = isforward(direction) ? "left" : "right"
  drho = nothing
  if noise > 0.0 && isforward(direction)
    drho = noise * noiseterm(reduced_operator, reduced_state, ortho)
  end
  spec = replacebond!(
    state,
    b,
    reduced_state;
    maxdim,
    mindim,
    cutoff,
    eigen_perturbation=drho,
    ortho=ortho,
    normalize,
    which_decomp,
    svd_alg,
  )
  maxtruncerr = max(maxtruncerr, spec.truncerr)
  return current_time, maxtruncerr, spec, info
end

function region_update!(
  nsite_val::Val{2},
  reverse_step_val::Val{true},
  reduced_operator,
  state,
  b;
  updater,
  updater_kwargs,
  current_time,
  time_step,
  outputlevel,
  normalize,
  direction,
  noise,
  which_decomp,
  svd_alg,
  cutoff,
  maxdim,
  mindim,
  maxtruncerr,
)
  N = length(state)
  nsite = 2
  # Do 'forwards' evolution step
  set_nsite!(reduced_operator, nsite)
  position!(reduced_operator, state, b)
  reduced_state = state[b] * state[b + 1]
  internal_kwargs = (; current_time, time_step, outputlevel)
  reduced_state, info = updater(
    reduced_operator, reduced_state; internal_kwargs, updater_kwargs...
  )
  current_time += time_step
  normalize && (reduced_state /= norm(reduced_state))
  spec = nothing
  ortho = isforward(direction) ? "left" : "right"
  drho = nothing
  if noise > 0.0 && isforward(direction)
    drho = noise * noiseterm(reduced_operator, phi, ortho)
  end
  spec = replacebond!(
    state,
    b,
    reduced_state;
    maxdim,
    mindim,
    cutoff,
    eigen_perturbation=drho,
    ortho=ortho,
    normalize,
    which_decomp,
    svd_alg,
  )
  maxtruncerr = max(maxtruncerr, spec.truncerr)
  if !is_half_sweep_done(direction, b, N; ncenter=nsite)
    # Do backwards evolution step
    b1 = (isforward(direction) ? b + 1 : b)
    Δ = (isforward(direction) ? +1 : -1)
    bond_reduced_state = state[b1]
    set_nsite!(reduced_operator, nsite - 1)
    position!(reduced_operator, state, b1)
    internal_kwargs = (; current_time, time_step=(-time_step), outputlevel)
    bond_reduced_state, info = updater(
      reduced_operator, bond_reduced_state; internal_kwargs, updater_kwargs...
    )
    current_time -= time_step
    normalize && (bond_reduced_state /= norm(bond_reduced_state))
    state[b1] = bond_reduced_state
    set_nsite!(reduced_operator, nsite)
  end
  return current_time, maxtruncerr, spec, info
end

function region_!(
  ::Val{nsite},
  ::Val{reverse_step},
  reduced_operator,
  state,
  b;
  updater,
  updater_kwargs,
  current_time,
  outputlevel,
  time_step,
  normalize,
  direction,
  noise,
  which_decomp,
  svd_alg,
  cutoff,
  maxdim,
  mindim,
  maxtruncerr,
) where {nsite,reverse_step}
  return error(
    "`tdvp`, `dmrg`, `linsolve`, etc. with `nsite=$nsite` and `reverse_step=$reverse_step` not implemented.",
  )
end
