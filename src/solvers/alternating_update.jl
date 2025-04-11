using ITensors: ITensors, permute

function _extend_sweeps_param(param, nsweeps)
  if param isa Number
    eparam = fill(param, nsweeps)
  else
    length(param) == nsweeps && return param
    eparam = Vector(undef, nsweeps)
    eparam[1:length(param)] = param
    eparam[(length(param) + 1):end] .= param[end]
  end
  return eparam
end

function process_sweeps(; nsweeps, maxdim, mindim, cutoff, noise)
  maxdim = _extend_sweeps_param(maxdim, nsweeps)
  mindim = _extend_sweeps_param(mindim, nsweeps)
  cutoff = _extend_sweeps_param(cutoff, nsweeps)
  noise = _extend_sweeps_param(noise, nsweeps)
  return (; maxdim, mindim, cutoff, noise)
end

function alternating_update(
  operator,
  init::MPS;
  updater,
  updater_kwargs=(;),
  nsweeps=default_nsweeps(),
  checkdone=default_checkdone(),
  write_when_maxdim_exceeds=default_write_when_maxdim_exceeds(),
  nsite=default_nsite(),
  reverse_step=default_reverse_step(),
  time_start=default_time_start(),
  time_step=default_time_step(),
  order=default_order(),
  (observer!)=default_observer(),
  (sweep_observer!)=default_sweep_observer(),
  outputlevel=default_outputlevel(),
  normalize=default_normalize(),
  maxdim=default_maxdim(),
  mindim=default_mindim(),
  cutoff=default_cutoff(ITensors.scalartype(init)),
  noise=default_noise(),
)
  reduced_operator = ITensorMPS.reduced_operator(operator)
  if isnothing(nsweeps)
    return error("Must specify `nsweeps`.")
  end
  maxdim, mindim, cutoff, noise = process_sweeps(; nsweeps, maxdim, mindim, cutoff, noise)
  forward_order = TDVPOrder(order, Base.Forward)
  state = copy(init)
  # Keep track of the start of the current time step.
  # Helpful for tracking the total time, for example
  # when using time-dependent updaters.
  # This will be passed as a keyword argument to the
  # `updater`.
  current_time = time_start
  info = nothing
  for sweep in 1:nsweeps
    if !isnothing(write_when_maxdim_exceeds) && maxdim[sweep] > write_when_maxdim_exceeds
      if outputlevel >= 2
        println(
          "write_when_maxdim_exceeds = $write_when_maxdim_exceeds and maxdim(sweeps, sw) = $(maxdim(sweeps, sweep)), writing environment tensors to disk",
        )
      end
      reduced_operator = disk(reduced_operator)
    end
    sweep_elapsed_time = @elapsed begin
      state, reduced_operator, info = sweep_update(
        forward_order,
        reduced_operator,
        state;
        updater,
        updater_kwargs,
        nsite,
        current_time,
        time_step,
        reverse_step,
        sweep,
        observer!,
        normalize,
        outputlevel,
        maxdim=maxdim[sweep],
        mindim=mindim[sweep],
        cutoff=cutoff[sweep],
        noise=noise[sweep],
      )
    end
    if !isnothing(time_step)
      current_time += time_step
    end
    update_observer!(
      sweep_observer!; state, reduced_operator, sweep, outputlevel, current_time
    )
    if outputlevel >= 1
      print("After sweep ", sweep, ":")
      print(" maxlinkdim=", maxlinkdim(state))
      @printf(" maxerr=%.2E", info.maxtruncerr)
      if !isnothing(current_time)
        print(" current_time=", round(current_time; digits=3))
      end
      print(" time=", round(sweep_elapsed_time; digits=3))
      println()
      flush(stdout)
    end
    isdone = checkdone(;
      state, sweep, outputlevel, observer=(observer!), sweep_observer=(sweep_observer!)
    )
    isdone && break
  end
  return state
end

# Assume it is already in a reduced basis.
reduced_operator(operator) = operator
reduced_operator(operators::Vector{MPO}) = ProjMPOSum(operators)
reduced_operator(operator::MPO) = ProjMPO(operator)
