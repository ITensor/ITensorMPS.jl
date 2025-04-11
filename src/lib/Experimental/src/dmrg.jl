using ..ITensorMPS:
  MPS,
  alternating_update,
  compose_observers,
  default_observer,
  eigsolve_updater,
  values_observer

function dmrg(
  operator, init::MPS; updater=eigsolve_updater, (observer!)=default_observer(), kwargs...
)
  info_ref! = Ref{Any}()
  info_observer! = values_observer(; info=(info_ref!))
  observer! = compose_observers(observer!, info_observer!)
  state = alternating_update(operator, init; updater, observer!, kwargs...)
  return info_ref![].eigval, state
end
