using ITensors: ITensors, ITensor, Index, @Algorithm_str, contract

function contract_operator_state_updater(operator, init; internal_kwargs)
  # TODO: Use `contract(operator)`.
  state = ITensor(true)
  for j in (operator.lpos + 1):(operator.rpos - 1)
    state *= operator.input_state[j]
  end
  state = contract(operator, state)
  return state, (;)
end

function default_contract_init(operator::MPO, input_state::MPS)
  input_state = deepcopy(input_state)
  s = only.(siteinds(uniqueinds, operator, input_state))
  # TODO: Fix issue with `replace_siteinds`, seems to be modifying in-place.
  return replace_siteinds(deepcopy(input_state), s)
end

function ITensors.contract(
  ::Algorithm"fit",
  operator::MPO,
  input_state::MPS;
  init=default_contract_init(operator, input_state),
  kwargs...,
)
  # Fix siteinds of `init` if needed.
  # This is needed to work around an issue that `apply`
  # can't be customized right now, and just uses the same `init`
  # as that of `contract`.
  # TODO: Allow customization of `apply` and remove this.
  s = only.(siteinds(uniqueinds, operator, input_state))
  if !all(p -> p[1] == [2], zip(s, siteinds(init)))
    # TODO: Fix issue with `replace_siteinds`, seems to be modifying in-place.
    init = replace_siteinds(deepcopy(init), s)
  end
  reduced_operator = ReducedContractProblem(input_state, operator)
  return alternating_update(
    reduced_operator, init; updater=contract_operator_state_updater, kwargs...
  )
end
