using ITensors: ITensors, Index, settag
using QuantumOperatorDefinitions: QuantumOperatorDefinitions

# TODO: Move to `ITensorSites.jl`?
function siteinds(t::String, n::Int; kwargs...)
  # TODO: QuantumOperatorAlgebra.jl defines its own  `QuantumOperatorAlgebra.sites`
  # with a different meaning, decide if we should keep both.
  return QuantumOperatorDefinitions.sites(Index, t, n; kwargs...)
end
