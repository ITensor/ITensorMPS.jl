using ITensors: ITensors, Index, settag
using QuantumOperatorDefinitions: sites

# TODO: Move to `ITensorSites.jl`?
function siteinds(t::String, n::Int; kwargs...)
  return sites(Index, t, n; kwargs...)
end
