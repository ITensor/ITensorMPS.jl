using ITensors: ITensors, Index, settag
using QuantumOperatorDefinitions: SiteType

# TODO: Move to `ITensorSites.jl`?
function siteinds(t::String, n::Int; kwargs...)
  return [settag(Index(SiteType(t; kwargs...)), "n", string(j)) for j in 1:n]
end
