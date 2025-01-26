using ITensors: Index, settag
using QuantumOperatorDefinitions: SiteType

# TODO: Move to `ITensorSites.jl`?
# TODO: Add support for symmetry sectors.
function siteinds(t::String, n::Int)
  return [settag(Index(length(SiteType(t))), "sitetype", t) for _ in 1:n]
end
