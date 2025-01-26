using Adapt: adapt
using ITensors:
  ITensors,
  Algorithm,
  Index,
  ITensor,
  @Algorithm_str,
  δ,
  commonind,
  dag,
  denseblocks,
  directsum,
  hasqns,
  prime,
  uniqueinds
using LinearAlgebra: normalize, svd, tr
using TypeParameterAccessors: unwrap_array_type
using VectorInterface: scalartype

# Possible improvements:
#  - Allow a maxdim argument to be passed to `expand`.
#  - Current behavior is letting bond dimension get too
#    big when used in imaginary time evolution.
#  - Consider switching the default to variational/fit apply
#    when building Krylov vectors.
#  - Use (1-tau*operator)|state> to generate Krylov vectors
#    instead of operator|state>. Is that needed?

function expand(state, reference; alg, kwargs...)
  return expand(Algorithm(alg), state, reference; kwargs...)
end

function expand_cutoff_doctring()
  return """
  The cutoff is used to control the truncation of the expanded
  basis and defaults to half the precision of the scalar type
  of the input state, i.e. ~1e-8 for `Float64`.
  """
end

function expand_warning_doctring()
  return """
  !!! warning
      Users are not given many customization options just yet as we
      gain more experience on the right balance between efficacy of the
      expansion and performance in various scenarios, and default values
      and keyword arguments are subject to change as we learn more about
      how to best use the method.
  """
end

function expand_citation_docstring()
  return """
  [^global_expansion]: Time Dependent Variational Principle with Ancillary Krylov Subspace. Mingru Yang, Steven R. White, [arXiv:2005.06104](https://arxiv.org/abs/2005.06104)
  """
end

"""
    expand(state::MPS, references::Vector{MPS}; alg="orthogonalize", cutoff)

Given an MPS `state` and a collection of MPS `references`,
returns an MPS which is equal to `state`
(has fidelity 1 with `state`) but whose MPS basis
is expanded to contain a portion of the basis of
the `references` that is orthogonal to the MPS basis of `state`.
See [^global_expansion] for more details.

$(expand_cutoff_doctring())

$(expand_warning_doctring())

$(expand_citation_docstring())
"""
function expand(
  ::Algorithm"orthogonalize",
  state::MPS,
  references::Vector{MPS};
  cutoff=(√(eps(real(scalartype(state))))),
)
  n = length(state)
  state = orthogonalize(state, n)
  references = map(reference -> orthogonalize(reference, n), references)
  s = siteinds(state)
  for j in reverse(2:n)
    # SVD state[j] to compute basisⱼ
    linds = [s[j - 1]; linkinds(state, j - 1)]
    _, λⱼ, basisⱼ = svd(state[j], linds; righttags="bψ_$j,Link")
    rinds = uniqueinds(basisⱼ, λⱼ)
    # Make projectorⱼ
    idⱼ = prod(rinds) do r
      return adapt(unwrap_array_type(basisⱼ), denseblocks(δ(scalartype(state), r', dag(r))))
    end
    projectorⱼ = idⱼ - prime(basisⱼ, rinds) * dag(basisⱼ)
    # Sum reference density matrices
    ρⱼ = sum(reference -> prime(reference[j], rinds) * dag(reference[j]), references)
    ρⱼ /= tr(ρⱼ)
    # Apply projectorⱼ
    ρⱼ_projected = apply(apply(projectorⱼ, ρⱼ), projectorⱼ)
    expanded_basisⱼ = basisⱼ
    if norm(ρⱼ_projected) > 10^3 * eps(real(scalartype(state)))
      # Diagonalize projected density matrix ρⱼ_projected
      # to compute reference_basisⱼ, which spans part of right basis
      # of references which is orthogonal to right basis of state
      dⱼ, reference_basisⱼ = eigen(
        ρⱼ_projected; cutoff, ishermitian=true, righttags="bϕ_$j,Link"
      )
      state_indⱼ = only(commoninds(basisⱼ, λⱼ))
      reference_indⱼ = only(commoninds(reference_basisⱼ, dⱼ))
      expanded_basisⱼ, expanded_indⱼ = directsum(
        basisⱼ => state_indⱼ, reference_basisⱼ => reference_indⱼ
      )
    end
    # Shift ortho center one site left using dag(expanded_basisⱼ)
    # and replace tensor at site j with expanded_basisⱼ
    state[j - 1] = state[j - 1] * (state[j] * dag(expanded_basisⱼ))
    state[j] = expanded_basisⱼ
    for reference in references
      reference[j - 1] = reference[j - 1] * (reference[j] * dag(expanded_basisⱼ))
      reference[j] = expanded_basisⱼ
    end
  end
  return state
end

"""
    expand(state::MPS, reference::MPO; alg="global_krylov", krylovdim=2, cutoff)

Given an MPS `state` and an MPO `reference`,
returns an MPS which is equal to `state`
(has fidelity 1 with state) but whose MPS basis
is expanded to contain a portion of the basis of
the Krylov space built by repeated applications of
`reference` to `state` that is orthogonal
to the MPS basis of `state`.
The `reference` operator is applied to `state` `krylovdim`
number of times, with a default of 2 which should give
a good balance between reliability and performance.
See [^global_expansion] for more details.

$(expand_cutoff_doctring())

$(expand_warning_doctring())

$(expand_citation_docstring())
"""
function expand(
  ::Algorithm"global_krylov",
  state::MPS,
  operator::MPO;
  krylovdim=2,
  cutoff=(√(eps(real(scalartype(state))))),
  apply_kwargs=(; maxdim=maxlinkdim(state) + 1),
)
  # TODO: Try replacing this logic with `Base.accumulate`.
  references = Vector{MPS}(undef, krylovdim)
  for k in 1:krylovdim
    previous_reference = get(references, k - 1, state)
    references[k] = normalize(apply(operator, previous_reference; apply_kwargs...))
  end
  return expand(state, references; alg="orthogonalize", cutoff)
end
