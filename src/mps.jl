using Adapt: adapt
using GradedUnitRanges: dual
using ITensors: hasqns
using LinearAlgebra: qr
using QuantumOperatorDefinitions: QuantumOperatorDefinitions, state
using Random: Random, AbstractRNG
using SparseArraysBase: oneelement

## TODO: Add this back.
## using NDTensors: using_auto_fermion

"""
    MPS

A finite size matrix product state type.
Keeps track of the orthogonality center.
"""
mutable struct MPS <: AbstractMPS
  data::Vector{ITensor}
  llim::Int
  rlim::Int
end

function MPS(A::Vector{<:ITensor}; ortho_lims::UnitRange=1:length(A))
  return MPS(A, first(ortho_lims) - 1, last(ortho_lims) + 1)
end

set_data(A::MPS, data::Vector{ITensor}) = MPS(data, A.llim, A.rlim)

@doc """
    MPS(v::Vector{<:ITensor})

Construct an MPS from a Vector of ITensors.
""" MPS(v::Vector{<:ITensor})

"""
    MPS()

Construct an empty MPS with zero sites.
"""
MPS() = MPS(ITensor[], 0, 0)

"""
    MPS(N::Int)

Construct an MPS with N sites with default constructed
ITensors.
"""
function MPS(N::Int; ortho_lims::UnitRange=1:N)
  return MPS(Vector{ITensor}(undef, N); ortho_lims=ortho_lims)
end

function apply_random_staircase_circuit(
  rng::AbstractRNG, elt::Type, state::MPS; depth, kwargs...
)
  n = length(state)
  layer = [(j - 1, j) for j in reverse(2:n)]
  s = siteinds(state)
  gate_layers = mapreduce(vcat, 1:depth) do _
    # TODO: Pass `elt` and `rng` as kwargs to `op`, to be
    # used as parameters in `OpName` by `QuantumOperaterDefinitions.jl`.
    return map(((i, j),) -> op("RandomUnitary", (s[i], s[j])), layer)
  end
  # TODO: Use `apply`, for some reason that can't be found right now.
  return apply(gate_layers, state; kwargs...)
end

function random_mps(
  rng::AbstractRNG, elt::Type, state, sites::Vector{<:Index}; maxdim, kwargs...
)
  x = MPS(elt, state, sites)
  depth = ceil(Int, log(minimum(Int ∘ length, sites), maxdim))
  return apply_random_staircase_circuit(rng, elt, x; depth, maxdim, kwargs...)
end
function random_mps(rng::AbstractRNG, state, sites::Vector{<:Index}; kwargs...)
  return random_mps(rng, Float64, state, sites; kwargs...)
end
function random_mps(elt::Type, state, sites::Vector{<:Index}; kwargs...)
  return random_mps(Random.default_rng(), elt, state, sites; kwargs...)
end
function random_mps(state, sites::Vector{<:Index}; kwargs...)
  return random_mps(Random.default_rng(), Float64, state, sites; kwargs...)
end
function random_mps(sites; kwargs...)
  state = fill("0", length(sites))
  return random_mps(Random.default_rng(), Float64, state, sites; kwargs...)
end

"""
    MPS(ivals::Vector{<:Pair{<:Index}})

Construct a product state MPS with element type `Float64` and
nonzero values determined from the input IndexVals.
"""
MPS(ivals::Vector{<:Pair{<:Index}}) = MPS(Float64, ivals)

"""
    MPS(::Type{T},
        sites::Vector{<:Index},
        states::Union{Vector{String},
                      Vector{Int},
                      String,
                      Int})

Construct a product state MPS of element type `T`, having
site indices `sites`, and which corresponds to the initial
state given by the array `states`. The input `states` may
be an array of strings or an array of ints recognized by the
`state` function defined for the relevant Index tag type.
In addition, a single string or int can be input to create
a uniform state.

# Examples

```julia
N = 10
sites = siteinds("S=1/2", N)
states = [isodd(n) ? "Up" : "Dn" for n in 1:N]
psi = MPS(ComplexF64, sites, states)
phi = MPS(sites, "Up")
```
"""
function MPS(elt::Type{<:Number}, states_, sites::Vector{<:Index})
  return MPS([state(elt, states_[j], sites[j]) for j in 1:length(sites)])
end

function MPS(
  ::Type{T}, state::Union{String,Integer}, sites::Vector{<:Index}
) where {T<:Number}
  return MPS(T, fill(state, length(sites)), sites)
end

function MPS(::Type{T}, states::Function, sites::Vector{<:Index}) where {T<:Number}
  states_vec = [states(n) for n in 1:length(sites)]
  return MPS(T, states_vec, sites)
end

"""
    MPS(states, sites::Vector{<:Index})

Construct a product state MPS having
site indices `sites`, and which corresponds to the initial
state given by the array `states`. The `states` array may
consist of either an array of integers or strings, as
recognized by the `state` function defined for the relevant
Index tag type.

# Examples

```julia
N = 10
sites = siteinds("S=1/2", N)
states = [isodd(n) ? "Up" : "Dn" for n in 1:N]
psi = MPS(sites, states)
```
"""
MPS(state, sites::Vector{<:Index}) = MPS(Float64, state, sites)

"""
    siteind(M::MPS, j::Int; kwargs...)

Get the first site Index of the MPS. Return `nothing` if none is found.
"""
siteind(M::MPS, j::Int; kwargs...) = siteind(first, M, j; kwargs...)

"""
    siteind(::typeof(only), M::MPS, j::Int; kwargs...)

Get the only site Index of the MPS. Return `nothing` if none is found.
"""
function siteind(::typeof(only), M::MPS, j::Int; kwargs...)
  is = siteinds(M, j; kwargs...)
  if isempty(is)
    return nothing
  end
  return only(is)
end

"""
    siteinds(M::MPS)
    siteinds(::typeof(first), M::MPS)

Get a vector of the first site Index found on each tensor of the MPS.

    siteinds(::typeof(only), M::MPS)

Get a vector of the only site Index found on each tensor of the MPS. Errors if more than one is found.

    siteinds(::typeof(all), M::MPS)

Get a vector of the all site Indices found on each tensor of the MPS. Returns a Vector of IndexSets.
"""
siteinds(M::MPS; kwargs...) = siteinds(first, M; kwargs...)

function replace_siteinds!(M::MPS, sites)
  for j in eachindex(M)
    sj = only(siteinds(M, j))
    M[j] = replaceinds(M[j], sj => sites[j])
  end
  return M
end

replace_siteinds(M::MPS, sites) = replace_siteinds!(copy(M), sites)

"""
    replacebond!(M::MPS, b::Int, phi::ITensor; kwargs...)

Factorize the ITensor `phi` and replace the ITensors
`b` and `b+1` of MPS `M` with the factors. Choose
the orthogonality with `ortho="left"/"right"`.
"""
function replacebond!(
  M::MPS,
  b::Int,
  phi::ITensor;
  normalize=nothing,
  swapsites=nothing,
  ortho=nothing,
  # Decomposition kwargs
  which_decomp=nothing,
  mindim=nothing,
  maxdim=nothing,
  cutoff=nothing,
  eigen_perturbation=nothing,
  # svd kwargs
  svd_alg=nothing,
  use_absolute_cutoff=nothing,
  use_relative_cutoff=nothing,
  min_blockdim=nothing,
)
  normalize = replace_nothing(normalize, false)
  swapsites = replace_nothing(swapsites, false)
  ortho = replace_nothing(ortho, "left")

  indsMb = inds(M[b])
  if swapsites
    sb = siteind(M, b)
    sbp1 = siteind(M, b + 1)
    indsMb = replaceind(indsMb, sb, sbp1)
  end
  L, R, spec = factorize(
    phi,
    indsMb;
    mindim,
    maxdim,
    cutoff,
    ortho,
    which_decomp,
    eigen_perturbation,
    svd_alg,
    tags=tags(linkind(M, b)),
    use_absolute_cutoff,
    use_relative_cutoff,
    min_blockdim,
  )
  M[b] = L
  M[b + 1] = R
  if ortho == "left"
    leftlim(M) == b - 1 && setleftlim!(M, leftlim(M) + 1)
    rightlim(M) == b + 1 && setrightlim!(M, rightlim(M) + 1)
    normalize && (M[b + 1] ./= norm(M[b + 1]))
  elseif ortho == "right"
    leftlim(M) == b && setleftlim!(M, leftlim(M) - 1)
    rightlim(M) == b + 2 && setrightlim!(M, rightlim(M) - 1)
    normalize && (M[b] ./= norm(M[b]))
  else
    error(
      "In replacebond!, got ortho = $ortho, only currently supports `left` and `right`."
    )
  end
  return spec
end

"""
    replacebond(M::MPS, b::Int, phi::ITensor; kwargs...)

Like `replacebond!`, but returns the new MPS.
"""
function replacebond(M0::MPS, b::Int, phi::ITensor; kwargs...)
  M = copy(M0)
  replacebond!(M, b, phi; kwargs...)
  return M
end

# Allows overloading `replacebond!` based on the projected
# MPO type. By default just calls `replacebond!` on the MPS.
function replacebond!(PH, M::MPS, b::Int, phi::ITensor; kwargs...)
  return replacebond!(M, b, phi; kwargs...)
end

"""
    sample!(m::MPS)

Given a normalized MPS m, returns a `Vector{Int}`
of `length(m)` corresponding to one sample
of the probability distribution defined by
squaring the components of the tensor
that the MPS represents. If the MPS does
not have an orthogonality center,
orthogonalize!(m,1) will be called before
computing the sample.
"""
function sample!(m::MPS)
  return sample!(Random.default_rng(), m)
end

function sample!(rng::AbstractRNG, m::MPS)
  orthogonalize!(m, 1)
  return sample(rng, m)
end

"""
    sample(m::MPS)

Given a normalized MPS m with `orthocenter(m)==1`,
returns a `Vector{Int}` of `length(m)`
corresponding to one sample of the
probability distribution defined by
squaring the components of the tensor
that the MPS represents
"""
function sample(m::MPS)
  return sample(Random.default_rng(), m)
end

function sample(rng::AbstractRNG, m::MPS)
  N = length(m)

  if orthocenter(m) != 1
    error("sample: MPS m must have orthocenter(m)==1")
  end
  if abs(1.0 - norm(m[1])) > 1E-8
    error("sample: MPS is not normalized, norm=$(norm(m[1]))")
  end

  ElT = scalartype(m)

  result = zeros(Int, N)
  A = m[1]

  for j in 1:N
    s = siteind(m, j)
    d = dim(s)
    # Compute the probability of each state
    # one-by-one and stop when the random
    # number r is below the total prob so far
    pdisc = zero(real(ElT))
    r = rand(rng)
    # Will need n,An, and pn below
    n = 1
    An = ITensor()
    pn = zero(real(ElT))
    while n <= d
      projn = ITensor(s)
      projn[s => n] = one(ElT)
      An = A * dag(adapt(datatype(A), projn))
      pn = real(scalar(dag(An) * An))
      pdisc += pn
      (r < pdisc) && break
      n += 1
    end
    result[j] = n
    if j < N
      A = m[j + 1] * An
      A *= (one(ElT) / sqrt(pn))
    end
  end
  return result
end

_op_prod(o1::AbstractString, o2::AbstractString) = "$o1 * $o2"
_op_prod(o1::Matrix{<:Number}, o2::Matrix{<:Number}) = o1 * o2

"""
    correlation_matrix(psi::MPS,
                       Op1::AbstractString,
                       Op2::AbstractString;
                       kwargs...)

    correlation_matrix(psi::MPS,
                       Op1::Matrix{<:Number},
                       Op2::Matrix{<:Number};
                       kwargs...)

Given an MPS psi and two strings denoting
operators (as recognized by the `op` function),
computes the two-point correlation function matrix
C[i,j] = <psi| Op1i Op2j |psi>
using efficient MPS techniques. Returns the matrix C.

# Optional Keyword Arguments

  - `sites = 1:length(psi)`: compute correlations only
     for sites in the given range
  - `ishermitian = false` : if `false`, force independent calculations of the
     matrix elements above and below the diagonal, while if `true` assume they are complex conjugates.

For a correlation matrix of size NxN and an MPS of typical
bond dimension m, the scaling of this algorithm is N^2*m^3.

# Examples

```julia
N = 30
m = 4

s = siteinds("S=1/2", N)
psi = random_mps(s; linkdims=m)
Czz = correlation_matrix(psi, "Sz", "Sz")
Czz = correlation_matrix(psi, [1/2 0; 0 -1/2], [1/2 0; 0 -1/2]) # same as above

s = siteinds("Electron", N; conserve_qns=true)
psi = random_mps(s, n -> isodd(n) ? "Up" : "Dn"; linkdims=m)
Cuu = correlation_matrix(psi, "Cdagup", "Cup"; sites=2:8)
```
"""
function correlation_matrix(
  psi::MPS, _Op1, _Op2; sites=1:length(psi), site_range=nothing, ishermitian=nothing
)
  if !isnothing(site_range)
    @warn "The `site_range` keyword arg. to `correlation_matrix` is deprecated: use the keyword `sites` instead"
    sites = site_range
  end
  if !(sites isa AbstractRange)
    sites = collect(sites)
  end

  start_site = first(sites)
  end_site = last(sites)

  N = length(psi)
  ElT = scalartype(psi)
  s = siteinds(psi)

  Op1 = _Op1 #make copies into which we can insert "F" string operators, and then restore.
  Op2 = _Op2
  onsiteOp = _op_prod(Op1, Op2)
  fermionic1 = has_fermion_string(Op1, s[start_site])
  fermionic2 = has_fermion_string(Op2, s[end_site])
  if fermionic1 != fermionic2
    error(
      "correlation_matrix: Mixed fermionic and bosonic operators are not supported yet."
    )
  end

  # Decide if we need to calculate a non-hermitian corr. matrix, which is roughly double the work.
  is_cm_hermitian = ishermitian
  if isnothing(is_cm_hermitian)
    # Assume correlation matrix is non-hermitian
    is_cm_hermitian = false
    O1 = op(Op1, s, start_site)
    O2 = op(Op2, s, start_site)
    O1 /= norm(O1)
    O2 /= norm(O2)
    #We need to decide if O1 ∝ O2 or O1 ∝ O2^dagger allowing for some round off errors.
    eps = 1e-10
    is_op_proportional = norm(O1 - O2) < eps
    is_op_hermitian = norm(O1 - dag(swapprime(O2, 0, 1))) < eps
    if is_op_proportional || is_op_hermitian
      is_cm_hermitian = true
    end
    # finally if they are both fermionic and proportional then the corr matrix will
    # be anti symmetric insterad of Hermitian. Handle things like <C_i*C_j>
    # at this point we know fermionic2=fermionic1, but we put them both in the if
    # to clarify the meaning of what we are doing.
    if is_op_proportional && fermionic1 && fermionic2
      is_cm_hermitian = false
    end
  end

  psi = orthogonalize(psi, start_site)
  norm2_psi = norm(psi[start_site])^2

  # Nb = size of block of correlation matrix
  Nb = length(sites)

  C = zeros(ElT, Nb, Nb)

  if start_site == 1
    L = ITensor(1.0)
  else
    lind = commonind(psi[start_site], psi[start_site - 1])
    L = delta(dual(lind), lind')
  end
  pL = start_site - 1

  for (ni, i) in enumerate(sites[1:(end - 1)])
    while pL < i - 1
      pL += 1
      sᵢ = siteind(psi, pL)
      L = (L * psi[pL]) * prime(dag(psi[pL]), !sᵢ)
    end

    Li = L * psi[i]

    # Get j == i diagonal correlations
    rind = commonind(psi[i], psi[i + 1])
    oᵢ = adapt(datatype(Li), op(onsiteOp, s, i))
    C[ni, ni] = ((Li * oᵢ) * prime(dag(psi[i]), !rind))[] / norm2_psi

    # Get j > i correlations
    if !using_auto_fermion() && fermionic2
      Op1 = "$Op1 * F"
    end

    oᵢ = adapt(datatype(Li), op(Op1, s, i))

    Li12 = (dag(psi[i])' * oᵢ) * Li
    pL12 = i

    for (n, j) in enumerate(sites[(ni + 1):end])
      nj = ni + n

      while pL12 < j - 1
        pL12 += 1
        if !using_auto_fermion() && fermionic2
          oᵢ = adapt(datatype(psi[pL12]), op("F", s[pL12]))
          Li12 *= (oᵢ * dag(psi[pL12])')
        else
          sᵢ = siteind(psi, pL12)
          Li12 *= prime(dag(psi[pL12]), !sᵢ)
        end
        Li12 *= psi[pL12]
      end

      lind = commonind(psi[j], Li12)
      Li12 *= psi[j]

      oⱼ = adapt(datatype(Li12), op(Op2, s, j))
      sⱼ = siteind(psi, j)
      val = (Li12 * oⱼ) * prime(dag(psi[j]), (sⱼ, lind))

      # XXX: This gives a different fermion sign with
      # ITensors.enable_auto_fermion()
      # val = prime(dag(psi[j]), (sⱼ, lind)) * (oⱼ * Li12)

      C[ni, nj] = scalar(val) / norm2_psi
      if is_cm_hermitian
        C[nj, ni] = conj(C[ni, nj])
      end

      pL12 += 1
      if !using_auto_fermion() && fermionic2
        oᵢ = adapt(datatype(psi[pL12]), op("F", s[pL12]))
        Li12 *= (oᵢ * dag(psi[pL12])')
      else
        sᵢ = siteind(psi, pL12)
        Li12 *= prime(dag(psi[pL12]), !sᵢ)
      end
      @assert pL12 == j
    end #for j
    Op1 = _Op1 #"Restore Op1 with no Fs"

    if !is_cm_hermitian #If isHermitian=false the we must calculate the below diag elements explicitly.

      #  Get j < i correlations by swapping the operators
      if !using_auto_fermion() && fermionic1
        Op2 = "$Op2 * F"
      end
      oᵢ = adapt(datatype(psi[i]), op(Op2, s, i))
      Li21 = (Li * oᵢ) * dag(psi[i])'
      pL21 = i
      if !using_auto_fermion() && fermionic1
        Li21 = -Li21 #Required because we swapped fermionic ops, instead of sweeping right to left.
      end

      for (n, j) in enumerate(sites[(ni + 1):end])
        nj = ni + n

        while pL21 < j - 1
          pL21 += 1
          if !using_auto_fermion() && fermionic1
            oᵢ = adapt(datatype(psi[pL21]), op("F", s[pL21]))
            Li21 *= oᵢ * dag(psi[pL21])'
          else
            sᵢ = siteind(psi, pL21)
            Li21 *= prime(dual(si[pL21]), !sᵢ)
          end
          Li21 *= psi[pL21]
        end

        lind = commonind(psi[j], Li21)
        Li21 *= psi[j]

        oⱼ = adapt(datatype(psi[j]), op(Op1, s, j))
        sⱼ = siteind(psi, j)
        val = (prime(dag(psi[j]), (sⱼ, lind)) * (oⱼ * Li21))[]
        C[nj, ni] = val / norm2_psi

        pL21 += 1
        if !using_auto_fermion() && fermionic1
          oᵢ = adapt(datatype(psi[pL21]), op("F", s[pL21]))
          Li21 *= (oᵢ * dag(psi[pL21])')
        else
          sᵢ = siteind(psi, pL21)
          Li21 *= prime(dag(psi[pL21]), !sᵢ)
        end
        @assert pL21 == j
      end #for j
      Op2 = _Op2 #"Restore Op2 with no Fs"
    end #if is_cm_hermitian

    pL += 1
    sᵢ = siteind(psi, i)
    L = Li * prime(dag(psi[i]), !sᵢ)
  end #for i

  # Get last diagonal element of C
  i = end_site
  while pL < i - 1
    pL += 1
    sᵢ = siteind(psi, pL)
    L = L * psi[pL] * prime(dag(psi[pL]), !sᵢ)
  end
  lind = commonind(psi[i], psi[i - 1])
  oᵢ = adapt(datatype(psi[i]), op(onsiteOp, s, i))
  sᵢ = siteind(psi, i)
  val = (L * (oᵢ * psi[i]) * prime(dag(psi[i]), (sᵢ, lind)))[]
  C[Nb, Nb] = val / norm2_psi

  return C
end

"""
    expect(psi::MPS, op::AbstractString...; kwargs...)
    expect(psi::MPS, op::Matrix{<:Number}...; kwargs...)
    expect(psi::MPS, ops; kwargs...)

Given an MPS `psi` and a single operator name, returns
a vector of the expected value of the operator on
each site of the MPS.

If multiple operator names are provided, returns a tuple
of expectation value vectors.

If a container of operator names is provided, returns the
same type of container with names replaced by vectors
of expectation values.

# Optional Keyword Arguments

  - `sites = 1:length(psi)`: compute expected values only for sites in the given range

# Examples

```julia
N = 10

s = siteinds("S=1/2", N)
psi = random_mps(s; linkdims=8)
Z = expect(psi, "Sz") # compute for all sites
Z = expect(psi, "Sz"; sites=2:4) # compute for sites 2,3,4
Z3 = expect(psi, "Sz"; sites=3)  # compute for site 3 only (output will be a scalar)
XZ = expect(psi, ["Sx", "Sz"]) # compute Sx and Sz for all sites
Z = expect(psi, [1/2 0; 0 -1/2]) # same as expect(psi,"Sz")

s = siteinds("Electron", N)
psi = random_mps(s; linkdims=8)
dens = expect(psi, "Ntot")
updens, dndens = expect(psi, "Nup", "Ndn") # pass more than one operator
```
"""
function expect(psi::MPS, ops; sites=1:length(psi), site_range=nothing)
  psi = copy(psi)
  N = length(psi)
  ElT = scalartype(psi)
  s = siteinds(psi)

  if !isnothing(site_range)
    @warn "The `site_range` keyword arg. to `expect` is deprecated: use the keyword `sites` instead"
    sites = site_range
  end

  site_range = (sites isa AbstractRange) ? sites : collect(sites)
  Ns = length(site_range)
  start_site = first(site_range)

  el_types = map(o -> ishermitian(op(o, s[start_site])) ? real(ElT) : ElT, ops)

  psi = orthogonalize(psi, start_site)
  norm2_psi = norm(psi)^2
  iszero(norm2_psi) && error("MPS has zero norm in function `expect`")

  ex = map((o, el_t) -> zeros(el_t, Ns), ops, el_types)
  for (entry, j) in enumerate(site_range)
    psi = orthogonalize(psi, j)
    for (n, opname) in enumerate(ops)
      oⱼ = adapt(datatype(psi[j]), op(opname, s[j]))
      val = inner(psi[j], apply(oⱼ, psi[j])) / norm2_psi
      ex[n][entry] = (el_types[n] <: Real) ? real(val) : val
    end
  end

  if sites isa Number
    return map(arr -> arr[1], ex)
  end
  return ex
end

function expect(psi::MPS, op::AbstractString; kwargs...)
  return first(expect(psi, (op,); kwargs...))
end

function expect(psi::MPS, op::Matrix{<:Number}; kwargs...)
  return first(expect(psi, (op,); kwargs...))
end

function expect(psi::MPS, op1::AbstractString, ops::AbstractString...; kwargs...)
  return expect(psi, (op1, ops...); kwargs...)
end

function expect(psi::MPS, op1::Matrix{<:Number}, ops::Matrix{<:Number}...; kwargs...)
  return expect(psi, (op1, ops...); kwargs...)
end
