module ITensorMPS
using Reexport: @reexport
@reexport using ITensorTDVP: TimeDependentSum, dmrg_x, expand, linsolve, tdvp, to_vec
# Not re-exported, but this makes these types and functions accessible
# as `ITensorMPS.x`.
using ITensors.ITensorMPS:
  AbstractProjMPO, AbstractSum, ProjMPS, makeL!, makeR!, set_terms, sortmergeterms, terms
include("Experimental.jl")
using .Experimental: Experimental
include("Deprecated.jl")
using .Deprecated: Deprecated, dmrg
export dmrg
# `ops` is defined in `ITensors.SiteTypes`.
# TODO: Maybe reexport from there.
@reexport using ITensors: contract, ops
@reexport using ITensors.ITensorMPS:
  @OpName_str,
  @SiteType_str,
  @StateName_str,
  @TagType_str,
  @ValName_str,
  @preserve_ortho,
  @visualize,
  @visualize!,
  @visualize_noeval,
  @visualize_noeval!,
  @visualize_sequence,
  @visualize_sequence_noeval,
  AbstractMPS,
  AbstractObserver,
  Apply,
  AutoMPO,
  DMRGMeasurement,
  DMRGObserver,
  Lattice,
  LatticeBond,
  MPO,
  MPS,
  NoObserver,
  Op,
  OpName,
  OpSum,
  Ops,
  Prod,
  ProjMPO,
  ProjMPOSum,
  ProjMPO_MPS,
  Scaled,
  SiteType,
  Spectrum,
  StateName,
  Sum,
  Sweeps,
  TagType, # deprecate
  Trotter,
  ValName,
  add,
  add!,
  apply,
  applyMPO,
  applympo,
  argsdict,
  checkdone!, # remove export
  coefficient,
  common_siteind,
  common_siteinds,
  convert_leaf_eltype, # remove export
  correlation_matrix,
  cutoff,
  cutoff!, # deprecate
  disk,
  dot, # remove export
  eigs, # deprecate
  energies, # deprecate
  entropy, # deprecate
  errorMPOprod, # deprecate
  error_contract,
  error_mpoprod, # deprecate
  error_mul, # deprecate
  expect,
  findfirstsiteind, # deprecate
  findfirstsiteinds, # deprecate
  findsite, # deprecate
  findsites, # deprecate
  firstsiteind, # deprecate
  firstsiteinds, # deprecate
  get_cutoffs, # deprecate
  get_maxdims, # deprecate
  get_mindims, # deprecate
  get_noises, # deprecate
  has_fermion_string, # remove export
  hassameinds,
  inner,
  isortho,
  linkdim,
  linkdims,
  linkind,
  linkindex,
  linkinds,
  logdot,
  loginner,
  lognorm,
  lproj,
  maxdim,
  maxdim!,
  maxlinkdim,
  measure!,
  measurements,
  mindim,
  mindim!,
  movesite,
  movesites,
  mul, # deprecate
  multMPO,
  multmpo,
  noise,
  noise!,
  noiseterm,
  nsite,
  nsweep,
  op,
  orthoCenter,
  ortho_lims,
  orthocenter,
  orthogonalize,
  orthogonalize!,
  outer,
  position!,
  product,
  primelinks!,
  productMPS,
  projector,
  promote_itensor_eltype,
  randomMPO,
  randomMPS,
  random_mpo,
  random_mps,
  replace_siteinds,
  replace_siteinds!,
  replacebond,
  replacebond!,
  replaceprime,
  replacesites!,
  reset_ortho_lims!,
  rproj,
  sample,
  sample!,
  set_leftlim!,
  set_ortho_lims!,
  set_rightlim!,
  setcutoff!,
  setmaxdim!,
  setmindim!,
  setnoise!,
  simlinks!,
  siteind,
  siteindex,
  siteinds,
  splitblocks,
  square_lattice,
  state,
  sum,
  swapbondsites,
  sweepnext,
  tensors,
  toMPO,
  totalqn,
  tr,
  triangular_lattice,
  truncate,
  truncate!,
  truncerror,
  truncerrors,
  unique_siteind,
  unique_siteinds,
  val,
  ⋅
end
