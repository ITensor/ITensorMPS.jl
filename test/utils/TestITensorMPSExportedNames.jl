module TestITensorMPSExportedNames
const ITENSORMPS_EXPORTED_NAMES = [
  Symbol("@OpName_str"),
  Symbol("@SiteType_str"),
  Symbol("@StateName_str"),
  Symbol("@TagType_str"),
  Symbol("@ValName_str"),
  Symbol("@preserve_ortho"),
  Symbol("@visualize"),
  Symbol("@visualize!"),
  Symbol("@visualize_noeval"),
  Symbol("@visualize_noeval!"),
  Symbol("@visualize_sequence"),
  Symbol("@visualize_sequence_noeval"),
  :AbstractMPS,
  :AbstractObserver,
  :Apply,
  :AutoMPO,
  :DMRGMeasurement,
  :DMRGObserver,
  :ITensorMPS,
  :Lattice,
  :LatticeBond,
  :MPO,
  :MPS,
  :NoObserver,
  :Op,
  :OpName,
  :OpSum,
  :Ops,
  :Prod,
  :ProjMPO,
  :ProjMPOSum,
  :ProjMPO_MPS,
  :Scaled,
  :SiteType,
  :Spectrum,
  :StateName,
  :Sum,
  :Sweeps,
  :Trotter,
  :ValName,
  :add,
  :add!,
  :apply,
  :applyMPO,
  :applympo,
  :argsdict,
  :checkdone!, # remove export
  :coefficient,
  :common_siteind,
  :common_siteinds,
  :correlation_matrix,
  :cutoff,
  :cutoff!, # deprecate
  :disk,
  :dmrg,
  :dot, # remove export
  :eigs, # deprecate
  :energies, # deprecate
  :entropy, # deprecate
  :errorMPOprod, # deprecate
  :error_contract,
  :error_mpoprod, # deprecate
  :error_mul, # deprecate
  :expect,
  :findfirstsiteind, # deprecate
  :findfirstsiteinds, # deprecate
  :findsite, # deprecate
  :findsites, # deprecate
  :firstsiteind, # deprecate
  :firstsiteinds, # deprecate
  :get_cutoffs, # deprecate
  :get_maxdims, # deprecate
  :get_mindims, # deprecate
  :get_noises, # deprecate
  :has_fermion_string, # remove export
  :hassameinds,
  :inner,
  :isortho,
  :linkdim,
  :linkdims,
  :linkind,
  :linkindex,
  :linkinds,
  :logdot,
  :loginner,
  :lognorm,
  :lproj,
  :maxdim,
  :maxdim!,
  :maxlinkdim,
  :measure!,
  :measurements,
  :mindim,
  :mindim!,
  :movesite,
  :movesites,
  :mul, # deprecate
  :multMPO,
  :multmpo,
  :noise,
  :noise!,
  :noiseterm,
  :nsite,
  :nsweep,
  :op,
  :ops,
  :orthoCenter,
  :ortho_lims,
  :orthocenter,
  :orthogonalize,
  :orthogonalize!,
  :outer,
  :position!,
  :product,
  :primelinks!,
  :productMPS,
  :projector,
  :promote_itensor_eltype,
  :randomITensor,
  :randomMPO,
  :randomMPS,
  :replace_siteinds,
  :replace_siteinds!,
  :replacebond,
  :replacebond!,
  :replaceprime,
  :replacesites!,
  :reset_ortho_lims!,
  :rproj,
  :sample,
  :sample!,
  :set_leftlim!,
  :set_ortho_lims!,
  :set_rightlim!,
  :setcutoff!,
  :setmaxdim!,
  :setmindim!,
  :setnoise!,
  :simlinks!,
  :siteind,
  :siteindex,
  :siteinds,
  :splitblocks,
  :square_lattice,
  :state,
  :sum,
  :swapbondsites,
  :sweepnext,
  :tensors,
  :toMPO,
  :totalqn,
  :tr,
  :triangular_lattice,
  :truncate,
  :truncate!,
  :truncerror,
  :truncerrors,
  :unique_siteind,
  :unique_siteinds,
  :val,
  :⋅,
]
end
