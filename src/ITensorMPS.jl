module ITensorMPS
using ITensorTDVP: TimeDependentSum, dmrg_x, linsolve, tdvp, to_vec
export TimeDependentSum, dmrg_x, linsolve, tdvp, to_vec
using ITensorTDVP: ITensorTDVP
alternating_update_dmrg(args...; kwargs...) = ITensorTDVP.dmrg(args...; kwargs...)
using ITensors.ITensorMPS: dmrg
export dmrg
end
