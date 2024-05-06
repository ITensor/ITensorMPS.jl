@eval module $(gensym())
using ITensorMPS: ITensorMPS
using ITensorTDVP: ITensorTDVP
using ITensors: ITensors
include("utils/TestITensorMPSExportedNames.jl")
using Test: @test, @testset
@testset "ITensorMPS.jl" begin
  @testset "exports" begin
    @test issetequal(
      names(ITensorMPS),
      [
        [:ITensorMPS]
        # ITensorTDVP reexports
        [:TimeDependentSum, :dmrg_x, :linsolve, :tdvp, :to_vec]
        # ITensors and ITensors.ITensorMPS reexports
        TestITensorMPSExportedNames.ITENSORMPS_EXPORTED_NAMES
      ],
    )
  end
  @testset "aliases" begin
    @test ITensorMPS.alternating_update_dmrg === ITensorTDVP.dmrg
  end
end
end
