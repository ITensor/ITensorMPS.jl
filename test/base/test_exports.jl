@eval module $(gensym())
using ITensorMPS: ITensorMPS
using ITensors: ITensors
include("utils/TestITensorMPSExportedNames.jl")
using Test: @test, @test_broken, @testset
@testset "Exports and aliases" begin
  @testset "Exports" begin
    @test issetequal(
      names(ITensorMPS),
      [
        [:ITensorMPS]
        # ITensorTDVP reexports
        [:TimeDependentSum, :dmrg_x, :expand, :linsolve, :tdvp, :to_vec]
        # ITensors and ITensors.ITensorMPS reexports
        TestITensorMPSExportedNames.ITENSORMPS_EXPORTED_NAMES
      ],
    )
  end
  @testset "Not exported" begin
    for f in [
      :AbstractProjMPO,
      :AbstractMPS,
      :ProjMPS,
      :makeL!,
      :makeR!,
      :set_terms,
      :sortmergeterms,
      :terms,
    ]
      @test getfield(ITensorMPS, f) === getfield(ITensors.ITensorMPS, f)
    end
  end
end
end
