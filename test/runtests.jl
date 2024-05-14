@eval module $(gensym())
using ITensorMPS: ITensorMPS
using ITensorTDVP: ITensorTDVP
using ITensors: ITensors
include("utils/TestITensorMPSExportedNames.jl")
using Test: @test, @test_broken, @testset
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
  @testset "Aliases" begin
    @test ITensorMPS.Experimental.dmrg === ITensorTDVP.dmrg
    @test ITensorMPS.dmrg === ITensors.ITensorMPS.dmrg
  end
  @testset "Not exported" begin
    @test ITensorMPS.sortmergeterms === ITensors.ITensorMPS.sortmergeterms
    # Should we fix this in ITensors.jl by adding:
    # ```julia
    # using .ITensorMPS: sortmergeterms
    # ```
    # ?
    @test_broken ITensorMPS.sortmergeterms === ITensors.sortmergeterms
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
