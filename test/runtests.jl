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
    @test ITensorMPS.alternating_update_dmrg === ITensorTDVP.dmrg
  end
  @testset "Not exported" begin
    @test ITensorMPS.sortmergeterms === ITensors.ITensorMPS.sortmergeterms
    # Should we fix this in ITensors.jl by adding:
    # ```julia
    # using .ITensorMPS: sortmergeterms
    # ```
    # ?
    @test_broken ITensorMPS.sortmergeterms === ITensors.sortmergeterms
    @test ITensorMPS.AbstractSum === ITensors.ITensorMPS.AbstractSum
  end
end
end
