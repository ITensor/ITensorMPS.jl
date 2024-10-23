@eval module $(gensym())
using ITensorMPS: ITensorMPS
include("utils/TestITensorMPSExportedNames.jl")
using Test: @test, @test_broken, @testset
@testset "Exports and aliases" begin
  @testset "Exports" begin
    # @show setdiff(names(ITensorMPS), TestITensorMPSExportedNames.ITENSORMPS_EXPORTED_NAMES)
    # @show setdiff(TestITensorMPSExportedNames.ITENSORMPS_EXPORTED_NAMES, names(ITensorMPS))
    @test issetequal(
      names(ITensorMPS), TestITensorMPSExportedNames.ITENSORMPS_EXPORTED_NAMES
    )
  end
  @testset "Not exported" begin
    for f in
        [:AbstractProjMPO, :ProjMPS, :makeL!, :makeR!, :set_terms, :sortmergeterms, :terms]
      @test isdefined(ITensorMPS, f)
      @test !Base.isexported(ITensorMPS, f)
    end
  end
end
end
