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
    # Test the names are actually defined, if not we might need to import them
    # from ITensors.jl.
    for name in TestITensorMPSExportedNames.ITENSORMPS_EXPORTED_NAMES
      @test isdefined(ITensorMPS, name)
    end
  end
  @testset "Not exported" begin
    for name in
        [:AbstractProjMPO, :ProjMPS, :makeL!, :makeR!, :set_terms, :sortmergeterms, :terms]
      @test isdefined(ITensorMPS, name)
      @test !Base.isexported(ITensorMPS, name)
    end
  end
end
end
