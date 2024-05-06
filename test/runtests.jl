@eval module $(gensym())
using ITensorMPS: ITensorMPS
using ITensorTDVP: ITensorTDVP
using Test: @test, @testset
@testset "ITensorMPS.jl" begin
  @testset "exports" begin
    @test issetequal(
      names(ITensorMPS),
      [
        :ITensorMPS,
        # ITensorMPS reexports
        :dmrg,
        # ITensorTDVP reexports
        :TimeDependentSum,
        :dmrg_x,
        :linsolve,
        :tdvp,
        :to_vec,
      ],
    )
  end
  @testset "aliases" begin
    @test ITensorMPS.alternating_update_dmrg === ITensorTDVP.dmrg
  end
end
end
