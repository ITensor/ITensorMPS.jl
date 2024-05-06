@eval module $(gensym())
using ITensorMPS: ITensorMPS
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
end
end
