@eval module $(gensym())
using Suppressor: @suppress
using ITensorMPS: ITensorMPS
using Test: @testset
@testset "Run examples" begin
  examples_files = [
    "01_tdvp.jl", "02_dmrg-x.jl", "03_tdvp_time_dependent.jl", "04_tdvp_observers.jl"
  ]
  examples_path = joinpath(pkgdir(ITensorMPS), "examples")
  @testset "Running example file $f" for f in examples_files
    println("Running example file $f")
    @suppress include(joinpath(examples_path, f))
  end
end
end