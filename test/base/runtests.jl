using ITensors: ITensors
using Test: @testset

ITensors.Strided.disable_threads()
ITensors.BLAS.set_num_threads(1)
ITensors.disable_threaded_blocksparse()

@testset "$(@__DIR__)" begin
  filenames = filter(readdir(@__DIR__)) do file
    return startswith("test_")(file) && endswith(".jl")(file)
  end
  @testset "Test $(@__DIR__)/$filename" for filename in filenames
    println("Running $(@__DIR__)/$filename")
    @time include(filename)
  end

  test_dirs = ["test_solvers"]
  @testset "Test $(@__DIR__)/$test_dir" for test_dir in test_dirs
    println("Running $(@__DIR__)/$test_dir/runtests.jl")
    @time include(joinpath(@__DIR__, test_dir, "runtests.jl"))
  end
end
