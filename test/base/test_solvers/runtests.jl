@eval module $(gensym())
using Test: @testset
using ITensorMPS: ITensorMPS
test_path = joinpath(pkgdir(ITensorMPS), "test")
test_files = filter(
  file -> startswith(file, "test_") && endswith(file, ".jl"), readdir(test_path)
)
@testset "ITensorMPS.jl" begin
  @testset "$filename" for filename in test_files
    println("Running $filename")
    @time include(joinpath(test_path, filename))
  end
end
end
