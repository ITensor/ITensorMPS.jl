@eval module $(gensym())
using Test: @testset
using ITensorMPS: ITensorMPS
test_path = @__DIR__
test_files = filter(readdir(test_path)) do file
  return startswith("test_")(file) && endswith(".jl")(file)
end
@testset "$test_path" begin
  @testset "$filename" for filename in test_files
    println("Running $filename")
    @time include(joinpath(test_path, filename))
  end
end
end
