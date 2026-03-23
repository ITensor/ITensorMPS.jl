using Aqua: Aqua
using ITensorMPS: ITensorMPS
using Test: @testset

@testset "Code quality (Aqua.jl)" begin
    Aqua.test_all(ITensorMPS)
end
