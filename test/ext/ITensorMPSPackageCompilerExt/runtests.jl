@eval module $(gensym())
using ITensorMPS: ITensorMPS
using ITensors: ITensors
using PackageCompiler: PackageCompiler
using Test: @test, @testset
@testset "ITensorMPSPackageCompilerExt" begin
    # Testing `ITensors.compile` would take too long so we just check
    # that `ITensorsPackageCompilerExt` overloads `ITensors.compile`.
    @test hasmethod(ITensors.compile, Tuple{ITensors.Algorithm"PackageCompiler"})
end
end
