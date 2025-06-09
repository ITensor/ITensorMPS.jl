using ITensors, Test, ITensorMPS

@test LatticeBond(1, 2) == LatticeBond(1, 2, 0.0, 0.0, 0.0, 0.0, "")
@testset "Square lattice" begin
  sL = square_lattice(3, 4)
  @test length(sL) == 17
end

@testset "Triangular lattice" begin
  tL = triangular_lattice(3, 4)
  @test length(tL) == 23
  tL = triangular_lattice(3, 4; yperiodic=true)
  @test length(tL) == 28 # inc. periodic vertical bonds
end

@testset "Honeycomb XC lattice" begin
  hL = Honeycomb_XC(3, 4)
  @test length(hL) == 13
  hL = Honeycomb_XC(3, 4; yperiodic=true)
  @test length(hL) == 16
end

@testset "Honeycomb YC lattice" begin
  hL = Honeycomb_YC(4, 4)
  @test length(hL) == 18
  hL = Honeycomb_YC(4, 4; yperiodic=true)
  @test length(hL) == 20
end
