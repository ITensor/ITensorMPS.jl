using ITensorMPS
using ITensors
using Random
using Test

function heisenberg_mpo(s)
    os = OpSum()
    for j in 1:(length(s) - 1)
        os += "Sz", j, "Sz", j + 1
        os += 1 / 2, "S+", j, "S-", j + 1
        os += 1 / 2, "S-", j, "S+", j + 1
    end
    return MPO(os, s)
end

@testset "CheckpointedProjMPO (conserve_qns=$conserve_qns)" for conserve_qns in
    (false, true)
    n = 20
    s = siteinds("S=1/2", n; conserve_qns)
    H = heisenberg_mpo(s)
    psi0 = MPS(s, j -> isodd(j) ? "Up" : "Dn")

    @testset "dmrg matches ProjMPO (k=$k)" for k in (1, 3, 4, 7)
        kw = (; nsweeps = 4, maxdim = [10, 20, 40], cutoff = 1.0e-12, outputlevel = 0)
        Random.seed!(1)
        e0, psi_ref = dmrg(ProjMPO(H), psi0; kw...)
        Random.seed!(1)
        e1, psi = dmrg(CheckpointedProjMPO(H; k), psi0; kw...)
        @test e1 ≈ e0 rtol = 1.0e-12
        @test abs(inner(psi, psi_ref)) ≈ 1 rtol = 1.0e-10
    end

    @testset "tdvp matches ProjMPO" begin
        kw = (; nsweeps = 4, maxdim = 32, cutoff = 1.0e-12, nsite = 2, outputlevel = 0)
        psi_ref = tdvp(ProjMPO(H), -0.2im, complex(psi0); kw...)
        psi = tdvp(CheckpointedProjMPO(H; k = 4), -0.2im, complex(psi0); kw...)
        @test abs(inner(psi, psi_ref)) ≈ 1 rtol = 1.0e-10
    end

    @testset "stores fewer blocks" begin
        P = CheckpointedProjMPO(H; k = 4)
        calls = Ref(0)
        P.after_drop = () -> (calls[] += 1)
        dmrg(P, psi0; nsweeps = 2, maxdim = 16, outputlevel = 0)
        @test count(!isnothing, P.LR) < n ÷ 2
        @test calls[] > 0
    end

    @test_throws ArgumentError CheckpointedProjMPO(H; k = 0)
end
