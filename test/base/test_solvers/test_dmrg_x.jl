@eval module $(gensym())
using ITensors: ITensors
using ITensorMPS: MPO, MPS, OpSum, ProjMPO, dmrg_x, inner, siteinds
using Random: Random
using StableRNGs: StableRNG
using Test: @test, @test_throws, @testset
@testset "DMRG-X (eltype=$elt, conserve_qns=$conserve_qns)" for elt in (
            Float32, Float64, Complex{Float32}, Complex{Float64},
        ),
        conserve_qns in [false, true]

    function heisenberg(n; h = zeros(n))
        os = OpSum()
        for j in 1:(n - 1)
            os += 0.5, "S+", j, "S-", j + 1
            os += 0.5, "S-", j, "S+", j + 1
            os += "Sz", j, "Sz", j + 1
        end
        for j in 1:n
            if h[j] ≠ 0
                os -= h[j], "Sz", j
            end
        end
        return os
    end
    n = 10
    s = siteinds("S=1/2", n; conserve_qns)
    Random.seed!(12)
    W = 12
    # Random fields h ∈ [-W, W]
    rng = StableRNG(1234)
    h = W * (2 * rand(rng, real(elt), n) .- 1)
    H = MPO(elt, heisenberg(n; h), s)
    initstate = rand(rng, ["↑", "↓"], n)
    ψ = MPS(elt, s, initstate)
    @test_throws ErrorException dmrg_x(H, ψ; nsite = 2, maxdim = 20, cutoff = 1.0e-10)
    dmrg_x_kwargs = (; nsweeps = 20, normalize = true, maxdim = 20, cutoff = 1.0e-10, outputlevel = 0)
    e, ϕ = dmrg_x(H, ψ; nsite = 2, dmrg_x_kwargs...)
    @test ITensors.scalartype(ϕ) == elt
    @test inner(ϕ', H, ϕ) / inner(ϕ, ϕ) ≈ e
    @test inner(H, ψ, H, ψ) ≉ inner(ψ', H, ψ)^2 rtol = √eps(real(elt))
    @test inner(ψ', H, ψ) / inner(ψ, ψ) ≈ inner(ϕ', H, ϕ) / inner(ϕ, ϕ) rtol = 1.0e-1
    @test inner(H, ϕ, H, ϕ) ≈ inner(ϕ', H, ϕ)^2 rtol = √eps(real(elt))
    ẽ, ϕ̃ = dmrg_x(ProjMPO(H), ϕ; nsite = 1, dmrg_x_kwargs...)
    @test ITensors.scalartype(ϕ̃) == elt
    @test inner(ϕ̃', H, ϕ̃) / inner(ϕ̃, ϕ̃) ≈ ẽ
    @test inner(ψ', H, ψ) / inner(ψ, ψ) ≈ inner(ϕ̃', H, ϕ̃) / inner(ϕ̃, ϕ̃) rtol = 1.0e-1
    scale(::Type{Float32}) = 10^2
    scale(::Type{Float64}) = 10^5
    @test inner(H, ϕ̃, H, ϕ̃) ≈ inner(ϕ̃', H, ϕ̃)^2 rtol =
        √(eps(real(elt))) * scale(real(elt))
    # Sometimes broken, sometimes not
    # @test abs(loginner(ϕ̃, ϕ) / n) ≈ 0.0 atol = 1e-6
end
end
