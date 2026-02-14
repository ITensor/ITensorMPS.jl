@eval module $(gensym())
using ITensorMPS: ITensorMPS, MPO, MPS, ProjMPO, ProjMPOSum, TimeDependentSum, position!,
    random_mps, siteinds, tdvp
using ITensors: ITensors, Index, QN, contract, scalartype
using LinearAlgebra: norm
using StableRNGs: StableRNG
using Test: @test, @test_skip, @testset
include(joinpath(pkgdir(ITensorMPS), "examples", "solvers", "03_models.jl"))
include(joinpath(pkgdir(ITensorMPS), "examples", "solvers", "03_updaters.jl"))
@testset "TDVP with ODE local updater" begin
    @testset "TimeDependentSum (eltype=$elt)" for elt in (
                Float32, Float64, Complex{Float32}, Complex{Float64},
            ),
            conserve_qns in [false, true]

        n = 4
        s = siteinds("S=1/2", 4; conserve_qns)
        H = MPO(elt, s, "I")
        H⃗ = (H, H)
        region = 2:3
        rng = StableRNG(1234)
        ψ = random_mps(rng, elt, s, j -> isodd(j) ? "↑" : "↓"; linkdims = 2)
        H⃗ᵣ = ProjMPO.(H⃗)
        map(Hᵣ -> position!(Hᵣ, ψ, first(region)), H⃗ᵣ)
        ∑Hᵣ = ProjMPOSum(collect(H⃗))
        position!(∑Hᵣ, ψ, first(region))
        f⃗ₜ = (t -> sin(elt(0.1) * t), t -> cos(elt(0.2) * t))
        α = elt(0.5)
        ∑Hₜ = α * TimeDependentSum(f⃗ₜ, ITensors.terms(∑Hᵣ))
        t₀ = elt(0.5)
        ∑Hₜ₀ = ∑Hₜ(t₀)
        ψᵣ = reduce(*, map(v -> ψ[v], region))
        Hψ = ∑Hₜ₀(ψᵣ)
        @test eltype(Hψ) == elt
        @test Hψ ≈ sum(i -> α * f⃗ₜ[i](t₀) * H⃗ᵣ[i](ψᵣ), eachindex(H⃗))
    end
    @testset "Time dependent Hamiltonian (eltype=$elt, conserve_qns=$conserve_qns)" for elt in
            (
                Float32, Float64, Complex{Float32}, Complex{Float64},
            ),
            conserve_qns in [false, true]

        n = 4
        J₁ = elt(1)
        J₂ = elt(0.1)
        ω₁ = real(elt)(0.1)
        ω₂ = real(elt)(0.2)
        ω⃗ = (ω₁, ω₂)
        f⃗ = map(ω -> (t -> cos(ω * t)), ω⃗)
        time_step = real(elt)(0.1)
        time_stop = real(elt)(1)
        nsite = 2
        maxdim = 100
        cutoff = √(eps(real(elt)))
        tol = √eps(real(elt))
        s = siteinds("S=1/2", n)
        ℋ₁₀ = heisenberg(n; J = J₁, J2 = zero(elt))
        ℋ₂₀ = heisenberg(n; J = zero(elt), J2 = J₂)
        ℋ⃗₀ = (ℋ₁₀, ℋ₂₀)
        H⃗₀ = map(ℋ₀ -> MPO(elt, ℋ₀, s), ℋ⃗₀)
        ψ₀ = complex.(MPS(elt, s, j -> isodd(j) ? "↑" : "↓"))
        ψₜ_ode = tdvp(
            -im * TimeDependentSum(f⃗, H⃗₀),
            time_stop,
            ψ₀;
            updater = ode_updater,
            updater_kwargs = (; reltol = tol, abstol = tol),
            time_step,
            maxdim,
            cutoff,
            nsite
        )
        ψₜ_krylov = tdvp(
            -im * TimeDependentSum(f⃗, H⃗₀),
            time_stop,
            ψ₀;
            updater = krylov_updater,
            updater_kwargs = (; tol, eager = true),
            time_step,
            maxdim,
            cutoff,
            nsite
        )
        ψₜ_full, _ = ode_updater(
            -im * TimeDependentSum(f⃗, contract.(H⃗₀)),
            contract(ψ₀);
            internal_kwargs = (; time_step = time_stop),
            reltol = tol,
            abstol = tol
        )

        @test scalartype(ψ₀) == complex(elt)
        @test scalartype(ψₜ_ode) == complex(elt)
        @test scalartype(ψₜ_krylov) == complex(elt)
        @test scalartype(ψₜ_full) == complex(elt)
        @test norm(ψ₀) ≈ 1
        @test norm(ψₜ_ode) ≈ 1
        @test norm(ψₜ_krylov) ≈ 1 rtol = √(eps(real(elt)))
        @test norm(ψₜ_full) ≈ 1

        ode_err = norm(contract(ψₜ_ode) - ψₜ_full)
        krylov_err = norm(contract(ψₜ_krylov) - ψₜ_full)

        @test krylov_err > ode_err
        @test ode_err < √(eps(real(elt))) * 10^4
        @test krylov_err < √(eps(real(elt))) * 10^5
    end
end
end
