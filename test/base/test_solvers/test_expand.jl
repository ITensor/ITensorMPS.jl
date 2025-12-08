@eval module $(gensym())
using ITensors: ITensor, Index, scalartype
using ITensorMPS:
    OpSum, MPO, MPS, apply, expand, inner, linkdims, maxlinkdim, random_mps, siteinds, tdvp
using ITensorMPS.Experimental: dmrg
using LinearAlgebra: normalize
using StableRNGs: StableRNG
using Test: @test, @testset
const elts = (Float32, Float64, Complex{Float32}, Complex{Float64})
@testset "expand (eltype=$elt)" for elt in elts
    @testset "expand (alg=\"orthogonalize\", conserve_qns=$conserve_qns, eltype=$elt)" for conserve_qns in
        (
            false, true,
        )
        n = 6
        s = siteinds("S=1/2", n; conserve_qns)
        rng = StableRNG(1234)
        state = random_mps(rng, elt, s, j -> isodd(j) ? "↑" : "↓"; linkdims = 4)
        reference = random_mps(rng, elt, s, j -> isodd(j) ? "↑" : "↓"; linkdims = 2)
        state_expanded = expand(state, [reference]; alg = "orthogonalize")
        @test scalartype(state_expanded) === elt
        @test inner(state_expanded, state) ≈ inner(state, state)
        @test inner(state_expanded, reference) ≈ inner(state, reference)
    end
    @testset "expand (alg=\"global_krylov\", conserve_qns=$conserve_qns, eltype=$elt)" for conserve_qns in
        (
            false, true,
        )
        n = 10
        s = siteinds("S=1/2", n; conserve_qns)
        opsum = OpSum()
        for j in 1:(n - 1)
            opsum += 0.5, "S+", j, "S-", j + 1
            opsum += 0.5, "S-", j, "S+", j + 1
            opsum += "Sz", j, "Sz", j + 1
        end
        operator = MPO(elt, opsum, s)
        state = MPS(elt, s, j -> isodd(j) ? "↑" : "↓")
        state_expanded = expand(state, operator; alg = "global_krylov")
        @test scalartype(state_expanded) === elt
        @test maxlinkdim(state_expanded) > 1
        @test inner(state_expanded, state) ≈ inner(state, state)
    end
    @testset "Decoupled ladder (alg=\"global_krylov\", eltype=$elt)" begin
        nx = 10
        ny = 2
        n = nx * ny
        s = siteinds("S=1/2", n)
        opsum = OpSum()
        for j in 1:2:(n - 2)
            opsum += 1 / 2, "S+", j, "S-", j + 2
            opsum += 1 / 2, "S-", j, "S+", j + 2
            opsum += "Sz", j, "Sz", j + 2
        end
        for j in 2:2:(n - 2)
            opsum += 1 / 2, "S+", j, "S-", j + 2
            opsum += 1 / 2, "S-", j, "S+", j + 2
            opsum += "Sz", j, "Sz", j + 2
        end
        operator = MPO(elt, opsum, s)
        rng = StableRNG(1234)
        init = random_mps(rng, elt, s; linkdims = 30)
        reference_energy, reference_state = dmrg(
            operator,
            init;
            nsweeps = 15,
            maxdim = [10, 10, 20, 20, 40, 80, 100],
            cutoff = (√(eps(real(elt)))),
            noise = (√(eps(real(elt)))),
        )
        rng = StableRNG(1234)
        state = random_mps(rng, elt, s)
        nexpansions = 10
        tau = elt(0.5)
        for step in 1:nexpansions
            # TODO: Use `fourthroot`/`∜` in Julia 1.10 and above.
            state = expand(
                state, operator; alg = "global_krylov", krylovdim = 3, cutoff = eps(real(elt))^(1 // 4)
            )
            state = tdvp(
                operator,
                -4tau,
                state;
                nsteps = 4,
                cutoff = 1.0e-5,
                updater_kwargs = (; tol = 1.0e-3, krylovdim = 5),
            )
            state = normalize(state)
        end
        @test scalartype(state) === elt
        # TODO: Use `fourthroot`/`∜` in Julia 1.10 and above.
        @test inner(state', operator, state) ≈ reference_energy rtol = 5 * eps(real(elt))^(1 // 4)
    end
    if elt == Float64
        @testset "Regression test issue #160" begin
            # Regression test for https://github.com/ITensor/ITensorMPS.jl/issues/160
            s = siteinds("S=1/2", 4)
            i = Index(2, "Link,l=1")
            j = Index(3, "Link,l=2")
            k = Index(2, "Link,l=3")
            l = Index(4, "HLink,l=1")
            m = Index(6, "HLink,l=2")
            n = Index(4, "HLink,l=3")
            ψ_data = [
                [-0.8946112605757434 - 3.6983198682267044e-19im 0.12770990068113935 + 0.3385178997037456im;;; -0.09831866026969982 + 9.596903243670168e-18im -0.0858106795611182 - 0.22745653126540122im],
                [-0.8910784777311369 - 0.15423796632294098im 0.07170978595132114 + 0.37416085631695717im; 0.28118596675443397 + 0.048673375447632186im 0.10671963134029532 + 0.5568576166502048im;;; 0.0574469949727872 + 0.012881670393400376im 0.020337396850339782 + 0.14414434717491714im; -0.5227159163029482 - 0.11721159907883279im 0.030267642036559694 + 0.21452644770567245im;;; -0.05787580381295044 - 0.004553341135313366im -0.02645607106024196 - 0.091233732757172im; -0.4916242418311841 - 0.038679460765149255im -0.039370853144135375 - 0.13578326382500433im],
                [-0.9076467289406478 + 0.0im 0.13496759758350133 + 0.35775609151581605im; 0.26511388984285783 + 0.0im 0.2313864094611697 + 0.6133310423992547im; 0.3254105727468315 + 0.0im 0.18794511667614308 + 0.49818130260948507im;;; -0.07783131185640604 - 6.718564066141929e-7im -0.05458384118196475 - 0.14468805481202962im; 0.6555262202240988 - 2.9032095377810825e-12im -0.09357940472115472 - 0.24804890035886243im; -0.7511508912806617 - 1.8739618375970262e-6im -0.076006129623389 - 0.20148297391321887im],
                [-0.9270540711350147 + 0.0im 0.13234125887211426 + 0.3507941567246226im; 0.3749276586116248 + 0.0im 0.3272298001989081 + 0.8673810631261137im;;;],
            ]
            ψ = MPS(
                [
                    ITensor(ψ_data[1], s[1], i),
                    ITensor(ψ_data[2], i, s[2], j),
                    ITensor(ψ_data[3], j, s[3], k),
                    ITensor(ψ_data[4], k, s[4]),
                ]
            )
            H_data = [
                [0.0 + 0.0im 0.0 + 0.0im;;; 0.0 + 0.0im 1.0681415022205298 + 0.0im;;;; 4.133529502045017 + 0.0im 0.0 + 0.0im;;; 0.0 + 0.0im 0.0 + 0.0im;;;; 0.0 + 0.0im 2.922846741130685 + 0.0im;;; 2.922846741130685 + 0.0im 0.0 + 0.0im;;;; 1.0 + 0.0im 0.0 + 0.0im;;; 0.0 + 0.0im 1.0 + 0.0im],
                [1.0 + 0.0im 0.0 + 0.0im; 0.0 + 0.0im 0.0 + 0.0im; 0.0 + 0.0im 0.0 + 0.0im; 0.0 + 0.0im 0.0 + 0.0im;;; 0.0 + 0.0im 1.0 + 0.0im; 0.0 + 0.0im 0.0 + 0.0im; 0.0 + 0.0im 0.0 + 0.0im; 0.0 + 0.0im 1.0681415022205298 + 0.0im;;;; 0.0 + 0.0im 0.0 + 0.0im; 0.0 + 0.0im 0.7071067811865475 + 0.0im; 0.0 + 0.0im 0.0 + 0.0im; 0.0 + 0.0im 0.0 + 0.0im;;; 0.0 + 0.0im 0.0 + 0.0im; 0.7071067811865475 + 0.0im 0.0 + 0.0im; 0.0 + 0.0im 0.0 + 0.0im; 0.0 + 0.0im 0.0 + 0.0im;;;; 0.0 + 0.0im 0.0 + 0.0im; 0.7071067811865475 + 0.0im 0.0 + 0.0im; 0.0 + 0.0im 0.0 + 0.0im; 0.0 + 0.0im 0.0 + 0.0im;;; 0.0 + 0.0im 0.0 + 0.0im; 0.0 + 0.0im 0.7071067811865475 + 0.0im; 0.0 + 0.0im 0.0 + 0.0im; 0.0 + 0.0im 0.0 + 0.0im;;;; 0.0 + 0.0im 0.0 + 0.0im; 0.0 + 0.0im 0.0 + 0.0im; 1.0 + 0.0im 0.0 + 0.0im; 0.0 + 0.0im 0.0 + 0.0im;;; 0.0 + 0.0im 0.0 + 0.0im; 0.0 + 0.0im 0.0 + 0.0im; 0.0 + 0.0im 0.0 + 0.0im; 0.0 + 0.0im 0.0 + 0.0im;;;; 0.0 + 0.0im 0.0 + 0.0im; 0.0 + 0.0im 0.0 + 0.0im; 0.0 + 0.0im 0.0 + 0.0im; 2.922846741130685 + 0.0im 0.0 + 0.0im;;; 0.0 + 0.0im 0.0 + 0.0im; 0.0 + 0.0im 0.0 + 0.0im; 0.0 + 0.0im 0.0 + 0.0im; 0.0 + 0.0im 0.0 + 0.0im;;;; 0.0 + 0.0im 0.0 + 0.0im; 0.0 + 0.0im 0.0 + 0.0im; 0.0 + 0.0im 0.0 + 0.0im; 1.0 + 0.0im 0.0 + 0.0im;;; 0.0 + 0.0im 0.0 + 0.0im; 0.0 + 0.0im 0.0 + 0.0im; 0.0 + 0.0im 0.0 + 0.0im; 0.0 + 0.0im 1.0 + 0.0im],
                [1.0 + 0.0im 0.0 + 0.0im; 0.0 + 0.0im 0.0 + 0.0im; 0.0 + 0.0im 0.0 + 0.0im; 1.0 + 0.0im 0.0 + 0.0im; 0.0 + 0.0im 0.0 + 0.0im; 0.0 + 0.0im 0.0 + 0.0im;;; 0.0 + 0.0im 1.0 + 0.0im; 0.0 + 0.0im 0.0 + 0.0im; 0.0 + 0.0im 0.0 + 0.0im; 0.0 + 0.0im 0.0 + 0.0im; 0.0 + 0.0im 0.0 + 0.0im; 0.0 + 0.0im 1.0681415022205298 + 0.0im;;;; 0.0 + 0.0im 0.0 + 0.0im; -1.0 + 0.0im 0.0 + 0.0im; 0.0 + 0.0im -1.0 + 0.0im; 0.0 + 0.0im 0.0 + 0.0im; 0.0 + 0.0im 0.0 + 0.0im; 0.0 + 0.0im 0.0 + 0.0im;;; 0.0 + 0.0im 0.0 + 0.0im; 0.0 + 0.0im -1.0 + 0.0im; -1.0 + 0.0im 0.0 + 0.0im; 0.0 + 0.0im 0.0 + 0.0im; 0.0 + 0.0im 0.0 + 0.0im; 0.0 + 0.0im 0.0 + 0.0im;;;; 0.0 + 0.0im 0.0 + 0.0im; 0.0 + 0.0im 0.0 + 0.0im; 0.0 + 0.0im 0.0 + 0.0im; 0.0 + 0.0im 0.0 + 0.0im; -1.0 + 0.0im 0.0 + 0.0im; 0.0 + 0.0im 0.0 + 0.0im;;; 0.0 + 0.0im 0.0 + 0.0im; 0.0 + 0.0im 0.0 + 0.0im; 0.0 + 0.0im 0.0 + 0.0im; 0.0 + 0.0im 0.0 + 0.0im; 0.0 + 0.0im 0.0 + 0.0im; 0.0 + 0.0im 0.0 + 0.0im;;;; 0.0 + 0.0im 0.0 + 0.0im; 0.0 + 0.0im 0.0 + 0.0im; 0.0 + 0.0im 0.0 + 0.0im; 0.0 + 0.0im 0.0 + 0.0im; 0.0 + 0.0im 0.0 + 0.0im; 1.0 + 0.0im 0.0 + 0.0im;;; 0.0 + 0.0im 0.0 + 0.0im; 0.0 + 0.0im 0.0 + 0.0im; 0.0 + 0.0im 0.0 + 0.0im; 0.0 + 0.0im 0.0 + 0.0im; 0.0 + 0.0im 0.0 + 0.0im; 0.0 + 0.0im 1.0 + 0.0im],
                [1.0 + 0.0im 0.0 + 0.0im; -1.0 + 0.0im 0.0 + 0.0im; 0.0 + 0.0im -1.0 + 0.0im; 0.0 + 0.0im 0.0 + 0.0im;;; 0.0 + 0.0im 1.0 + 0.0im; 0.0 + 0.0im 0.0 + 0.0im; -1.0 + 0.0im 0.0 + 0.0im; 0.0 + 0.0im 1.0681415022205298 + 0.0im;;;;],
            ]
            H = MPO(
                [
                    ITensor(H_data[1], s[1]', s[1], l),
                    ITensor(H_data[2], l, s[2]', s[2], m),
                    ITensor(H_data[3], m, s[3]', s[3], n),
                    ITensor(H_data[4], n, s[4]', s[4]),
                ]
            )

            Hψ = apply(H, ψ)
            ψ_expanded = expand(ψ, [Hψ]; alg = "orthogonalize")
            @test inner(ψ_expanded, ψ) ≈ 1.0
            ψ_expanded2 = expand(ψ, [normalize(Hψ)]; alg = "orthogonalize")
            @test inner(ψ_expanded2, ψ) ≈ 1
            ψ_expanded3 = expand(ψ, H; alg = "global_krylov", krylovdim = 1)
            @test inner(ψ_expanded3, ψ) ≈ 1
            ψ_expanded4 = expand(ψ, H; alg = "global_krylov", krylovdim = 1, cutoff = 1.0e-6)
            @test inner(ψ_expanded4, ψ) ≈ 1
        end
    end
end
end
