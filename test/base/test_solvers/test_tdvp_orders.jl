@eval module $(gensym())
using ITensorMPS: ITensorMPS, MPO, MPS, OpSum, maxlinkdim, siteinds, tdvp
using ITensors: ITensor, prime
using LinearAlgebra: kron, norm
using Test: @test, @testset

# Site 1 varies fastest in Array(ITensor, sites...), so it is the rightmost
# factor in a Kronecker product.
function dense_site_operator(operator, site, nsites)
    identity = [1.0 0.0; 0.0 1.0]
    return reduce(kron, (j == site ? operator : identity for j in nsites:-1:1))
end

function dense_hamiltonian(nsites)
    sx = [0.0 0.5; 0.5 0.0]
    sy = [0.0 -0.5im; 0.5im 0.0]
    sz = [0.5 0.0; 0.0 -0.5]
    xs = [dense_site_operator(sx, j, nsites) for j in 1:nsites]
    ys = [dense_site_operator(sy, j, nsites) for j in 1:nsites]
    zs = [dense_site_operator(sz, j, nsites) for j in 1:nsites]
    h = zeros(ComplexF64, 2^nsites, 2^nsites)
    for j in 1:(nsites - 1)
        h += xs[j] * xs[j + 1] + ys[j] * ys[j + 1] + zs[j] * zs[j + 1]
    end
    for j in 1:nsites
        h += 0.13 * j * zs[j]
    end
    h += 0.23 * zs[1] * zs[4]
    return h
end

function mpo_hamiltonian(sites)
    terms = OpSum()
    for j in 1:(length(sites) - 1)
        terms += 0.5, "S+", j, "S-", j + 1
        terms += 0.5, "S-", j, "S+", j + 1
        terms += 1.0, "Sz", j, "Sz", j + 1
    end
    for j in eachindex(sites)
        terms += 0.13 * j, "Sz", j
    end
    terms += 0.23, "Sz", 1, "Sz", 4
    return MPO(ComplexF64, terms, sites)
end

dense_state(state, sites) = vec(Array(prod(state), sites...))

function evolve(operator, init, t; nsteps, order, direction, nsite, tol = 1.0e-13)
    updater_kwargs = (; tol, krylovdim = 32, maxiter = 100, ishermitian = true)
    kwargs = (;
        nsite, updater_kwargs, reverse_step = true, normalize = false,
        cutoff = 0.0, maxdim = 2^(length(init) ÷ 2), outputlevel = 0,
    )
    if direction == Base.Forward
        return tdvp(operator, t, init; nsteps, order, kwargs...)
    end
    # The public tdvp wrapper always starts forward. Exercise the reverse
    # ordering through the same sweep implementation that tdvp calls.
    state = copy(init)
    reduced_operator = ITensorMPS.reduced_operator(operator)
    time_step = t / nsteps
    current_time = zero(t)
    for sweep in 1:nsteps
        state, reduced_operator, _ = ITensorMPS.sweep_update(
            ITensorMPS.TDVPOrder(order, direction), reduced_operator, state;
            updater = ITensorMPS.exponentiate_updater,
            current_time, time_step, sweep, kwargs...
        )
        current_time += time_step
    end
    return state
end

@testset "TDVP orders against an independent dense exponential" begin
    nsites = 6
    h = dense_hamiltonian(nsites)
    # All amplitudes lie in the total-Sz=0 sector, with generic coefficients
    # to give full Schmidt ranks. Both QN representations use this same vector.
    initial = ComplexF64[
        if count_ones(k) == nsites ÷ 2
                sin((k + 1)^2) + im * cos(sqrt(2.0) * (k + 1))
        else
                0.0
        end for k in 0:(2^nsites - 1)
    ]
    initial /= norm(initial)
    t = -0.4im
    exact = exp(t * h) * initial
    @testset "conserve_qns=$conserve_qns" for conserve_qns in (false, true)
        sites = siteinds("S=1/2", nsites; conserve_qns)
        operator = mpo_hamiltonian(sites)
        # Establish the basis convention and agreement of the independent
        # matrix and MPO before testing time evolution.
        h_mpo = reshape(Array(prod(operator), prime.(sites)..., sites...), size(h))
        @test norm(h_mpo - h) < 1.0e-12
        init = MPS(ITensor(initial, sites...), sites; cutoff = 0.0, maxdim = 8)
        @test norm(dense_state(init, sites) - initial) < 1.0e-12
        @test maxlinkdim(init) == 8
        @testset "order=$order, direction=$direction, nsite=$nsite" for order in (1, 2, 4),
                direction in (Base.Forward, Base.Reverse),
                nsite in (1, 2)

            state = evolve(operator, init, t; nsteps = 2, order, direction, nsite)
            evolved = dense_state(state, sites)
            # Full-state errors include the global phase; norm conservation
            # or fidelity alone would not detect an incorrect evolution time.
            @test norm(evolved - exact) < 1.0e-10
            @test abs(norm(evolved) - 1) < 1.0e-11
            if order == 4
                tighter = evolve(
                    operator, init, t; nsteps = 2, order, direction, nsite,
                    tol = 1.0e-14
                )
                @test norm(dense_state(tighter, sites) - evolved) < 1.0e-11
            end
        end
    end
end

@testset "TDVP timestep convergence on a fixed-rank manifold" begin
    nsites = 6
    sites = siteinds("S=1/2", nsites)
    operator = mpo_hamiltonian(sites)
    # Generic, non-collinear spinors give nontrivial interacting product-state
    # TDVP dynamics. One-site updates keep every bond dimension equal to one.
    # A Sz-conserving Up/Dn product state would instead be stationary here.
    spinors = [
        ComplexF64[cos((0.4 + 0.2 * j) / 2), cis(0.3 * j) * sin((0.4 + 0.2 * j) / 2)]
            for j in 1:nsites
    ]
    initial = reduce(kron, reverse(spinors))
    init = MPS(ITensor(initial, sites...), sites; cutoff = 0.0, maxdim = 1)
    @test norm(dense_state(init, sites) - initial) < 1.0e-12
    @test maxlinkdim(init) == 1
    nsteps_values = (4, 8, 16, 32)
    @testset "order=$order, direction=$direction" for order in (1, 2, 4),
            direction in (Base.Forward, Base.Reverse)

        states = map(nsteps_values) do nsteps
            state = evolve(operator, init, -1.0im; nsteps, order, direction, nsite = 1)
            @test maxlinkdim(state) == 1
            @test abs(norm(state) - 1) < 1.0e-11
            return dense_state(state, sites)
        end
        # Compare successive refinements of the same projected TDVP flow.
        # Its error against exp(t*H) contains a variational error that does
        # not vanish with the timestep, so it is not a convergence reference.
        differences = [norm(states[j] - states[j + 1]) for j in 1:3]
        @test minimum(differences) > 1.0e-11
        @test differences[1] > differences[2] > differences[3]
        ratios = differences[1:2] ./ differences[2:3]
        @testset "refinement=$j" for j in eachindex(ratios)
            @test ratios[j] ≈ 2.0^order rtol = 0.25
        end
    end
end
end
