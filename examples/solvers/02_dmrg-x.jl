using ITensorMPS: MPO, MPS, OpSum, dmrg_x, inner, siteinds
using Random: Random

function main()
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
    s = siteinds("S=1/2", n)

    Random.seed!(12)

    # MBL when W > 3.5-4
    W = 12
    # Random fields h ∈ [-W, W]
    h = W * (2 * rand(n) .- 1)
    H = MPO(heisenberg(n; h), s)

    initstate = rand(["↑", "↓"], n)
    ψ = MPS(s, initstate)
    e, ϕ = dmrg_x(H, ψ; nsweeps = 10, maxdim = 20, cutoff = 1.0e-10, normalize = true, outputlevel = 1)

    @show inner(ψ', H, ψ) / inner(ψ, ψ)
    @show inner(H, ψ, H, ψ) - inner(ψ', H, ψ)^2
    @show inner(ϕ', H, ϕ) / inner(ϕ, ϕ), e
    @show inner(H, ϕ, H, ϕ) - inner(ϕ', H, ϕ)^2
    return nothing
end

main()
