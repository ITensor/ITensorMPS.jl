"""
    CheckpointedProjMPO(H::MPO; k = round(Int, sqrt(length(H))), after_drop = () -> nothing)

A drop-in replacement for [`ProjMPO`](@ref) that keeps only every `k`-th environment
block (plus the blocks next to the current position) and recomputes a dropped block
from the nearest kept one when a sweep reaches it.

The result is exact: the rebuilt blocks are the same contractions `ProjMPO` performs.
Memory is about `N/k + 2k` environment blocks instead of `N` (minimal for `k ≈ √N`),
at the cost of about one extra environment update per site and sweep.

`after_drop` is called whenever blocks were dropped, for example
`() -> (GC.gc(false); CUDA.reclaim())` to return GPU memory right away.

```julia
dmrg(CheckpointedProjMPO(H), psi0; nsweeps, maxdim)
tdvp(CheckpointedProjMPO(H), t, psi0; nsweeps)
```
"""
mutable struct CheckpointedProjMPO <: AbstractProjMPO
    lpos::Int
    rpos::Int
    nsite::Int
    H::MPO
    LR::Vector{Union{ITensor, Nothing}}
    k::Int
    after_drop::Function
end

function CheckpointedProjMPO(
        H::MPO; k::Int = max(1, round(Int, sqrt(length(H)))),
        after_drop::Function = () -> nothing
    )
    k ≥ 1 || throw(ArgumentError("k must be ≥ 1, got $k"))
    return CheckpointedProjMPO(
        0, length(H) + 1, 2, H, Vector{Union{ITensor, Nothing}}(nothing, length(H)), k,
        after_drop
    )
end

function copy(P::CheckpointedProjMPO)
    return CheckpointedProjMPO(
        P.lpos, P.rpos, P.nsite, copy(P.H), copy(P.LR), P.k, P.after_drop
    )
end

function set_nsite!(P::CheckpointedProjMPO, nsite)
    P.nsite = nsite
    return P
end

_isstored(P::CheckpointedProjMPO, j::Int) = 1 ≤ j ≤ length(P.LR) && !isnothing(P.LR[j])

lproj(P::CheckpointedProjMPO) = P.lpos ≤ 0 ? OneITensor() : P.LR[P.lpos]
rproj(P::CheckpointedProjMPO) = P.rpos ≥ length(P.H) + 1 ? OneITensor() : P.LR[P.rpos]

# Rebuild the left block `l` (sites 1..l) from the nearest stored block below it.
function _rebuildL!(P::CheckpointedProjMPO, psi::MPS, l::Int)
    c = l - 1
    while c ≥ 1 && !_isstored(P, c)
        c -= 1
    end
    L = c ≥ 1 ? P.LR[c] : OneITensor()
    for j in (c + 1):l
        L = L * psi[j] * P.H[j] * dag(prime(psi[j]))
        P.LR[j] = L
    end
    return P
end

# Rebuild the right block `r` (sites r..N) from the nearest stored block above it.
function _rebuildR!(P::CheckpointedProjMPO, psi::MPS, r::Int)
    N = length(P.H)
    c = r + 1
    while c ≤ N && !_isstored(P, c)
        c += 1
    end
    R = c ≤ N ? P.LR[c] : OneITensor()
    for j in (c - 1):-1:r
        R = R * psi[j] * P.H[j] * dag(prime(psi[j]))
        P.LR[j] = R
    end
    return P
end

# Keep the checkpoints (multiples of `k`) and the open segments next to the current
# position, which the sweep uses in its next steps; drop everything else.
function _prune!(P::CheckpointedProjMPO)
    k = P.k
    lo = P.lpos - mod(P.lpos, k)
    hi = P.rpos + mod(-P.rpos, k)
    dropped = false
    for j in eachindex(P.LR)
        isnothing(P.LR[j]) && continue
        keep = if j ≤ P.lpos
            mod(j, k) == 0 || j > lo
        elseif j ≥ P.rpos
            mod(j, k) == 0 || j < hi
        else
            false
        end
        if !keep
            P.LR[j] = nothing
            dropped = true
        end
    end
    dropped && P.after_drop()
    return P
end

function position!(P::CheckpointedProjMPO, psi::MPS, pos::Int)
    l, r = pos - 1, pos + nsite(P)
    # Moving back onto a dropped block: rebuild it from the nearest stored one.
    (1 ≤ l < P.lpos && !_isstored(P, l)) && _rebuildL!(P, psi, l)
    (P.rpos < r ≤ length(P.H) && !_isstored(P, r)) && _rebuildR!(P, psi, r)
    makeL!(P, psi, l)
    makeR!(P, psi, r)
    _prune!(P)
    return P
end
