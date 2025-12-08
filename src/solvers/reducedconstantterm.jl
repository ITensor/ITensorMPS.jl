using ITensors: ITensors, ITensor, dag, dim, prime

"""
Holds the following data where basis
is the MPS being optimized and state is the 
MPS held constant by the ProjMPS.
```
     o--o--o--o--o--o--o--o--o--o--o <state|
     |  |  |  |  |  |  |  |  |  |  |
     *--*--*-      -*--*--*--*--*--* |reduced_basis>
```
"""
mutable struct ReducedConstantTerm <: AbstractProjMPO
    lpos::Int
    rpos::Int
    nsite::Int
    state::MPS
    environments::Vector{ITensor}
end

function ReducedConstantTerm(state::MPS)
    lpos = 0
    rpos = length(state) + 1
    nsite = 2
    environments = Vector{ITensor}(undef, length(state))
    return ReducedConstantTerm(lpos, rpos, nsite, state, environments)
end

function Base.getproperty(reduced_state::ReducedConstantTerm, sym::Symbol)
    # This is for compatibility with `AbstractProjMPO`.
    # TODO: Don't use `reduced_state.H`, `reduced_state.LR`, etc.
    # in `AbstractProjMPO`.
    if sym == :LR
        return getfield(reduced_state, :environments)
    end
    return getfield(reduced_state, sym)
end

Base.length(reduced_state::ReducedConstantTerm) = length(reduced_state.state)

function Base.copy(reduced_state::ReducedConstantTerm)
    return ReducedConstantTerm(
        reduced_state.lpos,
        reduced_state.rpos,
        reduced_state.nsite,
        copy(reduced_state.state),
        copy(reduced_state.environments),
    )
end

function ITensorMPS.set_nsite!(reduced_state::ReducedConstantTerm, nsite)
    reduced_state.nsite = nsite
    return reduced_state
end

function ITensorMPS.makeL!(reduced_state::ReducedConstantTerm, basis::MPS, position::Int)
    # Save the last `L` that is made to help with caching
    # for DiskProjMPO
    ll = reduced_state.lpos
    if ll ≥ position
        # Special case when nothing has to be done.
        # Still need to change the position if lproj is
        # being moved backward.
        reduced_state.lpos = position
        return nothing
    end
    # Make sure ll is at least 0 for the generic logic below
    ll = max(ll, 0)
    L = lproj(reduced_state)
    while ll < position
        L = L * basis[ll + 1] * dag(prime(reduced_state.state[ll + 1], "Link"))
        reduced_state.environments[ll + 1] = L
        ll += 1
    end
    # Needed when moving lproj backward.
    reduced_state.lpos = position
    return reduced_state
end

function ITensorMPS.makeR!(reduced_state::ReducedConstantTerm, basis::MPS, position::Int)
    # Save the last `R` that is made to help with caching
    # for DiskProjMPO
    environment_position = reduced_state.rpos
    if environment_position ≤ position
        # Special case when nothing has to be done.
        # Still need to change the position if rproj is
        # being moved backward.
        reduced_state.rpos = position
        return nothing
    end
    N = length(reduced_state.state)
    # Make sure environment_position is no bigger than `N + 1` for the generic logic below
    environment_position = min(environment_position, N + 1)
    right_environment = rproj(reduced_state)
    while environment_position > position
        right_environment =
            right_environment *
            basis[environment_position - 1] *
            dag(prime(reduced_state.state[environment_position - 1], "Link"))
        reduced_state.environments[environment_position - 1] = right_environment
        environment_position -= 1
    end
    reduced_state.rpos = position
    return reduced_state
end

function ITensors.contract(reduced_state::ReducedConstantTerm, v::ITensor)
    reduced_state_tensors = Union{ITensor, OneITensor}[lproj(reduced_state)]
    append!(
        reduced_state_tensors,
        [prime(t, "Link") for t in reduced_state.state[site_range(reduced_state)]],
    )
    push!(reduced_state_tensors, rproj(reduced_state))

    # Reverse the contraction order of the map if
    # the first tensor is a scalar (for example we
    # are at the left edge of the system)
    if dim(first(reduced_state_tensors)) == 1
        reverse!(reduced_state_tensors)
    end

    # Apply the map
    inner = v
    for t in reduced_state_tensors
        inner *= t
    end
    return inner
end

# Contract the reduced constant term down to a since ITensor.
function ITensors.contract(reduced_state::ReducedConstantTerm)
    reduced_state_tensors = Union{ITensor, OneITensor}[lproj(reduced_state)]
    append!(
        reduced_state_tensors,
        [dag(prime(t, "Link")) for t in reduced_state.state[site_range(reduced_state)]],
    )
    push!(reduced_state_tensors, rproj(reduced_state))

    # Reverse the contraction order of the map if
    # the first tensor is a scalar (for example we
    # are at the left edge of the system)
    if dim(first(reduced_state_tensors)) == 1
        reverse!(reduced_state_tensors)
    end

    # Apply the map
    contracted_reduced_state = ITensor(true)
    for t in reduced_state_tensors
        contracted_reduced_state *= t
    end
    return contracted_reduced_state
end
