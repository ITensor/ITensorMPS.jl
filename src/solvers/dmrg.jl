using ITensors: ITensors
using KrylovKit: eigsolve

function eigsolve_updater(
        operator,
        init;
        internal_kwargs,
        which_eigval = :SR,
        ishermitian = true,
        tol = 10^2 * eps(real(ITensors.scalartype(init))),
        krylovdim = 3,
        maxiter = 1,
        verbosity = 0,
        eager = false,
    )
    howmany = 1
    eigvals, eigvecs, info = eigsolve(
        operator, init, howmany, which_eigval; ishermitian, tol, krylovdim, maxiter, verbosity
    )
    return eigvecs[1], (; info, eigval = eigvals[1])
end

# A version of `dmrg` based on `alternating_update` and `eigsolve_updater`
# is defined in `src/lib/Experimental` as `ITensorMPS.Experimental.dmrg` to not conflict
# with `ITensorMPS.dmrg`.
