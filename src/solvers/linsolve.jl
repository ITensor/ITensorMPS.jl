using KrylovKit: KrylovKit, linsolve

function linsolve_updater(problem, init; internal_kwargs, coefficients, kwargs...)
  x, info = linsolve(
    operator(problem),
    constant_term(problem),
    init,
    coefficients[1],
    coefficients[2];
    kwargs...,
  )
  return x, (; info)
end

"""
Compute a solution x to the linear system:

(a₀ + a₁ * A)*x = b

using starting guess x₀. Leaving a₀, a₁
set to their default values solves the 
system A*x = b.

To adjust the balance between accuracy of solution
and speed of the algorithm, it is recommed to first try
adjusting the updater keyword arguments as descibed below.

Keyword arguments:
  - `nsweeps`, `cutoff`, `maxdim`, etc. (like for other MPO/MPS updaters).
  - `updater_kwargs=(;)` - a `NamedTuple` containing keyword arguments that will get forwarded to the local updater,
    in this case `KrylovKit.linsolve` which is a GMRES linear updater. For example:
    ```juli
    linsolve(A, b, x; maxdim=100, cutoff=1e-8, nsweeps=10, updater_kwargs=(; ishermitian=true, tol=1e-6, maxiter=20, krylovdim=30))
    ```
    See `KrylovKit.jl` documentation for more details on available keyword arguments.
"""
function KrylovKit.linsolve(
  operator,
  constant_term::MPS,
  init::MPS,
  coefficient1::Number=false,
  coefficient2::Number=true;
  updater=linsolve_updater,
  updater_kwargs=(;),
  kwargs...,
)
  reduced_problem = ReducedLinearProblem(operator, constant_term)
  updater_kwargs = (; coefficients=(coefficient1, coefficient2), updater_kwargs...)
  return alternating_update(reduced_problem, init; updater, updater_kwargs, kwargs...)
end
