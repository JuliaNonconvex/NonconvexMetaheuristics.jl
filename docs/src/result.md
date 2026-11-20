# Result

`optimize` returns a `MetaheuristicsResult` with the following fields:

- `minimizer`: the best solution found by the algorithm
- `minimum`: the best objective function value found
- `result`: the raw `Metaheuristics.jl` optimization result, e.g. to inspect
  the final population or the convergence history (see the
  [`Metaheuristics` documentation](https://jmejia8.github.io/Metaheuristics.jl/dev/))
- `alg`: the `MetaheuristicsAlg` instance used
- `options`: the `MetaheuristicsOptions` instance used
