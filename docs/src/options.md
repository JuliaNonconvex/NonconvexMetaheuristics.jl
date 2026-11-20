# Options

The options keyword argument to the `optimize` function shown above must be an instance of the `MetaheuristicsOptions` struct when the algorihm is a `MetaheuristicsAlg`. To specify options use keyword arguments in the constructor of `MetaheuristicsOptions`, e.g:
```julia
options = MetaheuristicsOptions(N = 1000)
```
The following options have defaults when not specified:

- `N`: the population size (default `100`)
- `f_calls_limit`: the maximum number of objective function evaluations
  (default `100 * N`)
- `rng`: the random number generator used to sample the initial population
  and by the algorithm itself (default `Random.GLOBAL_RNG`)
- `multiple_initial_solutions`: when `true`, `x0` is interpreted as a
  population of solutions rather than a single solution (default `false`)

All the other options that can be set for each algorithm (such as
`f_tol`, `x_tol` and `iterations`, which control the stopping criteria of
the `Metaheuristics.jl` solvers) can be found in the [algorithms section](https://jmejia8.github.io/Metaheuristics.jl/dev/algorithms/) of the documentation of `Metaheuristics.jl`. Note that one notable difference between using `Metaheuristics` directly and using it through `Nonconvex` is that in `Nonconvex`, all the options must be passed in through the `options` struct and only the algorithm type is part of the `alg` struct.
