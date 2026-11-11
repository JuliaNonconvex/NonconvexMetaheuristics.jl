# A collection of meta-heuristic algorithms in pure Julia

```@raw html
<p align="center">
  <img src="https://raw.githubusercontent.com/JuliaNonconvex/NonconvexMetaheuristics.jl/main/docs/src/assets/logo.png" alt="JuliaNonconvex logo" width="180"/>
</p>
```



## Description

[Metaheuristics.jl](https://github.com/jmejia8/Metaheuristics.jl) is an optimization library with a collection of [metaheuristic optimization algorithms](https://en.wikipedia.org/wiki/Metaheuristic) implemented. `NonconvexMetaheuristics.jl` allows the use of all the algorithms in the `Metaheuristics.jl` using the `MetaheuristicsAlg` struct.

The main advantage of metaheuristic algorithms is that they don't require the objective and constraint functions to be differentiable. One advantage of the `Metaheuristics.jl` package compared to other black-box optimization or metaheuristic algorithm packages is that a large number of the algorithms implemented in `Metaheuristics.jl` support bounds, inequality and equality constraints using constraint handling techniques for metaheuristic algorithms.

## Supported algorithms

`Nonconvex.jl` only supports the single objective optimization algorithms in `Metaheuristics.jl`. The following algorithms are supported:
- Evolutionary Centers Algorithm (`ECA`)
- Differential Evolution (`DE`)
- Particle Swarm Optimization (`PSO`)
- Artificial Bee Colony (`ABC`)
- Gravitational Search Algorithm (`CGSA`)
- Simulated Annealing (`SA`)
- Whale Optimization Algorithm (`WOA`)
- Machine-coded Compact Genetic Algorithm (`MCCGA`)
- Genetic Algorithm (`GA`)

For a summary of the strengths and weaknesses of each algorithm above, please refer to the table in the [algorithms page](https://jmejia8.github.io/Metaheuristics.jl/dev/algorithms/) in the `Metaheuristics` documentation. To define a `Metaheuristics` algorithm, you can use the `MetaheuristicsAlg` algorithm struct which wraps one of the above algorithm types, e.g. `MetaheuristicsAlg(ECA)` or `MetaheuristicsAlg(DE)`.

## Quick start

Given a model `model` and an initial solution `x0`, the following can be used to optimize the model using `Metaheuristics`.
```julia
using NonconvexMetaheuristics

alg = MetaheuristicsAlg(ECA)
options = MetaheuristicsOptions(N = 1000) # population size
result = optimize(model, alg, x0, options = options)
```
`Metaheuristics` is an optional dependency of Nonconvex so you need to load the package to be able to use it.

## Options

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

## Variable bounds

The initial population is sampled randomly in the interval of each variable, so finite variable bounds are strongly recommended. If a variable has an infinite bound, the wrapper emits a warning and samples that variable in a finite surrogate interval instead of failing.

## Initialization

Most metaheuristic algorithms are population algorithms which can accept multiple initial solutions to be part of the initial population. In `Nonconvex`, you can specify multiple initial solutions by making `x0` a vector of solutions. However, since `Nonconvex` models support arbitrary collections as decision variables, you must specify that the `x0` passed in is indeed a population of solutions rather than a single solution that's a vector of vectors for instance. To specify that `x0` is a vector of solutions, you can set the `multiple_initial_solutions` option to `true` in the `options` struct, e.g:
```julia
options = MetaheuristicsOptions(N = 1000, multiple_initial_solutions = true)
x0 = [[1.0, 1.0], [0.0, 0.0]]
```
When fewer solutions are passed in `x0` compared to the population size, random initial solutions between the lower and upper bounds are sampled to complete the initial population.

## Result

`optimize` returns a `MetaheuristicsResult` with the following fields:

- `minimizer`: the best solution found by the algorithm
- `minimum`: the best objective function value found
- `result`: the raw `Metaheuristics.jl` optimization result, e.g. to inspect
  the final population or the convergence history (see the
  [`Metaheuristics` documentation](https://jmejia8.github.io/Metaheuristics.jl/dev/))
- `alg`: the `MetaheuristicsAlg` instance used
- `options`: the `MetaheuristicsOptions` instance used
