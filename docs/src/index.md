# NonconvexMetaheuristics

![JuliaNonconvex logo](https://raw.githubusercontent.com/JuliaNonconvex/NonconvexMetaheuristics.jl/main/docs/src/assets/logo.png)

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

See [Quick start](quickstart.md) for the first steps, [Options](options.md)
for the algorithm options, [Variable bounds](bounds.md) and
[Initialization](init.md) for the model requirements, [Result](result.md) for
the fields of the returned result and [Docstrings](docstrings.md) for the
full API reference.
