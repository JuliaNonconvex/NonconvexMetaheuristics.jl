# Initialization

Most metaheuristic algorithms are population algorithms which can accept multiple initial solutions to be part of the initial population. In `Nonconvex`, you can specify multiple initial solutions by making `x0` a vector of solutions. However, since `Nonconvex` models support arbitrary collections as decision variables, you must specify that the `x0` passed in is indeed a population of solutions rather than a single solution that's a vector of vectors for instance. To specify that `x0` is a vector of solutions, you can set the `multiple_initial_solutions` option to `true` in the `options` struct, e.g:
```julia
options = MetaheuristicsOptions(N = 1000, multiple_initial_solutions = true)
x0 = [[1.0, 1.0], [0.0, 0.0]]
```
When fewer solutions are passed in `x0` compared to the population size, random initial solutions between the lower and upper bounds are sampled to complete the initial population.
