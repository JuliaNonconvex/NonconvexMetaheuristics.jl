# Quick start

Given a model `model` and an initial solution `x0`, the following can be used to optimize the model using `Metaheuristics`.
```julia
using NonconvexMetaheuristics

alg = MetaheuristicsAlg(ECA)
options = MetaheuristicsOptions(N = 1000) # population size
result = optimize(model, alg, x0, options = options)
```
`Metaheuristics` is an optional dependency of Nonconvex so you need to load the package to be able to use it.
