@testitem "Population truncation warning" begin
    using NonconvexMetaheuristics, NonconvexCore, LinearAlgebra, Test, Random

    Random.seed!(1)
    m = NonconvexCore.Model(x -> sum(x .^ 2))
    addvar!(m, [-5.0, -5.0], [5.0, 5.0])
    # More initial solutions than the population size N triggers the warning
    # and truncates the initial population.
    x0 = [rand(2) .* 4 .- 2 for _ in 1:40]
    options = MetaheuristicsOptions(N=10, f_calls_limit=1000, multiple_initial_solutions=true)
    r = NonconvexMetaheuristics.optimize(m, MetaheuristicsAlg(GA), x0; options=options)
    @test isfinite(r.minimum)
end
