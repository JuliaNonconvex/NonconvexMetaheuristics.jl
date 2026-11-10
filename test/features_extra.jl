# Additional feature coverage for the exported MetaheuristicsAlg /
# MetaheuristicsOptions API and the reexported algorithm types. These
# @testitem blocks exercise the MetaheuristicsOptions defaults and merging,
# the MetaheuristicsAlg wrapper fields, the DictModel path of `optimize`,
# the Workspace API, the MetaheuristicsResult fields and the rng option.

@testitem "features: MetaheuristicsOptions defaults and merging" begin
    using NonconvexMetaheuristics, Test, Random

    o = MetaheuristicsOptions()
    @test o isa NonconvexMetaheuristics.MetaheuristicsOptions
    @test o.nt.multiple_initial_solutions === false
    @test o.nt.N == 100
    @test o.nt.f_calls_limit == 100 * 100
    @test o.nt.rng === Random.GLOBAL_RNG

    # f_calls_limit defaults to N * 100 and can be overridden.
    o2 = MetaheuristicsOptions(N=10)
    @test o2.nt.N == 10
    @test o2.nt.f_calls_limit == 1000

    o3 = MetaheuristicsOptions(N=10, f_calls_limit=500, multiple_initial_solutions=true)
    @test o3.nt.N == 10
    @test o3.nt.f_calls_limit == 500
    @test o3.nt.multiple_initial_solutions === true

    # Extra kwargs (Metaheuristics.jl algorithm options) are merged.
    o4 = MetaheuristicsOptions(f_tol=1e-4, x_tol=1e-4, iterations=100)
    @test o4.nt.f_tol == 1e-4
    @test o4.nt.x_tol == 1e-4
    @test o4.nt.iterations == 100

    # A custom rng is stored as-is.
    rng = MersenneTwister(42)
    o5 = MetaheuristicsOptions(rng=rng)
    @test o5.nt.rng === rng
end

@testitem "features: MetaheuristicsAlg wraps every algorithm type" begin
    using NonconvexMetaheuristics, Test

    @test Metaheuristics isa Module
    for algT in (Metaheuristics, ECA, DE, PSO, ABC, CGSA, SA, WOA, MCCGA, GA)
        alg = MetaheuristicsAlg(algT)
        @test alg isa NonconvexMetaheuristics.MetaheuristicsAlg
        @test alg.algT === algT
    end
end

@testitem "features: DictModel optimize path" begin
    using NonconvexMetaheuristics, LinearAlgebra, Test, Random

    Random.seed!(1)
    dm = DictModel(x -> x[:a]^2 + x[:b]^2)
    addvar!(dm, :a, -5.0, 5.0)
    addvar!(dm, :b, -5.0, 5.0)
    options = MetaheuristicsOptions(N=50, f_calls_limit=2000)
    r = NonconvexCore.optimize(
        dm, MetaheuristicsAlg(DE), Dict(:a => 1.0, :b => -1.0), options=options
    )
    @test r.minimizer isa AbstractDict
    @test sort(collect(keys(r.minimizer))) == [:a, :b]
    @test isfinite(r.minimum)
    @test r.minimum < 0.8
    @test norm([r.minimizer[:a], r.minimizer[:b]]) < 1.0
end

@testitem "features: Workspace API and MetaheuristicsResult fields" begin
    using NonconvexMetaheuristics, NonconvexCore, LinearAlgebra, Test, Random

    Random.seed!(1)
    m = Model(x -> sum(x .^ 2))
    addvar!(m, [-5.0, -5.0], [5.0, 5.0])
    alg = MetaheuristicsAlg(DE)
    options = MetaheuristicsOptions(N=50, f_calls_limit=2000)

    # The Workspace constructor forwards kwargs positionally; this exercises
    # the positional-NamedTuple MetaheuristicsWorkspace method. As in the
    # `optimize` path, the initial point is passed as a population (a vector
    # of solutions).
    vm, vx0, _ = NonconvexCore.tovecmodel(m, [1.0, -1.0])
    ws = NonconvexMetaheuristics.Workspace(vm, alg, [vx0]; options=options)
    @test ws isa NonconvexMetaheuristics.MetaheuristicsWorkspace
    @test ws.alg === alg
    @test ws.options === options
    @test ws.x0 == [vx0]

    r = NonconvexMetaheuristics.optimize!(ws)
    @test r isa NonconvexMetaheuristics.MetaheuristicsResult
    @test isfinite(r.minimum)
    @test r.minimum < 0.8
    @test norm(r.minimizer) < 1.0
    @test r.alg === alg
    @test r.options === options
    @test Metaheuristics.minimizer(r.result) ≈ r.minimizer
    @test Metaheuristics.minimum(r.result) ≈ r.minimum
end

@testitem "features: rng option and integer-variable warning path" begin
    using NonconvexMetaheuristics, LinearAlgebra, Test, Random

    # A custom rng drives the population sampling.
    m = Model(x -> sum(x .^ 2))
    addvar!(m, [-5.0, -5.0], [5.0, 5.0])
    r = NonconvexCore.optimize(
        m,
        MetaheuristicsAlg(DE),
        [1.0, -1.0];
        options=MetaheuristicsOptions(N=30, f_calls_limit=1000, rng=MersenneTwister(7)),
    )
    @test isfinite(r.minimum)
    @test r.minimum < 0.8

    # Integer variables are treated as continuous with a warning.
    m2 = Model(x -> sum(x .^ 2))
    addvar!(m2, [-5.0, -5.0], [5.0, 5.0]; integer=[true, false])
    r2 = NonconvexCore.optimize(
        m2,
        MetaheuristicsAlg(DE),
        [1.0, -1.0];
        options=MetaheuristicsOptions(N=30, f_calls_limit=1000),
    )
    @test isfinite(r2.minimum)
end
