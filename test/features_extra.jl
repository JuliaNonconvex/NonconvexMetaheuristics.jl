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

@testitem "features: rng option and integer-variable validation" begin
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

    # Integer variables are not supported (integer=false in
    # solver_capabilities); the Model path validates the model against the
    # declared capabilities and errors instead of silently treating the
    # variables as continuous.
    m2 = Model(x -> sum(x .^ 2))
    addvar!(m2, [-5.0, -5.0], [5.0, 5.0]; integer=[true, false])
    @test_throws ArgumentError NonconvexCore.optimize(
        m2,
        MetaheuristicsAlg(DE),
        [1.0, -1.0];
        options=MetaheuristicsOptions(N=30, f_calls_limit=1000),
    )
end

@testitem "features: sd constraints are rejected on the Model and DictModel paths" begin
    using NonconvexMetaheuristics, LinearAlgebra, Test

    # Regression test: the custom optimize(::Model/::DictModel, ::MetaheuristicsAlg, x0)
    # paths used to bypass checksubproblem! and silently ignore unsupported
    # semidefinite constraints (sd=false in solver_capabilities).
    m = Model(x -> sum(x .^ 2))
    addvar!(m, [-2.0, -2.0], [2.0, 2.0])
    add_sd_constraint!(m, x -> [x[1]^2 + 1.0 0.0; 0.0 x[2]^2 + 1.0], 2)
    msg = try
        NonconvexCore.optimize(
            m,
            MetaheuristicsAlg(DE),
            [1.0, -1.0];
            options=MetaheuristicsOptions(N=10, f_calls_limit=100),
        )
        ""
    catch e
        sprint(showerror, e)
    end
    @test occursin("semidefinite", msg)
    @test occursin("MetaheuristicsAlg", msg)

    dm = DictModel(x -> x[:a]^2 + x[:b]^2)
    addvar!(dm, :a, -2.0, 2.0)
    addvar!(dm, :b, -2.0, 2.0)
    add_sd_constraint!(dm, x -> [x[:a]^2 + 1.0 0.0; 0.0 x[:b]^2 + 1.0], 2)
    msg2 = try
        NonconvexCore.optimize(
            dm,
            MetaheuristicsAlg(DE),
            Dict(:a => 1.0, :b => -1.0);
            options=MetaheuristicsOptions(N=10, f_calls_limit=100),
        )
        ""
    catch e
        sprint(showerror, e)
    end
    @test occursin("semidefinite", msg2)
end

@testitem "features: VecModel optimize without options" begin
    using NonconvexMetaheuristics, LinearAlgebra, Test, Random

    # Regression test: the default options used to be built as
    # MetaheuristicsOptions((;)) (an empty NamedTuple), so optimize! then
    # failed with a FieldError on options.nt.N.
    Random.seed!(1)
    m = Model(x -> sum(x .^ 2))
    addvar!(m, [-5.0, -5.0], [5.0, 5.0])
    vm, vx0, _ = NonconvexCore.tovecmodel(m, [1.0, -1.0])
    r = NonconvexCore.optimize(vm, MetaheuristicsAlg(DE), vx0)
    @test isfinite(r.minimum)
    @test r.minimum < 0.8

    # Same without an explicit initial point.
    r2 = NonconvexCore.optimize(vm, MetaheuristicsAlg(DE))
    @test isfinite(r2.minimum)
    @test r2.minimum < 0.8
end

@testitem "features: GA accepts a user-supplied mutation option" begin
    using NonconvexMetaheuristics, LinearAlgebra, Test, Random

    # Regression test: a user-supplied `mutation` option used to produce a
    # duplicate-key NamedTuple error when building the GA options.
    Random.seed!(1)
    m = Model(x -> sum(x .^ 2))
    addvar!(m, [-5.0, -5.0], [5.0, 5.0])
    options = MetaheuristicsOptions(
        N=20,
        f_calls_limit=500,
        mutation=Metaheuristics.PolynomialMutation(; bounds=[-5.0 -5.0; 5.0 5.0]),
    )
    r = NonconvexCore.optimize(m, MetaheuristicsAlg(GA), [1.0, -1.0], options=options)
    @test isfinite(r.minimum)
end

@testitem "features: finite surrogate bounds are used as the solver search space" begin
    using NonconvexMetaheuristics, LinearAlgebra, Test, Random

    # Regression test: the finite surrogate bounds for infinite variable
    # bounds were only used for the random initial population and not passed
    # to Metaheuristics.optimize. PSO then flew to x[2] ≈ 10 (the true
    # minimizer); with the finite search space it is constrained to [-1, 1].
    Random.seed!(1)
    m = Model(x -> x[1]^2 + (x[2] - 10)^2)
    addvar!(m, [-Inf, -Inf], [Inf, Inf])
    r = NonconvexCore.optimize(
        m,
        MetaheuristicsAlg(PSO),
        [1.0, 1.0];
        options=MetaheuristicsOptions(N=20, f_calls_limit=1000),
    )
    @test isfinite(r.minimum)
    @test all(isfinite, r.minimizer)
    @test all(abs.(r.minimizer) .<= 1.0 + 1e-8)
    @test r.minimum >= 81.0 - 1e-8
end
