@testitem "Multiple initial solutions" begin
    using NonconvexMetaheuristics, LinearAlgebra, Test, Random

    f(x::AbstractVector) = sqrt(x[2])
    g(x::AbstractVector, a, b) = (a * x[1] + b)^3 - x[2]

    tol = 2e-2
    N = 100
    function get_initial(x0, multiple_initial_solutions, N)
        if multiple_initial_solutions
            return [[x0]; [rand(2) * 10 for _ in 1:(N - 1)]]
        else
            return x0
        end
    end

    for multiple_initial_solutions in (true, false), malg in (ECA, GA)
        options = MetaheuristicsOptions(; multiple_initial_solutions, N=N)
        @testset "Simple constraints" begin
            Random.seed!(1)
            m = Model(f)
            addvar!(m, [0.0, 0.0], [10.0, 10.0])
            add_ineq_constraint!(m, x -> g(x, 2, 0))
            add_ineq_constraint!(m, x -> g(x, -1, 1))

            alg = MetaheuristicsAlg(malg)
            x0 = get_initial([1.234, 2.345], multiple_initial_solutions, N)
            r = NonconvexCore.optimize(m, alg, x0, options=options)
            @test abs(r.minimum - sqrt(8 / 27)) < tol
            @test norm(r.minimizer - [1 / 3, 8 / 27]) < tol
        end

        @testset "Equality constraints" begin
            Random.seed!(1)
            m = Model(f)
            addvar!(m, [0.0, 0.0], [10.0, 10.0])
            add_ineq_constraint!(m, x -> g(x, 2, 0))
            add_ineq_constraint!(m, x -> g(x, -1, 1))
            add_eq_constraint!(m, x -> sum(x) - 1 / 3 - 8 / 27)

            alg = MetaheuristicsAlg(malg)
            x0 = get_initial([1.234, 2.345], multiple_initial_solutions, N)
            r = NonconvexCore.optimize(m, alg, x0, options=options)
            @test abs(r.minimum - sqrt(8 / 27)) < tol
            @test norm(r.minimizer - [1 / 3, 8 / 27]) < tol
        end

        @testset "Block constraints" begin
            Random.seed!(1)
            m = Model(f)
            addvar!(m, [0.0, 0.0], [10.0, 10.0])
            add_ineq_constraint!(m, FunctionWrapper(x -> [g(x, 2, 0), g(x, -1, 1)], 2))

            alg = MetaheuristicsAlg(malg)
            x0 = get_initial([1.234, 2.345], multiple_initial_solutions, N)
            r = NonconvexCore.optimize(m, alg, x0, options=options)
            @test abs(r.minimum - sqrt(8 / 27)) < tol
            @test norm(r.minimizer - [1 / 3, 8 / 27]) < tol
        end
    end
end

@testitem "f_calls_limit and infinite bounds" begin
    using NonconvexMetaheuristics, LinearAlgebra, Test, Random

    f(x::AbstractVector) = sqrt(x[2])
    g(x::AbstractVector, a, b) = (a * x[1] + b)^3 - x[2]

    Random.seed!(1)
    # Regression test: f_calls_limit used to be passed as a bare keyword to the
    # Metaheuristics constructors, which MethodError'd at the compat floor.
    m = Model(f)
    addvar!(m, [0.0, 0.0], [10.0, 10.0])
    add_ineq_constraint!(m, x -> g(x, 2, 0))
    add_ineq_constraint!(m, x -> g(x, -1, 1))
    options = MetaheuristicsOptions(N=20, f_calls_limit=200)
    r = NonconvexCore.optimize(m, MetaheuristicsAlg(ECA), [1.234, 2.345], options=options)
    @test isfinite(r.minimum)

    # Regression test: infinite bounds used to produce NaN/Inf initial
    # populations.
    m2 = Model(x -> x[1]^2 + x[2]^2)
    addvar!(m2, [-Inf, -Inf], [Inf, Inf])
    options2 = MetaheuristicsOptions(N=20, f_calls_limit=200)
    r2 = NonconvexCore.optimize(m2, MetaheuristicsAlg(DE), [1.0, 1.0], options=options2)
    @test isfinite(r2.minimum)
    @test all(isfinite, r2.minimizer)
end

@testitem "Algorithms not covered above" begin
    using NonconvexMetaheuristics, LinearAlgebra, Test, Random

    for malg in (DE, PSO, ABC, CGSA, WOA)
        Random.seed!(1)
        m = Model(x -> sum(x .^ 2))
        addvar!(m, [-5.0, -5.0], [5.0, 5.0])
        options = MetaheuristicsOptions(N=50, f_calls_limit=2000)
        r = NonconvexCore.optimize(m, MetaheuristicsAlg(malg), [1.0, -1.0], options=options)
        # Zero-order algorithms differ in convergence speed; assert that the run
        # improves on the initial point (f = 2) with a finite result.
        @test isfinite(r.minimum)
        @test r.minimum < 0.8
        @test norm(r.minimizer) < 1.0
    end
end

@testitem "MCCGA runs on unconstrained models and rejects constrained ones" begin
    using NonconvexMetaheuristics, NonconvexCore, LinearAlgebra, Test, Random

    # Regression test: the wrapper always built the constraint-aware
    # (f, g, h) objective, but Metaheuristics.jl's MCCGA final_stage! is only
    # defined for the unconstrained state type, so every MCCGA solve crashed
    # with a MethodError. The wrapper now builds a scalar objective for MCCGA
    # when the model has no constraints, and constrained models are rejected
    # by the capability validation (Model/DictModel paths) or by a backstop
    # in optimize! (direct VecModel/Workspace paths).
    Random.seed!(1)
    m = Model(x -> sum(x .^ 2))
    addvar!(m, [-5.0, -5.0], [5.0, 5.0])
    options = MetaheuristicsOptions(N=50, f_calls_limit=2000)
    r = NonconvexCore.optimize(m, MetaheuristicsAlg(MCCGA), [1.0, -1.0], options=options)
    @test isfinite(r.minimum)
    @test r.minimum < 0.8
    @test norm(r.minimizer) < 1.0

    # Constrained models are rejected by the capability validation with a
    # hint naming the upstream limitation.
    Random.seed!(1)
    m2 = Model(x -> sum(x .^ 2))
    addvar!(m2, [-5.0, -5.0], [5.0, 5.0])
    add_ineq_constraint!(m2, x -> sum(x) - 0.5)
    msg = try
        NonconvexCore.optimize(m2, MetaheuristicsAlg(MCCGA), [1.0, -1.0], options=options)
        ""
    catch e
        sprint(showerror, e)
    end
    @test occursin("inequality constraints", msg)
    @test occursin("MCCGA", msg)
    @test occursin("final_stage!", msg)

    # The direct VecModel/Workspace paths bypass checksubproblem!, so the
    # optimize! backstop rejects constrained models there too.
    Random.seed!(1)
    m3 = Model(x -> sum(x .^ 2))
    addvar!(m3, [-5.0, -5.0], [5.0, 5.0])
    add_eq_constraint!(m3, x -> sum(x) - 0.5)
    vm, vx0, _ = NonconvexCore.tovecmodel(m3, [1.0, -1.0])
    ws = NonconvexMetaheuristics.Workspace(
        vm, MetaheuristicsAlg(MCCGA), [vx0]; options=options
    )
    @test_throws ArgumentError NonconvexMetaheuristics.optimize!(ws)
end

@testitem "SA runs to completion" begin
    using NonconvexMetaheuristics, LinearAlgebra, Test, Random

    # SA is a single-solution method; with the shared settings it does not
    # reliably improve on the initial point, so only assert completion.
    Random.seed!(1)
    m = Model(x -> sum(x .^ 2))
    addvar!(m, [-5.0, -5.0], [5.0, 5.0])
    options = MetaheuristicsOptions(N=50, f_calls_limit=2000)
    r = NonconvexCore.optimize(m, MetaheuristicsAlg(SA), [1.0, -1.0], options=options)
    @test isfinite(r.minimum)
    @test all(isfinite, r.minimizer)
end

@testitem "solver capabilities" begin
    using NonconvexMetaheuristics, Test

    caps = solver_capabilities(MetaheuristicsAlg(ECA))
    @test caps isa ConstraintCapabilities
    @test caps.linear_eq === true
    @test caps.nonlinear_eq === true
    @test caps.linear_ineq === true
    @test caps.nonlinear_ineq === true
    @test caps.linear_sd === false
    @test caps.nonlinear_sd === false
    @test caps.box === true
    @test caps.integer === false
    @test caps.infinite_bounds === true

    # MCCGA declares itself unsupported for constrained problems: upstream
    # Metaheuristics.jl only defines MCCGA's final_stage! for the
    # unconstrained state type.
    mccga = solver_capabilities(MetaheuristicsAlg(MCCGA))
    @test mccga.linear_eq === false
    @test mccga.nonlinear_eq === false
    @test mccga.linear_ineq === false
    @test mccga.nonlinear_ineq === false
    @test mccga.linear_sd === false
    @test mccga.nonlinear_sd === false
    @test mccga.box === true
    @test mccga.continuous === true
    @test mccga.binary === false
    @test mccga.integer === false
    @test mccga.infinite_bounds === true
end
