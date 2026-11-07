using NonconvexMetaheuristics, LinearAlgebra, Test, Random

f(x::AbstractVector) = sqrt(x[2])
g(x::AbstractVector, a, b) = (a * x[1] + b)^3 - x[2]

tol = 2e-2
N = 100
function get_initial(x0, multiple_initial_solutions, N)
    if multiple_initial_solutions
        return [[x0]; [rand(2) * 10 for _ = 1:N-1]]
    else
        return x0
    end
end

@testset "Multiple initial solutions: $multiple_initial_solutions, alg: $malg" for multiple_initial_solutions in
                                                                                   (
        true,
        false,
    ),
    malg in (ECA, GA)

    options = MetaheuristicsOptions(; multiple_initial_solutions, N = N)
    @testset "Simple constraints" begin
        Random.seed!(1)
        m = Model(f)
        addvar!(m, [0.0, 0.0], [10.0, 10.0])
        add_ineq_constraint!(m, x -> g(x, 2, 0))
        add_ineq_constraint!(m, x -> g(x, -1, 1))

        alg = MetaheuristicsAlg(malg)
        x0 = get_initial([1.234, 2.345], multiple_initial_solutions, N)
        r = NonconvexCore.optimize(m, alg, x0, options = options)
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
        r = NonconvexCore.optimize(m, alg, x0, options = options)
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
        r = NonconvexCore.optimize(m, alg, x0, options = options)
        @test abs(r.minimum - sqrt(8 / 27)) < tol
        @test norm(r.minimizer - [1 / 3, 8 / 27]) < tol
    end
end

@testset "f_calls_limit and infinite bounds" begin
    Random.seed!(1)
    # Regression test: f_calls_limit used to be passed as a bare keyword to the
    # Metaheuristics constructors, which MethodError'd at the compat floor.
    m = Model(f)
    addvar!(m, [0.0, 0.0], [10.0, 10.0])
    add_ineq_constraint!(m, x -> g(x, 2, 0))
    add_ineq_constraint!(m, x -> g(x, -1, 1))
    options = MetaheuristicsOptions(N = 20, f_calls_limit = 200)
    r = NonconvexCore.optimize(m, MetaheuristicsAlg(ECA), [1.234, 2.345], options = options)
    @test isfinite(r.minimum)

    # Regression test: infinite bounds used to produce NaN/Inf initial
    # populations.
    m2 = Model(x -> x[1]^2 + x[2]^2)
    addvar!(m2, [-Inf, -Inf], [Inf, Inf])
    options2 = MetaheuristicsOptions(N = 20, f_calls_limit = 200)
    r2 = NonconvexCore.optimize(m2, MetaheuristicsAlg(DE), [1.0, 1.0], options = options2)
    @test isfinite(r2.minimum)
    @test all(isfinite, r2.minimizer)
end

@testset "Algorithms not covered above: $malg" for malg in (DE, PSO, ABC, CGSA, WOA)
    Random.seed!(1)
    m = Model(x -> sum(x .^ 2))
    addvar!(m, [-5.0, -5.0], [5.0, 5.0])
    options = MetaheuristicsOptions(N = 50, f_calls_limit = 2000)
    r = NonconvexCore.optimize(m, MetaheuristicsAlg(malg), [1.0, -1.0], options = options)
    # Zero-order algorithms differ in convergence speed; assert that the run
    # improves on the initial point (f = 2) with a finite result.
    @test isfinite(r.minimum)
    @test r.minimum < 0.8
    @test norm(r.minimizer) < 1.0
end

@testset "MCCGA constraint-state incompatibility" begin
    # MCCGA's final_stage! is not defined for the constraint-aware state type
    # the wrapper builds, so the solve fails with a MethodError. Tracked in
    # the issue about MCCGA support through the wrapper.
    Random.seed!(1)
    m = Model(x -> sum(x .^ 2))
    addvar!(m, [-5.0, -5.0], [5.0, 5.0])
    options = MetaheuristicsOptions(N = 50, f_calls_limit = 2000)
    @test_broken isfinite(
        NonconvexCore.optimize(m, MetaheuristicsAlg(MCCGA), [1.0, -1.0], options = options).minimum,
    )
end

@testset "SA runs to completion" begin
    # SA is a single-solution method; with the shared settings it does not
    # reliably improve on the initial point, so only assert completion.
    Random.seed!(1)
    m = Model(x -> sum(x .^ 2))
    addvar!(m, [-5.0, -5.0], [5.0, 5.0])
    options = MetaheuristicsOptions(N = 50, f_calls_limit = 2000)
    r = NonconvexCore.optimize(m, MetaheuristicsAlg(SA), [1.0, -1.0], options = options)
    @test isfinite(r.minimum)
    @test all(isfinite, r.minimizer)
end
