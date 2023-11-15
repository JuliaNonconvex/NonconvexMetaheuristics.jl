using NonconvexMetaheuristics, LinearAlgebra, Test, Random

f(x::AbstractVector) = 100sqrt(x[2])
g(x::AbstractVector, a, b) = (a * x[1] + b)^3 - x[2]

tol = 2e-2
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

    options = MetaheuristicsOptions(; multiple_initial_solutions, N = 1000)
    @testset "Simple constraints" begin
        Random.seed!(1)
        m = Model(f)
        addvar!(m, [0.0, 0.0], [10.0, 10.0])
        add_ineq_constraint!(m, x -> g(x, 2, 0))
        add_ineq_constraint!(m, x -> g(x, -1, 1))

        alg = MetaheuristicsAlg(malg)
        x0 = get_initial([1.234, 2.345], multiple_initial_solutions, 1000)
        r = NonconvexCore.optimize(m, alg, x0, options = options)
        @test abs(r.minimum - 100sqrt(8 / 27)) < tol
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
        x0 = get_initial([1.234, 2.345], multiple_initial_solutions, 1000)
        r = NonconvexCore.optimize(m, alg, x0, options = options)
        @test abs(r.minimum - 100sqrt(8 / 27)) < tol
        @test norm(r.minimizer - [1 / 3, 8 / 27]) < tol
    end

    @testset "Block constraints" begin
        Random.seed!(1)
        m = Model(f)
        addvar!(m, [0.0, 0.0], [10.0, 10.0])
        add_ineq_constraint!(m, FunctionWrapper(x -> [g(x, 2, 0), g(x, -1, 1)], 2))

        alg = MetaheuristicsAlg(malg)
        x0 = get_initial([1.234, 2.345], multiple_initial_solutions, 1000)
        r = NonconvexCore.optimize(m, alg, x0, options = options)
        @test abs(r.minimum - 100sqrt(8 / 27)) < tol
        @test norm(r.minimizer - [1 / 3, 8 / 27]) < tol
    end
end
