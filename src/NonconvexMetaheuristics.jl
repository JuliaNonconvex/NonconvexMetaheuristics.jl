module NonconvexMetaheuristics

export MetaheuristicsAlg, MetaheuristicsOptions, Metaheuristics, ECA, DE, PSO, ABC, CGSA, SA, WOA, MCCGA, GA

using Reexport, Parameters, Setfield, Random
@reexport using NonconvexCore
using NonconvexCore: @params, VecModel, AbstractResult
using NonconvexCore: AbstractOptimizer, CountingFunction
import NonconvexCore: optimize, optimize!, Workspace
import Metaheuristics: Metaheuristics, ECA, DE, PSO, ABC, CGSA, SA, WOA, MCCGA, GA

struct MetaheuristicsAlg{A} <: AbstractOptimizer
    algT::A
end

@params struct MetaheuristicsOptions
    nt::NamedTuple
end
function MetaheuristicsOptions(; multiple_initial_solutions = false, N = 100, rng = Random.GLOBAL_RNG, kwargs...)
    return MetaheuristicsOptions(merge(NamedTuple(kwargs), (; rng, N, multiple_initial_solutions)))
end

@params mutable struct MetaheuristicsWorkspace <: Workspace
    model::VecModel
    x0::AbstractVector
    options::MetaheuristicsOptions
    alg::MetaheuristicsAlg
end
function MetaheuristicsWorkspace(
    model::VecModel, optimizer::MetaheuristicsAlg,
    x0::AbstractVector = getinit(model);
    options = MetaheuristicsOptions(), kwargs...,
)
    return MetaheuristicsWorkspace(model, copy(x0), options, optimizer)
end
@params struct MetaheuristicsResult <: AbstractResult
    minimizer
    minimum
    result
    alg
    options
end

function optimize(model::NonconvexCore.Model, optimizer::MetaheuristicsAlg, x0::Vector, args...; options = MetaheuristicsOptions())
    N = options.nt.N
    @assert N > 0
    if !(options.nt.multiple_initial_solutions)
        x0 = [x0]
    end
    @assert eltype(x0) <: AbstractVector
    first_x0 = identity.(first(x0))
    flat_x0 = first.(NonconvexCore.flatten.(map(x -> identity.(x), x0)))
    NonconvexCore._optimize_precheck(model, optimizer, first_x0)
    _model, _, unflatten = NonconvexCore.tovecmodel(model, first_x0)
    r = optimize(_model, optimizer, flat_x0; options)
    return @set r.minimizer = unflatten(r.minimizer)
end
function optimize(model::NonconvexCore.DictModel, optimizer::MetaheuristicsAlg, x0, args...; options = MetaheuristicsOptions())
    N = options.nt.N
    @assert N > 0
    if !(options.nt.multiple_initial_solutions)
        x0 = [x0]
    end
    @assert typeof(x0) <: AbstractVector
    @assert eltype(x0) <: AbstractDict
    first_x0 = first(x0)
    flat_x0 = first.(NonconvexCore.flatten.(x0))
    NonconvexCore._optimize_precheck(model, optimizer, first_x0)
    _model, _, unflatten = NonconvexCore.tovecmodel(model, first_x0)
    r = optimize(_model, optimizer, flat_x0; options)
    return @set r.minimizer = unflatten(r.minimizer)
end

@generated function drop_ks(nt::NamedTuple{names}, ::Val{ks}) where {names, ks}
    ns = Tuple(setdiff(names, ks))
    return :(NamedTuple{$ns}(nt))
end

function optimize!(workspace::MetaheuristicsWorkspace)
    @unpack model, options, x0, alg = workspace
    _options = drop_ks(options.nt, Val((:multiple_initial_solutions, :rng)))
    bounds = hcat(getmin(model), getmax(model))'
    if alg.algT === GA
        __options = (; _options..., mutation = Metaheuristics.PolynomialMutation(; bounds))
    else
        __options = _options
    end
    _alg = alg.algT(; __options...)
    N = options.nt.N
    obj = NonconvexCore.getobjective(model)
    ineq = NonconvexCore.getineqconstraints(model)
    eq = NonconvexCore.geteqconstraints(model)
    mh_func = x -> begin
        obj(x), (length(ineq.fs) === 0 ? [0.0] : ineq(x)), (length(eq.fs) === 0 ? [0.0] : eq(x))
    end
    newx = [rand(options.nt.rng, length(first(x0))) .* (getmax(model) .- getmin(model)) .+ getmin(model) for _ in 1:N-length(x0)]
    x0 = [x0; newx]
    population = [Metaheuristics.create_child(x, mh_func(x)) for x in x0]
    prev_status = Metaheuristics.State(Metaheuristics.get_best(population), population)
    _alg.status = prev_status
    result = Metaheuristics.optimize(mh_func, bounds, _alg)

    return MetaheuristicsResult(Metaheuristics.minimizer(result), Metaheuristics.minimum(result), result, alg, options)
end

function Workspace(model::VecModel, optimizer::MetaheuristicsAlg, args...; kwargs...,)
    return MetaheuristicsWorkspace(model, optimizer, args...; kwargs...)
end

end
