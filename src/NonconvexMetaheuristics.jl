module NonconvexMetaheuristics

export MetaheuristicsAlg,
    MetaheuristicsOptions, Metaheuristics, ECA, DE, PSO, ABC, CGSA, SA, WOA, MCCGA, GA

using Reexport, Parameters, Setfield, Random
@reexport using NonconvexCore
using NonconvexCore: @params, VecModel, AbstractResult
using NonconvexCore: AbstractOptimizer, CountingFunction
import NonconvexCore: optimize, optimize!, Workspace
import Metaheuristics: Metaheuristics, ECA, DE, PSO, ABC, CGSA, SA, WOA, MCCGA, GA, Options

"""
    MetaheuristicsAlg(algT)

A gradient-free metaheuristic algorithm from
[Metaheuristics.jl](https://github.com/jmejia8/Metaheuristics.jl), exposed
through the Nonconvex interface. `algT` is one of the supported
single-objective algorithm types: [`ECA`](@ref), [`DE`](@ref),
[`PSO`](@ref), [`ABC`](@ref), [`CGSA`](@ref), [`SA`](@ref), [`WOA`](@ref),
[`MCCGA`](@ref) or [`GA`](@ref), e.g. `MetaheuristicsAlg(ECA)`.

Most algorithms support bounds and nonlinear inequality and equality
constraints (handled via Metaheuristics.jl's penalty aggregation); `MCCGA`
is unconstrained-only. Semidefinite constraints are not supported. Options
are set through [`MetaheuristicsOptions`](@ref).
"""
struct MetaheuristicsAlg{A} <: AbstractOptimizer
    algT::A
end

# The algorithm types below are re-exported from Metaheuristics.jl; give each
# of them a local docstring so they appear in the Nonconvex API reference.
@doc """
    ECA

Evolutionary Centers Algorithm from
[Metaheuristics.jl](https://github.com/jmejia8/Metaheuristics.jl): a
gradient-free metaheuristic. Use as `MetaheuristicsAlg(ECA)`.
""" ECA
@doc """
    DE

Differential Evolution from
[Metaheuristics.jl](https://github.com/jmejia8/Metaheuristics.jl): a
gradient-free metaheuristic. Use as `MetaheuristicsAlg(DE)`.
""" DE
@doc """
    PSO

Particle Swarm Optimization from
[Metaheuristics.jl](https://github.com/jmejia8/Metaheuristics.jl): a
gradient-free metaheuristic. Use as `MetaheuristicsAlg(PSO)`.
""" PSO
@doc """
    ABC

Artificial Bee Colony from
[Metaheuristics.jl](https://github.com/jmejia8/Metaheuristics.jl): a
gradient-free metaheuristic. Use as `MetaheuristicsAlg(ABC)`.
""" ABC
@doc """
    CGSA

Gravitational Search Algorithm (also known as GSA) from
[Metaheuristics.jl](https://github.com/jmejia8/Metaheuristics.jl): a
gradient-free metaheuristic. Use as `MetaheuristicsAlg(CGSA)`.
""" CGSA
@doc """
    SA

Simulated Annealing from
[Metaheuristics.jl](https://github.com/jmejia8/Metaheuristics.jl): a
gradient-free metaheuristic. Use as `MetaheuristicsAlg(SA)`.
""" SA
@doc """
    WOA

Whale Optimization Algorithm from
[Metaheuristics.jl](https://github.com/jmejia8/Metaheuristics.jl): a
gradient-free metaheuristic. Use as `MetaheuristicsAlg(WOA)`.
""" WOA
@doc """
    MCCGA

Machine-coded Compact Genetic Algorithm from
[Metaheuristics.jl](https://github.com/jmejia8/Metaheuristics.jl): a
gradient-free metaheuristic. Unconstrained problems only. Use as
`MetaheuristicsAlg(MCCGA)`.
""" MCCGA
@doc """
    GA

Genetic Algorithm from
[Metaheuristics.jl](https://github.com/jmejia8/Metaheuristics.jl): a
gradient-free metaheuristic. Use as `MetaheuristicsAlg(GA)`.
""" GA

# Equality and inequality constraints are handled via Metaheuristics.jl's
# penalty aggregation of the (f, g, h) triple; sd constraints are not handled
# and integer variables are treated as continuous (with a warning).
function NonconvexCore.solver_capabilities(alg::MetaheuristicsAlg)
    if alg.algT === MCCGA
        # MCCGA cannot solve constrained problems: Metaheuristics.jl only
        # defines its final_stage! (Hooke-Jeeves local search) for the
        # unconstrained state type, so a constrained solve crashes with a
        # MethodError at the end of the run.
        # Upstream: https://github.com/jmejia8/Metaheuristics.jl/issues/135
        return ConstraintCapabilities(;
            linear_eq=false,
            nonlinear_eq=false,
            linear_ineq=false,
            nonlinear_ineq=false,
            linear_sd=false,
            nonlinear_sd=false,
            box=true,
            continuous=true,
            binary=false,
            integer=false,
            infinite_bounds=true,
        )
    end
    return ConstraintCapabilities(;
        linear_eq=true,
        nonlinear_eq=true,
        linear_ineq=true,
        nonlinear_ineq=true,
        linear_sd=false,
        nonlinear_sd=false,
        box=true,
        continuous=true,
        binary=false,
        integer=false,
        infinite_bounds=true,
    )
end

# Hint appended to capability-validation errors naming the specific upstream
# limitation, so users see why MCCGA is rejected for constrained models.
function _capability_hint(alg::MetaheuristicsAlg)
    alg.algT === MCCGA || return ""
    return "MCCGA does not support constrained problems: Metaheuristics.jl " *
           "only defines its final_stage! for unconstrained states " *
           "(https://github.com/jmejia8/Metaheuristics.jl/issues/135)."
end

"""
    MetaheuristicsOptions(; options=Metaheuristics.Options(), kwargs...)

Options for [`MetaheuristicsAlg`](@ref), passed to `optimize` as
`options = ...`. The keyword arguments are forwarded to the wrapped
Metaheuristics.jl `Options` (population size, iterations, tolerances, etc.);
see the [Metaheuristics.jl options documentation](https://jmejia8.github.io/Metaheuristics.jl/dev/api/#Metaheuristics.Options).
"""
struct MetaheuristicsOptions{N<:NamedTuple}
    nt::N
end
function MetaheuristicsOptions(;
    multiple_initial_solutions=false,
    N=100,
    f_calls_limit=N * 100,
    rng=Random.GLOBAL_RNG,
    kwargs...,
)
    return MetaheuristicsOptions(
        merge((; rng, N, f_calls_limit, multiple_initial_solutions), NamedTuple(kwargs))
    )
end

@params mutable struct MetaheuristicsWorkspace <: Workspace
    model::VecModel
    x0::AbstractVector
    options::MetaheuristicsOptions
    alg::MetaheuristicsAlg
end
function MetaheuristicsWorkspace(
    model::VecModel,
    optimizer::MetaheuristicsAlg,
    x0::AbstractVector=NonconvexCore.getinit(model);
    options=MetaheuristicsOptions(),
    kwargs...,
)
    return _metaheuristics_workspace(model, x0, options, optimizer)
end
# Keyword-forwarding path from `Workspace(model, optimizer, x0; kwargs...)`.
# Passing the keywords as a positional NamedTuple avoids a `Core.kwcall` on
# a `Base.Pairs` that fails on Julia 1.13.
function MetaheuristicsWorkspace(
    model::VecModel, optimizer::MetaheuristicsAlg, x0::AbstractVector, nt::NamedTuple
)
    options = haskey(nt, :options) ? nt[:options] : MetaheuristicsOptions()
    return _metaheuristics_workspace(model, x0, options, optimizer)
end
function _metaheuristics_workspace(model::VecModel, x0, options, optimizer)
    return MetaheuristicsWorkspace(model, copy(x0), options, optimizer)
end
@params struct MetaheuristicsResult <: AbstractResult
    minimizer::Any
    minimum::Any
    result::Any
    alg::Any
    options::Any
end

function optimize(
    model::NonconvexCore.Model,
    optimizer::MetaheuristicsAlg,
    x0::AbstractVector,
    args...;
    options=MetaheuristicsOptions(),
)
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
    # Validate the model against the declared capabilities as the generic
    # NonconvexCore `optimize` path does; otherwise unsupported features
    # (e.g. semidefinite constraints) would be silently ignored here.
    NonconvexCore.checksubproblem!(_model, optimizer; hint=_capability_hint(optimizer))
    r = optimize(_model, optimizer, flat_x0; options)
    return @set r.minimizer = unflatten(r.minimizer)
end
function optimize(
    model::NonconvexCore.DictModel,
    optimizer::MetaheuristicsAlg,
    x0,
    args...;
    options=MetaheuristicsOptions(),
)
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
    # Same as the Model path above: enforce the declared capabilities.
    NonconvexCore.checksubproblem!(_model, optimizer; hint=_capability_hint(optimizer))
    r = optimize(_model, optimizer, flat_x0; options)
    return @set r.minimizer = unflatten(r.minimizer)
end

@generated function drop_ks(nt::NamedTuple{names}, ::Val{ks}) where {names,ks}
    ns = Tuple(setdiff(names, ks))
    return :(NamedTuple{$ns}(nt))
end

function optimize!(workspace::MetaheuristicsWorkspace)
    @unpack model, options, x0, alg = workspace
    _options = drop_ks(
        options.nt,
        Val((
            :multiple_initial_solutions, :rng, :f_calls_limit, :f_tol, :x_tol, :iterations
        )),
    )
    mh_options = Metaheuristics.Options(;
        f_calls_limit=get(options.nt, :f_calls_limit, 0.0),
        f_tol=get(options.nt, :f_tol, 0.0),
        x_tol=get(options.nt, :x_tol, 0.0),
        iterations=get(options.nt, :iterations, 0),
    )
    N = options.nt.N
    obj = NonconvexCore.getobjective(model)
    ineq = NonconvexCore.getineqconstraints(model)
    eq = NonconvexCore.geteqconstraints(model)
    constrained = length(ineq.fs) > 0 || length(eq.fs) > 0
    if alg.algT === MCCGA && constrained
        # Backstop for the direct VecModel / Workspace paths, which bypass
        # the capability validation in checksubproblem!: Metaheuristics.jl
        # only defines MCCGA's final_stage! (its Hooke-Jeeves local search)
        # for the unconstrained state type, so a constrained solve would
        # crash with a MethodError at the end of the run.
        # Upstream: https://github.com/jmejia8/Metaheuristics.jl/issues/135
        throw(
            ArgumentError(
                "MCCGA does not support constrained problems through the " *
                "Metaheuristics wrapper (Metaheuristics.jl's final_stage! " *
                "is only defined for unconstrained states). Use an " *
                "unconstrained model or a different algorithm.",
            ),
        )
    end
    mh_func = if alg.algT === MCCGA
        # Unconstrained MCCGA: build a scalar objective so the population
        # state is the unconstrained type supported by MCCGA.final_stage!.
        x -> Float64(obj(x))
    else
        x -> begin
            Float64(obj(x)),
            Float64.(length(ineq.fs) === 0 ? [0.0] : ineq(x)),
            Float64.(length(eq.fs) === 0 ? [0.0] : eq(x))
        end
    end
    box_min, box_max = getmin(model), getmax(model)
    if !(all(isfinite, box_min) && all(isfinite, box_max))
        @warn "Some variable bounds are not finite. Using finite surrogate bounds for the random initial population and the solver search bounds." maxlog =
            1
        finite_min = isfinite.(box_min)
        finite_max = isfinite.(box_max)
        box_min = [
            finite_min[i] ? box_min[i] : (finite_max[i] ? box_max[i] - 1 : -1.0) for
            i in eachindex(box_min)
        ]
        box_max = [
            finite_max[i] ? box_max[i] : (finite_min[i] ? box_min[i] + 1 : 1.0) for
            i in eachindex(box_max)
        ]
    end
    bounds = hcat(box_min, box_max)'
    if alg.algT === GA
        # Override a user-supplied `mutation` with the bounds-aware polynomial
        # mutation; merging avoids the duplicate-key NamedTuple error when the
        # user passes their own `mutation` option.
        __options = merge(
            _options, (; mutation=Metaheuristics.PolynomialMutation(; bounds))
        )
    else
        __options = _options
    end
    _alg = alg.algT(; options=mh_options, __options...)
    if length(x0) > N
        @warn "More initial solutions ($(length(x0))) than the population size N ($N). Truncating." maxlog =
            1
        x0 = x0[1:N]
    end
    newx = [
        rand(options.nt.rng, length(first(x0))) .* (box_max .- box_min) .+ box_min for
        _ in 1:(N - length(x0))
    ]
    x0 = [x0; newx]
    if any(model.integer)
        @warn "Integer variables are not supported by the Metaheuristics wrapper; they will be treated as continuous." maxlog =
            1
    end
    population = [Metaheuristics.create_child(x, mh_func(x)) for x in x0]
    prev_status = Metaheuristics.State(Metaheuristics.get_best(population), population)
    _alg.status = prev_status
    result = Metaheuristics.optimize(mh_func, bounds, _alg)

    return MetaheuristicsResult(
        Metaheuristics.minimizer(result),
        Metaheuristics.minimum(result),
        result,
        alg,
        options,
    )
end

function Workspace(model::VecModel, optimizer::MetaheuristicsAlg, args...; kwargs...)
    x0 = isempty(args) ? NonconvexCore.getinit(model) : args[1]
    if !(first(x0) isa AbstractVector)
        # A single flat initial point: store it as a one-element population,
        # as the Model/DictModel `optimize` paths do.
        x0 = [x0]
    end
    return MetaheuristicsWorkspace(model, optimizer, x0, (; kwargs...))
end

end
