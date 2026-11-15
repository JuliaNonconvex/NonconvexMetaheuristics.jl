# NEWS

## 0.3.2

- Added docstrings for `MetaheuristicsAlg`, `MetaheuristicsOptions` and the re-exported algorithm names (REPL help and API reference).


## 0.3.1

- Enforced the solver capabilities and fixed options handling in the
  Metaheuristics workspace (`Workspace` kwargs forwarding, JET isolation);
  README and QA parity (docs badges, Aqua/JET, formatting); codecov OIDC/v5
  uploads and the organization logo.

## 0.3.0

- NonconvexCore 2 with strict solver capability declarations
  (`infinite_bounds`); dropped the local `[sources]` test override.
- DocumenterFragments docs, tests migrated to TestItems, JET/Aqua/formatting
  quality gates, and Julia 1.10+.

## 0.2.1

- Added tests running DE, PSO, ABC, CGSA, WOA and SA on a simple quadratic
  problem. MCCGA is marked broken because Metaheuristics.jl does not define
  `final_stage!` for the constraint-aware state the wrapper builds.
