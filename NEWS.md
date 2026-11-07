# NEWS

## 0.2.1

- Added tests running DE, PSO, ABC, CGSA, WOA and SA on a simple quadratic
  problem. MCCGA is marked broken because Metaheuristics.jl does not define
  `final_stage!` for the constraint-aware state the wrapper builds.
