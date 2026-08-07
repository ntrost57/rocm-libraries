# `rocalution_opt` — configuration reference

`rocalution_opt` loads a matrix **once** and benchmarks a set of PCG/Krylov + AMG
configurations described by a JSON(C) file. This document lists every supported option.

```
rocalution_opt <config.json> [results.json] [--dry-run]
```

- `results.json` defaults to `rocalution_opt_results.json`.
- `--dry-run` expands and lists the configurations without loading the matrix or solving.
- The config parser accepts `//` line and `/* ... */` block comments (JSONC).

---

## 1. Config structure

| Key | Type | Description |
|---|---|---|
| `global` | object | Applied-once settings (matrix, rhs, warm-up) plus `defaults`. May be omitted; the keys can also live at the top level. |
| `global.defaults` | object | Fallback values for any run parameter not otherwise set. |
| `sweep` | object | A single grid. Any run field may be a scalar (fixed) or an array (swept axis). |
| `sweeps` | array | A list of independent grid objects, each expanded on its own. |
| `runs` | array | Explicit, fully specified one-off configurations, appended after the sweeps. |

Runs are collected from `sweep` + `sweeps` + `runs` (all optional). If none are present,
the top-level object is treated as a single run.

**Resolution precedence** (high → low): explicit `runs` value → `sweep`/`sweeps` value →
`global.defaults` → library default.

**Sweep expansion**: array-valued fields are Cartesian-expanded; parameters not relevant to
the chosen AMG method are dropped; the resulting set is de-duplicated. Each run is
auto-named from its `group` prefix and the varying relevant axes (e.g.
`sa/coarsen=PMIS/eps=0.01`); an explicit `name` overrides this.

| Naming key | Type | Description |
|---|---|---|
| `group` | string | Name prefix for the runs produced by a sweep/run object. |
| `name` | string | Explicit run name (verbatim in the output). |

---

## 2. Global options

| Key | Type | Default | Values / notes |
|---|---|---|---|
| `matrix` | string | — (required) | Path to the matrix file. |
| `matrix_format` | string | `auto` | `auto` \| `mtx` \| `csr` \| `rsio` \| `binary`. Case-insensitive; see below. |
| `omp_threads` | int | `0` | Host OpenMP threads; `0` keeps the library default. |
| `rhs` | string | `ones` | `ones` → `rhs = A · 1` (exact solution is the all-ones vector). |
| `initial_guess` | string | `zeros` | Initial guess for `x`. |
| `warmup` | bool | `true` | Run a discarded warm-up per distinct method before timing. |
| `warmup_iters` | int | `2` | Outer iterations per warm-up solve. |

### Matrix input formats

| `matrix_format` | Reader | Notes |
|---|---|---|
| `auto` (default) | from file extension | `.mtx`/`.mm` → `mtx`, `.csr` → `csr`, `.rsio`/`.bin` → `rsio`. Errors out if the extension is unknown. |
| `mtx` | `ReadFileMTX` | MatrixMarket text. Portable but slow for very large matrices. |
| `csr` | `ReadFileCSR` | Legacy rocALUTION **binary** CSR. Deprecated in the library, still supported here for existing data sets. |
| `rsio` | `ReadFileRSIO` | Modern rocALUTION **binary** format. Recommended for large matrices. |
| `binary` | `ReadFileRSIO` | Alias for `rsio`. |

Binary input is dramatically faster than MTX text: for the same matrix, RSIO read in
0.00013 s versus 0.0036 s for MTX (~28x). Since the matrix is read only once per batch,
this matters most for the huge matrices this tool targets.

**64-bit nnz.** The number of nonzeros is limited by rocALUTION's row-pointer width, which
is a *library build option*, not a tool setting. The tool prints the active width, e.g.:

```
nnz index width: 32 bit (rebuild rocALUTION with -DBUILD_PTRTYPE_64=ON for nnz > 2147483647)
```

To read matrices with more than 2^31 nonzeros, build rocALUTION with
`-DBUILD_PTRTYPE_64=ON`. The number of rows and columns must still stay below 2^31 unless
rocALUTION is also built with `-DBUILD_LOCALTYPE_64=ON`, because local column indices are
32-bit. Note that only CSR carries a 64-bit row pointer; the `operator_format` /
`smoother_format` alternatives (COO, ELL, MCSR, DIA, HYB, BCSR) index with 32-bit types.

---

## 3. Solver and preconditioner selection

| Key | Type | Default | Values |
|---|---|---|---|
| `solver` | string | `CG` | `CG` \| `FCG` \| `CR` \| `GMRES` \| `FGMRES` \| `BiCGStab` \| `QMRCGStab` \| `IDR` |
| `preconditioner` | string | `AMG` | `None` \| `AMG` \| `Jacobi` \| `GS` \| `SGS` \| `MCGS` \| `MCSGS` \| `MCILU` \| `ILU` \| `IC` \| `FSAI`. Also accepts an AMG method name (`RugeStuebenAMG` \| `SAAMG` \| `UAAMG` \| `PairwiseAMG`), which sets `preconditioner = AMG` and the matching `amg_type`. |
| `amg_type` | string | `RugeStueben` | `RugeStueben`/`RS`/`RugeStuebenAMG` \| `SA`/`SAAMG` \| `UA`/`UAAMG` \| `Pairwise`/`PW`/`PairwiseAMG` (used when `preconditioner = AMG`). |

Non-AMG preconditioners (`None`, `Jacobi`, …) ignore all AMG/multigrid parameters; only
the solver, stopping criteria, and `repeats` apply.

---

## 4. AMG method-specific parameters

Each parameter applies **only** to the AMG methods listed; it is ignored (and dropped from
de-duplication/naming) for the others.

| Key | Type | Default | Methods | Values / notes |
|---|---|---|---|---|
| `coarsening_strategy` | string | library default (`Greedy`) | RS, SA, UA | `Greedy` \| `PMIS` |
| `interpolation_type` | string | `ExtPI` | RS | `Direct` \| `ExtPI` |
| `strength_threshold` | float | `0.0` | RS | Applied only when `> 0`. |
| `ff1_limit` | bool | `false` | RS | Limit FF interpolation. |
| `coupling_strength` | float | `0.001` | SA, UA | Aggregation coupling strength. |
| `interp_relax` | float | `0.6667` | SA | Interpolation smoother relaxation. |
| `lumping_strategy` | string | `AddWeakConnections` | SA | `AddWeakConnections` \| `SubtractWeakConnections` |
| `over_interp` | float | `2.0` | UA | Over-interpolation parameter. |
| `beta` | float | `0.25` | Pairwise | (Pairwise is deprecated in the library.) |
| `coarsening_factor` | float | `4.0` | Pairwise | Target coarsening factor. |
| `ordering` | string | `NoOrdering` | Pairwise | `NoOrdering` \| `Connectivity` \| `CMK` \| `RCMK` \| `MIS` \| `MultiColoring` |

---

## 5. Multigrid cycle and hierarchy (all AMG methods)

| Key | Type | Default | Values / notes |
|---|---|---|---|
| `cycle` | string | `V` | `V` \| `W` \| `K` \| `F` |
| `kcycle_full` | bool | library default | Full vs. truncated K-cycle; applied only when set, relevant only for `cycle = K`. |
| `pre_smooth_iter` | int | `1` | Pre-smoothing steps. |
| `post_smooth_iter` | int | `2` | Post-smoothing steps. |
| `scaling` | bool | `false` | Intergrid transfer scaling. |
| `coarsest_level` | int | `200` | Max unknowns on the coarsest level. |
| `host_levels` | int | `0` | Coarsest levels computed on the host. |
| `operator_format` | string | `CSR` | Hierarchy operator storage: `DENSE` \| `CSR` \| `MCSR` \| `BCSR` \| `COO` \| `DIA` \| `ELL` \| `HYB`. Applied only when set. |
| `operator_blockdim` | int | `1` | Block dimension; used only when `operator_format = BCSR`. |
| `smoother_format` | string | `CSR` | Smoother operator storage (same options). Applied only when set. |

### Smoother (per AMG level)

| Key | Type | Default | Values / notes |
|---|---|---|---|
| `smoother` | string | `Default` | `Default` \| `Jacobi` \| `GS` \| `SGS` \| `MCGS` \| `MCSGS` \| `MCILU` \| `ILU` \| `IC` \| `FSAI`. `Default` keeps the library smoother. |
| `smoother_relax` | float | `1.0` | FixedPoint relaxation; used only when `smoother != Default`. |

### Coarse-grid solver

| Key | Type | Default | Values / notes |
|---|---|---|---|
| `coarse_solver` | string | `Default` | `Default` \| `CG` \| `GMRES` \| `FGMRES` \| `BiCGStab` \| `FCG` \| `QMRCGStab` \| `LU` \| `QR`. |
| `coarse_solver_tol` | float | `1e-8` | Relative tolerance (iterative coarse solvers). |
| `coarse_solver_max_iter` | int | `1000` | Max iterations (iterative coarse solvers). |

---

## 6. Outer solver stopping criteria and timing

| Key | Type | Default | Notes |
|---|---|---|---|
| `abs_tol` | float | `1e-8` | Absolute tolerance. |
| `rel_tol` | float | `1e-8` | Relative tolerance. |
| `div_tol` | float | `1e8` | Divergence tolerance. |
| `max_iter` | int | `10000` | Maximum outer iterations. |
| `repeats` | int | `1` | Solve repetitions per run (build once, solve N times) for timing stats. |

---

## 7. Output

For each run the tool records: the resolved `setup` (relevance-filtered), `num_levels`,
`build_time_sec`, `solve_time_sec` (`mean`/`min`/`max` over `repeats`), `total_time_sec`
(build + mean solve), `iterations`, `final_residual`, `solver_status`
(`no-criteria`/`abs-tol`/`rel-tol`/`diverged`/`max-iter`), `converged`, `error_l2` (when
`rhs = ones`), and `error` (`null` on success).

The results JSON also contains matrix metadata and a `winners` block (`fastest_build`,
`fastest_solve`, `fastest_total` among converged runs, each with the winning `setup`
embedded). A summary table and the winners are printed to stdout as well.
