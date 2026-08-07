/* ************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */

/*! \file
 *  \brief Pure PCG + AMG solver piece for the rocalution_opt benchmarking tool.
 *
 *  This module is deliberately free of any JSON, file, or sweep logic. It takes a
 *  single, fully-resolved SolverInput plus an already-loaded operator / vectors and
 *  returns a SolverOutput with the collected metrics. The optimization box
 *  (rocalution_opt.cpp) is responsible for parsing configs, expanding sweeps, loading
 *  the matrix once, and looping over solve().
 */

#ifndef ROCALUTION_SOLVER_HPP_SAMPLE_
#define ROCALUTION_SOLVER_HPP_SAMPLE_

#include <string>
#include <vector>

#include <rocalution/rocalution.hpp>

namespace rocalution_opt
{
    // Bring rocALUTION's public types (LocalMatrix, CoarseningStrategy, Vcycle, ...) into
    // scope; this is a client sample, matching the convention used by the other samples.
    using namespace rocalution;

    /*! \brief Value type used throughout the tool (single GPU, LocalMatrix). */
    using ValueType = double;

    /*! \brief Outer Krylov solver, preconditioned by the AMG hierarchy.
     *
     *  CG / FCG / CR are appropriate for symmetric positive-definite systems; the others
     *  also handle non-symmetric operators.
     */
    enum class KrylovKind
    {
        CG,
        FCG,
        CR,
        GMRES,
        FGMRES,
        BiCGStab,
        QMRCGStab,
        IDR
    };

    /*! \brief Preconditioner driving the outer Krylov solver.
     *
     *  AMG selects an algebraic multigrid preconditioner whose specific method is chosen
     *  separately via AMGKind; the remaining entries are the standard single-level
     *  preconditioners. None runs the outer solver unpreconditioned.
     */
    enum class PreconditionerKind
    {
        None,
        AMG,
        Jacobi,
        GS,
        SGS,
        MCGS, /**< multi-colored Gauss-Seidel */
        MCSGS, /**< multi-colored symmetric Gauss-Seidel */
        MCILU, /**< multi-colored ILU */
        ILU,
        IC,
        FSAI
    };

    /*! \brief Which AMG method is used when the preconditioner is AMG. */
    enum class AMGKind
    {
        RugeStueben, /**< RugeStuebenAMG */
        SA, /**< SAAMG (smoothed aggregation) */
        UA, /**< UAAMG (unsmoothed aggregation) */
        Pairwise /**< PairwiseAMG (deprecated in the library) */
    };

    /*! \brief Smoother used on each AMG level. Default keeps the library-chosen smoother. */
    enum class SmootherKind
    {
        Default,
        Jacobi,
        GS,
        SGS,
        MCGS,
        MCSGS,
        MCILU,
        ILU,
        IC,
        FSAI
    };

    /*! \brief Coarse-grid solver. Default keeps the library-chosen coarse solver. */
    enum class CoarseSolverKind
    {
        Default,
        CG,
        GMRES,
        FGMRES,
        BiCGStab,
        FCG,
        QMRCGStab,
        LU,
        QR
    };

    /*! \brief One fully-resolved solver configuration.
     *
     *  All fields carry concrete values (typed enums, no arrays, nothing left to parse).
     *  Fields that are irrelevant to the chosen preconditioner are simply ignored by
     *  solve(). Defaults below are sensible starting points for a symmetric SPD system.
     */
    struct SolverInput
    {
        /*! \brief Human-readable name for this configuration (used in reports). */
        std::string name = "run";

        /*! \brief Outer Krylov solver selection. */
        KrylovKind solver = KrylovKind::CG;

        /*! \brief Preconditioner category (currently always AMG). */
        PreconditionerKind preconditioner = PreconditionerKind::AMG;

        /*! \brief AMG method used when preconditioner == AMG. */
        AMGKind amg_type = AMGKind::RugeStueben;

        // ---- AMG hierarchy knobs (relevance depends on preconditioner) ----
        /*! \brief Coarsening strategy (RS, SA, UA). Only applied when
         *  has_coarsening_strategy is true; otherwise each AMG method keeps its library
         *  default (Greedy). NOTE: SA-AMG + PMIS is a valid combination but currently hits a
         *  rocALUTION bug (smoothed-aggregation prolongation) that faults on some matrices. */
        CoarseningStrategy coarsening_strategy     = CoarseningStrategy::Greedy;
        bool               has_coarsening_strategy = false;
        /*! \brief Interpolation type (RS). */
        InterpolationType interpolation_type = InterpolationType::ExtPI;
        /*! \brief Strength-of-connection threshold (RS). Applied only when > 0. */
        float strength_threshold = 0.0f;
        /*! \brief Limit FF interpolation (RS). */
        bool ff1_limit = false;
        /*! \brief Coupling strength (SA, UA). */
        ValueType coupling_strength = 0.001;
        /*! \brief Interpolation relaxation (SA). */
        ValueType interp_relax = 2.0 / 3.0;
        /*! \brief Lumping strategy (SA). */
        LumpingStrategy lumping_strategy = LumpingStrategy::AddWeakConnections;
        /*! \brief Over-interpolation parameter (UA). */
        ValueType over_interp = 2.0;
        /*! \brief Beta factor (Pairwise, deprecated). */
        ValueType beta = 0.25;
        /*! \brief Target coarsening factor (Pairwise, deprecated). */
        double coarsening_factor = 4.0;
        /*! \brief Aggregation ordering (Pairwise, deprecated). See _aggregation_ordering. */
        unsigned int ordering = 0;
        /*! \brief Maximum number of unknowns on the coarsest level. */
        int coarsest_level = 200;
        /*! \brief Number of coarsest levels forced onto the host backend (0 = none). */
        int host_levels = 0;

        // ---- Multigrid cycle / smoothing ----
        /*! \brief Multigrid cycle (see _cycle: Vcycle/Wcycle/Kcycle/Fcycle). */
        unsigned int cycle = Vcycle;
        /*! \brief Number of pre-smoothing steps. */
        int pre_smooth_iter = 1;
        /*! \brief Number of post-smoothing steps. */
        int post_smooth_iter = 2;
        /*! \brief Enable intergrid transfer scaling. */
        bool scaling = false;
        /*! \brief Full (vs. truncated) K-cycle; only relevant when cycle == Kcycle. Applied
         *  only when has_kcycle_full is true, otherwise the library default is kept. */
        bool kcycle_full     = false;
        bool has_kcycle_full = false;

        // ---- Storage-format perf knobs (applied only when the matching has_* is set) ----
        /*! \brief Operator storage format for the AMG hierarchy
         *  (DENSE|CSR|MCSR|BCSR|COO|DIA|ELL|HYB). Library default is CSR. */
        unsigned int operator_format     = CSR;
        int          operator_blockdim   = 1; /**< block dimension, only used for BCSR. */
        bool         has_operator_format = false;
        /*! \brief Smoother operator storage format. Library default is CSR. */
        unsigned int smoother_format     = CSR;
        bool         has_smoother_format = false;

        // ---- Smoother (manual path used when != Default) ----
        SmootherKind smoother       = SmootherKind::Default;
        ValueType    smoother_relax = 1.0; /**< FixedPoint relaxation for the smoother. */

        // ---- Coarse-grid solver (manual path used when != Default) ----
        CoarseSolverKind coarse_solver         = CoarseSolverKind::Default;
        double           coarse_solver_tol     = 1e-8;
        int              coarse_solver_max_iter = 1000;

        // ---- Outer PCG stopping criteria ----
        double abs_tol  = 1e-8;
        double rel_tol  = 1e-8;
        double div_tol  = 1e+8;
        int    max_iter = 10000;

        /*! \brief Number of times to repeat the solve (reusing the build) for timing stats. */
        int repeats = 1;
    };

    /*! \brief Metrics collected from a single solve() invocation. */
    struct SolverOutput
    {
        /*! \brief Echoed configuration name. */
        std::string name = "run";

        /*! \brief Number of levels in the AMG hierarchy. */
        int num_levels = 0;

        /*! \brief Time spent in Build() (hierarchy + smoothers + coarse solver), seconds. */
        double build_time_sec = 0.0;

        /*! \brief Solve timing over `repeats`, seconds. */
        double solve_time_mean_sec = 0.0;
        double solve_time_min_sec  = 0.0;
        double solve_time_max_sec  = 0.0;

        /*! \brief Total time to solution: build once + one (mean) solve, seconds. */
        double total_time_sec = 0.0;

        /*! \brief Iterations of the last solve. */
        int iterations = 0;

        /*! \brief Final residual reported by the iteration control. */
        double final_residual = 0.0;

        /*! \brief Solver status text (no-criteria / abs-tol / rel-tol / diverged / max-iter). */
        std::string solver_status = "";

        /*! \brief True if the solve reached an absolute or relative tolerance. */
        bool converged = false;

        /*! \brief L2 error ||x - exact||, only meaningful when an exact solution was provided. */
        double error_l2 = 0.0;
        /*! \brief Whether error_l2 was computed. */
        bool has_error_l2 = false;

        /*! \brief Error message; empty on success. A non-empty value means the run failed. */
        std::string error = "";
    };

    /*! \brief Solve A x = rhs with PCG preconditioned by the requested AMG configuration.
     *
     *  The operator and vectors are expected to already reside on the desired backend
     *  (typically the accelerator). The function never mutates \p A, \p rhs, or \p x0; it
     *  works on an internal copy of \p x0 as the initial guess.
     *
     *  The entire body is wrapped in a try/catch: on any failure the returned
     *  SolverOutput has a non-empty \p error and \p converged == false, so a single bad
     *  configuration never aborts a batch.
     *
     *  \param A     System operator (must already be built / on backend).
     *  \param rhs   Right-hand side.
     *  \param x0    Initial guess (copied internally; reset before every repeat).
     *  \param in    Fully-resolved configuration.
     *  \param exact Optional exact solution; when non-null, error_l2 is computed.
     */
    SolverOutput solve(const LocalMatrix<ValueType>& A,
                       const LocalVector<ValueType>& rhs,
                       const LocalVector<ValueType>& x0,
                       const SolverInput&            in,
                       const LocalVector<ValueType>* exact = nullptr);

    // ---- String <-> enum helpers (shared with the optimization box's JSON mapping) ----

    bool parse_solver(const std::string& s, KrylovKind& out);
    bool parse_preconditioner(const std::string& s, PreconditionerKind& out);
    bool parse_amg_type(const std::string& s, AMGKind& out);
    bool parse_coarsening(const std::string& s, CoarseningStrategy& out);
    bool parse_interpolation(const std::string& s, InterpolationType& out);
    bool parse_lumping(const std::string& s, LumpingStrategy& out);
    bool parse_cycle(const std::string& s, unsigned int& out);
    bool parse_ordering(const std::string& s, unsigned int& out);
    bool parse_smoother(const std::string& s, SmootherKind& out);
    bool parse_coarse_solver(const std::string& s, CoarseSolverKind& out);
    bool parse_matrix_format(const std::string& s, unsigned int& out);

    const char* to_string(KrylovKind v);
    const char* to_string(PreconditionerKind v);
    const char* to_string(AMGKind v);
    const char* to_string(CoarseningStrategy v);
    const char* to_string(InterpolationType v);
    const char* to_string(LumpingStrategy v);
    const char* cycle_to_string(unsigned int v);
    const char* ordering_to_string(unsigned int v);
    const char* matrix_format_to_string(unsigned int v);
    const char* to_string(SmootherKind v);
    const char* to_string(CoarseSolverKind v);

} // namespace rocalution_opt

#endif // ROCALUTION_SOLVER_HPP_SAMPLE_
