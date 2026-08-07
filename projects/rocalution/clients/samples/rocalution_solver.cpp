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

#include "rocalution_solver.hpp"

#include <algorithm>
#include <exception>
#include <limits>
#include <memory>

// PairwiseAMG setters are deprecated in the library but still supported here; silence the
// deprecation warnings only around those calls.
#if defined(__GNUC__)
#define ROCALUTION_OPT_PUSH_DEPRECATED \
    _Pragma("GCC diagnostic push") _Pragma("GCC diagnostic ignored \"-Wdeprecated-declarations\"")
#define ROCALUTION_OPT_POP_DEPRECATED _Pragma("GCC diagnostic pop")
#else
#define ROCALUTION_OPT_PUSH_DEPRECATED
#define ROCALUTION_OPT_POP_DEPRECATED
#endif

namespace rocalution_opt
{
    using Mat   = LocalMatrix<ValueType>;
    using Vec   = LocalVector<ValueType>;
    using Slv   = Solver<Mat, Vec, ValueType>;
    using ItSlv = IterativeLinearSolver<Mat, Vec, ValueType>;
    using Prec  = Preconditioner<Mat, Vec, ValueType>;
    using AMG   = BaseAMG<Mat, Vec, ValueType>;

    namespace
    {
        // Create the inner preconditioner used inside a FixedPoint smoother.
        // Returns nullptr for SmootherKind::Default (caller should not build a manual smoother).
        Prec* create_inner_precond(SmootherKind kind)
        {
            switch(kind)
            {
            case SmootherKind::Jacobi:
                return new Jacobi<Mat, Vec, ValueType>;
            case SmootherKind::GS:
                return new GS<Mat, Vec, ValueType>;
            case SmootherKind::SGS:
                return new SGS<Mat, Vec, ValueType>;
            case SmootherKind::MCGS:
                return new MultiColoredGS<Mat, Vec, ValueType>;
            case SmootherKind::MCSGS:
                return new MultiColoredSGS<Mat, Vec, ValueType>;
            case SmootherKind::MCILU:
                return new MultiColoredILU<Mat, Vec, ValueType>;
            case SmootherKind::ILU:
                return new ILU<Mat, Vec, ValueType>;
            case SmootherKind::IC:
                return new IC<Mat, Vec, ValueType>;
            case SmootherKind::FSAI:
                return new FSAI<Mat, Vec, ValueType>;
            case SmootherKind::Default:
            default:
                return nullptr;
            }
        }

        // Create the coarse-grid solver. Iterative solvers are initialized with the
        // requested tolerance / iteration budget. Returns nullptr for Default.
        Slv* create_coarse_solver(const SolverInput& in)
        {
            const double tol      = in.coarse_solver_tol;
            const int    max_iter = in.coarse_solver_max_iter;

            switch(in.coarse_solver)
            {
            case CoarseSolverKind::CG:
            {
                auto* s = new CG<Mat, Vec, ValueType>;
                s->Init(tol, tol, 1e+8, max_iter);
                s->Verbose(0);
                return s;
            }
            case CoarseSolverKind::GMRES:
            {
                auto* s = new GMRES<Mat, Vec, ValueType>;
                s->Init(tol, tol, 1e+8, max_iter);
                s->Verbose(0);
                return s;
            }
            case CoarseSolverKind::FGMRES:
            {
                auto* s = new FGMRES<Mat, Vec, ValueType>;
                s->Init(tol, tol, 1e+8, max_iter);
                s->Verbose(0);
                return s;
            }
            case CoarseSolverKind::BiCGStab:
            {
                auto* s = new BiCGStab<Mat, Vec, ValueType>;
                s->Init(tol, tol, 1e+8, max_iter);
                s->Verbose(0);
                return s;
            }
            case CoarseSolverKind::FCG:
            {
                auto* s = new FCG<Mat, Vec, ValueType>;
                s->Init(tol, tol, 1e+8, max_iter);
                s->Verbose(0);
                return s;
            }
            case CoarseSolverKind::QMRCGStab:
            {
                auto* s = new QMRCGStab<Mat, Vec, ValueType>;
                s->Init(tol, tol, 1e+8, max_iter);
                s->Verbose(0);
                return s;
            }
            case CoarseSolverKind::LU:
            {
                auto* s = new LU<Mat, Vec, ValueType>;
                s->Verbose(0);
                return s;
            }
            case CoarseSolverKind::QR:
            {
                auto* s = new QR<Mat, Vec, ValueType>;
                s->Verbose(0);
                return s;
            }
            case CoarseSolverKind::Default:
            default:
                return nullptr;
            }
        }

        std::string status_text(int status)
        {
            switch(status)
            {
            case 1:
                return "abs-tol";
            case 2:
                return "rel-tol";
            case 3:
                return "diverged";
            case 4:
                return "max-iter";
            case 0:
            default:
                return "no-criteria";
            }
        }

        // Apply the multigrid / hierarchy settings that are common to every AMG subtype.
        void apply_common_amg_settings(AMG& amg, const SolverInput& in)
        {
            amg.SetCoarsestLevel(in.coarsest_level);
            amg.SetCycle(in.cycle);
            amg.SetSmootherPreIter(in.pre_smooth_iter);
            amg.SetSmootherPostIter(in.post_smooth_iter);
            amg.SetScaling(in.scaling);

            // Storage-format / K-cycle perf knobs. These must be set before the build, which
            // is where apply_common_amg_settings() runs. Only applied when requested so the
            // library defaults are otherwise preserved.
            if(in.has_smoother_format)
            {
                amg.SetDefaultSmootherFormat(in.smoother_format);
            }
            if(in.has_operator_format)
            {
                amg.SetOperatorFormat(in.operator_format, in.operator_blockdim);
            }
            if(in.has_kcycle_full)
            {
                amg.SetKcycleFull(in.kcycle_full);
            }

            amg.Verbose(0);
        }

        // Runs the outer Krylov solver preconditioned by the (already type-configured) AMG
        // object and fills the metrics. Owns and frees any manual smoother / coarse-solver
        // objects it allocates. Templated on the outer solver type (CG, GMRES, ...).
        template <typename Outer>
        SolverOutput run_amg(AMG&               amg,
                             const Mat&         A,
                             const Vec&         rhs,
                             const Vec&         x0,
                             const SolverInput& in,
                             const Vec*         exact)
        {
            SolverOutput out;
            out.name = in.name;

            const bool manual_smoother = (in.smoother != SmootherKind::Default);
            const bool manual_coarse   = (in.coarse_solver != CoarseSolverKind::Default);
            const bool manual          = manual_smoother || manual_coarse;

            // Objects that must outlive Build()/Solve() and be freed after Clear().
            ItSlv**             sm_array = nullptr;
            int                 sm_count = 0;
            std::vector<Slv*>   inner_precs;
            Slv*                coarse = nullptr;

            Outer ls;

            auto cleanup = [&]() {
                ls.Clear();
                if(sm_array != nullptr)
                {
                    for(int i = 0; i < sm_count; ++i)
                    {
                        delete sm_array[i];
                    }
                    delete[] sm_array;
                    sm_array = nullptr;
                }
                for(Slv* p : inner_precs)
                {
                    delete p;
                }
                inner_precs.clear();
                delete coarse;
                coarse = nullptr;
            };

            try
            {
                apply_common_amg_settings(amg, in);

                double build_tick = rocalution_time();

                if(manual)
                {
                    // Manual smoother / coarse-solver path (see amg.cpp): the hierarchy must
                    // exist before per-level smoothers can be attached.
                    amg.SetOperator(A);

                    if(manual_smoother)
                    {
                        amg.SetManualSmoothers(true);
                    }
                    if(manual_coarse)
                    {
                        amg.SetManualSolver(true);
                    }

                    amg.BuildHierarchy();

                    const int levels = amg.GetNumLevels();

                    if(manual_smoother && levels > 1)
                    {
                        sm_count = levels - 1;
                        sm_array = new ItSlv*[sm_count];

                        for(int i = 0; i < sm_count; ++i)
                        {
                            auto* fp = new FixedPoint<Mat, Vec, ValueType>;
                            fp->SetRelaxation(in.smoother_relax);

                            Prec* inner = create_inner_precond(in.smoother);
                            if(inner != nullptr)
                            {
                                inner_precs.push_back(inner);
                                fp->SetPreconditioner(*inner);
                            }

                            fp->Verbose(0);
                            sm_array[i] = fp;
                        }

                        amg.SetSmoother(sm_array);
                    }

                    if(manual_coarse)
                    {
                        coarse = create_coarse_solver(in);
                        if(coarse != nullptr)
                        {
                            amg.SetSolver(*coarse);
                        }
                    }

                    // Re-assert smoothing iterations (SetSmoother may reset internal state).
                    amg.SetSmootherPreIter(in.pre_smooth_iter);
                    amg.SetSmootherPostIter(in.post_smooth_iter);
                }

                // Wire the AMG as the PCG preconditioner and build.
                ls.SetPreconditioner(amg);
                ls.SetOperator(A);
                ls.Build();

                // Placement of the coarsest levels on the host requires a built solver.
                if(in.host_levels > 0)
                {
                    amg.SetHostLevels(in.host_levels);
                }

                double build_tack = rocalution_time();
                out.build_time_sec = (build_tack - build_tick) / 1e6;

                out.num_levels = amg.GetNumLevels();

                ls.Init(in.abs_tol, in.rel_tol, in.div_tol, in.max_iter);
                ls.Verbose(0);

                // Solve, repeating for timing statistics while reusing the build.
                Vec x;
                x.CloneFrom(x0);

                const int    repeats  = std::max(1, in.repeats);
                double       t_sum    = 0.0;
                double       t_min    = std::numeric_limits<double>::max();
                double       t_max    = 0.0;

                for(int r = 0; r < repeats; ++r)
                {
                    x.CopyFrom(x0);

                    double solve_tick = rocalution_time();
                    ls.Solve(rhs, &x);
                    double solve_tack = rocalution_time();

                    double t = (solve_tack - solve_tick) / 1e6;
                    t_sum += t;
                    t_min = std::min(t_min, t);
                    t_max = std::max(t_max, t);
                }

                out.solve_time_mean_sec = t_sum / repeats;
                out.solve_time_min_sec  = t_min;
                out.solve_time_max_sec  = t_max;
                out.total_time_sec      = out.build_time_sec + out.solve_time_mean_sec;

                out.iterations     = ls.GetIterationCount();
                out.final_residual = ls.GetCurrentResidual();

                int status         = ls.GetSolverStatus();
                out.solver_status  = status_text(status);
                out.converged      = (status == 1 || status == 2);

                if(exact != nullptr)
                {
                    Vec e;
                    e.CloneFrom(*exact);
                    e.ScaleAdd(-1.0, x); // e = x - exact
                    out.error_l2     = e.Norm();
                    out.has_error_l2 = true;
                }

                cleanup();
            }
            catch(...)
            {
                cleanup();
                throw;
            }

        return out;
    }

        // Select the outer Krylov solver type from the config and run the AMG path.
        SolverOutput run_dispatch_amg(AMG&               amg,
                                      const Mat&         A,
                                      const Vec&         rhs,
                                      const Vec&         x0,
                                      const SolverInput& in,
                                      const Vec*         exact)
        {
            switch(in.solver)
            {
            case KrylovKind::CG:
                return run_amg<CG<Mat, Vec, ValueType>>(amg, A, rhs, x0, in, exact);
            case KrylovKind::FCG:
                return run_amg<FCG<Mat, Vec, ValueType>>(amg, A, rhs, x0, in, exact);
            case KrylovKind::CR:
                return run_amg<CR<Mat, Vec, ValueType>>(amg, A, rhs, x0, in, exact);
            case KrylovKind::GMRES:
                return run_amg<GMRES<Mat, Vec, ValueType>>(amg, A, rhs, x0, in, exact);
            case KrylovKind::FGMRES:
                return run_amg<FGMRES<Mat, Vec, ValueType>>(amg, A, rhs, x0, in, exact);
            case KrylovKind::BiCGStab:
                return run_amg<BiCGStab<Mat, Vec, ValueType>>(amg, A, rhs, x0, in, exact);
            case KrylovKind::QMRCGStab:
                return run_amg<QMRCGStab<Mat, Vec, ValueType>>(amg, A, rhs, x0, in, exact);
            case KrylovKind::IDR:
                return run_amg<IDR<Mat, Vec, ValueType>>(amg, A, rhs, x0, in, exact);
            }
            return run_amg<CG<Mat, Vec, ValueType>>(amg, A, rhs, x0, in, exact);
        }

        // Build a single-level preconditioner object (nullptr for None). Caller owns it.
        Prec* create_simple_precond(PreconditionerKind kind)
        {
            switch(kind)
            {
            case PreconditionerKind::Jacobi:
                return new Jacobi<Mat, Vec, ValueType>;
            case PreconditionerKind::GS:
                return new GS<Mat, Vec, ValueType>;
            case PreconditionerKind::SGS:
                return new SGS<Mat, Vec, ValueType>;
            case PreconditionerKind::MCGS:
                return new MultiColoredGS<Mat, Vec, ValueType>;
            case PreconditionerKind::MCSGS:
                return new MultiColoredSGS<Mat, Vec, ValueType>;
            case PreconditionerKind::MCILU:
                return new MultiColoredILU<Mat, Vec, ValueType>;
            case PreconditionerKind::ILU:
                return new ILU<Mat, Vec, ValueType>;
            case PreconditionerKind::IC:
                return new IC<Mat, Vec, ValueType>;
            case PreconditionerKind::FSAI:
                return new FSAI<Mat, Vec, ValueType>;
            case PreconditionerKind::None:
            case PreconditionerKind::AMG:
            default:
                return nullptr;
            }
        }

        // Runs the outer Krylov solver with an optional single-level preconditioner
        // (precond == nullptr means unpreconditioned). Templated on the outer solver type.
        template <typename Outer>
        SolverOutput run_simple(Slv*               precond,
                                const Mat&         A,
                                const Vec&         rhs,
                                const Vec&         x0,
                                const SolverInput& in,
                                const Vec*         exact)
        {
            SolverOutput out;
            out.name = in.name;

            Outer ls;

            try
            {
                if(precond != nullptr)
                {
                    ls.SetPreconditioner(*precond);
                }
                ls.SetOperator(A);

                double build_tick = rocalution_time();
                ls.Build();
                double build_tack = rocalution_time();
                out.build_time_sec = (build_tack - build_tick) / 1e6;
                out.num_levels     = 0;

                ls.Init(in.abs_tol, in.rel_tol, in.div_tol, in.max_iter);
                ls.Verbose(0);

                Vec x;
                x.CloneFrom(x0);

                const int repeats = std::max(1, in.repeats);
                double    t_sum   = 0.0;
                double    t_min   = std::numeric_limits<double>::max();
                double    t_max   = 0.0;

                for(int r = 0; r < repeats; ++r)
                {
                    x.CopyFrom(x0);

                    double solve_tick = rocalution_time();
                    ls.Solve(rhs, &x);
                    double solve_tack = rocalution_time();

                    double t = (solve_tack - solve_tick) / 1e6;
                    t_sum += t;
                    t_min = std::min(t_min, t);
                    t_max = std::max(t_max, t);
                }

                out.solve_time_mean_sec = t_sum / repeats;
                out.solve_time_min_sec  = t_min;
                out.solve_time_max_sec  = t_max;
                out.total_time_sec      = out.build_time_sec + out.solve_time_mean_sec;

                out.iterations     = ls.GetIterationCount();
                out.final_residual = ls.GetCurrentResidual();

                int status        = ls.GetSolverStatus();
                out.solver_status = status_text(status);
                out.converged     = (status == 1 || status == 2);

                if(exact != nullptr)
                {
                    Vec e;
                    e.CloneFrom(*exact);
                    e.ScaleAdd(-1.0, x); // e = x - exact
                    out.error_l2     = e.Norm();
                    out.has_error_l2 = true;
                }

                ls.Clear();
            }
            catch(...)
            {
                ls.Clear();
                throw;
            }

            return out;
        }

        // Select the outer Krylov solver type and run the single-level / unpreconditioned path.
        SolverOutput run_dispatch_simple(Slv*               precond,
                                         const Mat&         A,
                                         const Vec&         rhs,
                                         const Vec&         x0,
                                         const SolverInput& in,
                                         const Vec*         exact)
        {
            switch(in.solver)
            {
            case KrylovKind::CG:
                return run_simple<CG<Mat, Vec, ValueType>>(precond, A, rhs, x0, in, exact);
            case KrylovKind::FCG:
                return run_simple<FCG<Mat, Vec, ValueType>>(precond, A, rhs, x0, in, exact);
            case KrylovKind::CR:
                return run_simple<CR<Mat, Vec, ValueType>>(precond, A, rhs, x0, in, exact);
            case KrylovKind::GMRES:
                return run_simple<GMRES<Mat, Vec, ValueType>>(precond, A, rhs, x0, in, exact);
            case KrylovKind::FGMRES:
                return run_simple<FGMRES<Mat, Vec, ValueType>>(precond, A, rhs, x0, in, exact);
            case KrylovKind::BiCGStab:
                return run_simple<BiCGStab<Mat, Vec, ValueType>>(precond, A, rhs, x0, in, exact);
            case KrylovKind::QMRCGStab:
                return run_simple<QMRCGStab<Mat, Vec, ValueType>>(precond, A, rhs, x0, in, exact);
            case KrylovKind::IDR:
                return run_simple<IDR<Mat, Vec, ValueType>>(precond, A, rhs, x0, in, exact);
            }
            return run_simple<CG<Mat, Vec, ValueType>>(precond, A, rhs, x0, in, exact);
        }

    } // namespace

    SolverOutput solve(const Mat&         A,
                       const Vec&         rhs,
                       const Vec&         x0,
                       const SolverInput& in,
                       const Vec*         exact)
    {
        SolverOutput out;
        out.name = in.name;

        try
        {
            // Non-AMG preconditioners (incl. None) take the simple single-level path.
            if(in.preconditioner != PreconditionerKind::AMG)
            {
                std::unique_ptr<Prec> p(create_simple_precond(in.preconditioner));
                return run_dispatch_simple(p.get(), A, rhs, x0, in, exact);
            }

            // AMG preconditioner: the specific method is selected by amg_type.
            switch(in.amg_type)
            {
            case AMGKind::RugeStueben:
            {
                RugeStuebenAMG<Mat, Vec, ValueType> p;
                if(in.has_coarsening_strategy)
                {
                    p.SetCoarseningStrategy(in.coarsening_strategy);
                }
                p.SetInterpolationType(in.interpolation_type);
                p.SetInterpolationFF1Limit(in.ff1_limit);
                if(in.strength_threshold > 0.0f)
                {
                    p.SetStrengthThreshold(in.strength_threshold);
                }
                return run_dispatch_amg(p, A, rhs, x0, in, exact);
            }
            case AMGKind::SA:
            {
                SAAMG<Mat, Vec, ValueType> p;
                if(in.has_coarsening_strategy)
                {
                    p.SetCoarseningStrategy(in.coarsening_strategy);
                }
                p.SetCouplingStrength(in.coupling_strength);
                p.SetInterpRelax(in.interp_relax);
                p.SetLumpingStrategy(in.lumping_strategy);
                return run_dispatch_amg(p, A, rhs, x0, in, exact);
            }
            case AMGKind::UA:
            {
                UAAMG<Mat, Vec, ValueType> p;
                if(in.has_coarsening_strategy)
                {
                    p.SetCoarseningStrategy(in.coarsening_strategy);
                }
                p.SetCouplingStrength(in.coupling_strength);
                p.SetOverInterp(in.over_interp);
                return run_dispatch_amg(p, A, rhs, x0, in, exact);
            }
            case AMGKind::Pairwise:
            {
                ROCALUTION_OPT_PUSH_DEPRECATED
                PairwiseAMG<Mat, Vec, ValueType> p;
                p.SetBeta(in.beta);
                p.SetCoarseningFactor(in.coarsening_factor);
                p.SetOrdering(in.ordering);
                ROCALUTION_OPT_POP_DEPRECATED
                return run_dispatch_amg(p, A, rhs, x0, in, exact);
            }
            }
        }
        catch(const std::exception& e)
        {
            out.error     = e.what();
            out.converged = false;
        }
        catch(...)
        {
            out.error     = "unknown error during solve";
            out.converged = false;
        }

        return out;
    }

    // ---------------------------------------------------------------------------
    // String <-> enum helpers
    // ---------------------------------------------------------------------------

    bool parse_solver(const std::string& s, KrylovKind& out)
    {
        if(s == "CG")
            out = KrylovKind::CG;
        else if(s == "FCG")
            out = KrylovKind::FCG;
        else if(s == "CR")
            out = KrylovKind::CR;
        else if(s == "GMRES")
            out = KrylovKind::GMRES;
        else if(s == "FGMRES")
            out = KrylovKind::FGMRES;
        else if(s == "BiCGStab")
            out = KrylovKind::BiCGStab;
        else if(s == "QMRCGStab")
            out = KrylovKind::QMRCGStab;
        else if(s == "IDR")
            out = KrylovKind::IDR;
        else
            return false;
        return true;
    }

    bool parse_preconditioner(const std::string& s, PreconditionerKind& out)
    {
        if(s == "None")
            out = PreconditionerKind::None;
        else if(s == "AMG")
            out = PreconditionerKind::AMG;
        else if(s == "Jacobi")
            out = PreconditionerKind::Jacobi;
        else if(s == "GS")
            out = PreconditionerKind::GS;
        else if(s == "SGS")
            out = PreconditionerKind::SGS;
        else if(s == "MCGS")
            out = PreconditionerKind::MCGS;
        else if(s == "MCSGS")
            out = PreconditionerKind::MCSGS;
        else if(s == "MCILU")
            out = PreconditionerKind::MCILU;
        else if(s == "ILU")
            out = PreconditionerKind::ILU;
        else if(s == "IC")
            out = PreconditionerKind::IC;
        else if(s == "FSAI")
            out = PreconditionerKind::FSAI;
        else
            return false;
        return true;
    }

    bool parse_amg_type(const std::string& s, AMGKind& out)
    {
        if(s == "RugeStueben" || s == "RugeStuebenAMG" || s == "RS")
        {
            out = AMGKind::RugeStueben;
        }
        else if(s == "SA" || s == "SAAMG")
        {
            out = AMGKind::SA;
        }
        else if(s == "UA" || s == "UAAMG")
        {
            out = AMGKind::UA;
        }
        else if(s == "Pairwise" || s == "PairwiseAMG" || s == "PW")
        {
            out = AMGKind::Pairwise;
        }
        else
        {
            return false;
        }
        return true;
    }

    bool parse_coarsening(const std::string& s, CoarseningStrategy& out)
    {
        if(s == "Greedy")
        {
            out = CoarseningStrategy::Greedy;
        }
        else if(s == "PMIS")
        {
            out = CoarseningStrategy::PMIS;
        }
        else
        {
            return false;
        }
        return true;
    }

    bool parse_interpolation(const std::string& s, InterpolationType& out)
    {
        if(s == "Direct")
        {
            out = InterpolationType::Direct;
        }
        else if(s == "ExtPI")
        {
            out = InterpolationType::ExtPI;
        }
        else
        {
            return false;
        }
        return true;
    }

    bool parse_lumping(const std::string& s, LumpingStrategy& out)
    {
        if(s == "AddWeakConnections")
        {
            out = LumpingStrategy::AddWeakConnections;
        }
        else if(s == "SubtractWeakConnections")
        {
            out = LumpingStrategy::SubtractWeakConnections;
        }
        else
        {
            return false;
        }
        return true;
    }

    bool parse_cycle(const std::string& s, unsigned int& out)
    {
        if(s == "V" || s == "Vcycle")
        {
            out = Vcycle;
        }
        else if(s == "W" || s == "Wcycle")
        {
            out = Wcycle;
        }
        else if(s == "K" || s == "Kcycle")
        {
            out = Kcycle;
        }
        else if(s == "F" || s == "Fcycle")
        {
            out = Fcycle;
        }
        else
        {
            return false;
        }
        return true;
    }

    bool parse_ordering(const std::string& s, unsigned int& out)
    {
        if(s == "NoOrdering")
        {
            out = NoOrdering;
        }
        else if(s == "Connectivity")
        {
            out = Connectivity;
        }
        else if(s == "CMK")
        {
            out = CMK;
        }
        else if(s == "RCMK")
        {
            out = RCMK;
        }
        else if(s == "MIS")
        {
            out = MIS;
        }
        else if(s == "MultiColoring")
        {
            out = MultiColoring;
        }
        else
        {
            return false;
        }
        return true;
    }

    bool parse_smoother(const std::string& s, SmootherKind& out)
    {
        if(s == "Default")
            out = SmootherKind::Default;
        else if(s == "Jacobi")
            out = SmootherKind::Jacobi;
        else if(s == "GS")
            out = SmootherKind::GS;
        else if(s == "SGS")
            out = SmootherKind::SGS;
        else if(s == "MCGS")
            out = SmootherKind::MCGS;
        else if(s == "MCSGS")
            out = SmootherKind::MCSGS;
        else if(s == "MCILU")
            out = SmootherKind::MCILU;
        else if(s == "ILU")
            out = SmootherKind::ILU;
        else if(s == "IC")
            out = SmootherKind::IC;
        else if(s == "FSAI")
            out = SmootherKind::FSAI;
        else
            return false;
        return true;
    }

    bool parse_matrix_format(const std::string& s, unsigned int& out)
    {
        if(s == "DENSE")
            out = DENSE;
        else if(s == "CSR")
            out = CSR;
        else if(s == "MCSR")
            out = MCSR;
        else if(s == "BCSR")
            out = BCSR;
        else if(s == "COO")
            out = COO;
        else if(s == "DIA")
            out = DIA;
        else if(s == "ELL")
            out = ELL;
        else if(s == "HYB")
            out = HYB;
        else
            return false;
        return true;
    }

    bool parse_coarse_solver(const std::string& s, CoarseSolverKind& out)
    {
        if(s == "Default")
            out = CoarseSolverKind::Default;
        else if(s == "CG")
            out = CoarseSolverKind::CG;
        else if(s == "GMRES")
            out = CoarseSolverKind::GMRES;
        else if(s == "FGMRES")
            out = CoarseSolverKind::FGMRES;
        else if(s == "BiCGStab")
            out = CoarseSolverKind::BiCGStab;
        else if(s == "FCG")
            out = CoarseSolverKind::FCG;
        else if(s == "QMRCGStab")
            out = CoarseSolverKind::QMRCGStab;
        else if(s == "LU")
            out = CoarseSolverKind::LU;
        else if(s == "QR")
            out = CoarseSolverKind::QR;
        else
            return false;
        return true;
    }

    const char* to_string(KrylovKind v)
    {
        switch(v)
        {
        case KrylovKind::CG:
            return "CG";
        case KrylovKind::FCG:
            return "FCG";
        case KrylovKind::CR:
            return "CR";
        case KrylovKind::GMRES:
            return "GMRES";
        case KrylovKind::FGMRES:
            return "FGMRES";
        case KrylovKind::BiCGStab:
            return "BiCGStab";
        case KrylovKind::QMRCGStab:
            return "QMRCGStab";
        case KrylovKind::IDR:
            return "IDR";
        }
        return "unknown";
    }

    const char* to_string(PreconditionerKind v)
    {
        switch(v)
        {
        case PreconditionerKind::None:
            return "None";
        case PreconditionerKind::AMG:
            return "AMG";
        case PreconditionerKind::Jacobi:
            return "Jacobi";
        case PreconditionerKind::GS:
            return "GS";
        case PreconditionerKind::SGS:
            return "SGS";
        case PreconditionerKind::MCGS:
            return "MCGS";
        case PreconditionerKind::MCSGS:
            return "MCSGS";
        case PreconditionerKind::MCILU:
            return "MCILU";
        case PreconditionerKind::ILU:
            return "ILU";
        case PreconditionerKind::IC:
            return "IC";
        case PreconditionerKind::FSAI:
            return "FSAI";
        }
        return "unknown";
    }

    const char* to_string(AMGKind v)
    {
        switch(v)
        {
        case AMGKind::RugeStueben:
            return "RugeStueben";
        case AMGKind::SA:
            return "SA";
        case AMGKind::UA:
            return "UA";
        case AMGKind::Pairwise:
            return "Pairwise";
        }
        return "unknown";
    }

    const char* to_string(CoarseningStrategy v)
    {
        switch(v)
        {
        case CoarseningStrategy::Greedy:
            return "Greedy";
        case CoarseningStrategy::PMIS:
            return "PMIS";
        }
        return "unknown";
    }

    const char* to_string(InterpolationType v)
    {
        switch(v)
        {
        case InterpolationType::Direct:
            return "Direct";
        case InterpolationType::ExtPI:
            return "ExtPI";
        }
        return "unknown";
    }

    const char* to_string(LumpingStrategy v)
    {
        switch(v)
        {
        case LumpingStrategy::AddWeakConnections:
            return "AddWeakConnections";
        case LumpingStrategy::SubtractWeakConnections:
            return "SubtractWeakConnections";
        }
        return "unknown";
    }

    const char* cycle_to_string(unsigned int v)
    {
        switch(v)
        {
        case Vcycle:
            return "V";
        case Wcycle:
            return "W";
        case Kcycle:
            return "K";
        case Fcycle:
            return "F";
        }
        return "unknown";
    }

    const char* matrix_format_to_string(unsigned int v)
    {
        switch(v)
        {
        case DENSE:
            return "DENSE";
        case CSR:
            return "CSR";
        case MCSR:
            return "MCSR";
        case BCSR:
            return "BCSR";
        case COO:
            return "COO";
        case DIA:
            return "DIA";
        case ELL:
            return "ELL";
        case HYB:
            return "HYB";
        }
        return "unknown";
    }

    const char* ordering_to_string(unsigned int v)
    {
        switch(v)
        {
        case NoOrdering:
            return "NoOrdering";
        case Connectivity:
            return "Connectivity";
        case CMK:
            return "CMK";
        case RCMK:
            return "RCMK";
        case MIS:
            return "MIS";
        case MultiColoring:
            return "MultiColoring";
        }
        return "unknown";
    }

    const char* to_string(SmootherKind v)
    {
        switch(v)
        {
        case SmootherKind::Default:
            return "Default";
        case SmootherKind::Jacobi:
            return "Jacobi";
        case SmootherKind::GS:
            return "GS";
        case SmootherKind::SGS:
            return "SGS";
        case SmootherKind::MCGS:
            return "MCGS";
        case SmootherKind::MCSGS:
            return "MCSGS";
        case SmootherKind::MCILU:
            return "MCILU";
        case SmootherKind::ILU:
            return "ILU";
        case SmootherKind::IC:
            return "IC";
        case SmootherKind::FSAI:
            return "FSAI";
        }
        return "unknown";
    }

    const char* to_string(CoarseSolverKind v)
    {
        switch(v)
        {
        case CoarseSolverKind::Default:
            return "Default";
        case CoarseSolverKind::CG:
            return "CG";
        case CoarseSolverKind::GMRES:
            return "GMRES";
        case CoarseSolverKind::FGMRES:
            return "FGMRES";
        case CoarseSolverKind::BiCGStab:
            return "BiCGStab";
        case CoarseSolverKind::FCG:
            return "FCG";
        case CoarseSolverKind::QMRCGStab:
            return "QMRCGStab";
        case CoarseSolverKind::LU:
            return "LU";
        case CoarseSolverKind::QR:
            return "QR";
        }
        return "unknown";
    }

} // namespace rocalution_opt
