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
 *  \brief Optimization box driver for the PCG + AMG benchmarking tool.
 *
 *  Reads a JSON(C) config that describes a matrix (loaded exactly ONCE) plus a set of
 *  PCG + AMG configurations to benchmark. Configurations come from any combination of a
 *  single \c sweep object, a \c sweeps array, and explicit \c runs; array-valued fields
 *  are Cartesian-expanded, irrelevant parameters are dropped, and the resulting set is
 *  de-duplicated. Every resolved configuration is handed to solve() (the JSON-free solver
 *  piece) and the collected metrics are written to a results JSON file and summarized on
 *  stdout.
 *
 *  Usage:
 *    rocalution_opt <config.json> [results.json] [--dry-run]
 *
 *  --dry-run lists the resolved configurations without loading the matrix or solving.
 */

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <limits>
#include <iomanip>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include <rocalution/rocalution.hpp>

#include "rocalution_opt_json.hpp"
#include "rocalution_solver.hpp"

using namespace rocalution;
using namespace rocalution_opt;

namespace
{
    /*! \brief Global (applied-once) settings from the config. */
    struct GlobalConfig
    {
        std::string matrix;
        // auto (detect from extension) | mtx | csr (legacy binary) | rsio (binary) | binary
        std::string matrix_format = "auto";
        int         omp_threads   = 0; // 0 = leave library default
        std::string rhs           = "ones"; // ones -> rhs = A * 1
        std::string initial_guess = "zeros"; // zeros
        bool        warmup        = true; // run a discarded warm-up before timing
        int         warmup_iters  = 2; // outer iterations for each warm-up solve
    };

    // ---- Typed accessors that validate the JSON value type ----

    std::string as_string(const JsonValue& v, const std::string& key)
    {
        if(!v.is_string())
        {
            throw std::runtime_error("config field '" + key + "' must be a string");
        }
        return v.str;
    }

    double as_number(const JsonValue& v, const std::string& key)
    {
        if(!v.is_number())
        {
            throw std::runtime_error("config field '" + key + "' must be a number");
        }
        return v.number;
    }

    bool as_bool(const JsonValue& v, const std::string& key)
    {
        if(!v.is_bool())
        {
            throw std::runtime_error("config field '" + key + "' must be a boolean");
        }
        return v.boolean;
    }

    int as_int(const JsonValue& v, const std::string& key)
    {
        return static_cast<int>(as_number(v, key));
    }

    // Enum parse helper: throws with a clear message if the string is not recognized.
    template <typename Fn>
    void parse_enum(const JsonValue& v, const std::string& key, Fn parser)
    {
        std::string s = as_string(v, key);
        if(!parser(s))
        {
            throw std::runtime_error("config field '" + key + "' has invalid value '" + s + "'");
        }
    }

    void map_global(const JsonValue& obj, GlobalConfig& g)
    {
        if(const JsonValue* v = obj.find("matrix"))
        {
            g.matrix = as_string(*v, "matrix");
        }
        if(const JsonValue* v = obj.find("matrix_format"))
        {
            g.matrix_format = as_string(*v, "matrix_format");
        }
        if(const JsonValue* v = obj.find("omp_threads"))
        {
            g.omp_threads = as_int(*v, "omp_threads");
        }
        if(const JsonValue* v = obj.find("rhs"))
        {
            g.rhs = as_string(*v, "rhs");
        }
        if(const JsonValue* v = obj.find("initial_guess"))
        {
            g.initial_guess = as_string(*v, "initial_guess");
        }
        if(const JsonValue* v = obj.find("warmup"))
        {
            g.warmup = as_bool(*v, "warmup");
        }
        if(const JsonValue* v = obj.find("warmup_iters"))
        {
            g.warmup_iters = as_int(*v, "warmup_iters");
        }
    }

    // Map a flat JSON object onto a SolverInput. Unknown keys are ignored; global-only
    // keys are handled separately by map_global(). The "preconditioner" key additionally
    // accepts AMG method names (RugeStuebenAMG / SAAMG / UAAMG / PairwiseAMG), which set
    // preconditioner = AMG and the matching amg_type, so both the flat and the
    // preconditioner+amg_type spellings work.
    void map_solver_input(const JsonValue& obj, SolverInput& in)
    {
        if(const JsonValue* v = obj.find("name"))
        {
            in.name = as_string(*v, "name");
        }
        if(const JsonValue* v = obj.find("solver"))
        {
            parse_enum(*v, "solver",
                       [&](const std::string& s) { return parse_solver(s, in.solver); });
        }
        if(const JsonValue* v = obj.find("preconditioner"))
        {
            std::string s = as_string(*v, "preconditioner");
            if(parse_preconditioner(s, in.preconditioner))
            {
                // Accepted as a preconditioner category (None / AMG / Jacobi / ...).
            }
            else if(parse_amg_type(s, in.amg_type))
            {
                // Accepted as an AMG method name -> AMG preconditioner of that type.
                in.preconditioner = PreconditionerKind::AMG;
            }
            else
            {
                throw std::runtime_error("config field 'preconditioner' has invalid value '" + s
                                         + "'");
            }
        }
        if(const JsonValue* v = obj.find("amg_type"))
        {
            parse_enum(*v, "amg_type",
                       [&](const std::string& s) { return parse_amg_type(s, in.amg_type); });
        }

        // AMG hierarchy knobs
        if(const JsonValue* v = obj.find("coarsening_strategy"))
        {
            parse_enum(*v, "coarsening_strategy", [&](const std::string& s) {
                return parse_coarsening(s, in.coarsening_strategy);
            });
            in.has_coarsening_strategy = true;
        }
        if(const JsonValue* v = obj.find("interpolation_type"))
        {
            parse_enum(*v, "interpolation_type", [&](const std::string& s) {
                return parse_interpolation(s, in.interpolation_type);
            });
        }
        if(const JsonValue* v = obj.find("strength_threshold"))
        {
            in.strength_threshold = static_cast<float>(as_number(*v, "strength_threshold"));
        }
        if(const JsonValue* v = obj.find("ff1_limit"))
        {
            in.ff1_limit = as_bool(*v, "ff1_limit");
        }
        if(const JsonValue* v = obj.find("coupling_strength"))
        {
            in.coupling_strength = as_number(*v, "coupling_strength");
        }
        if(const JsonValue* v = obj.find("interp_relax"))
        {
            in.interp_relax = as_number(*v, "interp_relax");
        }
        if(const JsonValue* v = obj.find("lumping_strategy"))
        {
            parse_enum(*v, "lumping_strategy", [&](const std::string& s) {
                return parse_lumping(s, in.lumping_strategy);
            });
        }
        if(const JsonValue* v = obj.find("over_interp"))
        {
            in.over_interp = as_number(*v, "over_interp");
        }
        if(const JsonValue* v = obj.find("beta"))
        {
            in.beta = as_number(*v, "beta");
        }
        if(const JsonValue* v = obj.find("coarsening_factor"))
        {
            in.coarsening_factor = as_number(*v, "coarsening_factor");
        }
        if(const JsonValue* v = obj.find("ordering"))
        {
            parse_enum(*v, "ordering",
                       [&](const std::string& s) { return parse_ordering(s, in.ordering); });
        }
        if(const JsonValue* v = obj.find("coarsest_level"))
        {
            in.coarsest_level = as_int(*v, "coarsest_level");
        }
        if(const JsonValue* v = obj.find("host_levels"))
        {
            in.host_levels = as_int(*v, "host_levels");
        }

        // Multigrid cycle / smoothing
        if(const JsonValue* v = obj.find("cycle"))
        {
            parse_enum(*v, "cycle",
                       [&](const std::string& s) { return parse_cycle(s, in.cycle); });
        }
        if(const JsonValue* v = obj.find("pre_smooth_iter"))
        {
            in.pre_smooth_iter = as_int(*v, "pre_smooth_iter");
        }
        if(const JsonValue* v = obj.find("post_smooth_iter"))
        {
            in.post_smooth_iter = as_int(*v, "post_smooth_iter");
        }
        if(const JsonValue* v = obj.find("scaling"))
        {
            in.scaling = as_bool(*v, "scaling");
        }
        if(const JsonValue* v = obj.find("kcycle_full"))
        {
            in.kcycle_full     = as_bool(*v, "kcycle_full");
            in.has_kcycle_full = true;
        }

        // Storage-format perf knobs
        if(const JsonValue* v = obj.find("operator_format"))
        {
            parse_enum(*v, "operator_format", [&](const std::string& s) {
                return parse_matrix_format(s, in.operator_format);
            });
            in.has_operator_format = true;
        }
        if(const JsonValue* v = obj.find("operator_blockdim"))
        {
            in.operator_blockdim = as_int(*v, "operator_blockdim");
        }
        if(const JsonValue* v = obj.find("smoother_format"))
        {
            parse_enum(*v, "smoother_format", [&](const std::string& s) {
                return parse_matrix_format(s, in.smoother_format);
            });
            in.has_smoother_format = true;
        }

        // Smoother
        if(const JsonValue* v = obj.find("smoother"))
        {
            parse_enum(*v, "smoother",
                       [&](const std::string& s) { return parse_smoother(s, in.smoother); });
        }
        if(const JsonValue* v = obj.find("smoother_relax"))
        {
            in.smoother_relax = as_number(*v, "smoother_relax");
        }

        // Coarse-grid solver
        if(const JsonValue* v = obj.find("coarse_solver"))
        {
            parse_enum(*v, "coarse_solver", [&](const std::string& s) {
                return parse_coarse_solver(s, in.coarse_solver);
            });
        }
        if(const JsonValue* v = obj.find("coarse_solver_tol"))
        {
            in.coarse_solver_tol = as_number(*v, "coarse_solver_tol");
        }
        if(const JsonValue* v = obj.find("coarse_solver_max_iter"))
        {
            in.coarse_solver_max_iter = as_int(*v, "coarse_solver_max_iter");
        }

        // Outer PCG
        if(const JsonValue* v = obj.find("abs_tol"))
        {
            in.abs_tol = as_number(*v, "abs_tol");
        }
        if(const JsonValue* v = obj.find("rel_tol"))
        {
            in.rel_tol = as_number(*v, "rel_tol");
        }
        if(const JsonValue* v = obj.find("div_tol"))
        {
            in.div_tol = as_number(*v, "div_tol");
        }
        if(const JsonValue* v = obj.find("max_iter"))
        {
            in.max_iter = as_int(*v, "max_iter");
        }
        if(const JsonValue* v = obj.find("repeats"))
        {
            in.repeats = as_int(*v, "repeats");
        }
    }

    // ------------------------------------------------------------------------------------
    // Sweep expansion
    // ------------------------------------------------------------------------------------

    // One resolved (all-scalar) configuration, before it is mapped onto a SolverInput.
    struct ResolvedRun
    {
        std::map<std::string, JsonValue> fields; // scalar field values
        std::set<std::string>            axis_keys; // keys that were swept (arrays)
        std::string                      group; // optional name prefix
        bool                             has_name = false;
        std::string                      name; // explicit name, if provided
    };

    // Build a JsonValue object from a field map so map_solver_input() can consume it.
    JsonValue make_object(const std::map<std::string, JsonValue>& fields)
    {
        JsonValue o;
        o.type   = JsonValue::Type::Object;
        o.object = fields;
        return o;
    }

    // Cartesian-expand a single sweep/run object: every array-valued field becomes a swept
    // axis, every scalar field is fixed. "group" and "name" are metadata, not solver fields.
    std::vector<ResolvedRun> expand_object(const JsonValue& obj)
    {
        ResolvedRun                                          base;
        std::vector<std::pair<std::string, const JsonValue*>> axes;

        for(const auto& kv : obj.object)
        {
            const std::string& key = kv.first;
            const JsonValue&    val = kv.second;

            if(key == "group")
            {
                if(val.is_string())
                {
                    base.group = val.str;
                }
                continue;
            }
            if(key == "name")
            {
                if(val.is_string())
                {
                    base.has_name = true;
                    base.name     = val.str;
                }
                continue;
            }
            // Skip container / global-only keys so a flat top-level object can be expanded
            // directly (Phase 1 back-compat).
            if(key == "global" || key == "sweep" || key == "sweeps" || key == "runs"
               || key == "defaults" || key == "matrix" || key == "matrix_format"
               || key == "omp_threads" || key == "rhs" || key == "initial_guess")
            {
                continue;
            }

            if(val.is_array())
            {
                axes.emplace_back(key, &val);
            }
            else
            {
                base.fields[key] = val;
            }
        }

        std::vector<ResolvedRun> runs;
        runs.push_back(base);

        for(const auto& ax : axes)
        {
            std::vector<ResolvedRun> next;
            next.reserve(runs.size() * ax.second->array.size());
            for(const auto& r : runs)
            {
                for(const auto& choice : ax.second->array)
                {
                    ResolvedRun c    = r;
                    c.fields[ax.first] = choice;
                    c.axis_keys.insert(ax.first);
                    next.push_back(std::move(c));
                }
            }
            runs.swap(next);
        }

        return runs;
    }

    void merge_defaults(ResolvedRun& r, const JsonValue* defaults)
    {
        if(defaults == nullptr || !defaults->is_object())
        {
            return;
        }
        for(const auto& kv : defaults->object)
        {
            if(!r.fields.count(kv.first))
            {
                r.fields[kv.first] = kv.second;
            }
        }
    }

    // ------------------------------------------------------------------------------------
    // Relevance, signature, and naming
    // ------------------------------------------------------------------------------------

    std::string numfmt(double x)
    {
        std::ostringstream o;
        o << std::setprecision(10) << x;
        return o.str();
    }

    // Set of config-key names that actually affect the solve for this configuration. Used
    // to (a) drop swept-but-irrelevant axes from auto-generated names and (b) collapse
    // configs that differ only in irrelevant parameters.
    std::set<std::string> relevant_keys(const SolverInput& in)
    {
        std::set<std::string> keys = {"solver",
                                      "preconditioner",
                                      "abs_tol",
                                      "rel_tol",
                                      "div_tol",
                                      "max_iter",
                                      "repeats"};

        if(in.preconditioner != PreconditionerKind::AMG)
        {
            return keys;
        }

        keys.insert({"amg_type",
                     "cycle",
                     "pre_smooth_iter",
                     "post_smooth_iter",
                     "scaling",
                     "coarsest_level",
                     "host_levels",
                     "smoother",
                     "coarse_solver",
                     "operator_format",
                     "smoother_format"});
        if(in.smoother != SmootherKind::Default)
        {
            keys.insert("smoother_relax");
        }
        if(in.coarse_solver != CoarseSolverKind::Default)
        {
            keys.insert("coarse_solver_tol");
            keys.insert("coarse_solver_max_iter");
        }
        // Block dimension only matters for the BCSR operator format.
        if(in.operator_format == BCSR)
        {
            keys.insert("operator_blockdim");
        }
        // Full-vs-truncated only matters for the K-cycle.
        if(in.cycle == Kcycle)
        {
            keys.insert("kcycle_full");
        }

        switch(in.amg_type)
        {
        case AMGKind::RugeStueben:
            keys.insert({"coarsening_strategy", "interpolation_type", "strength_threshold",
                         "ff1_limit"});
            break;
        case AMGKind::SA:
            keys.insert({"coarsening_strategy", "coupling_strength", "interp_relax",
                         "lumping_strategy"});
            break;
        case AMGKind::UA:
            keys.insert({"coarsening_strategy", "coupling_strength", "over_interp"});
            break;
        case AMGKind::Pairwise:
            keys.insert({"beta", "coarsening_factor", "ordering"});
            break;
        }
        return keys;
    }

    // Canonical string over the RELEVANT resolved parameters, used for de-duplication.
    std::string canonical_signature(const SolverInput& in)
    {
        std::set<std::string> rel = relevant_keys(in);
        std::ostringstream    o;

        o << "solver=" << to_string(in.solver);
        o << ";precond=" << to_string(in.preconditioner);

        if(in.preconditioner == PreconditionerKind::AMG)
        {
            o << ";amg=" << to_string(in.amg_type);
            o << ";cycle=" << cycle_to_string(in.cycle);
            if(in.cycle == Kcycle)
            {
                o << ";kfull=" << (in.kcycle_full ? 1 : 0);
            }
            o << ";pre=" << in.pre_smooth_iter << ";post=" << in.post_smooth_iter;
            o << ";scaling=" << (in.scaling ? 1 : 0);
            o << ";coarsest=" << in.coarsest_level << ";hostlv=" << in.host_levels;
            o << ";opfmt=" << matrix_format_to_string(in.operator_format);
            if(in.operator_format == BCSR)
            {
                o << ";blk=" << in.operator_blockdim;
            }
            o << ";smfmt=" << matrix_format_to_string(in.smoother_format);
            o << ";sm=" << to_string(in.smoother);
            if(rel.count("smoother_relax"))
            {
                o << ";smrelax=" << numfmt(in.smoother_relax);
            }
            o << ";cs=" << to_string(in.coarse_solver);
            if(rel.count("coarse_solver_tol"))
            {
                o << ";cstol=" << numfmt(in.coarse_solver_tol) << ";csmax="
                  << in.coarse_solver_max_iter;
            }
            if(rel.count("coarsening_strategy"))
            {
                o << ";coarsen="
                  << (in.has_coarsening_strategy ? to_string(in.coarsening_strategy) : "default");
            }
            if(rel.count("interpolation_type"))
            {
                o << ";interp=" << to_string(in.interpolation_type);
            }
            if(rel.count("strength_threshold"))
            {
                o << ";theta=" << numfmt(in.strength_threshold);
            }
            if(rel.count("ff1_limit"))
            {
                o << ";ff1=" << (in.ff1_limit ? 1 : 0);
            }
            if(rel.count("coupling_strength"))
            {
                o << ";eps=" << numfmt(in.coupling_strength);
            }
            if(rel.count("interp_relax"))
            {
                o << ";relax=" << numfmt(in.interp_relax);
            }
            if(rel.count("lumping_strategy"))
            {
                o << ";lump=" << to_string(in.lumping_strategy);
            }
            if(rel.count("over_interp"))
            {
                o << ";ov=" << numfmt(in.over_interp);
            }
            if(rel.count("beta"))
            {
                o << ";beta=" << numfmt(in.beta);
            }
            if(rel.count("coarsening_factor"))
            {
                o << ";cf=" << numfmt(in.coarsening_factor);
            }
            if(rel.count("ordering"))
            {
                o << ";ord=" << ordering_to_string(in.ordering);
            }
        }

        o << ";abstol=" << numfmt(in.abs_tol) << ";reltol=" << numfmt(in.rel_tol)
          << ";divtol=" << numfmt(in.div_tol) << ";maxit=" << in.max_iter
          << ";repeats=" << in.repeats;
        return o.str();
    }

    std::string scalar_str(const JsonValue& v)
    {
        if(v.is_string())
        {
            return v.str;
        }
        if(v.is_bool())
        {
            return v.boolean ? "true" : "false";
        }
        if(v.is_number())
        {
            std::ostringstream o;
            o << v.number;
            return o.str();
        }
        return "?";
    }

    // Short label for a config key, used in auto-generated run names.
    std::string short_key(const std::string& key)
    {
        static const std::map<std::string, std::string> abbrev = {
            {"amg_type", "amg"},
            {"coarsening_strategy", "coarsen"},
            {"interpolation_type", "interp"},
            {"strength_threshold", "theta"},
            {"ff1_limit", "ff1"},
            {"coupling_strength", "eps"},
            {"interp_relax", "relax"},
            {"lumping_strategy", "lump"},
            {"over_interp", "ov"},
            {"coarsening_factor", "cf"},
            {"pre_smooth_iter", "pre"},
            {"post_smooth_iter", "post"},
            {"coarsest_level", "coarsest"},
            {"host_levels", "hostlv"},
            {"smoother", "sm"},
            {"smoother_relax", "smrelax"},
            {"coarse_solver", "cs"},
            {"coarse_solver_tol", "cstol"},
            {"coarse_solver_max_iter", "csmax"},
            {"max_iter", "maxit"},
            {"operator_format", "opfmt"},
            {"operator_blockdim", "blk"},
            {"smoother_format", "smfmt"},
            {"kcycle_full", "kfull"},
        };
        auto it = abbrev.find(key);
        return it == abbrev.end() ? key : it->second;
    }

    std::string default_prefix(const SolverInput& in)
    {
        if(in.preconditioner != PreconditionerKind::AMG)
        {
            return to_string(in.preconditioner);
        }
        switch(in.amg_type)
        {
        case AMGKind::RugeStueben:
            return "rs-amg";
        case AMGKind::SA:
            return "sa-amg";
        case AMGKind::UA:
            return "ua-amg";
        case AMGKind::Pairwise:
            return "pw-amg";
        }
        return "amg";
    }

    // Canonical ordering of keys for readable auto-names.
    const std::vector<std::string>& name_key_order()
    {
        static const std::vector<std::string> order = {"solver",
                                                        "preconditioner",
                                                        "amg_type",
                                                        "coarsening_strategy",
                                                        "interpolation_type",
                                                        "strength_threshold",
                                                        "ff1_limit",
                                                        "coupling_strength",
                                                        "interp_relax",
                                                        "lumping_strategy",
                                                        "over_interp",
                                                        "beta",
                                                        "coarsening_factor",
                                                        "ordering",
                                                        "cycle",
                                                        "kcycle_full",
                                                        "pre_smooth_iter",
                                                        "post_smooth_iter",
                                                        "scaling",
                                                        "coarsest_level",
                                                        "host_levels",
                                                        "operator_format",
                                                        "operator_blockdim",
                                                        "smoother_format",
                                                        "smoother",
                                                        "smoother_relax",
                                                        "coarse_solver",
                                                        "coarse_solver_tol",
                                                        "coarse_solver_max_iter",
                                                        "abs_tol",
                                                        "rel_tol",
                                                        "div_tol",
                                                        "max_iter",
                                                        "repeats"};
        return order;
    }

    std::string make_name(const ResolvedRun& r, const SolverInput& in)
    {
        if(r.has_name)
        {
            return r.name;
        }

        std::string prefix = r.group.empty() ? default_prefix(in) : r.group;

        std::set<std::string> rel = relevant_keys(in);
        std::string           suffix;
        for(const std::string& key : name_key_order())
        {
            if(r.axis_keys.count(key) && rel.count(key))
            {
                auto it = r.fields.find(key);
                if(it != r.fields.end())
                {
                    suffix += "/" + short_key(key) + "=" + scalar_str(it->second);
                }
            }
        }
        return prefix + suffix;
    }

    // Expand the config into a de-duplicated, named list of SolverInput.
    std::vector<SolverInput> configure(const JsonValue& root, const JsonValue* defaults)
    {
        std::vector<ResolvedRun> runs;
        bool                     any = false;

        if(const JsonValue* s = root.find("sweep"))
        {
            if(!s->is_object())
            {
                throw std::runtime_error("'sweep' must be an object");
            }
            auto e = expand_object(*s);
            runs.insert(runs.end(), e.begin(), e.end());
            any = true;
        }
        if(const JsonValue* s = root.find("sweeps"))
        {
            if(!s->is_array())
            {
                throw std::runtime_error("'sweeps' must be an array");
            }
            for(const auto& el : s->array)
            {
                if(!el.is_object())
                {
                    throw std::runtime_error("each entry of 'sweeps' must be an object");
                }
                auto e = expand_object(el);
                runs.insert(runs.end(), e.begin(), e.end());
            }
            any = true;
        }
        if(const JsonValue* s = root.find("runs"))
        {
            if(!s->is_array())
            {
                throw std::runtime_error("'runs' must be an array");
            }
            for(const auto& el : s->array)
            {
                if(!el.is_object())
                {
                    throw std::runtime_error("each entry of 'runs' must be an object");
                }
                auto e = expand_object(el);
                runs.insert(runs.end(), e.begin(), e.end());
            }
            any = true;
        }

        // Back-compat: a flat top-level config with no sweep/sweeps/runs is a single run.
        if(!any)
        {
            auto e = expand_object(root);
            runs.insert(runs.end(), e.begin(), e.end());
        }

        std::vector<SolverInput>   out;
        std::set<std::string>      seen;
        std::map<std::string, int> used_names;

        for(ResolvedRun& r : runs)
        {
            merge_defaults(r, defaults);

            SolverInput in;
            JsonValue   obj = make_object(r.fields);
            map_solver_input(obj, in);

            std::string sig = canonical_signature(in);
            if(seen.count(sig))
            {
                continue;
            }
            seen.insert(sig);

            std::string nm    = make_name(r, in);
            int&        count = used_names[nm];
            if(count > 0)
            {
                in.name = nm + "#" + std::to_string(count + 1);
            }
            else
            {
                in.name = nm;
            }
            ++count;

            out.push_back(in);
        }

        return out;
    }

    // ------------------------------------------------------------------------------------
    // Results output (hand-rolled JSON, matching the codebase's no-dependency convention)
    // ------------------------------------------------------------------------------------

    std::string json_string(const std::string& s)
    {
        std::string o = "\"";
        for(char c : s)
        {
            switch(c)
            {
            case '"':
                o += "\\\"";
                break;
            case '\\':
                o += "\\\\";
                break;
            case '\n':
                o += "\\n";
                break;
            case '\r':
                o += "\\r";
                break;
            case '\t':
                o += "\\t";
                break;
            default:
                o += c;
            }
        }
        o += "\"";
        return o;
    }

    // JSON has no Inf/NaN; emit null for non-finite values so the file stays valid.
    std::string json_number(double x)
    {
        if(!std::isfinite(x))
        {
            return "null";
        }
        std::ostringstream o;
        o << std::setprecision(12) << x;
        return o.str();
    }

    struct JField
    {
        std::string key;
        std::string val; // already-serialized JSON token
    };

    void write_object(std::ostream& os, const std::vector<JField>& fields, const std::string& indent)
    {
        os << "{\n";
        for(size_t i = 0; i < fields.size(); ++i)
        {
            os << indent << "  " << json_string(fields[i].key) << ": " << fields[i].val;
            os << (i + 1 < fields.size() ? ",\n" : "\n");
        }
        os << indent << "}";
    }

    // Serialize the RELEVANT parameters of a configuration into the "setup" object.
    std::vector<JField> setup_fields(const SolverInput& in)
    {
        std::set<std::string> rel = relevant_keys(in);
        std::vector<JField>   f;

        f.push_back({"name", json_string(in.name)});
        f.push_back({"solver", json_string(to_string(in.solver))});
        f.push_back({"preconditioner", json_string(to_string(in.preconditioner))});

        if(in.preconditioner == PreconditionerKind::AMG)
        {
            f.push_back({"amg_type", json_string(to_string(in.amg_type))});

            if(rel.count("coarsening_strategy"))
            {
                f.push_back({"coarsening_strategy",
                             json_string(in.has_coarsening_strategy
                                             ? to_string(in.coarsening_strategy)
                                             : "default")});
            }
            if(rel.count("interpolation_type"))
            {
                f.push_back({"interpolation_type", json_string(to_string(in.interpolation_type))});
            }
            if(rel.count("strength_threshold"))
            {
                f.push_back({"strength_threshold", json_number(in.strength_threshold)});
            }
            if(rel.count("ff1_limit"))
            {
                f.push_back({"ff1_limit", in.ff1_limit ? "true" : "false"});
            }
            if(rel.count("coupling_strength"))
            {
                f.push_back({"coupling_strength", json_number(in.coupling_strength)});
            }
            if(rel.count("interp_relax"))
            {
                f.push_back({"interp_relax", json_number(in.interp_relax)});
            }
            if(rel.count("lumping_strategy"))
            {
                f.push_back({"lumping_strategy", json_string(to_string(in.lumping_strategy))});
            }
            if(rel.count("over_interp"))
            {
                f.push_back({"over_interp", json_number(in.over_interp)});
            }
            if(rel.count("beta"))
            {
                f.push_back({"beta", json_number(in.beta)});
            }
            if(rel.count("coarsening_factor"))
            {
                f.push_back({"coarsening_factor", json_number(in.coarsening_factor)});
            }
            if(rel.count("ordering"))
            {
                f.push_back({"ordering", json_string(ordering_to_string(in.ordering))});
            }

            f.push_back({"cycle", json_string(cycle_to_string(in.cycle))});
            if(in.cycle == Kcycle)
            {
                f.push_back({"kcycle_full", in.kcycle_full ? "true" : "false"});
            }
            f.push_back({"pre_smooth_iter", std::to_string(in.pre_smooth_iter)});
            f.push_back({"post_smooth_iter", std::to_string(in.post_smooth_iter)});
            f.push_back({"scaling", in.scaling ? "true" : "false"});
            f.push_back({"coarsest_level", std::to_string(in.coarsest_level)});
            f.push_back({"host_levels", std::to_string(in.host_levels)});
            f.push_back({"operator_format", json_string(matrix_format_to_string(in.operator_format))});
            if(in.operator_format == BCSR)
            {
                f.push_back({"operator_blockdim", std::to_string(in.operator_blockdim)});
            }
            f.push_back({"smoother_format", json_string(matrix_format_to_string(in.smoother_format))});
            f.push_back({"smoother", json_string(to_string(in.smoother))});
            if(rel.count("smoother_relax"))
            {
                f.push_back({"smoother_relax", json_number(in.smoother_relax)});
            }
            f.push_back({"coarse_solver", json_string(to_string(in.coarse_solver))});
            if(rel.count("coarse_solver_tol"))
            {
                f.push_back({"coarse_solver_tol", json_number(in.coarse_solver_tol)});
                f.push_back({"coarse_solver_max_iter", std::to_string(in.coarse_solver_max_iter)});
            }
        }

        f.push_back({"abs_tol", json_number(in.abs_tol)});
        f.push_back({"rel_tol", json_number(in.rel_tol)});
        f.push_back({"div_tol", json_number(in.div_tol)});
        f.push_back({"max_iter", std::to_string(in.max_iter)});
        f.push_back({"repeats", std::to_string(in.repeats)});
        return f;
    }

    struct MatrixMeta
    {
        std::string path;
        std::string format;
        int64_t     rows          = 0;
        int64_t     cols          = 0;
        int64_t     nnz           = 0;
        double      read_time_sec = 0.0;
    };

    // Indices of the best runs (fastest build / solve / total), considering only runs that
    // succeeded and converged. A value of -1 means no eligible run was found.
    struct Winners
    {
        int build = -1;
        int solve = -1;
        int total = -1;
    };

    Winners find_winners(const std::vector<SolverOutput>& outs)
    {
        Winners w;
        for(size_t i = 0; i < outs.size(); ++i)
        {
            const SolverOutput& o = outs[i];
            if(!o.error.empty() || !o.converged)
            {
                continue;
            }
            if(w.build < 0 || o.build_time_sec < outs[w.build].build_time_sec)
            {
                w.build = static_cast<int>(i);
            }
            if(w.solve < 0 || o.solve_time_mean_sec < outs[w.solve].solve_time_mean_sec)
            {
                w.solve = static_cast<int>(i);
            }
            if(w.total < 0 || o.total_time_sec < outs[w.total].total_time_sec)
            {
                w.total = static_cast<int>(i);
            }
        }
        return w;
    }

    void write_results(const std::string&               path,
                       const MatrixMeta&                meta,
                       const std::vector<SolverInput>&  inputs,
                       const std::vector<SolverOutput>& outputs)
    {
        std::ofstream os(path);
        if(!os)
        {
            throw std::runtime_error("cannot open results file '" + path + "' for writing");
        }

        os << "{\n";

        // Matrix metadata.
        os << "  \"matrix\": ";
        std::vector<JField> m = {{"path", json_string(meta.path)},
                                 {"format", json_string(meta.format)},
                                 {"rows", std::to_string(meta.rows)},
                                 {"cols", std::to_string(meta.cols)},
                                 {"nnz", std::to_string(meta.nnz)},
                                 {"read_time_sec", json_number(meta.read_time_sec)}};
        write_object(os, m, "  ");
        os << ",\n";

        os << "  \"run_count\": " << outputs.size() << ",\n";

        // Winners: fastest build / solve / total among converged runs. Each winner embeds
        // its resolved setup so the file is self-contained.
        Winners w           = find_winners(outputs);
        auto    emit_winner = [&](const char* label, int idx, double value, bool last) {
            os << "    " << json_string(label) << ": ";
            if(idx < 0)
            {
                os << "null";
            }
            else
            {
                os << "{\n";
                os << "      \"name\": " << json_string(outputs[idx].name) << ",\n";
                os << "      \"seconds\": " << json_number(value) << ",\n";
                os << "      \"setup\": ";
                write_object(os, setup_fields(inputs[idx]), "      ");
                os << "\n    }";
            }
            os << (last ? "\n" : ",\n");
        };
        os << "  \"winners\": {\n";
        emit_winner("fastest_build", w.build, w.build >= 0 ? outputs[w.build].build_time_sec : 0.0,
                    false);
        emit_winner("fastest_solve", w.solve,
                    w.solve >= 0 ? outputs[w.solve].solve_time_mean_sec : 0.0, false);
        emit_winner("fastest_total", w.total, w.total >= 0 ? outputs[w.total].total_time_sec : 0.0,
                    true);
        os << "  },\n";

        os << "  \"results\": [\n";

        for(size_t i = 0; i < outputs.size(); ++i)
        {
            const SolverInput&  in  = inputs[i];
            const SolverOutput& out = outputs[i];

            os << "    {\n";
            os << "      \"setup\": ";
            write_object(os, setup_fields(in), "      ");
            os << ",\n";

            os << "      \"num_levels\": " << out.num_levels << ",\n";
            os << "      \"build_time_sec\": " << json_number(out.build_time_sec) << ",\n";
            os << "      \"solve_time_sec\": {\"mean\": " << json_number(out.solve_time_mean_sec)
               << ", \"min\": " << json_number(out.solve_time_min_sec)
               << ", \"max\": " << json_number(out.solve_time_max_sec) << "},\n";
            os << "      \"total_time_sec\": " << json_number(out.total_time_sec) << ",\n";
            os << "      \"iterations\": " << out.iterations << ",\n";
            os << "      \"final_residual\": " << json_number(out.final_residual) << ",\n";
            os << "      \"solver_status\": " << json_string(out.solver_status) << ",\n";
            os << "      \"converged\": " << (out.converged ? "true" : "false") << ",\n";
            os << "      \"error_l2\": " << (out.has_error_l2 ? json_number(out.error_l2) : "null")
               << ",\n";
            os << "      \"error\": " << (out.error.empty() ? "null" : json_string(out.error))
               << "\n";
            os << "    }" << (i + 1 < outputs.size() ? ",\n" : "\n");
        }

        os << "  ]\n";
        os << "}\n";
    }

    // ------------------------------------------------------------------------------------
    // Console output
    // ------------------------------------------------------------------------------------

    // Strip surrounding quotes from a JSON string token for human-readable display.
    std::string unquote(const std::string& s)
    {
        if(s.size() >= 2 && s.front() == '"' && s.back() == '"')
        {
            return s.substr(1, s.size() - 2);
        }
        return s;
    }

    // "key=value, key=value, ..." over the RELEVANT resolved parameters (reuses
    // setup_fields so it always matches the JSON echo). The name is omitted.
    std::string props_line(const SolverInput& in)
    {
        std::ostringstream o;
        bool               first = true;
        for(const JField& jf : setup_fields(in))
        {
            if(jf.key == "name")
            {
                continue;
            }
            if(!first)
            {
                o << ", ";
            }
            first = false;
            o << jf.key << "=" << unquote(jf.val);
        }
        return o.str();
    }

    std::string one_line_setup(const SolverInput& in)
    {
        std::ostringstream o;
        o << to_string(in.solver) << " + ";
        if(in.preconditioner == PreconditionerKind::AMG)
        {
            o << to_string(in.amg_type) << "-AMG";
            if(in.has_coarsening_strategy)
            {
                o << " [" << to_string(in.coarsening_strategy) << "]";
            }
            o << " cycle=" << cycle_to_string(in.cycle);
        }
        else
        {
            o << to_string(in.preconditioner);
        }
        return o.str();
    }

    // One-time GPU/runtime costs (HIP module load, memory-pool init, first-touch
    // allocations, per-kernel JIT) otherwise land entirely on the first timed run. Run a
    // short, discarded solve for each distinct method present in the batch so every
    // configuration is measured against an already-warm device. Iterations are capped
    // since only kernel/allocation warm-up matters here, not convergence.
    void run_warmup(const LocalMatrix<ValueType>&   mat,
                    const LocalVector<ValueType>&   rhs,
                    const LocalVector<ValueType>&   x,
                    const std::vector<SolverInput>& inputs,
                    int                             warmup_iters)
    {
        std::set<std::string> seen;
        double                t0    = rocalution_time();
        int                   count = 0;

        for(const SolverInput& in : inputs)
        {
            // Distinct kernel set is determined by preconditioner (and AMG method).
            std::string key = to_string(in.preconditioner);
            if(in.preconditioner == PreconditionerKind::AMG)
            {
                key += std::string("/") + to_string(in.amg_type);
            }
            if(seen.count(key))
            {
                continue;
            }
            seen.insert(key);

            SolverInput w = in;
            w.max_iter    = std::max(1, std::min(in.max_iter, warmup_iters));
            w.repeats     = 1;
            w.name        = "warmup:" + key;

            std::cout << "  warming up " << key << " ..." << std::endl;
            SolverOutput out = solve(mat, rhs, x, w, nullptr); // result discarded
            if(!out.error.empty())
            {
                // A warm-up failure is not fatal; the real run will report it too.
                std::cout << "    (warm-up reported: " << out.error << ")\n";
            }
            ++count;
        }

        double t1 = rocalution_time();
        std::cout << "Warm-up: initialized " << count << " method(s) in " << (t1 - t0) / 1e6
                  << " s\n";
    }

    void print_summary_table(const std::vector<SolverInput>&  inputs,
                             const std::vector<SolverOutput>& outputs)
    {
        std::cout << "\n===================================== summary "
                     "=====================================\n";
        std::cout << std::left << std::setw(34) << "name" << std::right << std::setw(4) << "lvl"
                  << std::setw(11) << "build[s]" << std::setw(11) << "solve[s]" << std::setw(11)
                  << "total[s]" << std::setw(7) << "iters" << "  " << std::left << std::setw(11)
                  << "status" << "conv\n";
        std::cout << std::string(99, '-') << "\n";

        for(size_t i = 0; i < outputs.size(); ++i)
        {
            const SolverOutput& out = outputs[i];
            std::string         nm  = out.name;
            if(nm.size() > 33)
            {
                nm = nm.substr(0, 30) + "...";
            }

            std::cout << std::left << std::setw(34) << nm << std::right;
            if(!out.error.empty())
            {
                std::cout << std::setw(4) << "-" << std::setw(11) << "-" << std::setw(11) << "-"
                          << std::setw(11) << "-" << std::setw(7) << "-" << "  " << std::left
                          << std::setw(11) << "FAILED" << "\n";
                continue;
            }
            std::cout << std::setw(4) << out.num_levels << std::setw(11) << std::fixed
                      << std::setprecision(4) << out.build_time_sec << std::setw(11)
                      << out.solve_time_mean_sec << std::setw(11) << out.total_time_sec
                      << std::setw(7) << out.iterations << std::defaultfloat << "  " << std::left
                      << std::setw(11) << out.solver_status << (out.converged ? "yes" : "no")
                      << std::right << "\n";
        }
        std::cout << std::string(99, '-') << "\n";
    }

    void print_winners(const std::vector<SolverInput>&  inputs,
                       const std::vector<SolverOutput>& outputs)
    {
        Winners w = find_winners(outputs);
        std::cout << "\nWinners (fastest among converged runs):\n";
        if(w.build < 0)
        {
            std::cout << "  none (no run converged)\n";
            return;
        }

        auto show = [&](const char* label, int idx, double seconds) {
            std::cout << "  " << label << " : " << outputs[idx].name << "  (" << std::fixed
                      << std::setprecision(6) << seconds << " s)" << std::defaultfloat << "\n";
            std::cout << "      " << props_line(inputs[idx]) << "\n";
        };

        show("fastest build", w.build, outputs[w.build].build_time_sec);
        show("fastest solve", w.solve, outputs[w.solve].solve_time_mean_sec);
        show("fastest total", w.total, outputs[w.total].total_time_sec);
    }

    std::string read_file(const std::string& path)
    {
        std::ifstream f(path);
        if(!f)
        {
            throw std::runtime_error("cannot open config file '" + path + "'");
        }
        std::stringstream ss;
        ss << f.rdbuf();
        return ss.str();
    }

    std::string to_lower(std::string s)
    {
        for(char& c : s)
        {
            c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        }
        return s;
    }

    // Resolve which reader to use. An explicit matrix_format wins; "auto" (the default)
    // derives it from the file extension so binary matrices need no extra configuration.
    // Returns one of "mtx", "csr", "rsio".
    std::string resolve_matrix_format(const GlobalConfig& g)
    {
        const std::string fmt = to_lower(g.matrix_format);

        if(fmt == "mtx" || fmt == "csr" || fmt == "rsio")
        {
            return fmt;
        }
        // Convenience alias for the modern rocALUTION binary format.
        if(fmt == "binary")
        {
            return "rsio";
        }
        if(!fmt.empty() && fmt != "auto")
        {
            throw std::runtime_error("unknown matrix_format '" + g.matrix_format
                                     + "' (expected auto|mtx|csr|rsio|binary)");
        }

        std::string ext;
        const size_t dot = g.matrix.find_last_of('.');
        if(dot != std::string::npos)
        {
            ext = to_lower(g.matrix.substr(dot + 1));
        }

        if(ext == "mtx" || ext == "mm")
        {
            return "mtx";
        }
        if(ext == "csr")
        {
            return "csr";
        }
        if(ext == "rsio" || ext == "bin")
        {
            return "rsio";
        }

        throw std::runtime_error(
            "cannot infer matrix_format from '" + g.matrix
            + "' (known extensions: .mtx/.mm, .csr, .rsio/.bin); set matrix_format explicitly");
    }

    void read_matrix(LocalMatrix<ValueType>& mat, const std::string& path, const std::string& format)
    {
        if(format == "mtx")
        {
            // MatrixMarket text. Convenient, but slow for very large matrices; prefer a
            // binary format (rsio) for those.
            mat.ReadFileMTX(path);
        }
        else if(format == "csr")
        {
            // Legacy rocALUTION binary CSR. Deprecated in the library in favour of RSIO,
            // but still read here since existing data sets use it.
#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
            mat.ReadFileCSR(path);
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
        }
        else if(format == "rsio")
        {
            // Modern rocALUTION binary format.
            mat.ReadFileRSIO(path);
        }
        else
        {
            throw std::runtime_error("unknown matrix_format '" + format
                                     + "' (expected auto|mtx|csr|rsio|binary)");
        }
    }
} // namespace

int main(int argc, char* argv[])
{
    std::string config_path;
    std::string results_path = "rocalution_opt_results.json";
    bool        dry_run      = false;

    // CLI: positional <config.json> [results.json], plus a --dry-run flag.
    int positional = 0;
    for(int i = 1; i < argc; ++i)
    {
        std::string a = argv[i];
        if(a == "--dry-run")
        {
            dry_run = true;
        }
        else if(!a.empty() && a[0] == '-')
        {
            std::cerr << "Unknown option '" << a << "'\n";
            return 1;
        }
        else if(positional == 0)
        {
            config_path = a;
            ++positional;
        }
        else if(positional == 1)
        {
            results_path = a;
            ++positional;
        }
    }

    if(config_path.empty())
    {
        std::cerr << argv[0] << " <config.json> [results.json] [--dry-run]\n";
        std::cerr << "  Loads the matrix once and benchmarks a sweep of PCG+AMG configurations.\n";
        return 1;
    }

    // Parse the config and expand it into the list of configurations to run.
    GlobalConfig             g;
    std::vector<SolverInput> inputs;
    try
    {
        JsonParser parser(read_file(config_path));
        JsonValue  root = parser.parse();
        if(!root.is_object())
        {
            throw std::runtime_error("top-level JSON value must be an object");
        }

        const JsonValue* defaults = nullptr;
        if(const JsonValue* gobj = root.find("global"))
        {
            map_global(*gobj, g);
            defaults = gobj->find("defaults");
        }
        else
        {
            map_global(root, g);
            defaults = root.find("defaults");
        }

        inputs = configure(root, defaults);
    }
    catch(const std::exception& e)
    {
        std::cerr << "Error reading config: " << e.what() << "\n";
        return 1;
    }

    if(inputs.empty())
    {
        std::cerr << "Error: config expands to zero configurations\n";
        return 1;
    }

    std::cout << "Resolved " << inputs.size() << " configuration(s) from " << config_path << "\n";

    if(dry_run)
    {
        std::cout << "\n--dry-run: listing configurations (matrix is NOT loaded)\n";
        for(size_t i = 0; i < inputs.size(); ++i)
        {
            std::cout << "  [" << std::setw(3) << i << "] " << std::left << std::setw(40)
                      << inputs[i].name << std::right << "  " << one_line_setup(inputs[i]) << "\n";
        }
        return 0;
    }

    if(g.matrix.empty())
    {
        std::cerr << "Error: config must specify a 'matrix' path\n";
        return 1;
    }

    init_rocalution();

    if(g.omp_threads > 0)
    {
        set_omp_threads_rocalution(g.omp_threads);
    }

    info_rocalution();

    int ret = 0;
    {
        LocalMatrix<ValueType> mat;
        LocalVector<ValueType> rhs;
        LocalVector<ValueType> x;
        LocalVector<ValueType> e;

        std::vector<SolverOutput> outputs;
        MatrixMeta                meta;

        try
        {
            // Read the matrix ONCE, using the explicit or auto-detected reader.
            const std::string format = resolve_matrix_format(g);

            double read_tick = rocalution_time();
            read_matrix(mat, g.matrix, format);
            double read_tack = rocalution_time();

            meta.path          = g.matrix;
            meta.format        = format;
            meta.rows          = mat.GetM();
            meta.cols          = mat.GetN();
            meta.nnz           = mat.GetNnz();
            meta.read_time_sec = (read_tack - read_tick) / 1e6;

            std::cout << "Matrix: " << meta.path << " (" << meta.format << ")\n";
            std::cout << "  rows=" << meta.rows << " cols=" << meta.cols << " nnz=" << meta.nnz
                      << " read_time[s]=" << meta.read_time_sec << "\n";

            // The row-pointer width is a rocALUTION build option and caps the local nnz.
            std::cout << "  nnz index width: " << (sizeof(PtrType) * 8) << " bit";
            if(sizeof(PtrType) < sizeof(int64_t))
            {
                std::cout << " (rebuild rocALUTION with -DBUILD_PTRTYPE_64=ON for nnz > "
                          << std::numeric_limits<int32_t>::max() << ")";
            }
            std::cout << "\n";

            // Move to the accelerator and build rhs / initial guess once.
            mat.MoveToAccelerator();
            rhs.MoveToAccelerator();
            x.MoveToAccelerator();
            e.MoveToAccelerator();

            rhs.Allocate("rhs", mat.GetM());
            x.Allocate("x", mat.GetN());
            e.Allocate("e", mat.GetN());

            bool have_exact = false;
            if(g.rhs == "ones")
            {
                // rhs = A * 1 so the exact solution is the all-ones vector.
                e.Ones();
                mat.Apply(e, &rhs);
                have_exact = true;
            }
            else
            {
                throw std::runtime_error("unsupported rhs mode '" + g.rhs + "' (supported: 'ones')");
            }

            if(g.initial_guess == "zeros")
            {
                x.Zeros();
            }
            else
            {
                throw std::runtime_error("unsupported initial_guess mode '" + g.initial_guess
                                         + "' (supported: 'zeros')");
            }

            // Warm up the device / runtime so the first timed run is not penalized by
            // one-time initialization costs.
            if(g.warmup)
            {
                std::cout << "\n--- warm-up ---\n";
                run_warmup(mat, rhs, x, inputs, g.warmup_iters);
            }

            // Run every configuration against the single resident matrix.
            outputs.reserve(inputs.size());
            for(size_t i = 0; i < inputs.size(); ++i)
            {
                const SolverInput& in = inputs[i];
                std::cout << "\n[" << (i + 1) << "/" << inputs.size() << "] " << in.name << "  ("
                          << one_line_setup(in) << ")" << std::endl;

                SolverOutput out = solve(mat, rhs, x, in, have_exact ? &e : nullptr);
                outputs.push_back(out);

                if(!out.error.empty())
                {
                    std::cout << "    FAILED: " << out.error << "\n";
                }
                else
                {
                    std::cout << "    levels=" << out.num_levels << " iters=" << out.iterations
                              << " build[s]=" << out.build_time_sec
                              << " solve[s]=" << out.solve_time_mean_sec
                              << " total[s]=" << out.total_time_sec
                              << " status=" << out.solver_status
                              << (out.converged ? " (converged)" : " (NOT converged)");
                    if(out.has_error_l2)
                    {
                        std::cout << " ||x-exact||=" << out.error_l2;
                    }
                    std::cout << "\n";
                }
            }

            write_results(results_path, meta, inputs, outputs);
            std::cout << "\nWrote results for " << outputs.size() << " run(s) to " << results_path
                      << "\n";

            print_summary_table(inputs, outputs);
            print_winners(inputs, outputs);

            // Non-zero exit if every run failed (helps scripting/CI catch a bad batch).
            bool any_ok = false;
            for(const auto& out : outputs)
            {
                if(out.error.empty())
                {
                    any_ok = true;
                    break;
                }
            }
            if(!any_ok)
            {
                ret = 2;
            }
        }
        catch(const std::exception& ex)
        {
            std::cerr << "Fatal error: " << ex.what() << "\n";
            ret = 1;
        }
    }

    stop_rocalution();
    return ret;
}
