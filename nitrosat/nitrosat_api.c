/*=====================================================================
  NitroSat Python API Implementation
  --------------------------------------------------------------------
  Implementation of the public C API for Python ctypes wrapper.
  
  Compile with:
      gcc -O3 -fPIC -shared -o libnitrosat.so nitrosat_api.c -lm
  
  Based on NitroSAT by Sethu Iyer (sethuiyer95@gmail.com)
=====================================================================*/

/* Include the full NitroSAT implementation */
#include "nitrosat.c"
#include "nitrosat_api.h"

/* Version string */
static const char* VERSION = "1.0.0-fractal";

/* ----------------------------------------------------------------
   API Implementation
---------------------------------------------------------------- */

const char* nitrosat_version(void) {
    return VERSION;
}

void nitrosat_set_seed(double seed) {
    rng_state = seed;
}

void nitrosat_free_result(NitroSatResult* result) {
    if (result) {
        if (result->assignment) free(result->assignment);
        free(result);
    }
}

NitroSatResult* nitrosat_solve_file(
    const char* filename,
    int max_steps,
    int use_dcw,
    int use_topo,
    int verbose
) {
    NitroSatResult* result = calloc(1, sizeof(NitroSatResult));
    if (!result) return NULL;
    
    Instance* inst = read_cnf(filename);
    if (!inst) {
        result->solved = 0;
        return result;
    }
    
    result->num_vars = inst->num_vars;
    result->num_clauses = inst->num_clauses;
    
    NitroSat* ns = nitrosat_new(inst, max_steps, verbose);
    ns->use_topology = use_topo;
    
    /* Build variable-to-clause CSR structure */
    ns->v2c_ptr = calloc(ns->num_vars + 2, sizeof(int));
    for (int i = 0; i < inst->num_clauses; ++i) {
        for (int j = inst->cl_offs[i]; j < inst->cl_offs[i+1]; ++j) {
            ns->v2c_ptr[abs(inst->cl_flat[j]) + 1]++;
        }
    }
    for (int i = 1; i <= ns->num_vars + 1; ++i) 
        ns->v2c_ptr[i] += ns->v2c_ptr[i-1];
    ns->v2c_data = malloc(ns->v2c_ptr[ns->num_vars+1] * sizeof(int));
    int* cur = calloc(ns->num_vars + 1, sizeof(int));
    for (int c = 0; c < inst->num_clauses; ++c) {
        for (int j = inst->cl_offs[c]; j < inst->cl_offs[c+1]; ++j) {
            int v = abs(inst->cl_flat[j]);
            int pos = ns->v2c_ptr[v] + cur[v];
            ns->v2c_data[pos] = c;
            cur[v]++;
        }
    }
    free(cur);
    
    /* Solve */
    double start = (double)clock() / CLOCKS_PER_SEC;
    int solved = use_dcw ? nitrosat_solve_dcw(ns, 5) : nitrosat_solve(ns);
    double end = (double)clock() / CLOCKS_PER_SEC;
    
    result->solve_time = end - start;
    result->solved = solved;
    
    recompute_sat_counts(ns);
    result->satisfied = check_satisfaction(ns);
    result->unsatisfied = ns->num_clauses - result->satisfied;
    
    /* Topology info */
    if (ns->use_topology && ns->pt.has_initial) {
        result->initial_beta0 = ns->pt.initial_beta0;
        result->final_beta0 = ns->pt.final_beta0;
        result->initial_beta1 = ns->pt.initial_beta1;
        result->final_beta1 = ns->pt.final_beta1;
        result->persistence_events = ns->pt.persistence_events;
        result->complexity_trend = ns->pt.final_complexity - ns->pt.initial_complexity;
    }
    
    /* Copy assignment */
    result->assignment_size = ns->num_vars;
    result->assignment = malloc((ns->num_vars + 1) * sizeof(int8_t));
    for (int i = 1; i <= ns->num_vars; ++i) {
        result->assignment[i] = (ns->x[i] > 0.5) ? 1 : 0;
    }
    
    nitrosat_free(ns);
    free(inst->cl_offs);
    free(inst->cl_flat);
    free(inst);
    
    return result;
}

NitroSatResult* nitrosat_solve_arrays(
    int num_vars,
    int num_clauses,
    const int* clause_sizes,
    const int* literals,
    int max_steps,
    int use_dcw,
    int use_topo,
    int verbose
) {
    NitroSatResult* result = calloc(1, sizeof(NitroSatResult));
    if (!result) return NULL;
    
    result->num_vars = num_vars;
    result->num_clauses = num_clauses;
    
    /* Build instance from arrays */
    Instance* inst = malloc(sizeof(Instance));
    inst->num_vars = num_vars;
    inst->num_clauses = num_clauses;
    
    /* Calculate total literals */
    int total_lits = 0;
    for (int c = 0; c < num_clauses; ++c) {
        total_lits += clause_sizes[c];
    }
    
    /* Build offsets and flat arrays */
    inst->cl_offs = malloc((num_clauses + 1) * sizeof(int));
    inst->cl_flat = malloc(total_lits * sizeof(int));
    
    inst->cl_offs[0] = 0;
    int lit_idx = 0;
    for (int c = 0; c < num_clauses; ++c) {
        for (int i = 0; i < clause_sizes[c]; ++i) {
            inst->cl_flat[lit_idx++] = literals[inst->cl_offs[c] + i];
        }
        inst->cl_offs[c + 1] = inst->cl_offs[c] + clause_sizes[c];
    }
    
    NitroSat* ns = nitrosat_new(inst, max_steps, verbose);
    ns->use_topology = use_topo;
    
    /* Build variable-to-clause CSR structure */
    ns->v2c_ptr = calloc(ns->num_vars + 2, sizeof(int));
    for (int i = 0; i < inst->num_clauses; ++i) {
        for (int j = inst->cl_offs[i]; j < inst->cl_offs[i+1]; ++j) {
            ns->v2c_ptr[abs(inst->cl_flat[j]) + 1]++;
        }
    }
    for (int i = 1; i <= ns->num_vars + 1; ++i) 
        ns->v2c_ptr[i] += ns->v2c_ptr[i-1];
    ns->v2c_data = malloc(ns->v2c_ptr[ns->num_vars+1] * sizeof(int));
    int* cur = calloc(ns->num_vars + 1, sizeof(int));
    for (int c = 0; c < inst->num_clauses; ++c) {
        for (int j = inst->cl_offs[c]; j < inst->cl_offs[c+1]; ++j) {
            int v = abs(inst->cl_flat[j]);
            int pos = ns->v2c_ptr[v] + cur[v];
            ns->v2c_data[pos] = c;
            cur[v]++;
        }
    }
    free(cur);
    
    /* Solve */
    double start = (double)clock() / CLOCKS_PER_SEC;
    int solved = use_dcw ? nitrosat_solve_dcw(ns, 5) : nitrosat_solve(ns);
    double end = (double)clock() / CLOCKS_PER_SEC;
    
    result->solve_time = end - start;
    result->solved = solved;
    
    recompute_sat_counts(ns);
    result->satisfied = check_satisfaction(ns);
    result->unsatisfied = ns->num_clauses - result->satisfied;
    
    /* Topology info */
    if (ns->use_topology && ns->pt.has_initial) {
        result->initial_beta0 = ns->pt.initial_beta0;
        result->final_beta0 = ns->pt.final_beta0;
        result->initial_beta1 = ns->pt.initial_beta1;
        result->final_beta1 = ns->pt.final_beta1;
        result->persistence_events = ns->pt.persistence_events;
        result->complexity_trend = ns->pt.final_complexity - ns->pt.initial_complexity;
    }
    
    /* Copy assignment */
    result->assignment_size = ns->num_vars;
    result->assignment = malloc((ns->num_vars + 1) * sizeof(int8_t));
    for (int i = 1; i <= ns->num_vars; ++i) {
        result->assignment[i] = (ns->x[i] > 0.5) ? 1 : 0;
    }
    
    nitrosat_free(ns);
    free(inst->cl_offs);
    free(inst->cl_flat);
    free(inst);
    
    return result;
}
