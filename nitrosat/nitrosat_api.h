/*=====================================================================
  NitroSat Python API Header
  --------------------------------------------------------------------
  Public C API for Python ctypes wrapper.
  
  This header provides a clean interface for the NitroSAT solver
  that can be easily wrapped by Python.
  
  Author: Generated for fractal_manifold project
  Based on NitroSAT by Sethu Iyer (sethuiyer95@gmail.com)
=====================================================================*/

#ifndef NITROSAT_API_H
#define NITROSAT_API_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ----------------------------------------------------------------
   Result structure returned by solver
---------------------------------------------------------------- */
typedef struct {
    int      solved;           /* 1 if all clauses satisfied */
    int      satisfied;        /* number of satisfied clauses */
    int      unsatisfied;      /* number of unsatisfied clauses */
    int      num_vars;         /* number of variables */
    int      num_clauses;      /* number of clauses */
    double   solve_time;       /* time in seconds */
    /* Topology info */
    int      initial_beta0;
    int      final_beta0;
    int      initial_beta1;
    int      final_beta1;
    int      persistence_events;
    double   complexity_trend;
    /* Assignment (first 1024 vars, or NULL if more) */
    int8_t  *assignment;       /* 0 or 1 for each variable (1-indexed) */
    int      assignment_size;
} NitroSatResult;

/* ----------------------------------------------------------------
   Public API Functions
---------------------------------------------------------------- */

/**
 * Solve a CNF formula from a DIMACS file.
 * 
 * @param filename  Path to DIMACS CNF file
 * @param max_steps Maximum optimization steps (default: 3000)
 * @param use_dcw   Use Dynamic Clause Weighting (default: 1)
 * @param use_topo  Use topology analysis (default: 1)
 * @param verbose   Print progress (default: 0)
 * @return NitroSatResult structure (caller must free with nitrosat_free_result)
 */
NitroSatResult* nitrosat_solve_file(
    const char* filename,
    int max_steps,
    int use_dcw,
    int use_topo,
    int verbose
);

/**
 * Solve a CNF formula from arrays.
 * 
 * @param num_vars     Number of variables
 * @param num_clauses  Number of clauses
 * @param clause_sizes Array of clause sizes (length = num_clauses)
 * @param literals     Flattened array of literals (signed integers)
 * @param max_steps    Maximum optimization steps
 * @param use_dcw      Use Dynamic Clause Weighting
 * @param use_topo     Use topology analysis
 * @param verbose      Print progress
 * @return NitroSatResult structure
 */
NitroSatResult* nitrosat_solve_arrays(
    int num_vars,
    int num_clauses,
    const int* clause_sizes,
    const int* literals,
    int max_steps,
    int use_dcw,
    int use_topo,
    int verbose
);

/**
 * Free a NitroSatResult structure.
 */
void nitrosat_free_result(NitroSatResult* result);

/**
 * Get version string.
 */
const char* nitrosat_version(void);

/**
 * Set random seed for reproducibility.
 */
void nitrosat_set_seed(double seed);

#ifdef __cplusplus
}
#endif

#endif /* NITROSAT_API_H */
