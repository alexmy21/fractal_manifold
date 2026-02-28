
/*=====================================================================
  NitroSat – Advanced MaxSAT Solver (C version)
  --------------------------------------------------------------------
  This file is a direct translation of the Lua source 
  It compiles as a standalone executable:

      gcc -O3 -march=native -std=c99 -lm -o nitrosat nitrosat.c

  Usage:
      ./nitrosat <cnf-file> [max-steps]

  The program reads a DIMACS CNF file, builds the solver instance and
  runs the full solver (gradient descent + BAHA + three‑phase finisher
  + core‑decomposition).  Progress is printed to stdout; the final line
  reports whether the formula was solved and how many steps were needed.

AUTHOR: Sethu Iyer (sethuiyer95@gmail.com)
Apache 2.0 License. 

Please cite https://zenodo.org/records/18753235 if you are using this software
=====================================================================*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <time.h>
#include <assert.h>

/* --------------------------------------------------------------------
   Helper macros / constants
-------------------------------------------------------------------- */
#define MAX_NAME_LEN    64
#define PI              3.14159265358979323846
#define E_INV           0.36787944117144232   /* 1/e   */
#define PHI             ((1.0 + sqrt(5.0)) / 2.0)
#define EPS             1e-12

/* --------------------------------------------------------------------
   Simple pseudo‑random generator – deterministic, fast.
   The Lua version used a “x = (x * 9973) mod 1” scheme.
-------------------------------------------------------------------- */
static double rng_state = 0.95607928791821;   /* matches LuaJIT math.randomseed(42); math.random() */

static double rng_double(void)
{
    double x = rng_state * 9973.0;
    rng_state = x - floor(x);
    return rng_state;
}
static int rng_int(int a, int b) /* inclusive */
{
    return a + (int)(rng_double() * (b - a + 1));
}

/* --------------------------------------------------------------------
   Utility functions
-------------------------------------------------------------------- */
static inline double clamp(double x, double lo, double hi)
{
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
}
static inline int   sign(double x) { return (x > 0) - (x < 0); }

/* --------------------------------------------------------------------
   1.  Prime generation (sieve of Eratosthenes)
-------------------------------------------------------------------- */
static int *sieve_primes(int n, int *out_cnt)
{
    /* Estimate an upper bound for the nth prime (Rosser–Schoenfeld) */
    int limit = (n < 6) ? 15 : (int)(n * (log((double)n) + log(log((double)n))) + 3);
    char *is_prime = calloc(limit + 1, 1);
    for (int i = 2; i <= limit; ++i) is_prime[i] = 1;
    for (int p = 2; p * p <= limit; ++p)
        if (is_prime[p])
            for (int q = p * p; q <= limit; q += p) is_prime[q] = 0;

    int *primes = malloc(sizeof(int) * n);
    int cnt = 0;
    for (int i = 2; i <= limit && cnt < n; ++i)
        if (is_prime[i]) primes[cnt++] = i;

    free(is_prime);
    *out_cnt = cnt;
    return primes;
}

/* --------------------------------------------------------------------
   2.  Zeta‑related helpers
-------------------------------------------------------------------- */
static double zeta_log_derivative(double s, const int *primes, int npr)
{
    double force = 0.0;
    for (int i = 0; i < npr; ++i) {
        double p = (double)primes[i];
        force += log(p) * pow(p, -s);
    }
    return force;
}

/* --------------------------------------------------------------------
   3.  Optimiser --------------------------------------------------------
-------------------------------------------------------------------- */
typedef struct {
    double *m;      /* 1st moment */
    double *v;      /* 2nd moment */
    double  lr;
    double  beta1, beta2, eps;
    int     t;
    int     nv;
    double  resonance_amplitude;
    char    name[MAX_NAME_LEN];   /* "nadam" or "harmonic_adam" … */
} Optimizer;

/* Allocate and initialise an optimiser */
static Optimizer *optimizer_new(const char *name, double *params, int nv,
                               double lr, double beta1, double beta2,
                               double eps, double resonance_amp)
{
    Optimizer *opt = calloc(1, sizeof(Optimizer));
    strncpy(opt->name, name, MAX_NAME_LEN-1);
    opt->nv    = nv;
    opt->lr    = lr;
    opt->beta1 = beta1;
    opt->beta2 = beta2;
    opt->eps   = eps;
    opt->m = calloc(nv + 1, sizeof(double));
    opt->v = calloc(nv + 1, sizeof(double));
    opt->resonance_amplitude = resonance_amp;
    return opt;
}

static void optimizer_step(Optimizer *opt, double *params, const double *grads)
{
    ++opt->t;
    double b1 = opt->beta1;
    double b2 = opt->beta2;
    double lr = opt->lr;
    double eps = opt->eps;
    int    n = opt->nv;

    double b1_t = pow(b1, opt->t);
    double b2_t = pow(b2, opt->t);
    double t_inv1 = 1.0 / (1.0 - b1_t);
    double t_inv2 = 1.0 / (1.0 - b2_t);

    int is_nadam = (strcmp(opt->name, "nadam") == 0);

    for (int i = 1; i <= n; ++i) {
        double m_hat_prev = 0.0;
        if (is_nadam) {
            m_hat_prev = b1 * opt->m[i] * t_inv1;
        }

        opt->m[i] = b1 * opt->m[i] + (1.0 - b1) * grads[i];
        opt->v[i] = b2 * opt->v[i] + (1.0 - b2) * grads[i] * grads[i];
        
        double m_hat = opt->m[i] * t_inv1;
        double v_hat = opt->v[i] * t_inv2;

        if (is_nadam) {
            double m_nadam = m_hat + b1 * (m_hat - m_hat_prev);
            params[i] -= lr * m_nadam / (sqrt(v_hat) + eps);
        } else {
            params[i] -= lr * m_hat / (sqrt(v_hat) + eps);
        }
    }

    /* ---- Resonance perturbation (identical to Lua) ------------------- */
    double phi = PHI;
    double t   = (double)opt->t * 0.1;
    double intensity =
          sin(PI/20.0 * t) * exp(-PI/20.0  * t)
        + sin(PI/10.0 * t) * exp(-PI/10.0  * t)
        + sin(PI/9.0  * t) * exp(-PI/9.0   * t)
        + sin((1.0/(phi*phi))*t) * exp(-(1.0/(phi*phi))*t);

    /* compute l2‑norm of the gradient */
    double gnorm2 = 0.0;
    for (int i = 1; i <= n; ++i) gnorm2 += grads[i]*grads[i];
    double gnorm = sqrt(gnorm2);
    if (gnorm > 1e-10) {
        double pert = opt->resonance_amplitude * intensity;
        for (int i = 1; i <= n; ++i) {
            params[i] -= pert * grads[i] / gnorm;
        }
    }
}

/* --------------------------------------------------------------------
   4.  BAHA utilities – fracture detection & Lambert‑W branch enumeration
-------------------------------------------------------------------- */
typedef struct {
    double sigma_factor;
    int    n;
    double mean, M2, last_rate;
    double pb, plZ;          /* previous β , lZ */
} FractureDetector;

static FractureDetector *fd_new(double sigma_factor)
{
    FractureDetector *fd = calloc(1, sizeof(FractureDetector));
    fd->sigma_factor = sigma_factor;
    return fd;
}
static void fd_record(FractureDetector *fd, double beta, double lZ)
{
    if (fd->pb != 0.0 && fd->plZ != 0.0) {
        double db = beta - fd->pb;
        if (fabs(db) > 1e-9) {
            double rate = fabs(lZ - fd->plZ) / db;
            fd->last_rate = rate;
            ++fd->n;
            double delta = rate - fd->mean;
            fd->mean += delta / fd->n;
            fd->M2   += delta * (rate - fd->mean);
        }
    }
    fd->pb = beta; fd->plZ = lZ;
}
static int fd_is_fracture(FractureDetector *fd)
{
    if (fd->n < 5) return 0;
    double s = sqrt(fd->M2 / (fd->n - 1));
    return fd->last_rate > fd->mean + fd->sigma_factor * s;
}
static void fd_clear(FractureDetector *fd)
{
    fd->n = fd->mean = fd->M2 = fd->last_rate = 0.0;
    fd->pb = fd->plZ = 0.0;
}

/* ---- simple Lambert‑W (principal branch) via Halley's method ------- */
static double lambert_W0(double z)
{
    if (z < -E_INV) return NAN;
    /* Initial guess */
    double w = (z < -0.3) ? (z * exp(1.0))
             : (z < 1.0) ? (z * (1.0 - z + z*z))
                        : (log(z) - log(log(z) + 1.0));
    for (int i = 0; i < 50; ++i) {
        double ew = exp(w);
        double f  = w*ew - z;
        double fp = ew * (w + 1.0);
        if (fabs(fp) < 1e-15) break;
        double denom = fp - f * (ew * (w + 2.0)) / (2.0 * fp);
        if (fabs(denom) < 1e-15) break;
        double w2 = w - f/denom;
        if (fabs(w2-w) < 1e-10) { w=w2; break; }
        w = w2;
    }
    return w;
}

/* Enumerate possible branches of “β – βc = u ” using the transformation
   u·e^u = ξ,  ξ = u·exp(u).  The Lua code returns at most two branches
   (k=0 → principal, k=-1 → secondary). */
typedef struct { int k; double beta; double score; } Branch;
static Branch *enumerate_branches(double bc, double b, int *out_cnt)
{
    double u = b - bc;
    if (fabs(u) < 1e-12) u = 1e-12;
    double xi = u * exp(u);
    Branch *br = malloc(2 * sizeof(Branch));
    int cnt = 0;
    double w0 = lambert_W0(xi);
    if (!isnan(w0)) {
        double b0 = bc + w0;
        if (b0 > 0) { br[cnt++] = (Branch){0, b0, 0.0}; }
    }
    if (xi >= -E_INV && xi < 0) {
        /* secondary branch (W_{-1}) – not needed for most instances */
        double w1 = NAN; /* stub – could implement W_{-1} similarly. */
        (void)w1;
    }
    *out_cnt = cnt;
    return br;
}

/* --------------------------------------------------------------------
   5.  Persistent homology – only Betti‑0/1 are needed.
-------------------------------------------------------------------- */
typedef struct {
    int *parent;
    int *rank;
    int  n, count;
} UnionFind;
static UnionFind *uf_new(int n)
{
    UnionFind *uf = calloc(1, sizeof(UnionFind));
    uf->n = n;
    uf->parent = malloc((n+1)*sizeof(int));
    uf->rank   = calloc(n+1, sizeof(int));
    uf->count  = n;
    for (int i = 1; i <= n; ++i) uf->parent[i] = i;
    return uf;
}
static int uf_find(UnionFind *uf, int x)
{
    if (uf->parent[x] != x) uf->parent[x] = uf_find(uf, uf->parent[x]);
    return uf->parent[x];
}
static int uf_union(UnionFind *uf, int a, int b)
{
    int pa = uf_find(uf, a);
    int pb = uf_find(uf, b);
    if (pa == pb) return 0;
    if (uf->rank[pa] < uf->rank[pb]) { int t=pa; pa=pb; pb=t; }
    else if (uf->rank[pa] == uf->rank[pb]) ++uf->rank[pa];
    uf->parent[pb] = pa;
    --uf->count;
    return 1;
}

/* --------------------------------------------------------------------
   6.  Topological Structures
-------------------------------------------------------------------- */
typedef struct {
    int    beta0, beta1;
    int    active_vars, edge_count;
    double avg_confidence, complexity_score;
} TopologyInfo;

typedef struct {
    int initial_beta0, final_beta0;
    int initial_beta1, final_beta1;
    double initial_complexity, final_complexity;
    int has_initial;
    int birth_step_0, birth_step_1;
    int prev_beta0, prev_beta1;
    int persistence_events;
} PersistenceTracker;

static void update_persistence(PersistenceTracker *pt, int beta0, int beta1, int step, double complexity) {
    if (!pt->has_initial) {
        pt->initial_beta0 = beta0;
        pt->initial_beta1 = beta1;
        pt->initial_complexity = complexity;
        pt->has_initial = 1;
        pt->prev_beta0 = beta0;
        pt->prev_beta1 = beta1;
    }
    pt->final_beta0 = beta0;
    pt->final_beta1 = beta1;
    pt->final_complexity = complexity;

    if (beta0 > pt->prev_beta0) pt->birth_step_0 = step;
    if (beta1 > pt->prev_beta1) pt->birth_step_1 = step;

    if (beta0 < pt->prev_beta0 && pt->birth_step_0 > 0) pt->persistence_events++;
    if (beta1 < pt->prev_beta1 && pt->birth_step_1 > 0) pt->persistence_events++;

    pt->prev_beta0 = beta0;
    pt->prev_beta1 = beta1;
}

/* --------------------------------------------------------------------
   7.  Instance representation (flattened clauses)
-------------------------------------------------------------------- */
typedef struct {
    int    num_vars;
    int    num_clauses;
    int   *cl_offs;    /* size num_clauses+1 */
    int   *cl_flat;    /* flattened literals */
} Instance;

/* --------------------------------------------------------------------
   7.  Solver (NitroSat) structure
-------------------------------------------------------------------- */
typedef struct {
    /* instance data */
    Instance *inst;
    /* flattened clause structures */
    const int *cl_offs;
    const int *cl_flat;
    int    num_vars, num_clauses;
    int    *v2c_ptr;
    int    *v2c_data;
    /* per‑variable data */
    double *x;          /* current (continuous) assignment */
    double *degrees;    /* degree = Σ(k−1) per variable */
    uint8_t *decimated;/* 0 = free, 1 = locked */
    /* clause data */
    double *cl_weights;     /* prime‑weighted base weight */
    double *zeta_weights;   /* log(p)/p   – used by Zeta perturbation   */
    uint8_t *is_hard;       /* 0 = soft, 1 = hard (unused in current code) */
    /* optimisation state */
    Optimizer *opt;
    /* BAHA utilities */
    FractureDetector *fd;
    double baha_beta_c;  /* β at max variance (critical β) */
    double baha_var_peak;/* max variance observed so far   */
    int    baha_var_n;
    double baha_var_mean, baha_var_M2;
    /* persistent homology */
    int    use_topology;
    TopologyInfo last_topology;
    PersistenceTracker pt;
    /* prime table */
    int    *primes;
    int    prime_cnt;
    double sigma_factor; /* for fracture detector – user tunable */
    /* other knobs */
    double heat_beta, heat_lambda;
    double entropy_weight;
    double dec_threshold;
    int    dec_freq;     /* lock variables every N steps */
    int    max_steps;
    int    verbose;
    /* hot-path buffers */
    double *grad_buffer;
    double *heat_mult_buffer;
    int    *sat_counts; /* number of satisfied literals per clause */
} NitroSat;

static void recompute_sat_counts(NitroSat *ns);

static TopologyInfo compute_topology(NitroSat *ns, double th)
{
    int num_vars = ns->num_vars;
    int num_clauses = ns->num_clauses;
    const int *cl_offs = ns->cl_offs;
    const int *cl_flat = ns->cl_flat;
    const double *x = ns->x;

    TopologyInfo topo = {0};
    UnionFind *uf = uf_new(num_vars);
    int *is_active = calloc(num_vars + 1, sizeof(int));
    int active_var_cnt = 0;

    /* 1. BFS/Union-Find for Beta-0 and identify active vars */
    recompute_sat_counts(ns);
    for (int c = 0; c < num_clauses; ++c) {
        if (ns->sat_counts[c] > 0) continue;
        int s = cl_offs[c], e = cl_offs[c+1];

        int first_v = -1;
        for (int i=s; i<e; ++i) {
            int v = abs(cl_flat[i]);
            if (!is_active[v]) { is_active[v] = 1; active_var_cnt++; }
            if (first_v == -1) first_v = v;
            else uf_union(uf, first_v, v);
        }
    }

    /* 2. Sparse Edge Counting for Beta-1 */
    long long edge_cnt = 0;
    unsigned char *seen = calloc((num_vars / 8) + 1, 1);
    #define SET_BIT(a, b) (a[(b)>>3] |= (1 << ((b)&7)))
    #define GET_BIT(a, b) (a[(b)>>3] & (1 << ((b)&7)))
    #define CLR_BIT(a, b) (a[(b)>>3] &= ~(1 << ((b)&7)))

    int *neighbor_list = malloc(1024 * sizeof(int));
    int cap = 1024;

    for (int u = 1; u <= num_vars; ++u) {
        if (!is_active[u]) continue;
        int neighbors_found = 0;

        for (int p = ns->v2c_ptr[u]; p < ns->v2c_ptr[u+1]; ++p) {
            int c = ns->v2c_data[p];
            if (ns->sat_counts[c] > 0) continue;

            for (int i=cl_offs[c]; i<cl_offs[c+1]; ++i) {
                int v = abs(cl_flat[i]);
                if (v > u && !GET_BIT(seen, v)) {
                    SET_BIT(seen, v);
                    edge_cnt++;
                    if (neighbors_found >= cap) {
                        cap *= 2;
                        neighbor_list = realloc(neighbor_list, cap * sizeof(int));
                    }
                    neighbor_list[neighbors_found++] = v;
                }
            }
        }
        for (int i=0; i<neighbors_found; ++i) CLR_BIT(seen, neighbor_list[i]);
    }
    free(neighbor_list);

    topo.beta0 = uf->count - (num_vars - active_var_cnt);
    if (topo.beta0 < 0) topo.beta0 = 0;
    topo.edge_count = (int)edge_cnt;
    topo.active_vars = active_var_cnt;
    
    int chi = active_var_cnt - (int)edge_cnt;
    topo.beta1 = (topo.beta0 - chi > 0) ? (topo.beta0 - chi) : 0;

    double conf_sum = 0.0;
    for (int i=1;i<=num_vars;++i) if (is_active[i]) conf_sum += fabs(x[i]-0.5);
    topo.avg_confidence = (active_var_cnt ? conf_sum/active_var_cnt : 0.0);
    topo.complexity_score = (topo.beta0>0 ? (double)topo.beta1 / topo.beta0 : 0.0);

    free(is_active); free(seen); free(uf->parent); free(uf->rank); free(uf);
    return topo;
}

static Instance *read_cnf(const char *filename)
{
    FILE *fp = fopen(filename, "r");
    if (!fp) { perror("fopen"); return NULL; }

    char line[4096];
    int vars = 0, cls = 0;
    while (fgets(line, sizeof(line), fp)) {
        if (line[0] == 'c') continue;
        if (line[0] == 'p') {
            if (sscanf(line, "p cnf %d %d", &vars, &cls) != 2) {
                fprintf(stderr, "Bad problem line.\n"); fclose(fp); return NULL;
            }
            break;
        }
    }
    if (!vars) { fprintf(stderr, "No problem line.\n"); fclose(fp); return NULL; }

    int *offs = malloc((cls+1) * sizeof(int));
    int *flat = NULL;
    int flat_sz = 0, flat_cap = 0;
    int cur = 0;
    offs[0] = 0;

    while (fgets(line, sizeof(line), fp)) {
        if (line[0] == 'c' || line[0] == 'p' || line[0]=='\n') continue;
        char *tok = strtok(line, " \t\r\n");
        while (tok) {
            int lit = atoi(tok);
            if (lit == 0) break;
            if (abs(lit) > vars) {
                fprintf(stderr, "Literal %d exceeds declared variable count %d\n", lit, vars);
                free(offs); free(flat); fclose(fp); return NULL;
            }
            if (flat_sz == flat_cap) {
                flat_cap = flat_cap ? flat_cap*2 : 1024;
                flat = realloc(flat, flat_cap * sizeof(int));
            }
            flat[flat_sz++] = lit;
            tok = strtok(NULL, " \t\r\n");
        }
        ++cur;
        offs[cur] = flat_sz;
        if (cur == cls) break;
    }
    fclose(fp);
    Instance *inst = malloc(sizeof(Instance));
    inst->num_vars    = vars;
    inst->num_clauses = cls;
    inst->cl_offs     = offs;
    inst->cl_flat     = flat;
    return inst;
}

static void nitrosat_free(NitroSat *ns)
{
    if (!ns) return;
    free(ns->x);
    free(ns->degrees);
    free(ns->decimated);
    free(ns->cl_weights);
    free(ns->zeta_weights);
    free(ns->is_hard);
    if (ns->opt) {
        free(ns->opt->m);
        free(ns->opt->v);
        free(ns->opt);
    }
    free(ns->fd);
    free(ns->primes);
    free(ns->v2c_ptr);
    free(ns->v2c_data);
    free(ns->grad_buffer);
    free(ns->heat_mult_buffer);
    free(ns->sat_counts);
    free(ns);
}

/* --------------------------------------------------------------------
   8.  Helper – compute vertex degrees (used for heat kernel)
-------------------------------------------------------------------- */
static void compute_degrees(NitroSat *ns)
{
    int n = ns->num_vars;
    ns->degrees = calloc(n+1, sizeof(double));
    for (int c = 0; c < ns->num_clauses; ++c) {
        int s = ns->cl_offs[c];
        int e = ns->cl_offs[c+1];
        int k = e - s;                     /* clause length */
        for (int i=s; i<e; ++i) {
            int lit = ns->cl_flat[i];
            int v = abs(lit);
            ns->degrees[v] += (double)(k-1);
        }
    }
}

/* --------------------------------------------------------------------
   8b. Spectral initialisation – XOR-Laplacian power iteration
        (matches gh.lua xor_laplacian.spectral_init)
-------------------------------------------------------------------- */
static void spectral_init(NitroSat *ns, int iters)
{
    int nv = ns->num_vars;
    int nc = ns->num_clauses;
    double *v = malloc((nv+1) * sizeof(double));
    double *y = malloc((nv+1) * sizeof(double));

    /* random initial vector */
    for (int i = 1; i <= nv; ++i) v[i] = rng_double();

    for (int it = 0; it < iters; ++it) {
        /* y = D*v  (degree-weighted) */
        for (int i = 1; i <= nv; ++i) y[i] = ns->degrees[i] * v[i];

        /* subtract off-diagonal: for each clause, y[vi] -= (sum - v[vi]) */
        for (int c = 0; c < nc; ++c) {
            int s = ns->cl_offs[c];
            int e = ns->cl_offs[c+1];
            double sum = 0.0;
            for (int i = s; i < e; ++i) sum += v[abs(ns->cl_flat[i])];
            for (int i = s; i < e; ++i) {
                int vi = abs(ns->cl_flat[i]);
                y[vi] -= (sum - v[vi]);
            }
        }

        /* normalise */
        double norm = 0.0;
        for (int i = 1; i <= nv; ++i) norm += y[i] * y[i];
        norm = sqrt(norm);
        if (norm > 1e-10)
            for (int i = 1; i <= nv; ++i) v[i] = y[i] / norm;
        else
            for (int i = 1; i <= nv; ++i) v[i] = y[i];
    }

    /* rescale to [0,1] */
    double mi = 1e300, ma = -1e300;
    for (int i = 1; i <= nv; ++i) {
        if (v[i] < mi) mi = v[i];
        if (v[i] > ma) ma = v[i];
    }
    double r = ma - mi;
    if (r > 1e-10)
        for (int i = 1; i <= nv; ++i) v[i] = (v[i] - mi) / r;

    /* store with small noise */
    for (int i = 1; i <= nv; ++i)
        ns->x[i] = v[i] + (rng_double() - 0.5) * 0.01;

    free(v);
    free(y);
}

/* --------------------------------------------------------------------
   9.  Initialise a NitroSat object
-------------------------------------------------------------------- */
static NitroSat *nitrosat_new(Instance *inst, int max_steps, int verbose)
{
    NitroSat *ns = calloc(1, sizeof(NitroSat));
    ns->inst       = inst;
    ns->cl_offs    = inst->cl_offs;
    ns->cl_flat    = inst->cl_flat;
    ns->num_vars   = inst->num_vars;
    ns->num_clauses= inst->num_clauses;
    ns->max_steps  = max_steps;
    ns->verbose    = verbose;

    /* degree vector (heat kernel) */
    compute_degrees(ns);

    /* continuous assignment – spectral initialisation (power iteration) */
    ns->x = malloc((ns->num_vars+1) * sizeof(double));
    spectral_init(ns, 50);  /* 50 iterations, matches gh.lua */

    /* clause weight initialisation (prime‑weighted) */
    ns->primes = sieve_primes(ns->num_clauses, &ns->prime_cnt);
    ns->cl_weights = malloc(ns->num_clauses * sizeof(double));
    ns->zeta_weights = malloc(ns->num_clauses * sizeof(double));
    for (int c=0;c<ns->num_clauses;++c) {
        int p = (c < ns->prime_cnt) ? ns->primes[c] : 2;
        ns->cl_weights[c]   = 1.0 / pow(1.0 + log((double)p), 1.0);   /* prime_alpha = 1.0 */
        ns->zeta_weights[c] = log((double)p) / (double)p;
    }
    ns->is_hard = malloc(ns->num_clauses);
    for (int c=0;c<ns->num_clauses;++c) ns->is_hard[c] = 1;   /* all hard initially, matches gh.lua */

    /* decimation vector */
    ns->decimated = calloc(ns->num_vars+1, 1);

    /* optimiser – NADAM (the Lua default) */
    ns->opt = optimizer_new("nadam", ns->x, ns->num_vars,
                            0.002, 0.9, 0.999, 1e-8, 0.02);

    /* BAHA detector */
    ns->fd = fd_new(1.5);
    ns->sigma_factor = 1.5;   /* same as BAHA default */
    ns->baha_var_peak = -1e100;

    /* parameters */
    ns->heat_beta     = 0.5;
    ns->heat_lambda   = 0.1;
    ns->entropy_weight= 0.01;
    ns->dec_threshold = 0.49;
    ns->dec_freq      = 100;
    ns->use_topology  = 1;   /* enable topology snapshots */

    ns->grad_buffer      = calloc(ns->num_vars + 1, sizeof(double));
    ns->heat_mult_buffer = malloc((ns->num_vars + 1) * sizeof(double));
    ns->sat_counts       = calloc(ns->num_clauses, sizeof(int));

    /* pre-calculate heat multipliers */
    for (int i=1;i<=ns->num_vars;++i) {
        double w = exp(-ns->heat_beta * ns->degrees[i]);
        ns->heat_mult_buffer[i] = 1.0 + ns->heat_lambda * w;
    }

    return ns;
}

/* heat‑kernel multipliers are pre‑calculated in nitrosat_new */

/* --------------------------------------------------------------------
   11. Compute the gradient of the smooth Log‑Barrier objective.
        Returns the gradient vector (allocated) and the current number
        of unsatisfied clauses.
-------------------------------------------------------------------- */
#define MAX_CLAUSE_SIZE 256
static int compute_gradients(NitroSat *ns)
{
    int n = ns->num_vars;
    int m = ns->num_clauses;
    double *grad = ns->grad_buffer;
    for (int i=0; i<=n; ++i) grad[i] = 0.0;
    int unsat = 0;

    for (int c = 0; c < ns->num_clauses; ++c) {
        int s = ns->cl_offs[c];
        int e = ns->cl_offs[c+1];
        double violation = 1.0;
        int sat = 0;

        for (int i = s; i < e; ++i) {
            int lit = ns->cl_flat[i];
            int v   = abs(lit);
            double val = ns->x[v];
            double lv = (lit > 0) ? (1.0 - val) : val;
            if (lv < 1e-12) lv = 1e-12;
            violation *= lv;
            if ((lit > 0 && val > 0.5) || (lit < 0 && val <= 0.5)) {
                sat = 1;
                /* If we only need the gradient for unsat clauses, we could skip here.
                   But NitroSAT needs gradients even for satisfied clauses to maintain smooth barrier. */
            }
        }
        if (!sat) ++unsat;

        double barrier = 1.0 / (1e-6 + (1.0 - violation));
        double w = ns->cl_weights[c];
        if (ns->is_hard[c]) w *= 50.0;
        double coef = w * barrier * violation;

        for (int i = s; i < e; ++i) {
            int lit = ns->cl_flat[i];
            int v   = abs(lit);
            double val = ns->x[v];
            double lv = (lit > 0) ? (1.0 - val) : val;
            if (lv < 1e-12) lv = 1e-12;
            grad[v] += coef * (((lit > 0) ? -1.0 : 1.0) / lv);
        }
    }

    /* heat‑kernel smoothing (pre‑calculated) + entropy regularisation */
    for (int i=1;i<=n;++i) {
        grad[i] *= ns->heat_mult_buffer[i];
        double v_clamped = clamp(ns->x[i], 1e-9, 1.0-1e-9);
        grad[i] += ns->entropy_weight * log((1.0 - v_clamped) / v_clamped);
    }
    return unsat;
}

/* --------------------------------------------------------------------
   12. Simple annealing schedule – produces a learning rate
-------------------------------------------------------------------- */
static double annealing_lr(double t, double A0)
{
    double phi = PHI;
    double s = sin(PI/20.0 * t) * exp(-PI/20.0 * t)
             + sin(PI/10.0 * t) * exp(-PI/10.0 * t)
             + sin(PI/9.0  * t) * exp(-PI/9.0  * t)
             + sin((1.0/(phi*phi))*t) * exp(-(1.0/(phi*phi))*t);
    return fmax(0.0, A0 * s);
}

/* --------------------------------------------------------------------
   13. Satisfaction test (discrete interpretation)
-------------------------------------------------------------------- */
static void recompute_sat_counts(NitroSat *ns)
{
    memset(ns->sat_counts, 0, ns->num_clauses * sizeof(int));
    for (int c=0; c<ns->num_clauses; ++c) {
        for (int i=ns->cl_offs[c]; i<ns->cl_offs[c+1]; ++i) {
            int lit = ns->cl_flat[i];
            int v = abs(lit);
            if ((lit > 0 && ns->x[v] > 0.5) || (lit < 0 && ns->x[v] <= 0.5)) {
                ns->sat_counts[c]++;
            }
        }
    }
}

/* --------------------------------------------------------------------
   13. Satisfaction test (discrete interpretation)
-------------------------------------------------------------------- */
static int check_satisfaction(NitroSat *ns)
{
    int sat = 0;
    for (int c=0; c<ns->num_clauses; ++c) {
        if (ns->sat_counts[c] > 0) sat++;
    }
    return sat;
}

/* --------------------------------------------------------------------
   14. Variable decimation (soft locking of confident variables)
-------------------------------------------------------------------- */
static int decimate(NitroSat *ns)
{
    int locked = 0;
    for (int i=1;i<=ns->num_vars;++i) if (!ns->decimated[i]) {
        double conf = fabs(ns->x[i] - 0.5);
        if (conf > ns->dec_threshold) {
            ns->x[i] = (ns->x[i] > 0.5) ? 1.0 : 0.0;
            ns->decimated[i] = 1;
            ++locked;
        }
    }
    return locked;
}

/* --------------------------------------------------------------------
   15. Zeta‑zero perturbation (standard & “nuclear” version)
-------------------------------------------------------------------- */
static double zeta_zero_perturb(int var_idx, int step, int max_steps,
                               const int *primes, int prime_cnt,
                               const TopologyInfo *topo, int num_vars,
                               int nuclear)
{
    double progress = (double)step / max_steps;
    double val = 0.0;
    int limit = (prime_cnt < 256) ? prime_cnt : 256;

    for (int i=0; i<limit; ++i) {
        int p = primes[i];
        double phase = 2.0*PI * ((double)p * var_idx) / ((log((double)p))*num_vars);
        double amplitude = 1.0 / sqrt((double)p);
        if (nuclear && topo) {
            double crit = sin(2.0*PI*0.5*progress * p); /* crude critical‑line term */
            val += amplitude * sin(phase) * crit;
        } else {
            val += amplitude * sin(phase);
        }
    }

    if (nuclear && topo && topo->complexity_score > 0.5)
        val *= 1.0 + topo->complexity_score;
    if (progress > 0.9) val *= 2.0;
    return 0.001 * val;   /* tiny step – same scale as Lua's 0.001 */
}

/* --------------------------------------------------------------------
   16. Zeta‑sweep – high‑precision stochastic repair
-------------------------------------------------------------------- */
static int zeta_sweep(NitroSat *ns, double beta)
{
    double *dF = calloc(ns->num_vars+1, sizeof(double));
    int unsat = 0;
    int *unsat_cls = malloc(ns->num_clauses * sizeof(int));

    for (int c=0;c<ns->num_clauses;++c) {
        int s = ns->cl_offs[c];
        int e = ns->cl_offs[c+1];
        int sat = 0;
        for (int i=s;i<e && !sat;++i) {
            int lit = ns->cl_flat[i];
            int v = abs(lit);
            double val = ns->x[v];
            if ((lit>0 && val>0.5) || (lit<0 && val<=0.5)) sat = 1;
        }
        if (!sat) {
            unsat_cls[unsat++] = c;
            double w = ns->zeta_weights[c] * ns->cl_weights[c] * 2.0;
            for (int i=s;i<e;++i) {
                int lit = ns->cl_flat[i];
                int v = abs(lit);
                dF[v] += (lit>0 ? 1.0 : -1.0) * w;
            }
        }
    }

    int flips = 0;
    for (int i=1;i<=ns->num_vars;++i) {
        if (fabs(dF[i]) > 1e-12) {
            double p = 1.0 / (1.0 + exp(-beta * fabs(dF[i])));
            if (rng_double() < p) {
                ns->x[i] = (dF[i] > 0.0) ? 0.99 : 0.01;
                ++flips;
            }
        }
    }

    free(dF); free(unsat_cls);
    return (unsat == 0) ? 1 : 0;   /* 1 = success */
}

/* --------------------------------------------------------------------
   17. BAHA‑WalkSAT – phase‑transition aware discrete local search
-------------------------------------------------------------------- */
static int baha_walksat(NitroSat *ns, int max_flips)
{
    int n = ns->num_vars, m = ns->num_clauses;
    recompute_sat_counts(ns);

    int *unsat = malloc(m * sizeof(int));
    int *unsat_pos = malloc(m * sizeof(int));
    for (int c = 0; c < m; ++c) unsat_pos[c] = -1;

    int unsz = 0;
    for (int c = 0; c < m; ++c) {
        if (ns->sat_counts[c] == 0) {
            unsat_pos[c] = unsz;
            unsat[unsz++] = c;
        }
    }

    if (unsz == 0) { free(unsat); free(unsat_pos); return 1; }

    for (int step = 1; step <= max_flips; ++step) {
        if (unsz == 0) break;

        double beta = 0.5 + 4.5 * ((double)step / max_flips);
        int choice = rng_int(0, unsz - 1);
        int c_idx = unsat[choice];

        int best_v = -1, best_delta = 999999;
        int cs = ns->cl_offs[c_idx];
        int ce = ns->cl_offs[c_idx + 1];

        for (int i = cs; i < ce; ++i) {
            int v = abs(ns->cl_flat[i]);
            int br = 0;
            int old_v_sat = (ns->x[v] > 0.5);

            for (int p = ns->v2c_ptr[v]; p < ns->v2c_ptr[v+1]; ++p) {
                int c = ns->v2c_data[p];
                if (ns->sat_counts[c] == 1) {
                    /* check if flipping v breaks c */
                    int lit_p = 0;
                    for (int q = ns->cl_offs[c]; q < ns->cl_offs[c+1]; ++q) {
                        if (abs(ns->cl_flat[q]) == v) { lit_p = (ns->cl_flat[q] > 0); break; }
                    }
                    if (lit_p == old_v_sat) br++;
                }
            }
            if (br < best_delta) { best_delta = br; best_v = v; }
        }

        if (best_v != -1 && (best_delta == 0 || rng_double() < exp(-beta * best_delta))) {
            int old_v_sat = (ns->x[best_v] > 0.5);
            ns->x[best_v] = (old_v_sat) ? 0.0 : 1.0;
            int new_v_sat = !old_v_sat;

            for (int p = ns->v2c_ptr[best_v]; p < ns->v2c_ptr[best_v + 1]; ++p) {
                int c = ns->v2c_data[p];
                int lit_p = 0;
                for (int q = ns->cl_offs[c]; q < ns->cl_offs[c+1]; ++q) {
                   if (abs(ns->cl_flat[q]) == best_v) { lit_p = (ns->cl_flat[q] > 0); break; }
                }
                int was_sat = ns->sat_counts[c] > 0;
                if (lit_p == new_v_sat) ns->sat_counts[c]++;
                else ns->sat_counts[c]--;
                int is_sat = ns->sat_counts[c] > 0;

                if (!was_sat && is_sat) {
                    int pos = unsat_pos[c];
                    if (pos != -1) {
                        int last_c = unsat[unsz - 1];
                        unsat[pos] = last_c;
                        unsat_pos[last_c] = pos;
                        unsat_pos[c] = -1;
                        unsz--;
                    }
                } else if (was_sat && !is_sat) {
                    if (unsat_pos[c] == -1) {
                        unsat_pos[c] = unsz;
                        unsat[unsz++] = c;
                    }
                }
            }
        }
    }
    int solved = (unsz == 0);
    free(unsat); free(unsat_pos);
    return solved;
}

/* --------------------------------------------------------------------
   18. Core decomposition – identify the unsatisfied core and “blast’’
        it by temporarily inflating clause weights.
-------------------------------------------------------------------- */
static int core_decomposition(NitroSat *ns)
{
    recompute_sat_counts(ns);
    int *core = malloc(ns->num_clauses * sizeof(int));
    int core_sz = 0;
    for (int c=0;c<ns->num_clauses;++c) {
        if (ns->sat_counts[c] == 0) core[core_sz++] = c;
    }
    if (core_sz==0) { free(core); return 1; }

    /* stochastic WalkSAT on the core */
    int *saved = malloc(ns->num_clauses * sizeof(double));
    memcpy(saved, ns->cl_weights, ns->num_clauses * sizeof(double));
    for (int i=0;i<core_sz;++i) ns->cl_weights[core[i]] *= 100.0;

    int max_flips = ns->num_vars * 5;
    if (max_flips > 1000000) max_flips = 1000000;
    int success = 0;

    for (int flips=0; flips < max_flips; ++flips) {
        int c_idx = rand() % core_sz;
        int c = core[c_idx];
        if (ns->sat_counts[c] > 0) continue;

        int cs = ns->cl_offs[c];
        int ce = ns->cl_offs[c+1];
        int var = abs(ns->cl_flat[cs + (rand() % (ce-cs))]);
        
        double old_x = ns->x[var];
        ns->x[var] = (ns->x[var] > 0.5) ? 0.0 : 1.0;
        
        /* Update sat_counts for var flip */
        for (int p = ns->v2c_ptr[var]; p < ns->v2c_ptr[var+1]; ++p) {
            int cl = ns->v2c_data[p];
            int sc = 0;
            for (int j=ns->cl_offs[cl]; j<ns->cl_offs[cl+1]; ++j) {
                int l = ns->cl_flat[j];
                if ((l>0 && ns->x[abs(l)]>0.5) || (l<0 && ns->x[abs(l)]<=0.5)) sc++;
            }
            ns->sat_counts[cl] = sc;
        }

        if (flips % 10000 == 0) {
            int still_unsat = 0;
            for (int i=0; i<core_sz; ++i) if (ns->sat_counts[core[i]] == 0) { still_unsat = 1; break; }
            if (!still_unsat) { success = 1; break; }
        }
    }

    memcpy(ns->cl_weights, saved, ns->num_clauses * sizeof(double));
    free(core); free(saved);
    return success;
}

static double score_branch(NitroSat *ns, double beta, int n_samples, double *best_x_out)
{
    if (beta <= 0) return -1e300;
    double total_score = 0.0;
    double best_seen = 1e300;
    double *buf = malloc((ns->num_vars + 1) * sizeof(double));
    
    for (int s = 0; s < n_samples; ++s) {
        for (int i = 1; i <= ns->num_vars; ++i) buf[i] = ns->decimated[i] ? ns->x[i] : rng_double();
        int uns = 0;
        for (int c = 0; c < ns->num_clauses; ++c) {
            int sat = 0;
            for (int q = ns->cl_offs[c]; q < ns->cl_offs[c+1]; ++q) {
                int lit = ns->cl_flat[q], v = abs(lit);
                if ((lit > 0 && buf[v] > 0.5) || (lit < 0 && buf[v] <= 0.5)) { sat = 1; break; }
            }
            if (!sat) uns++;
        }
        total_score += exp(-beta * uns);
        if (uns < best_seen) {
            best_seen = uns;
            if (best_x_out) {
                for (int i = 1; i <= ns->num_vars; ++i) best_x_out[i] = buf[i];
            }
        }
    }
    free(buf);
    return total_score / n_samples + 100.0 / (best_seen + 1.0);
}

/* --------------------------------------------------------------------
   [PORTED] Restricted WalkSAT (Local search on subset of variables)
-------------------------------------------------------------------- */
static int restricted_walksat(NitroSat *ns, int *var_list, int var_count, int max_flips)
{
    if (var_count == 0) return 1;
    
    int *clause_sat_counts = calloc(ns->num_clauses, sizeof(int));
    int *unsat_list = malloc(ns->num_clauses * sizeof(int));
    int *unsat_pos = calloc(ns->num_clauses + 1, sizeof(int));
    int unsz = 0;
    
    for (int c = 0; c < ns->num_clauses; ++c) {
        int count = 0;
        for (int i = ns->cl_offs[c]; i < ns->cl_offs[c+1]; ++i) {
            int lit = ns->cl_flat[i], var = abs(lit);
            if ((lit > 0 && ns->x[var] > 0.5) || (lit < 0 && ns->x[var] <= 0.5)) count++;
        }
        clause_sat_counts[c] = count;
        if (count == 0) {
            unsat_list[unsz] = c;
            unsat_pos[c] = unsz + 1;
            unsz++;
        }
    }
    if (unsz == 0) { free(clause_sat_counts); free(unsat_list); free(unsat_pos); return 1; }
    
    int *var_set = calloc(ns->num_vars + 1, sizeof(int));
    for (int i = 0; i < var_count; ++i) var_set[var_list[i]] = 1;

    for (int step = 1; step <= max_flips; ++step) {
        if (unsz == 0) break;
        int clause_idx = unsat_list[rng_int(0, unsz - 1)];
        int start = ns->cl_offs[clause_idx], stop = ns->cl_offs[clause_idx+1];
        int flip_var = -1;
        
        if (rng_double() < 0.3) {
            int cand[128], csz = 0;
            for (int i = start; i < stop; ++i) {
                int v = abs(ns->cl_flat[i]);
                if (var_set[v] && csz < 128) cand[csz++] = v;
            }
            if (csz > 0) flip_var = cand[rng_int(0, csz - 1)];
        } else {
            int min_breaks = 999999, cand[128], csz = 0;
            for (int i = start; i < stop; ++i) {
                int var = abs(ns->cl_flat[i]);
                if (!var_set[var]) continue;
                int breaks = 0;
                for (int p = ns->v2c_ptr[var]; p < ns->v2c_ptr[var+1]; ++p) {
                    int k = ns->v2c_data[p];
                    if (clause_sat_counts[k] == 1) {
                        for (int j = ns->cl_offs[k]; j < ns->cl_offs[k+1]; ++j) {
                            int k_lit = ns->cl_flat[j];
                            if (abs(k_lit) == var) {
                                if ((k_lit > 0 && ns->x[var] > 0.5) || (k_lit < 0 && ns->x[var] <= 0.5)) breaks++;
                                break;
                            }
                        }
                    }
                }
                if (breaks < min_breaks) { min_breaks = breaks; csz = 0; cand[csz++] = var; }
                else if (breaks == min_breaks && csz < 128) { cand[csz++] = var; }
            }
            if (csz > 0) flip_var = cand[rng_int(0, csz - 1)];
        }
        
        if (flip_var == -1) {
            for (int i = start; i < stop; ++i) {
                int v = abs(ns->cl_flat[i]);
                if (var_set[v]) { flip_var = v; break; }
            }
        }
        
        if (flip_var != -1) {
            double new_val = (ns->x[flip_var] > 0.5) ? 0.0 : 1.0;
            ns->x[flip_var] = new_val;
            for (int p = ns->v2c_ptr[flip_var]; p < ns->v2c_ptr[flip_var+1]; ++p) {
                int k = ns->v2c_data[p];
                int old_count = clause_sat_counts[k], delta = 0;
                for (int j = ns->cl_offs[k]; j < ns->cl_offs[k+1]; ++j) {
                    int lit = ns->cl_flat[j];
                    if (abs(lit) == flip_var) {
                        delta = ((lit > 0 && new_val > 0.5) || (lit < 0 && new_val <= 0.5)) ? 1 : -1;
                        break;
                    }
                }
                int new_count = old_count + delta;
                clause_sat_counts[k] = new_count;
                if (old_count == 0 && new_count > 0) {
                    int pos = unsat_pos[k];
                    if (pos) {
                        int last_c = unsat_list[unsz - 1];
                        unsat_list[pos - 1] = last_c;
                        unsat_pos[last_c] = pos;
                        unsat_pos[k] = 0; unsz--;
                    }
                } else if (old_count > 0 && new_count == 0) {
                    if (!unsat_pos[k]) { unsat_list[unsz] = k; unsat_pos[k] = unsz + 1; unsz++; }
                }
            }
        }
    }
    free(clause_sat_counts); free(unsat_list); free(unsat_pos); free(var_set);
    return unsz == 0;
}

/* --------------------------------------------------------------------
   [PORTED] Topological Repair Phase
-------------------------------------------------------------------- */
static int topological_repair_phase(NitroSat *ns, int max_steps)
{
    if (!ns->use_topology) return 0;
    recompute_sat_counts(ns);
    for (int step = 1; step <= max_steps; ++step) {
        ns->last_topology = compute_topology(ns, 0.3);
        
        int *hole_vars = malloc(ns->num_vars * sizeof(int));
        int hole_count = 0;
        
        if (ns->last_topology.beta1 == 0) {
            for (int i = 1; i <= ns->num_vars; ++i) {
                if (!ns->decimated[i]) hole_vars[hole_count++] = i;
            }
        } else {
            int *deg = calloc(ns->num_vars + 1, sizeof(int));
            for (int c = 0; c < ns->num_clauses; ++c) {
                if (ns->sat_counts[c] == 0) {
                    for (int i = ns->cl_offs[c]; i < ns->cl_offs[c+1]; ++i) deg[abs(ns->cl_flat[i])]++;
                }
            }
            for (int i = 1; i <= ns->num_vars; ++i) {
                if (deg[i] > 0 && !ns->decimated[i]) hole_vars[hole_count++] = i;
            }
            free(deg);
        }
        
        int flips = 0;
        for (int i = 0; i < hole_count; ++i) {
            int var = hole_vars[i];
            if (!ns->decimated[var]) {
                double force = 0.0;
                for (int p = ns->v2c_ptr[var]; p < ns->v2c_ptr[var+1]; ++p) {
                    int c = ns->v2c_data[p];
                    if (ns->sat_counts[c] == 0) {
                        for (int j = ns->cl_offs[c]; j < ns->cl_offs[c+1]; ++j) {
                            if (abs(ns->cl_flat[j]) == var) {
                                force += (ns->cl_flat[j] > 0 ? 1 : -1) * ns->cl_weights[c];
                                break;
                            }
                        }
                    }
                }
                if (fabs(force) > 0.01) {
                    double old_x = ns->x[var];
                    ns->x[var] = (force > 0) ? 0.99 : 0.01;
                    /* Update sat_counts for var flip */
                    if ((old_x > 0.5) != (ns->x[var] > 0.5)) {
                        for (int p = ns->v2c_ptr[var]; p < ns->v2c_ptr[var+1]; ++p) {
                            int c = ns->v2c_data[p];
                            /* Check if previously sat, now unsat or vice versa */
                            /* Exhaustive check for this clause only is O(k) */
                            int sc = 0;
                            for (int j=ns->cl_offs[c]; j<ns->cl_offs[c+1]; ++j) {
                                int l = ns->cl_flat[j];
                                if ((l>0 && ns->x[abs(l)]>0.5) || (l<0 && ns->x[abs(l)]<=0.5)) sc++;
                            }
                            ns->sat_counts[c] = sc;
                        }
                    }
                    flips++;
                }
            }
        }
        free(hole_vars);
        if (flips == 0) break;
        int uns = 0;
        for (int c=0; c<ns->num_clauses; ++c) if (ns->sat_counts[c]==0) { uns++; break; }
        if (uns == 0) return 1;
    }
    return 0;
}

/* --------------------------------------------------------------------
   [PORTED] Unit Propagation
-------------------------------------------------------------------- */
static int unit_propagation(NitroSat *ns)
{
    int changed = 0;
    for (int c = 0; c < ns->num_clauses; ++c) {
        int unassigned_lit = 0, satisfied = 0, num_unassigned = 0;
        for (int i = ns->cl_offs[c]; i < ns->cl_offs[c+1]; ++i) {
            int lit = ns->cl_flat[i], var = abs(lit);
            if (ns->decimated[var]) {
                if ((lit > 0 && ns->x[var] > 0.5) || (lit < 0 && ns->x[var] <= 0.5)) { satisfied = 1; break; }
            } else {
                num_unassigned++;
                if (num_unassigned == 1) unassigned_lit = lit;
                else { unassigned_lit = 0; break; }
            }
        }
        if (!satisfied && unassigned_lit) {
            int var = abs(unassigned_lit);
            double force_val = (unassigned_lit > 0) ? 1.0 : 0.0;
            if (fabs(ns->x[var] - force_val) > 0.5) {
                ns->x[var] = force_val;
                ns->decimated[var] = 1;
                changed = 1;
            }
        }
    }
    return changed;
}

/* --------------------------------------------------------------------
   [PORTED] Adelic Saturation Phase
-------------------------------------------------------------------- */
static int zeta_sweep_aggressive(NitroSat *ns, double beta, int *flips_out, int *unsat_out)
{
    recompute_sat_counts(ns);
    int m = ns->num_clauses;
    int *unsat = malloc(m * sizeof(int));
    int unsz = 0;
    for (int c = 0; c < m; ++c) if (ns->sat_counts[c] == 0) unsat[unsz++] = c;

    if (unsz == 0) { free(unsat); if (flips_out) *flips_out=0; if (unsat_out) *unsat_out=0; return 1; }

    int flips = 0;
    for (int i = 0; i < unsz; ++i) {
        int c = unsat[i];
        if (ns->sat_counts[c] > 0) continue;

        int best_v = -1, best_br = 999999;
        int cs = ns->cl_offs[c], ce = ns->cl_offs[c+1];
        for (int j = cs; j < ce; ++j) {
            int v = abs(ns->cl_flat[j]);
            int br = 0;
            int old_v_sat = (ns->x[v] > 0.5);
            for (int p = ns->v2c_ptr[v]; p < ns->v2c_ptr[v+1]; ++p) {
                int cl = ns->v2c_data[p];
                if (ns->sat_counts[cl] == 1) {
                    int lit_p = 0;
                    for (int q = ns->cl_offs[cl]; q < ns->cl_offs[cl+1]; ++q) {
                        if (abs(ns->cl_flat[q]) == v) { lit_p = (ns->cl_flat[q] > 0); break; }
                    }
                    if (lit_p == old_v_sat) br++;
                }
            }
            if (br < best_br) { best_br = br; best_v = v; }
        }

        if (best_v != -1 && (best_br == 0 || rng_double() < exp(-beta * best_br))) {
            int old_v_sat = (ns->x[best_v] > 0.5);
            ns->x[best_v] = (old_v_sat) ? 0.0 : 1.0;
            int new_v_sat = !old_v_sat;
            for (int p = ns->v2c_ptr[best_v]; p < ns->v2c_ptr[best_v+1]; ++p) {
                int cl = ns->v2c_data[p];
                int lit_p = 0;
                for (int q = ns->cl_offs[cl]; q < ns->cl_offs[cl+1]; ++q) {
                    if (abs(ns->cl_flat[q]) == best_v) { lit_p = (ns->cl_flat[q] > 0); break; }
                }
                if (lit_p == new_v_sat) ns->sat_counts[cl]++;
                else ns->sat_counts[cl]--;
            }
            flips++;
        }
    }
    if (flips_out) *flips_out = flips;
    int final_unsat = 0;
    for (int c = 0; c < m; ++c) if (ns->sat_counts[c] == 0) final_unsat++;
    if (unsat_out) *unsat_out = final_unsat;
    free(unsat);
    return final_unsat == 0;
}

static int adelic_saturation_phase(NitroSat *ns, int max_steps)
{
    recompute_sat_counts(ns);
    double beta_low = 0.1, beta_high = 10.0;
    int last_unsat = ns->num_clauses;
    for (int step = 1; step <= max_steps; ++step) {
        double beta = beta_low + (beta_high - beta_low) * ((double)step / max_steps);
        int unsat, flips;
        if (zeta_sweep_aggressive(ns, beta, &flips, &unsat)) return 1;
        
        if (flips == 0) beta_low = beta;
        else if (unsat > last_unsat) beta_high = beta;
        last_unsat = unsat;
        
        if (step % 10 == 0) unit_propagation(ns);
    }
    return check_satisfaction(ns) == ns->num_clauses;
}

/* --------------------------------------------------------------------
   19. Main solve routine – mirrors the Lua `NitroSat:solve` method.
-------------------------------------------------------------------- */
static int nitrosat_solve(NitroSat *ns)
{
    int max_steps = ns->max_steps;
    double lr0   = 0.2;               /* base learning‑rate before annealing */
    int    best_sat = 0;
    double *best_x  = malloc((ns->num_vars+1)*sizeof(double));

    for (int i=1;i<=ns->num_vars;++i) best_x[i] = ns->x[i];

    for (int step=1; step<=max_steps; ++step) {
        int unsat = compute_gradients(ns);
        int sat = ns->num_clauses - unsat;

        if (sat > best_sat) {
            best_sat = sat;
            for (int i=1;i<=ns->num_vars;++i) best_x[i] = ns->x[i];
        }
        if (unsat == 0) {
            if (ns->verbose) printf("[EARLY] 100%% satisfaction at step %d!\n", step);
            /* Restore best solution before breaking */
            for (int i = 1; i <= ns->num_vars; i++) {
                ns->x[i] = best_x[i];
            }
            break;
        }

        /* Debug output every step */
        if (ns->verbose) {
            double grad_norm = 0;
            for (int i = 1; i <= ns->num_vars; i++) grad_norm += ns->grad_buffer[i] * ns->grad_buffer[i];
            grad_norm = sqrt(grad_norm);
            printf("[%04d] sat=%d/%d (%.2f%%) lr=%.6f |grad|=%.4f unsat_ clauses=%d\n",
                   step, sat, ns->num_clauses, 100.0*sat/ns->num_clauses,
                   ns->opt->lr, grad_norm, unsat);
            fflush(stdout);
        }

        /* dynamic clause re‑weighting – SAPS-like (matches gh.lua) */
        if (step % 10 == 0) {
            recompute_sat_counts(ns);
            for (int c = 0; c < ns->num_clauses; ++c) {
                if (ns->sat_counts[c] == 0) {
                    ns->cl_weights[c] *= 1.1;  /* solve_gravity penalty */
                } else {
                    ns->cl_weights[c] *= 0.999; /* decay */
                }
            }
        } /* annealed learning‑rate */
        double t = step * (30.0 / max_steps);
        double lr = annealing_lr(t, lr0);
        if (lr <= 0.0) lr = 1e-6;
        ns->opt->lr = lr;
        optimizer_step(ns->opt, ns->x, ns->grad_buffer);
        /* clamp x to [0,1] after optimizer step (matches gh.lua line 992) */
        for (int i = 1; i <= ns->num_vars; ++i)
            ns->x[i] = clamp(ns->x[i], 0.0, 1.0);

        /* decimation (soft locking) */
        if (ns->dec_freq && step % ns->dec_freq == 0) {
            int locked = decimate(ns);
            if (ns->verbose && locked) printf("[dec] locked %d vars\n", locked);
        }

        /* Zeta‑zero guided perturbation (every 10 steps) */
        if (step % 10 == 0) {
            TopologyInfo *topo = ns->use_topology ? &ns->last_topology : NULL;
            for (int v=1; v<=ns->num_vars; ++v) if (!ns->decimated[v]) {
                double dz = zeta_zero_perturb(v, step, max_steps,
                                             ns->primes, ns->prime_cnt,
                                             topo, ns->num_vars,
                                             0 /* nuclear = false */);
                ns->x[v] = clamp(ns->x[v] + dz * 0.001, 0.0, 1.0);  /* gh.lua scales by 0.001 */
            }
        }

        /* Inline AdaBoost DCW (gh.lua lines 2146-2170) – second pass */
        if (step % 10 == 0) {
            for (int c = 0; c < ns->num_clauses; ++c) {
                int clause_sat = 0;
                for (int i = ns->cl_offs[c]; i < ns->cl_offs[c+1] && !clause_sat; ++i) {
                    int lit = ns->cl_flat[i];
                    int v   = abs(lit);
                    double val = ns->x[v];
                    if ((lit > 0 && val > 0.5) || (lit < 0 && val <= 0.5)) clause_sat = 1;
                }
                if (!clause_sat) {
                    ns->cl_weights[c] *= 2.5;
                }
            }
            for (int i = 0; i < ns->num_clauses; ++i) {
                ns->cl_weights[i] *= 0.99;
            }
        }

        /* BAHA fracture detection every 50 steps */
        if (step % 50 == 0) {
            double beta_proxy = ((double)step / max_steps) * 10.0;
            /* cheap thermodynamic estimator */
            double logZ = 0.0;
            for (int s=0; s<20; ++s) {  /* 20 samples */
                int e = 0;
                for (int c=0; c<ns->num_clauses; ++c) {
                    int sat = 0;
                    for (int i=ns->cl_offs[c]; i<ns->cl_offs[c+1] && !sat; ++i) {
                        int lit = ns->cl_flat[i];
                        int v   = abs(lit);
                        double val = ns->x[v];
                        if ((lit>0 && val>0.5) || (lit<0 && val<=0.5)) sat = 1;
                    }
                    if (!sat) ++e;
                }
                logZ += -beta_proxy * e;
            }
            logZ = logZ / 20.0;
            fd_record(ns->fd, beta_proxy, logZ);

            /* update variance/peak statistics */
            ++ns->baha_var_n;
            double delta = logZ - ns->baha_var_mean;
            ns->baha_var_mean += delta / ns->baha_var_n;
            ns->baha_var_M2   += delta * (logZ - ns->baha_var_mean);
            double std = (ns->baha_var_n>1) ?
                         sqrt(ns->baha_var_M2/(ns->baha_var_n-1)) : 0.0;
            double z   = (std>1e-12) ? (logZ - ns->baha_var_mean)/std : 0.0;

            if (fd_is_fracture(ns->fd) && z > 0.8) {
                if (logZ > ns->baha_var_peak) {
                    ns->baha_var_peak = logZ;
                    ns->baha_beta_c   = beta_proxy;
                }
                
                int nb;
                Branch *brs = enumerate_branches(ns->baha_beta_c, beta_proxy, &nb);
                if (nb > 0) {
                    double bestScore = -1e300;
                    int bestIdx = -1;
                    double *best_jump_x = malloc((ns->num_vars + 1) * sizeof(double));
                    double *temp_x = malloc((ns->num_vars + 1) * sizeof(double));
                    
                    for (int i = 0; i < nb; ++i) {
                        double score = score_branch(ns, brs[i].beta, 20, temp_x);
                        if (score > bestScore) {
                            bestScore = score;
                            bestIdx = i;
                            for (int j = 1; j <= ns->num_vars; ++j) best_jump_x[j] = temp_x[j];
                        }
                    }
                    
                    if (bestIdx != -1) {
                        if (ns->verbose) printf("[baha] jumping to branch beta=%.3f\n", brs[bestIdx].beta);
                        for (int i = 1; i <= ns->num_vars; ++i) {
                            if (!ns->decimated[i]) ns->x[i] = best_jump_x[i];
                        }
                        free(ns->opt->m); free(ns->opt->v); free(ns->opt);
                        ns->opt = optimizer_new("harmonic_adam", ns->x, ns->num_vars, 0.02, 0.9, 0.999, 1e-8, 0.02);
                    }
                    free(best_jump_x); free(temp_x);
                }
                free(brs);
                fd_clear(ns->fd);
            }
        }

        /* periodic topology snapshot (every topology_freq steps) */
        if (ns->use_topology && step % 50 == 0) {
            ns->last_topology = compute_topology(ns, 0.3);
            update_persistence(&ns->pt, ns->last_topology.beta0, ns->last_topology.beta1, step, ns->last_topology.complexity_score);
            if (ns->verbose)
                printf("[topo] step %d β0=%d β1=%d comp=%.3f\n",
                        step,
                        ns->last_topology.beta0,
                        ns->last_topology.beta1,
                        ns->last_topology.complexity_score);
        }
    }

    /* restore best solution */
    for (int i=1;i<=ns->num_vars;++i) ns->x[i] = best_x[i];

    /* -----------------------------------------------------------------
       3‑phase finisher
       ----------------------------------------------------------------- */
    int sat = check_satisfaction(ns);
    if (sat == ns->num_clauses) {
        printf("Freeing best_x in nitrosat_solve (early solve)... (sat: %d/%d)\n", sat, ns->num_clauses);
        free(best_x);
        return 1;
    }   /* already solved */

    /* Phase‑2 : topological repair (≈95 % satisfied) */
    if (sat >= (int)(0.95 * ns->num_clauses)) {
        if (ns->verbose) puts("[phase‑2] entering topological repair");
        
        for (int i = 1; i <= ns->num_vars; ++i) ns->x[i] = best_x[i];
        topological_repair_phase(ns, 1500);
        int phase2_sat = check_satisfaction(ns);
        
        if (phase2_sat > best_sat) {
            best_sat = phase2_sat;
            for (int i = 1; i <= ns->num_vars; ++i) best_x[i] = ns->x[i];
        }
        if (ns->verbose) printf("[phase‑2] finished, sat=%d/%d\n", best_sat, ns->num_clauses);
        if (best_sat == ns->num_clauses) { free(best_x); return 1; }
    }

    /* Phase‑3 : adelic saturation (≈98 % satisfied) */
    if (best_sat >= (int)(0.98 * ns->num_clauses)) {
        if (ns->verbose) puts("[phase‑3] entering adelic saturation");
        for (int i = 1; i <= ns->num_vars; ++i) ns->x[i] = best_x[i];
        
        adelic_saturation_phase(ns, 2000);
        int phase3_sat = check_satisfaction(ns);
        if (phase3_sat > best_sat) {
            best_sat = phase3_sat;
            for (int i = 1; i <= ns->num_vars; ++i) best_x[i] = ns->x[i];
        }
        if (ns->verbose) printf("[phase‑3] finished, sat=%d/%d\n", best_sat, ns->num_clauses);
        if (best_sat == ns->num_clauses) { free(best_x); return 1; }
    }

    /* Core decomposition “blast’’ */
    if (best_sat < ns->num_clauses) {
        if (ns->verbose) puts("[core] attempting core decomposition");
        for (int i = 1; i <= ns->num_vars; ++i) ns->x[i] = best_x[i];
        if (core_decomposition(ns)) {
            int core_sat = check_satisfaction(ns);
            if (core_sat > best_sat) {
                best_sat = core_sat;
                for (int i = 1; i <= ns->num_vars; ++i) best_x[i] = ns->x[i];
            }
        }
        if (best_sat == ns->num_clauses) { free(best_x); return 1; }
    }

    /* Final BAHA‑WalkSAT (discrete local search) */
    if (best_sat < ns->num_clauses) {
        if (ns->verbose) puts("[final] BAHA‑WalkSAT");
        for (int i = 1; i <= ns->num_vars; ++i) ns->x[i] = best_x[i] > 0.5 ? 1.0 : 0.0;
        baha_walksat(ns, ns->num_vars * 100);
        int bsat = check_satisfaction(ns);
        if (bsat > best_sat) {
            best_sat = bsat;
        }
    }
    
    free(best_x);
    return (best_sat == ns->num_clauses);
}

static int nitrosat_solve_dcw(NitroSat *ns, int num_passes)
{
    if (ns->num_vars == 0 || ns->num_clauses == 0) return nitrosat_solve(ns);
    
    int best_overall_sat = 0;
    double *best_overall_x = malloc((ns->num_vars + 1) * sizeof(double));
    int success = 0;
    
    for (int i = 0; i < ns->num_clauses; ++i) ns->cl_weights[i] = 1.0;
    
    for (int pass = 1; pass <= num_passes; ++pass) {
        if (ns->verbose) printf("--- DCW PASS %d ---\n", pass);
        
        success = nitrosat_solve(ns);
        int sat_count = check_satisfaction(ns);
        
        if (sat_count > best_overall_sat) {
            best_overall_sat = sat_count;
            for (int i = 1; i <= ns->num_vars; ++i) best_overall_x[i] = ns->x[i];
        }
        
        if (sat_count == ns->num_clauses) { success = 1; break; }
        
        if (pass < num_passes) {
            for (int c = 0; c < ns->num_clauses; ++c) {
                int sat = 0;
                for (int i = ns->cl_offs[c]; i < ns->cl_offs[c+1]; ++i) {
                    int lit = ns->cl_flat[i], v = abs(lit);
                    if ((lit > 0 && ns->x[v] > 0.5) || (lit < 0 && ns->x[v] <= 0.5)) { sat = 1; break; }
                }
                if (!sat) ns->cl_weights[c] *= 1.5;
            }
        }
        
        // Soft reset for next pass
        ns->opt->t = 0;
        for (int i = 1; i <= ns->num_vars; ++i) {
            ns->opt->m[i-1] = 0;
            ns->opt->v[i-1] = 0;
            if (!ns->decimated[i]) ns->x[i] = rng_double();
        }
    }
    
    for (int i = 1; i <= ns->num_vars; ++i) ns->x[i] = best_overall_x[i];
    free(best_overall_x);
    return success;
}

/* --------------------------------------------------------------------
   20. Entry point – read problem, build solver, run.
-------------------------------------------------------------------- */
int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <cnf-file> [max-steps] [--no-dcw] [--no-topo]\n", argv[0]);
        return EXIT_FAILURE;
    }
    const char *cnf_file = argv[1];
    int max_steps = 3000;
    int use_dcw = 1;
    int use_topo = 1;

    for (int i = 2; i < argc; ++i) {
        if (strcmp(argv[i], "--no-dcw") == 0) use_dcw = 0;
        else if (strcmp(argv[i], "--no-topo") == 0) use_topo = 0;
        else max_steps = atoi(argv[i]);
    }

    Instance *inst = read_cnf(cnf_file);
    if (!inst) return EXIT_FAILURE;

    NitroSat *ns = nitrosat_new(inst, max_steps, /*verbose=*/1);
    ns->use_topology = use_topo;

    /* Build variable‑to‑clause CSR structure (O(L) time) */
    ns->v2c_ptr = calloc(ns->num_vars + 2, sizeof(int));
    for (int i=0; i<inst->num_clauses; ++i) {
        for (int j=inst->cl_offs[i]; j<inst->cl_offs[i+1]; ++j) {
            ns->v2c_ptr[abs(inst->cl_flat[j]) + 1]++;
        }
    }
    for (int i=1; i<=ns->num_vars+1; ++i) ns->v2c_ptr[i] += ns->v2c_ptr[i-1];
    ns->v2c_data = malloc(ns->v2c_ptr[ns->num_vars+1] * sizeof(int));
    int *cur = calloc(ns->num_vars + 1, sizeof(int));
    for (int c=0; c<inst->num_clauses; ++c) {
        for (int j=inst->cl_offs[c]; j<inst->cl_offs[c+1]; ++j) {
            int v = abs(inst->cl_flat[j]);
            int pos = ns->v2c_ptr[v] + cur[v];
            ns->v2c_data[pos] = c;
            cur[v]++;
        }
    }
    free(cur);

    double start = (double)clock() / CLOCKS_PER_SEC;
    int solved = use_dcw ? nitrosat_solve_dcw(ns, 5) : nitrosat_solve(ns);
    double end   = (double)clock() / CLOCKS_PER_SEC;

    printf("\n=== RESULT ===\n");
    printf("Formula : %s\n", cnf_file);
    printf("Variables: %d   Clauses: %d\n", ns->num_vars, ns->num_clauses);

    int final_sat = check_satisfaction(ns);
    printf("Satisfied: %d   Unsatisfied: %d\n", final_sat, ns->num_clauses - final_sat);
    printf("Solved  : %s\n", solved ? "YES" : "NO");
    printf("Time    : %.2f s\n", end - start);
    
    if (ns->use_topology && ns->pt.has_initial) {
        printf("Topology Summary:\n");
        printf("  Beta0: %d -> %d\n", ns->pt.initial_beta0, ns->pt.final_beta0);
        printf("  Beta1: %d -> %d\n", ns->pt.initial_beta1, ns->pt.final_beta1);
        printf("  Persistence Events: %d\n", ns->pt.persistence_events);
        printf("  Complexity Trend: %.3f\n", ns->pt.final_complexity - ns->pt.initial_complexity);
    }
    
    printf("Final assignment (first 20 vars):\n");
    for (int i=1; i<=ns->num_vars && i<=20; ++i)
        printf("%d:%d ", i, (ns->x[i] > 0.5));
    puts("\n");

    nitrosat_free(ns);
    free(inst->cl_offs); free(inst->cl_flat); free(inst);
    return solved ? EXIT_SUCCESS : EXIT_FAILURE;
}
