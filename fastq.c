/* fastq.c -- fused quantize+dequantize+f64-accumulate kernel for
 * cosine-table builds (see PLAN_FUSED_CPP.md).
 *
 * Loads ggml-base.dll (the SAME dll the table fingerprint pins) and uses
 * its ggml_quantize_chunk / dequantize_row_* -- so produced bytes are
 * bit-exact with llama-quantize. The f64 dot/sumsq that the Python engine
 * used to do with 40+ B/elem/tier of numpy casts is fused into one pass
 * over the dequantized row: ~17 B/elem/tier total.
 *
 * Compiled:  cl /O2 /W3 /LD fastq.c /Fe:fastq.dll
 * No Python API -- plain C DLL, driven via ctypes (fastq.py).
 */
#include <windows.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <intrin.h>
#include <smmintrin.h>
#include <nmmintrin.h>

typedef size_t (*fn_quantize_chunk)(int, const float *, void *,
                                    int64_t, int64_t, int64_t,
                                    const float *);
typedef void (*fn_dequant_row)(void *, float *, int64_t);
typedef int64_t (*fn_blck_size)(int);
typedef size_t (*fn_type_size)(int);

#define FQAPI __declspec(dllexport)

static HMODULE g_ggml = NULL;
static fn_quantize_chunk g_quantize_chunk = NULL;
static fn_blck_size g_blck_size = NULL;
static fn_type_size g_type_size = NULL;

/* dequant resolver table, indexed by ggml_type (sparse set we support) */
#define N_TYPES 32
static fn_dequant_row g_deq[N_TYPES];

/* ggml enums (stable, ggml.h) */
enum {
    T_Q4_0 = 2, T_Q4_1 = 3, T_Q5_0 = 6, T_Q5_1 = 7, T_Q8_0 = 8,
    T_Q2_K = 10, T_Q3_K = 11, T_Q4_K = 12, T_Q5_K = 13,
    T_Q6_K = 14, T_IQ2_XXS = 16, T_IQ2_XS = 17, T_IQ3_XXS = 18,
    T_IQ1_S = 19, T_IQ4_NL = 20, T_IQ3_S = 21, T_IQ2_S = 22,
    T_IQ4_XS = 23, T_IQ1_M = 29
};
/* these abort inside ggml if imatrix == NULL (verified by probe) */
static int imatrix_required(int t) {
    return t == T_IQ1_S || t == T_IQ1_M || t == T_IQ2_XXS || t == T_IQ2_XS ||
           t == T_IQ2_S || t == T_IQ3_XXS;
}
static const char *sym_name(int t) {
    switch (t) {
        /* NOTE: exact export case -- this DLL exports K-quants with
         * uppercase K (dequantize_row_q4_K). GetProcAddress here is
         * case-sensitive. */
        case T_Q4_0: return "dequantize_row_q4_0";
        case T_Q4_1: return "dequantize_row_q4_1";
        case T_Q5_0: return "dequantize_row_q5_0";
        case T_Q5_1: return "dequantize_row_q5_1";
        case T_Q8_0: return "dequantize_row_q8_0";
        case T_Q2_K: return "dequantize_row_q2_K";
        case T_Q3_K: return "dequantize_row_q3_K";
        case T_Q4_K: return "dequantize_row_q4_K";
        case T_Q5_K: return "dequantize_row_q5_K";
        case T_Q6_K: return "dequantize_row_q6_K";
        case T_IQ2_XXS: return "dequantize_row_iq2_xxs";
        case T_IQ2_XS: return "dequantize_row_iq2_xs";
        case T_IQ3_XXS: return "dequantize_row_iq3_xxs";
        case T_IQ1_S: return "dequantize_row_iq1_s";
        case T_IQ4_NL: return "dequantize_row_iq4_nl";
        case T_IQ3_S: return "dequantize_row_iq3_s";
        case T_IQ2_S: return "dequantize_row_iq2_s";
        case T_IQ4_XS: return "dequantize_row_iq4_xs";
        case T_IQ1_M: return "dequantize_row_iq1_m";
        default: return NULL;
    }
}

/* Load ggml-base.dll and resolve all symbols. 0 ok, -1 fail. */
FQAPI int fq_init(const char *ggml_dll_path) {
    if (g_ggml) return 0;
    g_ggml = LoadLibraryA(ggml_dll_path);
    if (!g_ggml) return -1;
    g_quantize_chunk = (fn_quantize_chunk)
        GetProcAddress(g_ggml, "ggml_quantize_chunk");
    g_blck_size = (fn_blck_size)GetProcAddress(g_ggml, "ggml_blck_size");
    g_type_size = (fn_type_size)GetProcAddress(g_ggml, "ggml_type_size");
    if (!g_quantize_chunk || !g_blck_size || !g_type_size) return -1;
    for (int t = 0; t < N_TYPES; t++) g_deq[t] = NULL;
    for (int t = 0; t < N_TYPES; t++) {
        const char *nm = sym_name(t);
        if (nm) g_deq[t] = (fn_dequant_row)GetProcAddress(g_ggml, nm);
    }
    return 0;
}

FQAPI int fq_blck_size(int type)   { return g_blck_size(type); }
FQAPI int fq_type_size(int type)   { return (int)g_type_size(type); }
FQAPI int fq_has_deq(int type)     { return type < N_TYPES && g_deq[type] != NULL; }
FQAPI int fq_imatrix_required(int t) { return imatrix_required(t); }

/* f32 -> f16 (RNE) -> f32. CVTSS2SH with the RNE flag is the exact IEEE
 * round-trip, bit-identical to numpy astype(float16). Requires SSE4.1
 * (build with /arch:AVX2). */
static float f16_roundtrip(float x) {
    __m128 v = _mm_set_ss(x);
    __m128i hi = _mm_cvtps_ph(v, 0);   /* RC=00: round to nearest, ties even */
    return _mm_cvtph_ps(hi).m128_f32[0];
}

static const int64_t ACC_CHUNK = 1 << 24;   /* f64 accumulation segment */

/* number of set 8-byte lanes in an AVX2 predicate mask */
static int lane_cnt(__m256 m) {
    __m256i mi = _mm256_castps_si256(m);
    __m128i l = _mm256_castsi256_si128(mi);
    __m128i h = _mm256_extracti128_si256(mi, 1);
    return (_mm_popcnt_u32((unsigned)_mm_movemask_epi8(l)) +
            _mm_popcnt_u32((unsigned)_mm_movemask_epi8(h))) >> 2;
}

/* Neumaier-compensated sum: ~machine-precision regardless of n/order.
 * Plain sequential f64 accum drifts ~n*eps (~1e-10 at 8M elems), which
 * fails the 1e-12 test gate. */
static double acc_finish(double sum, double comp) { return sum + comp; }
static inline void acc_add(double *sum, double *comp, double x) {
    double t = *sum + x;
    if (fabs(*sum) >= fabs(x)) *comp += (*sum - t) + x;
    else                       *comp += (x - t) + *sum;
    *sum = t;
}

/* Quantize one SEGMENT (a[0..n), n a multiple of npr, npr a multiple of
 * blck(type)) to raw bytes. Returns bytes written, 0 on error. */
/* 0 ok, else: 1 no deq sym, 2 imx missing, 3 n%blck, 4 n%npr, 5 cap */
FQAPI int fq_gate(int type, int64_t n, int64_t npr, const float *imx,
                  size_t raw_cap) {
    if (type >= N_TYPES || !g_deq[type]) return 1;
    if (imatrix_required(type) && !imx) return 2;
    int64_t bs = g_blck_size(type);
    if (n % bs) return 3;
    if (n % npr) return 4;
    if ((size_t)(n / bs) * g_type_size(type) > raw_cap) return 5;
    return 0;
}

/* Diagnostic: time the A = ||a||^2 pass alone (seconds). */
FQAPI double fq_time_a(const float *a, int64_t n) {
    double acc = 0.0;
    LARGE_INTEGER fq, fq2, pc;
    QueryPerformanceFrequency(&pc);
    QueryPerformanceCounter(&fq);
    for (int64_t off = 0; off < n; off += ACC_CHUNK) {
        int64_t m = n - off < ACC_CHUNK ? n - off : ACC_CHUNK;
        double c = 0.0, cc = 0.0;
        for (int64_t i = 0; i < m; i++) {
            double d = a[off + i];
            acc_add(&c, &cc, d * d);
        }
        acc += acc_finish(c, cc);
    }
    QueryPerformanceCounter(&fq2);
    (void)acc;
    return (double)(fq2.QuadPart - fq.QuadPart) / pc.QuadPart;
}

/* Diagnostic: time one type's quantize / dequantize / f64-accumulate /
 * f16 phases separately (seconds, double[4]). 0 ok. */
FQAPI int fq_time(int type, const float *a, int64_t n, int64_t npr,
                  const float *imx, uint8_t *raw, float *bb, double out[4]) {
    out[0] = out[1] = out[2] = out[3] = 0.0;
    if (fq_gate(type, n, npr, imx, SIZE_MAX) != 0) return -1;
    int64_t bs = g_blck_size(type);
    size_t need = (size_t)(n / bs) * g_type_size(type);
    LARGE_INTEGER fq, fq2, pc;
    QueryPerformanceFrequency(&pc);
    QueryPerformanceCounter(&fq);
    size_t rc = g_quantize_chunk(type, a, raw, 0, n / npr, npr, imx);
    QueryPerformanceCounter(&fq2);
    if (rc != need) return -1;
    out[0] = (double)(fq2.QuadPart - fq.QuadPart) / pc.QuadPart;
    QueryPerformanceCounter(&fq);
    g_deq[type](raw, bb, n);
    QueryPerformanceCounter(&fq2);
    out[1] = (double)(fq2.QuadPart - fq.QuadPart) / pc.QuadPart;
    double accS = 0.0, accQ = 0.0;
    for (int64_t off = 0; off < n; off += ACC_CHUNK) {
        int64_t m = n - off < ACC_CHUNK ? n - off : ACC_CHUNK;
        double cs = 0.0, cc = 0.0, cq = 0.0, qc = 0.0;
        for (int64_t i = 0; i < m; i++) {
            double d = bb[off + i];
            double av = a[off + i];
            acc_add(&cs, &cc, av * d);
            acc_add(&cq, &qc, d * d);
        }
        accS += acc_finish(cs, cc);
        accQ += acc_finish(cq, qc);
    }
    QueryPerformanceCounter(&fq2);
    out[2] = (double)(fq2.QuadPart - fq.QuadPart) / pc.QuadPart;
    (void)accS; (void)accQ;
    return 0;
}

FQAPI size_t fq_quant(const float *a, int64_t n, int64_t npr, int type,
                const float *imx, uint8_t *raw, size_t raw_cap) {
    if (fq_gate(type, n, npr, imx, raw_cap)) return 0;
    int64_t bs = g_blck_size(type);
    size_t need = (size_t)(n / bs) * g_type_size(type);
    size_t rc = g_quantize_chunk(type, a, raw, 0, n / npr, npr, imx);
    return rc == need ? rc : 0;
}

/* Fused per-segment stats.
 * a: n f32 (the segment), npr: row width, types[ntypes]: ggml enums,
 * imx: f32[npr] or NULL, raw/bb: scratch (raw >= n bytes, bb >= n f32).
 * out: A (||a||^2), S[j] = <a, Q_j(a)>, Q[j] = ||Q_j(a)||^2, f64.
 * f16: if 1, also compute the F16 round-trip column into S[ntypes],Q[ntypes].
 * Returns 0 ok; -1 quantize mismatch; -3 imatrix missing; -4 f16 overflow.
 */
FQAPI int fq_segment(const float *a, int64_t n, int64_t npr,
               const int *types, int ntypes, const float *imx,
               double *A, double *S, double *Q,
               uint8_t *raw, float *bb, int with_f16) {
    /* A = ||a||^2, chunked compensated f64 */
    {
        double acc = 0.0, c = 0.0, cc = 0.0;
        for (int64_t off = 0; off < n; off += ACC_CHUNK) {
            int64_t m = n - off < ACC_CHUNK ? n - off : ACC_CHUNK;
            c = 0.0; cc = 0.0;
            for (int64_t i = 0; i < m; i++) {
                double d = a[off + i];
                acc_add(&c, &cc, d * d);
            }
            acc += acc_finish(c, cc);
        }
        *A = acc;
    }
    for (int j = 0; j < ntypes; j++) {
        int t = types[j];
        if (fq_gate(t, n, npr, imx, SIZE_MAX) != 0) return -1;
        int64_t bs = g_blck_size(t);
        size_t need = (size_t)(n / bs) * g_type_size(t);
        /* imx is passed to every type, matching tablebuild._do_batch which
         * passes the tensor's imatrix row to ALL tiers when imatrix is
         * loaded (K-quants use it in their quantize path). */
        size_t rc = g_quantize_chunk(t, a, raw, 0, n / npr, npr, imx);
        if (rc != need) return -1;
        g_deq[t](raw, bb, n);
        double accS = 0.0, accQ = 0.0;
        for (int64_t off = 0; off < n; off += ACC_CHUNK) {
            int64_t m = n - off < ACC_CHUNK ? n - off : ACC_CHUNK;
            double cs = 0.0, cc = 0.0, cq = 0.0, qc = 0.0;
            for (int64_t i = 0; i < m; i++) {
                double d = bb[off + i];
                double av = a[off + i];
                acc_add(&cs, &cc, av * d);
                acc_add(&cq, &qc, d * d);
            }
            accS += acc_finish(cs, cc);
            accQ += acc_finish(cq, qc);
        }
        S[j] = accS;
        Q[j] = accQ;
    }
    if (with_f16) {
        double accS = 0.0, accQ = 0.0, bad = 0.0;
        const float *pa = a;
        __m256 lim = _mm256_set1_ps(65504.0f);       /* largest finite f16 */
        __m256 sign = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
        for (int64_t off = 0; off < n; off += ACC_CHUNK) {
            int64_t m = n - off < ACC_CHUNK ? n - off : ACC_CHUNK;
            const float *p = pa + off;      /* chunk base (was missing off) */
            double cs = 0.0, cc = 0.0, cq = 0.0, qc = 0.0;
            int64_t i = 0;
            for (; i + 8 <= m; i += 8) {
                float d8[8];
                __m256 v = _mm256_loadu_ps(p + i);
                __m128i h = _mm256_cvtps_ph(v, 0);   /* 8 halves, RNE */
                _mm_storeu_ps(d8, _mm_cvtph_ps(h));
                _mm_storeu_ps(d8 + 4,
                              _mm_cvtph_ps(_mm_srli_si128(h, 8)));
                __m256 oob = _mm256_cmp_ps(_mm256_and_ps(v, sign),
                                           lim, _CMP_GT_OQ);
                __m256 nan = _mm256_cmp_ps(v, v, _CMP_UNORD_Q);
                bad += (double)lane_cnt(_mm256_or_ps(oob, nan));
                for (int k = 0; k < 8; k++) {
                    double d = d8[k];
                    acc_add(&cs, &cc, (double)p[i + k] * d);
                    acc_add(&cq, &qc, d * d);
                }
            }
            for (; i < m; i++) {
                double d = f16_roundtrip(p[i]);
                acc_add(&cs, &cc, (double)p[i] * d);
                acc_add(&cq, &qc, d * d);
                if (!isfinite(d)) bad += 1.0;
            }
            accS += acc_finish(cs, cc);
            accQ += acc_finish(cq, qc);
        }
        if (bad > 0.0) return -4;
        S[ntypes] = accS;
        Q[ntypes] = accQ;
    }
    return 0;
}
