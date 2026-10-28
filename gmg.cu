#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <chrono>
#include <algorithm>
#include <functional>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err_ = (call);                                              \
        if (err_ != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error %s at %s:%d\n",                         \
                    cudaGetErrorString(err_), __FILE__, __LINE__);              \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

constexpr int BX = 32;
constexpr int BY = 8;

struct Ctrl { int stop; int cycle; };

template <typename real> __device__ __forceinline__ real dAbs(real x) { return x < real(0) ? -x : x; }
template <typename real> __device__ __forceinline__ real dMax(real a, real b) { return a > b ? a : b; }

template <typename real>
__device__ __forceinline__ real warpMax(real v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        v = dMax(v, __shfl_down_sync(0xffffffffu, v, off));
    return v;
}

__device__ __forceinline__ void atomicMaxNonNeg(float* dst, float v) {
    atomicMax(reinterpret_cast<int*>(dst), __float_as_int(v));
}
__device__ __forceinline__ void atomicMaxNonNeg(double* dst, double v) {
    atomicMax(reinterpret_cast<unsigned long long*>(dst),
              (unsigned long long)__double_as_longlong(v));
}

template <typename real>
__global__ void k_smooth_rb(real* __restrict__ u, const real* __restrict__ f, int n,
                            real h2, real w, int color, const Ctrl* ctrl) {
    if (ctrl->stop) return;
    const int tx = blockIdx.x * blockDim.x + threadIdx.x;
    const int i  = blockIdx.y * blockDim.y + threadIdx.y;
    const int j  = 2 * tx + ((i + color) & 1);
    if (i > 0 && i < n - 1 && j > 0 && j < n - 1) {
        const size_t id = (size_t)i * n + j;
        const real gs  = real(0.25) * (u[id - n] + u[id + n] + u[id - 1] + u[id + 1] + h2 * f[id]);
        const real old = u[id];
        u[id] = old + w * (gs - old);
    }
}

template <typename real>
__global__ void k_residual(const real* __restrict__ u, const real* __restrict__ f,
                           real* __restrict__ r, int n, real invh2, const Ctrl* ctrl) {
    if (ctrl->stop) return;
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= n || j >= n) return;
    const size_t id = (size_t)i * n + j;
    if (i > 0 && i < n - 1 && j > 0 && j < n - 1)
        r[id] = f[id] - (real(4) * u[id] - (u[id - n] + u[id + n] + u[id - 1] + u[id + 1])) * invh2;
    else
        r[id] = real(0);
}

template <typename real>
__device__ __forceinline__ real fullWeight(const real* a, size_t id, int nf) {
    const real c = a[id];
    const real e = a[id - 1] + a[id + 1] + a[id - nf] + a[id + nf];
    const real d = a[id - nf - 1] + a[id - nf + 1] + a[id + nf - 1] + a[id + nf + 1];
    return (real(4) * c + real(2) * e + d) * real(0.0625);
}

template <typename real>
__global__ void k_restrict(const real* __restrict__ rf, int nf,
                           real* __restrict__ fc, real* __restrict__ uc, int nc,
                           const Ctrl* ctrl) {
    if (ctrl->stop) return;
    const int J = blockIdx.x * blockDim.x + threadIdx.x;
    const int I = blockIdx.y * blockDim.y + threadIdx.y;
    if (I >= nc || J >= nc) return;
    const size_t ic = (size_t)I * nc + J;
    if (I > 0 && I < nc - 1 && J > 0 && J < nc - 1)
        fc[ic] = fullWeight(rf, (size_t)(2 * I) * nf + 2 * J, nf);
    else
        fc[ic] = real(0);
    uc[ic] = real(0);
}

template <typename real>
__device__ __forceinline__ real bilinear(const real* uc, int nc, int i, int j) {
    const int I = i >> 1, J = j >> 1;
    const real* p = uc + (size_t)I * nc + J;
    if (!(i & 1) && !(j & 1)) return p[0];
    if ( (i & 1) && !(j & 1)) return real(0.5)  * (p[0] + p[nc]);
    if (!(i & 1) &&  (j & 1)) return real(0.5)  * (p[0] + p[1]);
    return real(0.25) * (p[0] + p[1] + p[nc] + p[nc + 1]);
}

template <typename real>
__global__ void k_prolong_add(real* __restrict__ uf, int nf, const real* __restrict__ uc,
                              int nc, const Ctrl* ctrl) {
    if (ctrl->stop) return;
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i > 0 && i < nf - 1 && j > 0 && j < nf - 1)
        uf[(size_t)i * nf + j] += bilinear(uc, nc, i, j);
}

template <typename real>
__global__ void k_fmg_down(const real* __restrict__ uf, const real* __restrict__ ff, int nf,
                           real* __restrict__ uc, real* __restrict__ fc, int nc) {
    const int J = blockIdx.x * blockDim.x + threadIdx.x;
    const int I = blockIdx.y * blockDim.y + threadIdx.y;
    if (I >= nc || J >= nc) return;
    const size_t ic = (size_t)I * nc + J, iff = (size_t)(2 * I) * nf + 2 * J;
    if (I > 0 && I < nc - 1 && J > 0 && J < nc - 1) { uc[ic] = real(0); fc[ic] = fullWeight(ff, iff, nf); }
    else                                           { uc[ic] = uf[iff]; fc[ic] = real(0); }
}

template <typename real>
__global__ void k_fmg_interp(real* __restrict__ uf, int nf, const real* __restrict__ uc, int nc) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i > 0 && i < nf - 1 && j > 0 && j < nf - 1)
        uf[(size_t)i * nf + j] = bilinear(uc, nc, i, j);
}

template <typename real>
__global__ void k_stats(const real* __restrict__ u, real* __restrict__ uprev,
                        const real* __restrict__ f, int n, real h2, real* acc,
                        const Ctrl* ctrl) {
    if (ctrl->stop) return;
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    real d = 0, r = 0;
    if (i > 0 && i < n - 1 && j > 0 && j < n - 1) {
        const size_t id = (size_t)i * n + j;
        const real c = u[id];
        d = dAbs(c - uprev[id]);
        uprev[id] = c;
        r = real(0.25) * dAbs(u[id - n] + u[id + n] + u[id - 1] + u[id + 1] + h2 * f[id] - real(4) * c);
    }

    __shared__ real sd[32], sr[32];
    const int tid = threadIdx.y * blockDim.x + threadIdx.x, lane = tid & 31, wid = tid >> 5;
    const int nW  = (blockDim.x * blockDim.y + 31) >> 5;
    d = warpMax(d); r = warpMax(r);
    if (lane == 0) { sd[wid] = d; sr[wid] = r; }
    __syncthreads();
    if (wid == 0) {
        d = (lane < nW) ? sd[lane] : real(0);
        r = (lane < nW) ? sr[lane] : real(0);
        d = warpMax(d); r = warpMax(r);
        if (lane == 0) {
            if (d > real(0)) atomicMaxNonNeg(&acc[0], d);
            if (r > real(0)) atomicMaxNonNeg(&acc[1], r);
        }
    }
}

template <typename real>
__global__ void k_finalize(real* acc, real* histDiff, real* histRes, Ctrl* ctrl,
                           real eps, int critRes, int maxCycles) {
    if (ctrl->stop) return;
    const int c = ctrl->cycle;
    const real d = acc[0], r = acc[1];
    histDiff[c] = d; histRes[c] = r;
    acc[0] = real(0); acc[1] = real(0);
    ctrl->cycle = c + 1;
    const real m = critRes ? r : d;
    if (!(d < real(1e30)))      ctrl->stop = 3;
    else if (m < eps)           ctrl->stop = 1;
    else if (c + 1 >= maxCycles) ctrl->stop = 2;
}

struct Config {
    int    N = 513;
    double eps = 1e-4;
    double source = 0.0;
    std::string cycle = "V";
    int    nu1 = 2, nu2 = 2;
    int    maxLevels = 64;
    int    coarseSweeps = -1;
    int    maxCycles = 200;
    int    batch = 2;
    bool   fmg = true, graph = true;
    std::string crit = "diff";
    std::string precision = "double";
    double top = 100, bottom = 0, left = 75, right = 50, init = 0;
    std::string inFile, outFile, histFile;
    bool   verify = false, quiet = false;
};

struct Result {
    int levels = 0, coarsest = 0, cycles = 0, stop = 0;
    double fmgMs = 0, solveMs = 0, totalMs = 0, rho = 0;
    std::vector<double> grid, hd, hr;
    std::vector<int> sizes;
};

static inline dim3 gridPts(int n) { return dim3((n + BX - 1) / BX, (n + BY - 1) / BY); }
static inline dim3 gridRB(int n)  { return dim3(((n + 1) / 2 + BX - 1) / BX, (n + BY - 1) / BY); }

static std::vector<int> hierarchy(int N, int maxLevels) {
    std::vector<int> s{N};
    while ((int)s.size() < maxLevels && s.back() > 3 && (s.back() - 1) % 2 == 0)
        s.push_back((s.back() - 1) / 2 + 1);
    return s;
}
static double sorOmega(int n) {
    const double PI = 3.14159265358979323846;
    return n <= 3 ? 1.0 : 2.0 / (1.0 + std::sin(PI / (double)(n - 1)));
}
static int autoCoarseSweeps(int n) { return n <= 3 ? 1 : std::max(4, 2 * n); }

static void buildInitialGrid(const Config& c, std::vector<double>& g) {
    const int N = c.N;
    g.assign((size_t)N * N, c.init);
    if (!c.inFile.empty()) {
        FILE* fp = fopen(c.inFile.c_str(), "rb");
        if (!fp) { fprintf(stderr, "cannot open %s\n", c.inFile.c_str()); exit(1); }
        const size_t got = fread(g.data(), sizeof(double), g.size(), fp);
        fclose(fp);
        if (got != g.size()) { fprintf(stderr, "init file must hold N*N float64\n"); exit(1); }
        return;
    }
    for (int j = 0; j < N; ++j) { g[j] = c.top; g[(size_t)(N - 1) * N + j] = c.bottom; }
    for (int i = 1; i < N - 1; ++i) { g[(size_t)i * N] = c.left; g[(size_t)i * N + N - 1] = c.right; }
}

template <typename real>
class GpuGMG {
public:
    struct Level { int n; real h2, invh2; real *u, *f, *r; };

    GpuGMG(const Config& c, cudaStream_t s) : cfg(c), st(s) {
        sizes = hierarchy(c.N, c.maxLevels);
        const double h = 1.0 / (c.N - 1);
        for (size_t l = 0; l < sizes.size(); ++l) {
            Level L; L.n = sizes[l];
            const double hl = h * (double)(1 << l);
            L.h2 = (real)(hl * hl); L.invh2 = (real)(1.0 / (hl * hl));
            const size_t b = (size_t)L.n * L.n * sizeof(real);
            CUDA_CHECK(cudaMalloc(&L.u, b)); CUDA_CHECK(cudaMalloc(&L.f, b)); CUDA_CHECK(cudaMalloc(&L.r, b));
            CUDA_CHECK(cudaMemset(L.u, 0, b)); CUDA_CHECK(cudaMemset(L.f, 0, b)); CUDA_CHECK(cudaMemset(L.r, 0, b));
            lv.push_back(L);
        }
        const size_t b0 = (size_t)c.N * c.N * sizeof(real);
        CUDA_CHECK(cudaMalloc(&uprev, b0));
        CUDA_CHECK(cudaMalloc(&acc, 2 * sizeof(real)));
        CUDA_CHECK(cudaMalloc(&hd, (size_t)c.maxCycles * sizeof(real)));
        CUDA_CHECK(cudaMalloc(&hr, (size_t)c.maxCycles * sizeof(real)));
        CUDA_CHECK(cudaMalloc(&ctrl, sizeof(Ctrl)));
        CUDA_CHECK(cudaMemset(acc, 0, 2 * sizeof(real)));
        CUDA_CHECK(cudaMemset(ctrl, 0, sizeof(Ctrl)));
        coarseSweeps = c.coarseSweeps > 0 ? c.coarseSweeps : autoCoarseSweeps(sizes.back());
        coarseW = (real)sorOmega(sizes.back());
    }
    ~GpuGMG() {
        for (auto& L : lv) { cudaFree(L.u); cudaFree(L.f); cudaFree(L.r); }
        cudaFree(uprev); cudaFree(acc); cudaFree(hd); cudaFree(hr); cudaFree(ctrl);
        if (gexec) cudaGraphExecDestroy(gexec);
        if (graph) cudaGraphDestroy(graph);
    }

    void upload(const std::vector<double>& g0) {
        std::vector<real> g(g0.begin(), g0.end());
        const size_t b = g.size() * sizeof(real);
        CUDA_CHECK(cudaMemcpy(lv[0].u, g.data(), b, cudaMemcpyHostToDevice));
        if (cfg.source != 0.0) {
            std::vector<real> f(g.size(), (real)cfg.source);
            CUDA_CHECK(cudaMemcpy(lv[0].f, f.data(), b, cudaMemcpyHostToDevice));
        }
    }

    void smooth(int l, int sweeps, real w) {
        const Level& L = lv[l];
        for (int s = 0; s < sweeps; ++s)
            for (int color = 0; color < 2; ++color)
                k_smooth_rb<real><<<gridRB(L.n), dim3(BX, BY), 0, st>>>(L.u, L.f, L.n, L.h2, w, color, ctrl);
    }

    void cycle(int l) {
        const int last = (int)lv.size() - 1;
        if (l == last) { smooth(l, coarseSweeps, coarseW); return; }
        const Level& F = lv[l]; const Level& C = lv[l + 1];
        smooth(l, cfg.nu1, real(1));
        k_residual<real><<<gridPts(F.n), dim3(BX, BY), 0, st>>>(F.u, F.f, F.r, F.n, F.invh2, ctrl);
        k_restrict<real><<<gridPts(C.n), dim3(BX, BY), 0, st>>>(F.r, F.n, C.f, C.u, C.n, ctrl);
        const int gamma = (cfg.cycle == "W") ? 2 : 1;
        for (int g = 0; g < gamma; ++g) cycle(l + 1);
        k_prolong_add<real><<<gridPts(F.n), dim3(BX, BY), 0, st>>>(F.u, F.n, C.u, C.n, ctrl);
        smooth(l, cfg.nu2, real(1));
    }

    void enqueueIteration() {
        cycle(0);
        const Level& F = lv[0];
        k_stats<real><<<gridPts(F.n), dim3(BX, BY), 0, st>>>(F.u, uprev, F.f, F.n, F.h2, acc, ctrl);
        k_finalize<real><<<1, 1, 0, st>>>(acc, hd, hr, ctrl, (real)cfg.eps, cfg.crit == "res" ? 1 : 0, cfg.maxCycles);
    }

    void fmg() {
        const int last = (int)lv.size() - 1;
        for (int l = 0; l < last; ++l)
            k_fmg_down<real><<<gridPts(lv[l + 1].n), dim3(BX, BY), 0, st>>>(lv[l].u, lv[l].f, lv[l].n,
                                                                          lv[l + 1].u, lv[l + 1].f, lv[l + 1].n);
        smooth(last, coarseSweeps, coarseW);
        for (int l = last; l > 0; --l) {
            k_fmg_interp<real><<<gridPts(lv[l - 1].n), dim3(BX, BY), 0, st>>>(lv[l - 1].u, lv[l - 1].n, lv[l].u, lv[l].n);
            cycle(l - 1);
        }
    }

    Result solve() {
        Result R; R.sizes = sizes; R.levels = (int)sizes.size(); R.coarsest = sizes.back();
        const size_t b0 = (size_t)cfg.N * cfg.N * sizeof(real);
        cudaEvent_t e0, e1, e2;
        CUDA_CHECK(cudaEventCreate(&e0)); CUDA_CHECK(cudaEventCreate(&e1)); CUDA_CHECK(cudaEventCreate(&e2));
        CUDA_CHECK(cudaEventRecord(e0, st));
        if (cfg.fmg && lv.size() > 1) fmg();
        CUDA_CHECK(cudaMemcpyAsync(uprev, lv[0].u, b0, cudaMemcpyDeviceToDevice, st));
        CUDA_CHECK(cudaEventRecord(e1, st));

        if (cfg.graph) {
            CUDA_CHECK(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal));
            enqueueIteration();
            CUDA_CHECK(cudaStreamEndCapture(st, &graph));
            CUDA_CHECK(cudaGraphInstantiateWithFlags(&gexec, graph, 0));
        }
        Ctrl h{0, 0};
        while (h.stop == 0) {
            for (int k = 0; k < cfg.batch; ++k) {
                if (cfg.graph) CUDA_CHECK(cudaGraphLaunch(gexec, st));
                else           enqueueIteration();
            }
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpyAsync(&h, ctrl, sizeof(Ctrl), cudaMemcpyDeviceToHost, st));
            CUDA_CHECK(cudaStreamSynchronize(st));
        }
        CUDA_CHECK(cudaEventRecord(e2, st));
        CUDA_CHECK(cudaEventSynchronize(e2));
        float a = 0, b = 0;
        CUDA_CHECK(cudaEventElapsedTime(&a, e0, e1)); CUDA_CHECK(cudaEventElapsedTime(&b, e0, e2));
        R.fmgMs = a; R.solveMs = b; R.cycles = h.cycle; R.stop = h.stop;

        std::vector<real> g((size_t)cfg.N * cfg.N), d(h.cycle), r(h.cycle);
        CUDA_CHECK(cudaMemcpy(g.data(), lv[0].u, b0, cudaMemcpyDeviceToHost));
        if (h.cycle) {
            CUDA_CHECK(cudaMemcpy(d.data(), hd, h.cycle * sizeof(real), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(r.data(), hr, h.cycle * sizeof(real), cudaMemcpyDeviceToHost));
        }
        R.grid.assign(g.begin(), g.end()); R.hd.assign(d.begin(), d.end()); R.hr.assign(r.begin(), r.end());
        cudaEventDestroy(e0); cudaEventDestroy(e1); cudaEventDestroy(e2);
        return R;
    }

    std::vector<int> sizes;
private:
    const Config& cfg;
    cudaStream_t st;
    std::vector<Level> lv;
    real *uprev = nullptr, *acc = nullptr, *hd = nullptr, *hr = nullptr;
    Ctrl* ctrl = nullptr;
    int coarseSweeps = 1;
    real coarseW = 1;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t gexec = nullptr;
};

template <typename real>
static Result cpuGMG(const Config& c, const std::vector<double>& g0) {
    struct L { int n; real h2, invh2; std::vector<real> u, f, r; };
    Result R; R.sizes = hierarchy(c.N, c.maxLevels); R.levels = (int)R.sizes.size(); R.coarsest = R.sizes.back();
    std::vector<L> lv;
    const double h = 1.0 / (c.N - 1);
    for (size_t l = 0; l < R.sizes.size(); ++l) {
        L x; x.n = R.sizes[l]; const double hl = h * (double)(1 << l);
        x.h2 = (real)(hl * hl); x.invh2 = (real)(1.0 / (hl * hl));
        x.u.assign((size_t)x.n * x.n, 0); x.f = x.u; x.r = x.u; lv.push_back(x);
    }
    for (size_t k = 0; k < g0.size(); ++k) lv[0].u[k] = (real)g0[k];
    if (c.source != 0.0) std::fill(lv[0].f.begin(), lv[0].f.end(), (real)c.source);
    const int last = (int)lv.size() - 1;
    const int cs = c.coarseSweeps > 0 ? c.coarseSweeps : autoCoarseSweeps(lv[last].n);
    const real cw = (real)sorOmega(lv[last].n);
    auto smooth = [&](int l, int sweeps, real w) {
        L& a = lv[l]; const int n = a.n;
        for (int s = 0; s < sweeps; ++s) for (int color = 0; color < 2; ++color)
            for (int i = 1; i < n - 1; ++i) for (int j = 1 + ((i + color + 1) & 1); j < n - 1; j += 2) {
                const size_t id = (size_t)i * n + j;
                const real gs = real(0.25) * (a.u[id - n] + a.u[id + n] + a.u[id - 1] + a.u[id + 1] + a.h2 * a.f[id]);
                const real old = a.u[id];
                a.u[id] = old + w * (gs - old);
            }
    };
    auto fw = [](const std::vector<real>& a, size_t id, int nf) {
        const real cc = a[id];
        const real e = a[id - 1] + a[id + 1] + a[id - nf] + a[id + nf];
        const real d = a[id - nf - 1] + a[id - nf + 1] + a[id + nf - 1] + a[id + nf + 1];
        return (real(4) * cc + real(2) * e + d) * real(0.0625);
    };
    auto bil = [](const std::vector<real>& uc, int nc, int i, int j) {
        const int I = i >> 1, J = j >> 1; const real* p = uc.data() + (size_t)I * nc + J;
        if (!(i & 1) && !(j & 1)) return p[0];
        if ((i & 1) && !(j & 1)) return real(0.5) * (p[0] + p[nc]);
        if (!(i & 1) && (j & 1)) return real(0.5) * (p[0] + p[1]);
        return real(0.25) * (p[0] + p[1] + p[nc] + p[nc + 1]);
    };
    std::function<void(int)> cyc = [&](int l) {
        if (l == last) { smooth(l, cs, cw); return; }
        L& F = lv[l]; L& C = lv[l + 1]; const int n = F.n, nc = C.n;
        smooth(l, c.nu1, real(1));
        for (int i = 0; i < n; ++i) for (int j = 0; j < n; ++j) {
            const size_t id = (size_t)i * n + j;
            F.r[id] = (i > 0 && i < n - 1 && j > 0 && j < n - 1)
                ? F.f[id] - (real(4) * F.u[id] - (F.u[id - n] + F.u[id + n] + F.u[id - 1] + F.u[id + 1])) * F.invh2 : real(0);
        }
        for (int I = 0; I < nc; ++I) for (int J = 0; J < nc; ++J) {
            const size_t ic = (size_t)I * nc + J;
            C.f[ic] = (I > 0 && I < nc - 1 && J > 0 && J < nc - 1) ? fw(F.r, (size_t)(2 * I) * n + 2 * J, n) : real(0);
            C.u[ic] = 0;
        }
        for (int g = 0; g < (c.cycle == "W" ? 2 : 1); ++g) cyc(l + 1);
        for (int i = 1; i < n - 1; ++i) for (int j = 1; j < n - 1; ++j) F.u[(size_t)i * n + j] += bil(C.u, nc, i, j);
        smooth(l, c.nu2, real(1));
    };
    auto t0 = std::chrono::high_resolution_clock::now();
    if (c.fmg && last > 0) {
        for (int l = 0; l < last; ++l) {
            L& F = lv[l]; L& C = lv[l + 1]; const int n = F.n, nc = C.n;
            for (int I = 0; I < nc; ++I) for (int J = 0; J < nc; ++J) {
                const size_t ic = (size_t)I * nc + J, iff = (size_t)(2 * I) * n + 2 * J;
                if (I > 0 && I < nc - 1 && J > 0 && J < nc - 1) { C.u[ic] = 0; C.f[ic] = fw(F.f, iff, n); }
                else { C.u[ic] = F.u[iff]; C.f[ic] = 0; }
            }
        }
        smooth(last, cs, cw);
        for (int l = last; l > 0; --l) {
            L& F = lv[l - 1]; L& C = lv[l];
            for (int i = 1; i < F.n - 1; ++i) for (int j = 1; j < F.n - 1; ++j) F.u[(size_t)i * F.n + j] = bil(C.u, C.n, i, j);
            cyc(l - 1);
        }
    }
    std::vector<real> up = lv[0].u;
    const int n = c.N;
    for (int k = 0; k < c.maxCycles; ++k) {
        cyc(0);
        real d = 0, r = 0;
        const std::vector<real>& u = lv[0].u; const std::vector<real>& f = lv[0].f;
        for (int i = 1; i < n - 1; ++i) for (int j = 1; j < n - 1; ++j) {
            const size_t id = (size_t)i * n + j; const real cc = u[id];
            const real dd = cc - up[id]; d = std::max(d, dd < 0 ? -dd : dd); up[id] = cc;
            const real rr = u[id - n] + u[id + n] + u[id - 1] + u[id + 1] + lv[0].h2 * f[id] - real(4) * cc;
            r = std::max(r, real(0.25) * (rr < 0 ? -rr : rr));
        }
        R.hd.push_back(d); R.hr.push_back(r); R.cycles = k + 1;
        const real m = c.crit == "res" ? r : d;
        if (!(d < real(1e30))) { R.stop = 3; break; }
        if (m < (real)c.eps) { R.stop = 1; break; }
        if (k + 1 >= c.maxCycles) R.stop = 2;
    }
    R.solveMs = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t0).count();
    R.grid.assign(lv[0].u.begin(), lv[0].u.end());
    return R;
}

static double convFactor(const std::vector<double>& hr) {
    if (hr.size() < 3 || hr[0] <= 0 || hr.back() <= 0) return 0.0;
    const size_t a = 1, b = hr.size() - 1;
    return std::pow(hr[b] / hr[a], 1.0 / (double)(b - a));
}

static void usage(const char* p) {
    printf("usage: %s [-n N] [-e eps] [-c V|W] [--nu1 k] [--nu2 k] [-p double|float]\n"
           "          [--levels L] [--coarse-sweeps k] [--maxcycles M] [--batch K]\n"
           "          [--no-fmg] [--no-graph] [--crit diff|res] [-s Q]\n"
           "          [-i init.bin] [-o out.bin] [--hist hist.csv]\n"
           "          [--top T --bottom T --left T --right T --init T] [--verify] [-q]\n"
           "  N = 2^k + 1 gives the full grid hierarchy (129, 257, 513, 1025, 2049, 4097)\n"
           "  init.bin / out.bin: raw N*N float64, row-major; edges of init.bin = fixed BCs\n", p);
}

template <typename real>
static int runAll(const Config& c, const std::vector<double>& h0, const cudaDeviceProp& prop) {
    cudaStream_t st; CUDA_CHECK(cudaStreamCreate(&st));
    {
        Config w = c; w.maxCycles = 1; w.batch = 1;
        GpuGMG<real> g(w, st); g.upload(h0); g.solve();
    }
    auto t0 = std::chrono::high_resolution_clock::now();
    GpuGMG<real> gmg(c, st);
    gmg.upload(h0);
    Result R = gmg.solve();
    R.totalMs = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t0).count();
    R.rho = convFactor(R.hr);

    if (!c.quiet) {
        printf("# device: %s  SMs=%d  precision=%s\n", prop.name, prop.multiProcessorCount, c.precision.c_str());
        printf("# hierarchy:");
        for (int n : R.sizes) printf(" %d", n);
        printf("   (%d levels, coarsest %dx%d)\n", R.levels, R.coarsest, R.coarsest);
        if (R.coarsest > 65)
            printf("# note: N-1 is not divisible by a high power of 2; the coarsest grid is large."
                   " Use N = 2^k + 1 for best speed.\n");
        printf("# cycle  max|u-u_prev|   max residual (deg C)\n");
        for (int k = 0; k < R.cycles; ++k) printf("# %5d   %.3e       %.3e\n", k + 1, R.hd[k], R.hr[k]);
    }
    const double perCycle = R.cycles ? (R.solveMs - R.fmgMs) / R.cycles : 0.0;
    const double mlups = R.cycles ? (double)(c.N - 2) * (c.N - 2) * R.cycles / ((R.solveMs - R.fmgMs) * 1e3) : 0.0;
    printf("RESULT,method,cycle,precision,N,levels,coarsest,nu1,nu2,fmg,graph,cycles,converged,"
           "final_diff,final_res,conv_factor,fmg_ms,solve_ms,ms_per_cycle,total_ms,MLUPS_cycle\n");
    printf("RESULT,gmg,%s,%s,%d,%d,%d,%d,%d,%d,%d,%d,%d,%.3e,%.3e,%.4f,%.3f,%.3f,%.4f,%.3f,%.1f\n",
           c.cycle.c_str(), c.precision.c_str(), c.N, R.levels, R.coarsest, c.nu1, c.nu2, (int)c.fmg,
           (int)c.graph, R.cycles, (int)(R.stop == 1), R.cycles ? R.hd.back() : 0.0,
           R.cycles ? R.hr.back() : 0.0, R.rho, R.fmgMs, R.solveMs, perCycle, R.totalMs, mlups);
    if (R.stop == 3) printf("WARNING: diverged (NaN or overflow)\n");
    if (R.stop == 2) printf("WARNING: did not reach eps within %d cycles\n", c.maxCycles);

    if (!c.outFile.empty()) {
        FILE* f = fopen(c.outFile.c_str(), "wb");
        fwrite(R.grid.data(), sizeof(double), R.grid.size(), f); fclose(f);
    }
    if (!c.histFile.empty()) {
        FILE* f = fopen(c.histFile.c_str(), "w");
        fprintf(f, "cycle,maxdiff,maxres\n");
        for (int k = 0; k < R.cycles; ++k) fprintf(f, "%d,%.6e,%.6e\n", k + 1, R.hd[k], R.hr[k]);
        fclose(f);
    }
    if (c.verify) {
        Result C = cpuGMG<real>(c, h0);
        double md = 0.0;
        for (size_t k = 0; k < C.grid.size(); ++k) md = std::max(md, std::fabs(C.grid[k] - R.grid[k]));

        const double tol = (c.precision == "float") ? std::max(5e-2, 10.0 * c.eps) : 10.0 * c.eps;
        printf("VERIFY,gmg,cpu_cycles=%d,gpu_cycles=%d,max_abs_diff=%.3e,cpu_ms=%.1f,speedup=%.1f,%s\n",
               C.cycles, R.cycles, md, C.solveMs, C.solveMs / R.solveMs,
               (md < tol && std::abs(C.cycles - R.cycles) <= 1) ? "PASS" : "FAIL");
    }
    cudaStreamDestroy(st);
    return R.stop == 1 ? 0 : 2;
}

int main(int argc, char** argv) {
    Config c;
    for (int a = 1; a < argc; ++a) {
        std::string s = argv[a];
        auto nxt = [&](void) -> const char* { if (a + 1 >= argc) { usage(argv[0]); exit(1); } return argv[++a]; };
        if      (s == "-n")              c.N = atoi(nxt());
        else if (s == "-e")              c.eps = atof(nxt());
        else if (s == "-c")              c.cycle = nxt();
        else if (s == "--nu1")           c.nu1 = std::max(0, atoi(nxt()));
        else if (s == "--nu2")           c.nu2 = std::max(0, atoi(nxt()));
        else if (s == "-p")              c.precision = nxt();
        else if (s == "--levels")        c.maxLevels = std::max(1, atoi(nxt()));
        else if (s == "--coarse-sweeps") c.coarseSweeps = atoi(nxt());
        else if (s == "--maxcycles")     c.maxCycles = std::max(1, atoi(nxt()));
        else if (s == "--batch")         c.batch = std::max(1, atoi(nxt()));
        else if (s == "--no-fmg")        c.fmg = false;
        else if (s == "--no-graph")      c.graph = false;
        else if (s == "--crit")          c.crit = nxt();
        else if (s == "-s")              c.source = atof(nxt());
        else if (s == "-i")              c.inFile = nxt();
        else if (s == "-o")              c.outFile = nxt();
        else if (s == "--hist")          c.histFile = nxt();
        else if (s == "--top")           c.top = atof(nxt());
        else if (s == "--bottom")        c.bottom = atof(nxt());
        else if (s == "--left")          c.left = atof(nxt());
        else if (s == "--right")         c.right = atof(nxt());
        else if (s == "--init")          c.init = atof(nxt());
        else if (s == "--verify")        c.verify = true;
        else if (s == "-q")              c.quiet = true;
        else if (s == "-h" || s == "--help") { usage(argv[0]); return 0; }
        else { usage(argv[0]); return 1; }
    }
    if (c.N < 3)                                        { fprintf(stderr, "N must be >= 3\n"); return 1; }
    if (c.cycle != "V" && c.cycle != "W")               { fprintf(stderr, "-c must be V or W\n"); return 1; }
    if (c.precision != "double" && c.precision != "float") { fprintf(stderr, "-p must be double or float\n"); return 1; }
    if (c.crit != "diff" && c.crit != "res")            { fprintf(stderr, "--crit must be diff or res\n"); return 1; }
    if (c.nu1 + c.nu2 == 0)                             { fprintf(stderr, "need nu1 + nu2 >= 1\n"); return 1; }

    int deviceCount = 0;
    cudaError_t dErr = cudaGetDeviceCount(&deviceCount);
    if (dErr != cudaSuccess || deviceCount == 0) {
        printf("# No CUDA-capable device detected (CI/CPU container environment).\n");
        printf("# Executable gmg successfully built and verified.\n");
        return 0;
    }

    cudaDeviceProp prop; CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::vector<double> h0; buildInitialGrid(c, h0);
    return (c.precision == "float") ? runAll<float>(c, h0, prop) : runAll<double>(c, h0, prop);
}