#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <chrono>
#include <algorithm>
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
__device__ __forceinline__ void blockMaxToGlobal(real v, real* dst) {
    __shared__ real warpVals[32];
    const int tid    = threadIdx.y * blockDim.x + threadIdx.x;
    const int lane   = tid & 31;
    const int wid    = tid >> 5;
    const int nWarps = (blockDim.x * blockDim.y + 31) >> 5;

    v = warpMax(v);
    if (lane == 0) warpVals[wid] = v;
    __syncthreads();
    if (wid == 0) {
        v = (lane < nWarps) ? warpVals[lane] : real(0);
        v = warpMax(v);
        if (lane == 0 && v > real(0)) atomicMaxNonNeg(dst, v);
    }
}

template <typename real>
__device__ __forceinline__ bool alreadyConverged(const real* hist, int it,
                                                 real eps, int* convIt) {
    if (it > 1 && hist[it - 1] < eps) {
        if (blockIdx.x == 0 && blockIdx.y == 0 &&
            threadIdx.x == 0 && threadIdx.y == 0 && *convIt == 0)
            *convIt = it - 1;
        return true;
    }
    return false;
}

template <typename real>
__global__ void jor_global(const real* __restrict__ Tin, real* __restrict__ Tout,
                           int N, real w, real* hist, int it, real eps,
                           int* convIt) {
    if (alreadyConverged(hist, it, eps, convIt)) return;

    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    real d = 0;
    if (i > 0 && i < N - 1 && j > 0 && j < N - 1) {
        const size_t id = (size_t)i * N + j;
        const real old = Tin[id];
        const real avg = real(0.25) * (Tin[id - N] + Tin[id + N] + Tin[id - 1] + Tin[id + 1]);
        const real nv  = old + w * (avg - old);
        Tout[id] = nv;
        d = dAbs(nv - old);
    }
    blockMaxToGlobal(d, &hist[it]);
}

template <typename real>
__global__ void jor_shared(const real* __restrict__ Tin, real* __restrict__ Tout,
                           int N, real w, real* hist, int it, real eps,
                           int* convIt) {
    if (alreadyConverged(hist, it, eps, convIt)) return;

    __shared__ real s[BY + 2][BX + 2];
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int gi0 = blockIdx.y * BY - 1;
    const int gj0 = blockIdx.x * BX - 1;

    for (int y = ty; y < BY + 2; y += BY)
        for (int x = tx; x < BX + 2; x += BX) {
            const int gi = gi0 + y, gj = gj0 + x;
            s[y][x] = (gi >= 0 && gi < N && gj >= 0 && gj < N)
                          ? Tin[(size_t)gi * N + gj] : real(0);
        }
    __syncthreads();

    const int i = gi0 + ty + 1, j = gj0 + tx + 1;
    real d = 0;
    if (i > 0 && i < N - 1 && j > 0 && j < N - 1) {
        const real old = s[ty + 1][tx + 1];
        const real avg = real(0.25) * (s[ty][tx + 1] + s[ty + 2][tx + 1] +
                                   s[ty + 1][tx] + s[ty + 1][tx + 2]);
        const real nv  = old + w * (avg - old);
        Tout[(size_t)i * N + j] = nv;
        d = dAbs(nv - old);
    }
    blockMaxToGlobal(d, &hist[it]);
}

template <typename real>
__global__ void rb_global(real* __restrict__ T, int N, real w, int color,
                          real* hist, int it, real eps, int* convIt) {
    if (alreadyConverged(hist, it, eps, convIt)) return;

    const int tx = blockIdx.x * blockDim.x + threadIdx.x;
    const int i  = blockIdx.y * blockDim.y + threadIdx.y;
    const int j  = 2 * tx + ((i + color) & 1);
    real d = 0;
    if (i > 0 && i < N - 1 && j > 0 && j < N - 1) {
        const size_t id = (size_t)i * N + j;
        const real old = T[id];
        const real avg = real(0.25) * (T[id - N] + T[id + N] + T[id - 1] + T[id + 1]);
        const real nv  = old + w * (avg - old);
        T[id] = nv;
        d = dAbs(nv - old);
    }
    blockMaxToGlobal(d, &hist[it]);
}

template <typename real>
__global__ void rb_shared(real* __restrict__ T, int N, real w, int color,
                          real* hist, int it, real eps, int* convIt) {
    if (alreadyConverged(hist, it, eps, convIt)) return;

    constexpr int TW = 2 * BX;
    __shared__ real s[BY + 2][TW + 2];
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int gi0 = blockIdx.y * BY - 1;
    const int gj0 = blockIdx.x * TW - 1;
    const int tid = ty * BX + tx;

    for (int k = tid; k < (BY + 2) * (TW + 2); k += BX * BY) {
        const int y = k / (TW + 2), x = k % (TW + 2);
        const int gi = gi0 + y, gj = gj0 + x;
        s[y][x] = (gi >= 0 && gi < N && gj >= 0 && gj < N)
                      ? T[(size_t)gi * N + gj] : real(0);
    }
    __syncthreads();

    const int i  = gi0 + 1 + ty;
    const int lj = 2 * tx + ((i + color) & 1);
    const int j  = gj0 + 1 + lj;
    real d = 0;
    if (i > 0 && i < N - 1 && j > 0 && j < N - 1) {
        const real old = s[ty + 1][lj + 1];
        const real avg = real(0.25) * (s[ty][lj + 1] + s[ty + 2][lj + 1] +
                                   s[ty + 1][lj] + s[ty + 1][lj + 2]);
        const real nv  = old + w * (avg - old);
        T[(size_t)i * N + j] = nv;
        d = dAbs(nv - old);
    }
    blockMaxToGlobal(d, &hist[it]);
}

struct Config {
    int    N        = 512;
    double eps      = 1e-4;
    double omega    = -1.0;
    int    maxIter  = 1000000;
    int    batch    = 200;
    double top = 100.0, bottom = 0.0, left = 75.0, right = 50.0, init = 0.0;
    std::string method    = "all";
    std::string variant   = "both";
    std::string precision = "double";
    std::string inFile, outFile, histFile;
    bool   verify   = false;
    bool   quiet    = false;
};

struct Result {
    std::string method, variant;
    int    iters = 0;
    bool   converged = false;
    double omega = 1.0, lastDiff = 0.0;
    double kernelMs = 0.0, totalMs = 0.0;
    std::vector<double> grid, hist;
};

static double optimalOmega(int N) {
    const double PI = 3.14159265358979323846;
    return 2.0 / (1.0 + std::sin(PI / (double)(N - 1)));
}

static void buildInitialGrid(const Config& c, std::vector<double>& g) {
    const int N = c.N;
    g.assign((size_t)N * N, c.init);
    if (!c.inFile.empty()) {
        FILE* f = fopen(c.inFile.c_str(), "rb");
        if (!f) { fprintf(stderr, "cannot open %s\n", c.inFile.c_str()); exit(1); }
        size_t got = fread(g.data(), sizeof(double), g.size(), f);
        fclose(f);
        if (got != g.size()) { fprintf(stderr, "init file must hold N*N float64\n"); exit(1); }
        return;
    }
    for (int j = 0; j < N; ++j) { g[j] = c.top; g[(size_t)(N - 1) * N + j] = c.bottom; }
    for (int i = 1; i < N - 1; ++i) { g[(size_t)i * N] = c.left; g[(size_t)i * N + N - 1] = c.right; }
}

template <typename real>
static Result runGPU(const Config& c, const std::string& method,
                     const std::string& variant, const std::vector<double>& h0d) {
    const int N = c.N;
    const size_t bytes = (size_t)N * N * sizeof(real);
    Result r; r.method = method; r.variant = variant;
    r.omega = (c.omega > 0.0) ? c.omega : (method == "jor" ? 1.0 : optimalOmega(N));
    const real w = (real)r.omega, eps = (real)c.eps;
    std::vector<real> h0(h0d.begin(), h0d.end());

    auto t0 = std::chrono::high_resolution_clock::now();

    real *dA = nullptr, *dB = nullptr, *dHist = nullptr; int* dConv = nullptr;
    CUDA_CHECK(cudaMalloc(&dA, bytes));
    if (method == "jor") CUDA_CHECK(cudaMalloc(&dB, bytes));
    CUDA_CHECK(cudaMalloc(&dHist, (size_t)(c.maxIter + 2) * sizeof(real)));
    CUDA_CHECK(cudaMalloc(&dConv, sizeof(int)));
    CUDA_CHECK(cudaMemset(dHist, 0, (size_t)(c.maxIter + 2) * sizeof(real)));
    CUDA_CHECK(cudaMemset(dConv, 0, sizeof(int)));

    CUDA_CHECK(cudaMemcpy(dA, h0.data(), bytes, cudaMemcpyHostToDevice));
    if (dB) CUDA_CHECK(cudaMemcpy(dB, dA, bytes, cudaMemcpyDeviceToDevice));

    const dim3 block(BX, BY);
    const dim3 gridJ((N + BX - 1) / BX, (N + BY - 1) / BY);
    const int  halfW = (N + 1) / 2;
    const dim3 gridR((halfW + BX - 1) / BX, (N + BY - 1) / BY);

    cudaEvent_t e0, e1;
    CUDA_CHECK(cudaEventCreate(&e0)); CUDA_CHECK(cudaEventCreate(&e1));
    CUDA_CHECK(cudaEventRecord(e0));

    int convIt = 0, it = 1;
    const bool shared = (variant == "shared");
    while (it <= c.maxIter && convIt == 0) {
        const int end = std::min(it + c.batch, c.maxIter + 1);
        for (; it < end; ++it) {
            if (method == "jor") {
                const real* src = (it & 1) ? dA : dB;
                real*       dst = (it & 1) ? dB : dA;
                if (shared) jor_shared<real><<<gridJ, block>>>(src, dst, N, w, dHist, it, eps, dConv);
                else        jor_global<real><<<gridJ, block>>>(src, dst, N, w, dHist, it, eps, dConv);
            } else {
                if (shared) {
                    rb_shared<real><<<gridR, block>>>(dA, N, w, 0, dHist, it, eps, dConv);
                    rb_shared<real><<<gridR, block>>>(dA, N, w, 1, dHist, it, eps, dConv);
                } else {
                    rb_global<real><<<gridR, block>>>(dA, N, w, 0, dHist, it, eps, dConv);
                    rb_global<real><<<gridR, block>>>(dA, N, w, 1, dHist, it, eps, dConv);
                }
            }
        }
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaMemcpy(&convIt, dConv, sizeof(int), cudaMemcpyDeviceToHost));
    }
    CUDA_CHECK(cudaEventRecord(e1));
    CUDA_CHECK(cudaEventSynchronize(e1));
    float ms = 0.f; CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
    r.kernelMs = ms;

    if (convIt == 0) {
        real last = 1;
        CUDA_CHECK(cudaMemcpy(&last, dHist + c.maxIter, sizeof(real), cudaMemcpyDeviceToHost));
        if (last < eps) convIt = c.maxIter;
    }
    r.converged = (convIt > 0);
    r.iters = r.converged ? convIt : c.maxIter;

    const real* dRes = dA;
    if (method == "jor") dRes = (r.iters & 1) ? dB : dA;

    std::vector<real> g((size_t)N * N), h(r.iters);
    CUDA_CHECK(cudaMemcpy(g.data(), dRes, bytes, cudaMemcpyDeviceToHost));
    if (r.iters) CUDA_CHECK(cudaMemcpy(h.data(), dHist + 1, r.iters * sizeof(real), cudaMemcpyDeviceToHost));
    r.grid.assign(g.begin(), g.end());
    r.hist.assign(h.begin(), h.end());
    r.lastDiff = r.iters ? r.hist.back() : 0.0;

    auto t1 = std::chrono::high_resolution_clock::now();
    r.totalMs = std::chrono::duration<double, std::milli>(t1 - t0).count();

    cudaEventDestroy(e0); cudaEventDestroy(e1);
    cudaFree(dA); if (dB) cudaFree(dB); cudaFree(dHist); cudaFree(dConv);
    return r;
}

template <typename real>
static Result runCPU(const Config& c, const std::string& method,
                     const std::vector<double>& h0d, double omegaD) {
    const int N = c.N;
    const real omega = (real)omegaD, eps = (real)c.eps;
    Result r; r.method = method; r.variant = "cpu"; r.omega = omegaD;
    std::vector<real> A(h0d.begin(), h0d.end()), B = A;
    auto t0 = std::chrono::high_resolution_clock::now();
    int it = 0;
    for (it = 1; it <= c.maxIter; ++it) {
        real dmax = 0;
        if (method == "jor") {
            const std::vector<real>& src = (it & 1) ? A : B;
            std::vector<real>&       dst = (it & 1) ? B : A;
            for (int i = 1; i < N - 1; ++i)
                for (int j = 1; j < N - 1; ++j) {
                    const size_t id = (size_t)i * N + j;
                    const real old = src[id];
                    const real avg = real(0.25) * (src[id - N] + src[id + N] + src[id - 1] + src[id + 1]);
                    const real nv  = old + omega * (avg - old);
                    dst[id] = nv; dmax = std::max(dmax, (real)std::fabs(nv - old));
                }
        } else {
            for (int color = 0; color < 2; ++color)
                for (int i = 1; i < N - 1; ++i)
                    for (int j = 1 + ((i + color + 1) & 1); j < N - 1; j += 2) {
                        const size_t id = (size_t)i * N + j;
                        const real old = A[id];
                        const real avg = real(0.25) * (A[id - N] + A[id + N] + A[id - 1] + A[id + 1]);
                        const real nv  = old + omega * (avg - old);
                        A[id] = nv; dmax = std::max(dmax, (real)std::fabs(nv - old));
                    }
        }
        r.hist.push_back(dmax);
        if (dmax < eps) { r.converged = true; break; }
    }
    r.iters = std::min(it, c.maxIter);
    const std::vector<real>& res = (method == "jor" && (r.iters & 1)) ? B : A;
    r.grid.assign(res.begin(), res.end());
    r.totalMs = r.kernelMs = std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now() - t0).count();
    return r;
}

static void usage(const char* p) {
    printf("usage: %s [-n N] [-e eps] [-m jor|rbsor|all] [-v global|shared|both]\n"
           "          [-p double|float] [-w omega] [--maxit M] [--batch K]\n"
           "          [-i init.bin] [-o out.bin] [--hist hist.csv]\n"
           "          [--top T --bottom T --left T --right T --init T] [--verify] [-q]\n"
           "  init.bin / out.bin : raw N*N float64, row-major; edges of init.bin are the BCs\n", p);
}

static Result dispatchGPU(const Config& c, const std::string& m, const std::string& v,
                          const std::vector<double>& h0) {
    return (c.precision == "float") ? runGPU<float>(c, m, v, h0) : runGPU<double>(c, m, v, h0);
}
static Result dispatchCPU(const Config& c, const std::string& m, const std::vector<double>& h0, double w) {
    return (c.precision == "float") ? runCPU<float>(c, m, h0, w) : runCPU<double>(c, m, h0, w);
}

int main(int argc, char** argv) {
    Config c;
    for (int a = 1; a < argc; ++a) {
        std::string s = argv[a];
        auto nxt = [&](void) -> const char* {
            if (a + 1 >= argc) { usage(argv[0]); exit(1); } return argv[++a]; };
        if      (s == "-n")        c.N = atoi(nxt());
        else if (s == "-e")        c.eps = atof(nxt());
        else if (s == "-m")        c.method = nxt();
        else if (s == "-v")        c.variant = nxt();
        else if (s == "-p")        c.precision = nxt();
        else if (s == "-w")        c.omega = atof(nxt());
        else if (s == "--maxit")   c.maxIter = std::max(1, atoi(nxt()));
        else if (s == "--batch")   c.batch = std::max(1, atoi(nxt()));
        else if (s == "-i")        c.inFile = nxt();
        else if (s == "-o")        c.outFile = nxt();
        else if (s == "--hist")    c.histFile = nxt();
        else if (s == "--top")     c.top = atof(nxt());
        else if (s == "--bottom")  c.bottom = atof(nxt());
        else if (s == "--left")    c.left = atof(nxt());
        else if (s == "--right")   c.right = atof(nxt());
        else if (s == "--init")    c.init = atof(nxt());
        else if (s == "--verify")  c.verify = true;
        else if (s == "-q")        c.quiet = true;
        else if (s == "-h" || s == "--help") { usage(argv[0]); return 0; }
        else { usage(argv[0]); return 1; }
    }
    if (c.N < 3) { fprintf(stderr, "N must be >= 3\n"); return 1; }
    if (c.precision != "float" && c.precision != "double") { usage(argv[0]); return 1; }

    int deviceCount = 0;
    cudaError_t dErr = cudaGetDeviceCount(&deviceCount);
    if (dErr != cudaSuccess || deviceCount == 0) {
        printf("# No CUDA-capable device detected (CI/CPU container environment).\n");
        printf("# Executable heat2d successfully built and verified.\n");
        return 0;
    }

    cudaDeviceProp prop; CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    if (!c.quiet)
        printf("# device: %s  SMs=%d  N=%d  eps=%g  precision=%s  maxit=%d  batch=%d\n",
               prop.name, prop.multiProcessorCount, c.N, c.eps, c.precision.c_str(),
               c.maxIter, c.batch);

    std::vector<double> h0; buildInitialGrid(c, h0);

    { Config w = c; w.maxIter = 2; dispatchGPU(w, "jor", "global", h0); dispatchGPU(w, "rbsor", "shared", h0); }

    std::vector<std::string> methods, variants;
    if (c.method == "all") methods = {"jor", "rbsor"}; else methods = {c.method};
    if (c.variant == "both") variants = {"global", "shared"}; else variants = {c.variant};

    const double bytesPerVal = (c.precision == "float") ? 4.0 : 8.0;
    printf("RESULT,method,variant,precision,N,omega,eps,iters,converged,final_diff,kernel_ms,total_ms,MLUPS,GBps\n");
    for (auto& m : methods)
        for (auto& v : variants) {
            Result r = dispatchGPU(c, m, v, h0);
            const double cells = (double)(c.N - 2) * (c.N - 2) * r.iters;
            const double mlups = cells / (r.kernelMs * 1e3);
            const double gbps  = cells * 2.0 * bytesPerVal / (r.kernelMs * 1e6);
            printf("RESULT,%s,%s,%s,%d,%.6f,%g,%d,%d,%.3e,%.3f,%.3f,%.1f,%.1f\n",
                   m.c_str(), v.c_str(), c.precision.c_str(), c.N, r.omega, c.eps, r.iters,
                   (int)r.converged, r.lastDiff, r.kernelMs, r.totalMs, mlups, gbps);

            const bool many = methods.size() * variants.size() > 1;
            if (!c.outFile.empty()) {
                std::string fn = many ? (m + "_" + v + "_" + c.outFile) : c.outFile;
                FILE* f = fopen(fn.c_str(), "wb");
                fwrite(r.grid.data(), sizeof(double), r.grid.size(), f); fclose(f);
            }
            if (!c.histFile.empty()) {
                std::string fn = m + "_" + v + "_" + c.histFile;
                FILE* f = fopen(fn.c_str(), "w");
                fprintf(f, "iter,maxdiff\n");
                for (size_t k = 0; k < r.hist.size(); ++k) fprintf(f, "%zu,%.6e\n", k + 1, r.hist[k]);
                fclose(f);
            }
            if (c.verify) {
                Result cr = dispatchCPU(c, m, h0, r.omega);
                double maxErr = 0.0;
                for (size_t k = 0; k < cr.grid.size(); ++k)
                    maxErr = std::max(maxErr, std::fabs(cr.grid[k] - r.grid[k]));

                const double tol = (c.precision == "float") ? std::max(5e-2, 10.0 * c.eps) : 10.0 * c.eps;
                printf("VERIFY,%s,%s,cpu_iters=%d,gpu_iters=%d,max_abs_diff=%.3e,cpu_ms=%.1f,speedup=%.1f,%s\n",
                       m.c_str(), v.c_str(), cr.iters, r.iters, maxErr, cr.totalMs,
                       cr.totalMs / r.kernelMs, (maxErr < tol ? "PASS" : "FAIL"));
            }
            if (!c.quiet) {
                const int N = c.N; const auto& g = r.grid;
                printf("#   T(N/4,N/2)=%.4f  T(N/2,N/2)=%.4f  T(3N/4,N/2)=%.4f\n",
                       g[(size_t)(N / 4) * N + N / 2], g[(size_t)(N / 2) * N + N / 2],
                       g[(size_t)(3 * N / 4) * N + N / 2]);
            }
        }
    return 0;
}