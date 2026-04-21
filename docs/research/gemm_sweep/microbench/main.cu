// main.cu — standalone NVFP4 GEMM microbench for Phase B sweep.
//
// Usage: ./gemm_microbench <config_name> <M> <N> <K>
// Prints one CSV row to stdout: <config_name>,<M>,<N>,<K>,<min_us>
//
// On can_implement / initialize failure, prints <config_name>,<M>,<N>,<K>,-1.000
// so a higher-level sweep script can mark the config as "not implementable"
// and move on.
//
// Build via CMake; see CMakeLists.txt. CUTLASS_ROOT env var must point at a
// CUTLASS source checkout (tested with the one used by production at
// /app/nvllm/.deps/cutlass-src inside the nvllm:gb10 container).

#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/util/packed_stride.hpp"

#include "configs.hpp"
// NOTE: main_dispatch.hpp is included AFTER run_one is defined, below.

// --------------------------------------------------------------------------
// Small CUDA error-check helper.
// --------------------------------------------------------------------------
#define CUDA_CHECK(expr)                                                     \
  do {                                                                       \
    cudaError_t _err = (expr);                                               \
    if (_err != cudaSuccess) {                                               \
      std::fprintf(stderr, "CUDA error %d (%s) at %s:%d: %s\n",              \
                   (int)_err, cudaGetErrorName(_err), __FILE__, __LINE__,    \
                   cudaGetErrorString(_err));                                \
      std::exit(2);                                                          \
    }                                                                        \
  } while (0)

// --------------------------------------------------------------------------
// Build Gemm::Arguments exactly like args_from_options<Gemm> in
// csrc/libtorch_stable/quantization/fp4/nvfp4_scaled_mm_sm120_kernels.cu.
// --------------------------------------------------------------------------
template <typename Gemm>
typename Gemm::Arguments make_args(int M, int N, int K,
                                   void* A_ptr, void* B_ptr,
                                   void* A_sf_ptr, void* B_sf_ptr,
                                   void* D_ptr,
                                   const float* alpha_ptr) {
  using ElementA = typename Gemm::ElementA;
  using ElementB = typename Gemm::ElementB;
  using ElementD = typename Gemm::ElementD;
  using ElementSFA = cutlass::float_ue4m3_t;
  using ElementSFB = cutlass::float_ue4m3_t;
  using ElementCompute = float;

  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideD = typename Gemm::GemmKernel::StrideD;

  using Sm1xxBlkScaledConfig =
      typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

  auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, {M, K, 1});
  auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, {N, K, 1});
  auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, {M, N, 1});

  auto layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(
      cute::make_shape(M, N, K, 1));
  auto layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(
      cute::make_shape(M, N, K, 1));

  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {M, N, K, 1},
      {static_cast<ElementA const*>(A_ptr), stride_A,
       static_cast<ElementB const*>(B_ptr), stride_B,
       static_cast<ElementSFA const*>(A_sf_ptr), layout_SFA,
       static_cast<ElementSFB const*>(B_sf_ptr), layout_SFB},
      {{},
       static_cast<ElementD const*>(D_ptr),
       stride_D,
       static_cast<ElementD*>(D_ptr),
       stride_D}};
  auto& fusion_args = arguments.epilogue.thread;
  fusion_args.alpha_ptr = alpha_ptr;
  return arguments;
}

// --------------------------------------------------------------------------
// run_one<Config>(M, N, K, warmup, timed)
//   -> minimum kernel time in microseconds over `timed` iters,
//      or -1.0 if the config cannot be implemented for this shape.
// --------------------------------------------------------------------------
template <typename ConfigT>
double run_one(int M, int N, int K, int warmup, int timed) {
  using Gemm = typename ConfigT::Gemm;
  using ElementD = typename Gemm::ElementD;

  // NVFP4 packed-byte sizes. A/B are packed 2 elems per byte; scale factors are
  // fp8 laid out in the padded/swizzled blockscale tile format required by
  // Sm1xxBlkScaledConfig. Per the production dispatch (cutlass_scaled_fp4_mm_sm120a
  // in nvfp4_scaled_mm_sm120_kernels.cu), scale tensors are padded to
  //   rounded_m = round_up(M, 128)
  //   rounded_n = round_up(N, 128)
  //   rounded_k = round_up(K/16, 4)
  // and must have that many fp8 bytes allocated, NOT the unpadded M*K/16.
  auto round_up = [](int x, int y) { return (x + y - 1) / y * y; };
  const int rounded_m = round_up(M, 128);
  const int rounded_n = round_up(N, 128);
  const int rounded_k = round_up(K / 16, 4);
  const size_t bytes_A    = static_cast<size_t>(M) * K / 2;
  const size_t bytes_B    = static_cast<size_t>(N) * K / 2;
  const size_t bytes_A_sf = static_cast<size_t>(rounded_m) * rounded_k;
  const size_t bytes_B_sf = static_cast<size_t>(rounded_n) * rounded_k;
  const size_t bytes_D    = static_cast<size_t>(M) * N * sizeof(ElementD);

  void *A = nullptr, *B = nullptr, *A_sf = nullptr, *B_sf = nullptr, *D = nullptr;
  float* alpha_dev = nullptr;

  CUDA_CHECK(cudaMalloc(&A, bytes_A));
  CUDA_CHECK(cudaMalloc(&B, bytes_B));
  CUDA_CHECK(cudaMalloc(&A_sf, bytes_A_sf));
  CUDA_CHECK(cudaMalloc(&B_sf, bytes_B_sf));
  CUDA_CHECK(cudaMalloc(&D, bytes_D));
  CUDA_CHECK(cudaMalloc(&alpha_dev, sizeof(float)));

  // Initialize buffers. Zero A/B/D (avoids NaNs), set scale factors to the fp8
  // "1.0" bit pattern (0x3C for e4m3). alpha = 1.0 on device.
  CUDA_CHECK(cudaMemset(A, 0, bytes_A));
  CUDA_CHECK(cudaMemset(B, 0, bytes_B));
  CUDA_CHECK(cudaMemset(D, 0, bytes_D));
  CUDA_CHECK(cudaMemset(A_sf, 0x3C, bytes_A_sf));
  CUDA_CHECK(cudaMemset(B_sf, 0x3C, bytes_B_sf));

  const float alpha_host = 1.0f;
  CUDA_CHECK(cudaMemcpy(alpha_dev, &alpha_host, sizeof(float),
                        cudaMemcpyHostToDevice));

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  // Assemble Gemm args.
  Gemm gemm;
  auto args = make_args<Gemm>(M, N, K, A, B, A_sf, B_sf, D, alpha_dev);

  auto can = gemm.can_implement(args);
  if (can != cutlass::Status::kSuccess) {
    std::fprintf(stderr, "can_implement failed: %s\n",
                 cutlassGetStatusString(can));
    cudaStreamDestroy(stream);
    cudaFree(A); cudaFree(B); cudaFree(A_sf); cudaFree(B_sf); cudaFree(D);
    cudaFree(alpha_dev);
    return -1.0;
  }

  size_t workspace_bytes = Gemm::get_workspace_size(args);
  void* workspace = nullptr;
  if (workspace_bytes > 0) {
    CUDA_CHECK(cudaMalloc(&workspace, workspace_bytes));
  }

  // Clear any latent CUDA error state before initialize so a post-failure
  // cudaGetLastError() diagnoses the right call.
  (void)cudaGetLastError();
  auto init = gemm.initialize(args, workspace, stream);
  if (init != cutlass::Status::kSuccess) {
    cudaError_t cuda_err = cudaGetLastError();
    std::fprintf(stderr, "initialize failed: %s (last CUDA: %s / %s)\n",
                 cutlassGetStatusString(init),
                 cudaGetErrorName(cuda_err),
                 cudaGetErrorString(cuda_err));
    if (workspace) cudaFree(workspace);
    cudaStreamDestroy(stream);
    cudaFree(A); cudaFree(B); cudaFree(A_sf); cudaFree(B_sf); cudaFree(D);
    cudaFree(alpha_dev);
    return -1.0;
  }

  // Warmup.
  for (int i = 0; i < warmup; ++i) {
    auto st = gemm.run(args, workspace, stream);
    if (st != cutlass::Status::kSuccess) {
      std::fprintf(stderr, "run (warmup) failed: %s\n",
                   cutlassGetStatusString(st));
      if (workspace) cudaFree(workspace);
      cudaStreamDestroy(stream);
      cudaFree(A); cudaFree(B); cudaFree(A_sf); cudaFree(B_sf); cudaFree(D);
      cudaFree(alpha_dev);
      return -1.0;
    }
  }
  CUDA_CHECK(cudaStreamSynchronize(stream));

  // Event-based timing, one event-pair per iter, take the minimum.
  std::vector<cudaEvent_t> starts(timed), stops(timed);
  for (int i = 0; i < timed; ++i) {
    CUDA_CHECK(cudaEventCreate(&starts[i]));
    CUDA_CHECK(cudaEventCreate(&stops[i]));
  }

  for (int i = 0; i < timed; ++i) {
    CUDA_CHECK(cudaEventRecord(starts[i], stream));
    auto st = gemm.run(args, workspace, stream);
    if (st != cutlass::Status::kSuccess) {
      std::fprintf(stderr, "run (timed) failed: %s\n",
                   cutlassGetStatusString(st));
      for (int j = 0; j < timed; ++j) {
        cudaEventDestroy(starts[j]);
        cudaEventDestroy(stops[j]);
      }
      if (workspace) cudaFree(workspace);
      cudaStreamDestroy(stream);
      cudaFree(A); cudaFree(B); cudaFree(A_sf); cudaFree(B_sf); cudaFree(D);
      cudaFree(alpha_dev);
      return -1.0;
    }
    CUDA_CHECK(cudaEventRecord(stops[i], stream));
  }
  CUDA_CHECK(cudaStreamSynchronize(stream));

  double min_us = 1e18;
  for (int i = 0; i < timed; ++i) {
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, starts[i], stops[i]));
    double us = static_cast<double>(ms) * 1000.0;
    if (us < min_us) min_us = us;
    cudaEventDestroy(starts[i]);
    cudaEventDestroy(stops[i]);
  }

  if (workspace) cudaFree(workspace);
  cudaStreamDestroy(stream);
  cudaFree(A); cudaFree(B); cudaFree(A_sf); cudaFree(B_sf); cudaFree(D);
  cudaFree(alpha_dev);

  return min_us;
}

// Include generated dispatcher AFTER run_one<> is defined so its template
// instantiations in main_dispatch.hpp see the definition.
#include "main_dispatch.hpp"

// --------------------------------------------------------------------------
// main — parse <config_name> <M> <N> <K>, dispatch, print CSV row.
// --------------------------------------------------------------------------
int main(int argc, char** argv) {
  if (argc != 5) {
    std::fprintf(stderr,
                 "Usage: %s <config_name> <M> <N> <K>\n"
                 "  config_name: smoke_M256 or any Cfg_<TM>x<TN>x<TK>_<Sched>_<TileSched>\n"
                 "               from configs_generated.hpp\n",
                 argv[0]);
    return 1;
  }

  const std::string cfg_name = argv[1];
  const int M = std::atoi(argv[2]);
  const int N = std::atoi(argv[3]);
  const int K = std::atoi(argv[4]);

  if (M <= 0 || N <= 0 || K <= 0) {
    std::fprintf(stderr, "M, N, K must all be positive (got %d %d %d)\n",
                 M, N, K);
    return 1;
  }

  constexpr int kWarmup = 10;
  constexpr int kTimed  = 100;

  double min_us = dispatch_config(cfg_name.c_str(), M, N, K, kWarmup, kTimed);
  if (min_us == -2.0) {
    std::fprintf(stderr, "Unknown config_name: %s\n", cfg_name.c_str());
    return 1;
  }

  std::printf("%s,%d,%d,%d,%.3f\n", cfg_name.c_str(), M, N, K, min_us);
  return 0;
}
