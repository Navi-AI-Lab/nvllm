// configs.hpp — NVFP4 GEMM config definitions for Phase B microbench sweep.
//
// Mirrors the production Fp4GemmSm120<Config, OutType> chain from
// csrc/libtorch_stable/quantization/fp4/nvfp4_scaled_mm_sm120_kernels.cu
// (sm120_fp4_config_M256), so this microbench binary uses the same
// CollectiveMainloop + CollectiveEpilogue + GemmKernel assembly as production.
//
// The SMOKE config reproduces sm120_fp4_config_M256:
//   MmaTileShape  = <128, 128, 128>
//   ClusterShape  = <1, 1, 1>
//   KernelSchedule   = KernelScheduleAuto
//   EpilogueSchedule = EpilogueScheduleAuto
//   TileScheduler    = void (default persistent)

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"

#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/dispatch_policy.hpp"

#include "cute/tensor.hpp"

// --------------------------------------------------------------------------
// Tile-shape configs. Each struct exports ClusterShape, MmaTileShape, and
// PerSmTileShape_MNK; Phase B.1.2 will add more configs following this pattern.
// --------------------------------------------------------------------------

struct smoke_M256_config {
  using ClusterShape      = cute::Shape<cute::_1,   cute::_1, cute::_1>;
  using MmaTileShape      = cute::Shape<cute::_128, cute::_128, cute::_128>;
  using PerSmTileShape_MNK = cute::Shape<cute::_128, cute::_128, cute::_128>;
};

// --------------------------------------------------------------------------
// GemmFactory<Config, KernelSchedule, EpilogueSchedule, TileScheduler, OutType>
//
// Assembles the full Gemm type stack exactly like production's
// Fp4GemmSm120<Config, OutType> (non-StreamK variant). Defaults reproduce
// the M256 production path.
// --------------------------------------------------------------------------

template <typename Config,
          typename KernelSchedule   = cutlass::gemm::collective::KernelScheduleAuto,
          typename EpilogueSchedule = cutlass::epilogue::collective::EpilogueScheduleAuto,
          typename TileScheduler    = void,
          typename OutType          = cutlass::bfloat16_t>
struct GemmFactory {
  using ElementA = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
  using LayoutATag = cutlass::layout::RowMajor;
  static constexpr int AlignmentA = 32;

  using ElementB = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
  using LayoutBTag = cutlass::layout::ColumnMajor;
  static constexpr int AlignmentB = 32;

  using ElementD = OutType;
  using ElementC = OutType;
  using LayoutCTag = cutlass::layout::RowMajor;
  using LayoutDTag = cutlass::layout::RowMajor;
  static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;
  static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;

  using ElementAccumulator = float;
  using ArchTag = cutlass::arch::Sm120;
  using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;

  using MmaTileShape = typename Config::MmaTileShape;
  using ClusterShape = typename Config::ClusterShape;
  using PerSmTileShape_MNK = typename Config::PerSmTileShape_MNK;

  using CollectiveEpilogue =
      typename cutlass::epilogue::collective::CollectiveBuilder<
          ArchTag, OperatorClass, PerSmTileShape_MNK, ClusterShape,
          cutlass::epilogue::collective::EpilogueTileAuto, ElementAccumulator,
          ElementAccumulator, ElementC, LayoutCTag, AlignmentC, ElementD,
          LayoutDTag, AlignmentD,
          EpilogueSchedule>::CollectiveOp;

  using CollectiveMainloop =
      typename cutlass::gemm::collective::CollectiveBuilder<
          ArchTag, OperatorClass, ElementA, LayoutATag, AlignmentA, ElementB,
          LayoutBTag, AlignmentB, ElementAccumulator, MmaTileShape,
          ClusterShape,
          cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
              sizeof(typename CollectiveEpilogue::SharedStorage))>,
          KernelSchedule>::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      cute::Shape<int, int, int, int>,
      CollectiveMainloop,
      CollectiveEpilogue,
      TileScheduler>;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

// --------------------------------------------------------------------------
// SmokeConfig: the one config this B.1.1 binary needs to instantiate.
// --------------------------------------------------------------------------

using SmokeConfig = GemmFactory<smoke_M256_config>;

// --------------------------------------------------------------------------
// Generated configs (12 tiles x 4 schedules x 2 schedulers, illegal combos
// pre-skipped by gen_configs.py). Regenerate with:
//   .venv/bin/python docs/research/gemm_sweep/microbench/gen_configs.py
// --------------------------------------------------------------------------
#include "configs_generated.hpp"
