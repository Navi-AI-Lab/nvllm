## Per-kernel comparison (PIECEWISE vs FULL+blessed)

Common kernels: 77. Top 20 by absolute total_ms shift.

| Kernel | PW calls | PW mean μs | FL calls | FL mean μs | Δ μs | Δ % | Δ total ms |
|---|--:|--:|--:|--:|--:|--:|--:|
| `DecodeKernel (CuTe paged attn)` | 2786 | 17244.189 | 2786 | 17128.719 | -115.470 | -0.7% | -321.699 |
| `FP4 GEMM (_ZN7cutlass13device_kernelINS_4gemm6kern…)` | 27988 | 310.295 | 27988 | 314.113 | +3.818 | +1.2% | +106.870 |
| `PhaseE_Beta_Kernel (β-coop fused attn+MLP)` | 398 | 40967.875 | 398 | 40799.316 | -168.559 | -0.4% | -67.086 |
| `std::enable_if<!(false), void>::type internal::gemvx::kernel<int, int, __nv_bflo…` | 28856 | 391.672 | 28856 | 393.839 | +2.167 | +0.6% | +62.533 |
| `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<int>, …` | 9806 | 1.003 | 9806 | 0.707 | -0.296 | -29.5% | -2.895 |
| `causal_conv1d_update` | 9552 | 2.760 | 9552 | 2.478 | -0.282 | -10.2% | -2.690 |
| `memcpy32_post` | 6368 | 0.796 | 9154 | 0.818 | +0.022 | +2.8% | +2.419 |
| `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<a…` | 64 | 6082.614 | 64 | 6114.708 | +32.094 | +0.5% | +2.054 |
| `GDN linear-attn (fused_recurrent_gated_delta_rule)` | 9552 | 18.168 | 9552 | 17.973 | -0.195 | -1.1% | -1.860 |
| `void vllm::reshape_and_cache_flash_kernel<__nv_bfloat16, unsigned char, (vllm::F…` | 3200 | 2.943 | 3200 | 2.367 | -0.576 | -19.6% | -1.843 |
| `triton_poi_fused_0` | 9600 | 0.921 | 9600 | 0.730 | -0.191 | -20.7% | -1.836 |
| `nvjet_sm121_tst_mma_128x96x64_3_64x24x64_tmaAB_bz_TNNN` | 48 | 727.928 | 48 | 761.613 | +33.685 | +4.6% | +1.616 |
| `cvt_fp16_to_fp4` | 18420 | 2.380 | 18420 | 2.305 | -0.075 | -3.2% | -1.369 |
| `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<float>…` | 3980 | 0.980 | 3980 | 0.697 | -0.283 | -28.9% | -1.127 |
| `triton_poi_fused_zeros_6` | 6400 | 0.738 | 6400 | 0.600 | -0.138 | -18.7% | -0.883 |
| `triton_poi_fused_mul_silu_slice_0` | 2802 | 1.796 | 2802 | 1.506 | -0.290 | -16.1% | -0.812 |
| `triton_poi_fused_zeros_2` | 3200 | 0.744 | 3200 | 0.601 | -0.143 | -19.2% | -0.458 |
| `triton_poi_fused__to_copy__unsafe_view_add_clone_mean_mm_mul_pow_rsqrt_silu_t_vi…` | 9600 | 0.899 | 9600 | 0.856 | -0.043 | -4.8% | -0.412 |
| `triton_per_fused_1` | 9600 | 1.093 | 9600 | 1.073 | -0.020 | -1.8% | -0.193 |
| `FP4 GEMM (void cutlass::Kernel2<cutlass_80_wmma_te…)` | 48 | 307.217 | 48 | 311.142 | +3.925 | +1.3% | +0.189 |

### Kernels in PIECEWISE only (top 10 by total_ms)

| Kernel | calls | mean μs | total ms |
|---|--:|--:|--:|
| `triton_poi_fused__to_copy_add_cat_clone_mean_mul_pow_rsqrt_slice_split_split_wit…` | 3200 | 1.652 | 5.285 |
| `triton_poi_fused__to_copy_add_cat_mean_mul_pow_rsqrt_slice_split_split_with_size…` | 3200 | 1.516 | 4.852 |
| `triton_poi_fused__to_copy_add_cat_mean_mul_pow_rsqrt_slice_split_with_sizes_view…` | 3200 | 1.130 | 3.617 |
| `triton_poi_fused__to_copy_add_cat_clone_mean_mul_pow_rsqrt_slice_split_split_wit…` | 3200 | 1.107 | 3.544 |

### Kernels in FULL only (top 10 by total_ms)

| Kernel | calls | mean μs | total ms |
|---|--:|--:|--:|
| `triton_poi_fused_10` | 3200 | 1.173 | 3.755 |
| `triton_poi_fused_8` | 3200 | 1.165 | 3.729 |
| `triton_poi_fused_9` | 3200 | 1.050 | 3.359 |

### Aggregate (sum across common kernels)

- PIECEWISE total kernel time: **85,205.049 ms**
- FULL total kernel time: **84,976.460 ms**
- Δ: **-228.589 ms** (-0.27%)
