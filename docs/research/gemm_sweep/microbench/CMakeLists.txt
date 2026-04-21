cmake_minimum_required(VERSION 3.20)
project(gemm_microbench CUDA CXX)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

# DGX Spark GB10 reports compute_cap 12.1 — use the "a" architecture variant
# for block-scaled tensor ops (matches the production docker/Dockerfile.gb10
# TORCH_CUDA_ARCH_LIST="12.1a").
set(CMAKE_CUDA_ARCHITECTURES 121a)

set(CUTLASS_ROOT "$ENV{CUTLASS_ROOT}" CACHE PATH "CUTLASS source root")
if(NOT EXISTS "${CUTLASS_ROOT}/include/cutlass/cutlass.h")
    message(FATAL_ERROR "Set CUTLASS_ROOT env var to a valid CUTLASS source checkout (need include/cutlass/cutlass.h).")
endif()

add_executable(gemm_microbench main.cu)

target_include_directories(gemm_microbench PRIVATE
    "${CUTLASS_ROOT}/include"
    "${CUTLASS_ROOT}/tools/util/include"
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_compile_options(gemm_microbench PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --expt-relaxed-constexpr
        --expt-extended-lambda
        -Xcompiler=-fPIC
        --generate-line-info
    >
)

target_compile_definitions(gemm_microbench PRIVATE
    CUTLASS_ARCH_MMA_SM120_SUPPORTED=1
)

target_link_libraries(gemm_microbench PRIVATE cuda cudart)
