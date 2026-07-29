cmake_minimum_required(VERSION 3.22)
project(flinke2c_gpu_kernels LANGUAGES CUDA)

set(FLINKE2C_CUDA_ARCHITECTURE "compute_80" CACHE STRING
        "Virtual CUDA architecture used for the generated PTX")
set(FLINKE2C_PTX "${CMAKE_CURRENT_SOURCE_DIR}/process_function.ptx")

add_custom_command(
        OUTPUT "${FLINKE2C_PTX}"
        COMMAND "${CMAKE_CUDA_COMPILER}"
                --ptx
                --std=c++17
                --gpu-architecture=${FLINKE2C_CUDA_ARCHITECTURE}
                -I"${CMAKE_CURRENT_SOURCE_DIR}"
                "${CMAKE_CURRENT_SOURCE_DIR}/process_function.cu"
                -o "${FLINKE2C_PTX}"
        DEPENDS
                process_function.cu
                process_map.cuh
                slot.h
        VERBATIM
        COMMENT "Compiling CUDA processing kernel to PTX"
)

add_custom_target(flinke2c_gpu ALL DEPENDS "${FLINKE2C_PTX}")
