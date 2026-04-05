cmake_minimum_required(VERSION 3.28)
project(flinke2c_gpu LANGUAGES CUDA C CXX)

add_library(flinke2c_gpu SHARED
        kernel_scaffold.cu
)

set_target_properties(flinke2c_gpu PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES native
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)

target_compile_options(flinke2c_gpu PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fPIC>
)

add_library(flinke2c_rdma_shim STATIC
        rdma_verbs_shim.c
)

set_target_properties(flinke2c_rdma_shim PROPERTIES
        POSITION_INDEPENDENT_CODE ON
)
