cmake_minimum_required(VERSION 3.22)
project(flinke2c_gpu_kernels LANGUAGES CUDA)

set(FLINKE2C_CUDA_ARCHITECTURE "compute_80" CACHE STRING
        "Virtual CUDA architecture used for the generated PTX")
set(FLINKE2C_DIRECT_CUDA_ARCHITECTURE "80" CACHE STRING
        "CUDA architecture used for the direct JNI imputation library")
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

# CMake 3.22's FindJNI requires AWT by default. This library only uses the
# invocation-interface headers, so locate jni.h and the platform jni_md.h
# directly. Prefer JAVA_HOME, then infer the JDK root from javac.
find_package(Java QUIET COMPONENTS Development)
set(FLINKE2C_JAVA_HOME_HINTS)
if(DEFINED ENV{JAVA_HOME} AND NOT "$ENV{JAVA_HOME}" STREQUAL "")
    list(APPEND FLINKE2C_JAVA_HOME_HINTS "$ENV{JAVA_HOME}")
endif()
if(Java_JAVAC_EXECUTABLE)
    get_filename_component(FLINKE2C_JAVAC_REALPATH
            "${Java_JAVAC_EXECUTABLE}" REALPATH)
    get_filename_component(FLINKE2C_JAVA_BIN_DIR
            "${FLINKE2C_JAVAC_REALPATH}" DIRECTORY)
    get_filename_component(FLINKE2C_DETECTED_JAVA_HOME
            "${FLINKE2C_JAVA_BIN_DIR}" DIRECTORY)
    list(APPEND FLINKE2C_JAVA_HOME_HINTS
            "${FLINKE2C_DETECTED_JAVA_HOME}")
endif()

find_path(FLINKE2C_JNI_INCLUDE_DIR
        NAMES jni.h
        HINTS ${FLINKE2C_JAVA_HOME_HINTS}
        PATH_SUFFIXES include)
find_path(FLINKE2C_JNI_PLATFORM_INCLUDE_DIR
        NAMES jni_md.h
        HINTS "${FLINKE2C_JNI_INCLUDE_DIR}"
        PATH_SUFFIXES linux darwin win32)
if(NOT FLINKE2C_JNI_INCLUDE_DIR OR
        NOT FLINKE2C_JNI_PLATFORM_INCLUDE_DIR)
    message(FATAL_ERROR
            "JNI headers not found. Install a JDK and set JAVA_HOME to its root "
            "(the directory containing include/jni.h).")
endif()

add_library(flinke2c_imputation_gpu SHARED direct_imputation_jni.cu)
target_include_directories(
        flinke2c_imputation_gpu
        PRIVATE
        "${FLINKE2C_JNI_INCLUDE_DIR}"
        "${FLINKE2C_JNI_PLATFORM_INCLUDE_DIR}")
set_target_properties(
        flinke2c_imputation_gpu
        PROPERTIES
        CUDA_ARCHITECTURES "${FLINKE2C_DIRECT_CUDA_ARCHITECTURE}"
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
        POSITION_INDEPENDENT_CODE ON
)
