cmake_minimum_required(VERSION 3.16)
project(replayssm_ar_ascendc LANGUAGES C CXX)

# ==============================================================================
# Basic configuration
# ==============================================================================

set(RUN_MODE "npu" CACHE STRING "AscendC run mode")
set(SOC_VERSION "Ascend910B4" CACHE STRING "Exact compiler SoC; confirm on server")
set(
    ASCEND_CANN_PACKAGE_PATH
    "/usr/local/Ascend/cann-9.2.0"
    CACHE PATH
    "CANN 9.2.0 root"
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Explicitly use Release when the caller does not provide a build type.
#
# This is important for CANN 9.2 AscendC legacy CMake. An empty
# CMAKE_BUILD_TYPE results in:
#
#   merge_mix_obj.sh ... --build-type
#
# with no value following --build-type, causing the merge stage to fail.
if(NOT CMAKE_BUILD_TYPE)
    set(
        CMAKE_BUILD_TYPE
        Release
        CACHE STRING
        "Build type"
        FORCE
    )
endif()

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Bisheng/cce-ld does not support CMake's GNU linker dependency-file flag.
set(CMAKE_LINK_DEPENDS_USE_LINKER FALSE)


# ==============================================================================
# Validate configuration
# ==============================================================================

if(NOT SOC_VERSION MATCHES "^[Aa]scend910[Bb]")
    message(
        FATAL_ERROR
        "This experiment targets Ascend910B only; got SOC_VERSION=${SOC_VERSION}"
    )
endif()

if(NOT RUN_MODE STREQUAL "npu")
    message(
        FATAL_ERROR
        "Only the real NPU build is supported; got RUN_MODE=${RUN_MODE}"
    )
endif()


# ==============================================================================
# AscendC
# ==============================================================================

set(ASCENDC_CMAKE_DIR "")

foreach(
    candidate
    compiler/tikcpp/ascendc_kernel_cmake
    tools/tikcpp/ascendc_kernel_cmake
)
    if(EXISTS "${ASCEND_CANN_PACKAGE_PATH}/${candidate}/ascendc.cmake")
        set(
            ASCENDC_CMAKE_DIR
            "${ASCEND_CANN_PACKAGE_PATH}/${candidate}"
        )
        break()
    endif()
endforeach()

if(NOT ASCENDC_CMAKE_DIR)
    message(
        FATAL_ERROR
        "ascendc.cmake missing under ASCEND_CANN_PACKAGE_PATH=${ASCEND_CANN_PACKAGE_PATH}"
    )
endif()

message(STATUS "CANN root: ${ASCEND_CANN_PACKAGE_PATH}")
message(STATUS "AscendC CMake: ${ASCENDC_CMAKE_DIR}")
message(STATUS "SOC_VERSION: ${SOC_VERSION}")
message(STATUS "RUN_MODE: ${RUN_MODE}")
message(STATUS "CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")

include("${ASCENDC_CMAKE_DIR}/ascendc.cmake")


# ==============================================================================
# AscendC ReplaySSM AR kernel
# ==============================================================================

ascendc_library(
    replayssm_ar_kernel
    SHARED
    ../../csrc/kernels/replayssm_ar.cpp
)

# CANN's ascendc_library() places this library under:
#
#   ${CMAKE_CURRENT_BINARY_DIR}/lib/libreplayssm_ar_kernel.so
#
# Keep this CANN-provided layout instead of overriding its output directory.


# ==============================================================================
# Python / PyTorch
# ==============================================================================

find_package(
    Python3
    REQUIRED
    COMPONENTS Interpreter
)

execute_process(
    COMMAND
        "${Python3_EXECUTABLE}"
        -c
        "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE torch_prefix
    OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY
)

list(PREPEND CMAKE_PREFIX_PATH "${torch_prefix}")

find_package(Torch REQUIRED)


# ==============================================================================
# torch_npu
# ==============================================================================

execute_process(
    COMMAND
        "${Python3_EXECUTABLE}"
        -c
        "import pathlib, torch_npu; print(pathlib.Path(torch_npu.__file__).parent)"
    OUTPUT_VARIABLE torch_npu_root
    OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY
)

find_library(
    TORCH_NPU_LIB
    torch_npu
    PATHS "${torch_npu_root}/lib"
    NO_DEFAULT_PATH
    REQUIRED
)


# ==============================================================================
# AscendCL
# ==============================================================================

find_library(
    ASCENDCL_LIB
    ascendcl
    PATHS "${ASCEND_CANN_PACKAGE_PATH}/lib64"
    NO_DEFAULT_PATH
    REQUIRED
)


# ==============================================================================
# PyTorch binding
# ==============================================================================

add_library(
    replayssm_ar_torch
    SHARED
    csrc/torch_binding.cpp
    ../../csrc/replayssm_ar.cpp
)

# The Torch binding depends on the AscendC kernel library being generated first.
add_dependencies(
    replayssm_ar_torch
    replayssm_ar_kernel
)

target_include_directories(
    replayssm_ar_torch
    PRIVATE

    "${CMAKE_CURRENT_SOURCE_DIR}/../../csrc"

    # Generated AscendC launch headers.
    "${CMAKE_CURRENT_BINARY_DIR}/include"

    # torch_npu headers.
    "${torch_npu_root}/include"
    "${torch_npu_root}/.."

    # CANN headers.
    "${ASCEND_CANN_PACKAGE_PATH}/include"

    # PyTorch headers.
    ${TORCH_INCLUDE_DIRS}
)

separate_arguments(
    torch_cxx_flags
    NATIVE_COMMAND
    "${TORCH_CXX_FLAGS}"
)

target_compile_options(
    replayssm_ar_torch
    PRIVATE
    ${torch_cxx_flags}
    -O2
    -Wall
)

target_link_libraries(
    replayssm_ar_torch
    PRIVATE

    # AscendC kernel:
    #
    #   build/lib/libreplayssm_ar_kernel.so
    replayssm_ar_kernel

    # PyTorch
    ${TORCH_LIBRARIES}

    # torch_npu
    "${TORCH_NPU_LIB}"

    # AscendCL
    "${ASCENDCL_LIB}"
)


# ==============================================================================
# Runtime library search path
# ==============================================================================
#
# Resulting build layout:
#
#   build/
#   ├── libreplayssm_ar_torch.so
#   └── lib/
#       └── libreplayssm_ar_kernel.so
#
# Therefore:
#
#   $ORIGIN
#
# refers to build/, while:
#
#   $ORIGIN/lib
#
# is required for libreplayssm_ar_torch.so to locate
# libreplayssm_ar_kernel.so.
#
# This removes the need for:
#
#   export LD_LIBRARY_PATH=$PWD/build/lib:$LD_LIBRARY_PATH
#
# when loading the Torch extension from the build tree.

set_target_properties(
    replayssm_ar_torch
    PROPERTIES

    BUILD_RPATH
    "$ORIGIN;$ORIGIN/lib;${TORCH_INSTALL_PREFIX}/lib;${torch_npu_root}/lib;${ASCEND_CANN_PACKAGE_PATH}/lib64"

    BUILD_RPATH_USE_ORIGIN
    TRUE
)
# Compile the same validated host launcher as the framework build.
target_compile_definitions(replayssm_ar_torch PRIVATE VLLM_ASCEND_ENABLE_REPLAYSSM_AR)
