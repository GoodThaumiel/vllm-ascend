cmake_minimum_required(VERSION 3.19)
project(replayssm_sd_mix_910b LANGUAGES C CXX)

set(SOC_VERSION "Ascend910B4" CACHE STRING "AscendC compiler SoC")
set(ASCEND_CANN_PACKAGE_PATH "/usr/local/Ascend/cann-9.2.0" CACHE PATH "CANN root")
set(REPLAYSSM_VERIFY_MAX_TOKENS "4" CACHE STRING "Candidate window compiled into the mixed verify library")
option(REPLAYSSM_ROOFLINE_CALIBRATION "Build the standalone AIV throughput calibration op" OFF)
if(NOT REPLAYSSM_VERIFY_MAX_TOKENS MATCHES "^(4|8)$")
    message(FATAL_ERROR "REPLAYSSM_VERIFY_MAX_TOKENS must be 4 or 8")
endif()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_LINK_DEPENDS_USE_LINKER FALSE)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
if(NOT (SOC_VERSION MATCHES "^[Aa]scend910[Bb]" OR SOC_VERSION STREQUAL "ascend910_9362"))
    message(FATAL_ERROR "Replay SD pilot targets Ascend910B or ascend910_9362; got ${SOC_VERSION}")
endif()

set(ASCENDC_CMAKE_DIR "")
foreach(candidate compiler/tikcpp/ascendc_kernel_cmake tools/tikcpp/ascendc_kernel_cmake)
    if(EXISTS "${ASCEND_CANN_PACKAGE_PATH}/${candidate}/ascendc.cmake")
        set(ASCENDC_CMAKE_DIR "${ASCEND_CANN_PACKAGE_PATH}/${candidate}")
        break()
    endif()
endforeach()
if(NOT ASCENDC_CMAKE_DIR)
    message(FATAL_ERROR "ascendc.cmake missing under ${ASCEND_CANN_PACKAGE_PATH}")
endif()
include("${ASCENDC_CMAKE_DIR}/ascendc.cmake")

set(mixed_kernels verify.cpp)
if(REPLAYSSM_ROOFLINE_CALIBRATION)
    list(APPEND mixed_kernels vector_calibration.cpp)
endif()
ascendc_library(replayssm_sd_mix_kernel SHARED ${mixed_kernels})
ascendc_compile_definitions(replayssm_sd_mix_kernel PRIVATE REPLAYSSM_VERIFY_MAX_TOKENS=${REPLAYSSM_VERIFY_MAX_TOKENS})
# CANN's legacy helper sends definitions only to the device compilations.
ascendc_compile_options(replayssm_sd_mix_kernel PRIVATE
    -forward-options-to-host-compiler "-DREPLAYSSM_VERIFY_MAX_TOKENS=${REPLAYSSM_VERIFY_MAX_TOKENS}")

find_package(Python3 REQUIRED COMPONENTS Interpreter)
execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE torch_prefix OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)
list(PREPEND CMAKE_PREFIX_PATH "${torch_prefix}")
find_package(Torch REQUIRED)
execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c "import pathlib, torch_npu; print(pathlib.Path(torch_npu.__file__).parent)"
    OUTPUT_VARIABLE torch_npu_root OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)
find_library(TORCH_NPU_LIB torch_npu PATHS "${torch_npu_root}/lib" NO_DEFAULT_PATH REQUIRED)
find_library(ASCENDCL_LIB ascendcl PATHS "${ASCEND_CANN_PACKAGE_PATH}/lib64" NO_DEFAULT_PATH REQUIRED)

add_library(replayssm_sd_mix_torch SHARED binding.cpp)
if(REPLAYSSM_ROOFLINE_CALIBRATION)
    target_compile_definitions(replayssm_sd_mix_torch PRIVATE REPLAYSSM_ROOFLINE_CALIBRATION)
endif()
target_compile_definitions(replayssm_sd_mix_torch PRIVATE REPLAYSSM_VERIFY_MAX_TOKENS=${REPLAYSSM_VERIFY_MAX_TOKENS})
add_dependencies(replayssm_sd_mix_torch replayssm_sd_mix_kernel)
target_include_directories(replayssm_sd_mix_torch PRIVATE
    "${CMAKE_CURRENT_BINARY_DIR}/include"
    "${torch_npu_root}/include"
    "${torch_npu_root}/.."
    "${ASCEND_CANN_PACKAGE_PATH}/include"
    ${TORCH_INCLUDE_DIRS})
separate_arguments(torch_cxx_flags NATIVE_COMMAND "${TORCH_CXX_FLAGS}")
target_compile_options(replayssm_sd_mix_torch PRIVATE ${torch_cxx_flags} -O2 -Wall)
target_link_libraries(replayssm_sd_mix_torch PRIVATE
    replayssm_sd_mix_kernel ${TORCH_LIBRARIES} "${TORCH_NPU_LIB}" "${ASCENDCL_LIB}")
# Runtime search path for the AscendC kernel dependency. ascendc_library writes
# the kernel under build/lib, while the binding remains in build/.
set_target_properties(replayssm_sd_mix_torch PROPERTIES
    BUILD_RPATH "$ORIGIN/lib"
    INSTALL_RPATH "$ORIGIN/lib"
    BUILD_RPATH_USE_ORIGIN TRUE
    SKIP_BUILD_RPATH FALSE
    BUILD_WITH_INSTALL_RPATH FALSE
)

# Explicitly request an ELF RUNPATH entry from the linker.
target_link_options(replayssm_sd_mix_torch PRIVATE
    "LINKER:-rpath,\$ORIGIN/lib"
    "LINKER:--enable-new-dtags"
)
