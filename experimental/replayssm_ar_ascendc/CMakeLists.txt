cmake_minimum_required(VERSION 3.20)
project(replayssm_ar_lab LANGUAGES CXX)

set(RUN_MODE "npu" CACHE STRING "AscendC run mode")
set(SOC_VERSION "Ascend910B4" CACHE STRING "Exact compiler SoC; confirm on server")
set(ASCEND_CANN_PACKAGE_PATH "/usr/local/Ascend/cann-9.2.0" CACHE PATH "CANN 9.2.0 root")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
# Bisheng/cce-ld does not support CMake's GNU linker dependency-file flag.
set(CMAKE_LINK_DEPENDS_USE_LINKER FALSE)

if(NOT SOC_VERSION MATCHES "^[Aa]scend910[Bb]")
  message(FATAL_ERROR "This experiment targets Ascend910B only")
endif()
if(NOT RUN_MODE STREQUAL "npu")
  message(FATAL_ERROR "Only the real NPU build is supported")
endif()
foreach(candidate compiler/tikcpp/ascendc_kernel_cmake tools/tikcpp/ascendc_kernel_cmake)
  if(EXISTS "${ASCEND_CANN_PACKAGE_PATH}/${candidate}/ascendc.cmake")
    set(ASCENDC_CMAKE_DIR "${ASCEND_CANN_PACKAGE_PATH}/${candidate}")
    break()
  endif()
endforeach()
if(NOT ASCENDC_CMAKE_DIR)
  message(FATAL_ERROR "ascendc.cmake missing under ASCEND_CANN_PACKAGE_PATH")
endif()
include("${ASCENDC_CMAKE_DIR}/ascendc.cmake")
ascendc_library(replayssm_ar_kernel SHARED csrc/replayssm_ar.cpp)

find_package(Python3 REQUIRED COMPONENTS Interpreter)
execute_process(COMMAND "${Python3_EXECUTABLE}" -c
  "import torch; print(torch.utils.cmake_prefix_path)"
  OUTPUT_VARIABLE torch_prefix OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)
list(PREPEND CMAKE_PREFIX_PATH "${torch_prefix}")
find_package(Torch REQUIRED)
execute_process(COMMAND "${Python3_EXECUTABLE}" -c
  "import pathlib, torch_npu; print(pathlib.Path(torch_npu.__file__).parent)"
  OUTPUT_VARIABLE torch_npu_root OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)
find_library(TORCH_NPU_LIB torch_npu PATHS "${torch_npu_root}/lib" NO_DEFAULT_PATH REQUIRED)
find_library(ASCENDCL_LIB ascendcl PATHS "${ASCEND_CANN_PACKAGE_PATH}/lib64" NO_DEFAULT_PATH REQUIRED)

add_library(replayssm_ar_torch SHARED csrc/torch_binding.cpp)
add_dependencies(replayssm_ar_torch replayssm_ar_kernel)
target_include_directories(replayssm_ar_torch PRIVATE
  "${CMAKE_CURRENT_BINARY_DIR}/include"
  "${torch_npu_root}/include" "${torch_npu_root}/.."
  "${ASCEND_CANN_PACKAGE_PATH}/include" ${TORCH_INCLUDE_DIRS})
separate_arguments(torch_cxx_flags NATIVE_COMMAND "${TORCH_CXX_FLAGS}")
target_compile_options(replayssm_ar_torch PRIVATE ${torch_cxx_flags} -O2 -Wall)
target_link_libraries(replayssm_ar_torch PRIVATE replayssm_ar_kernel
  ${TORCH_LIBRARIES} "${TORCH_NPU_LIB}" "${ASCENDCL_LIB}")
set_target_properties(replayssm_ar_torch PROPERTIES
  BUILD_RPATH "$ORIGIN;${TORCH_INSTALL_PREFIX}/lib;${torch_npu_root}/lib;${ASCEND_CANN_PACKAGE_PATH}/lib64")
