cmake_minimum_required(VERSION 3.19)
project(qkv_matmul_310p LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(SOC_VERSION "Ascend310P3" CACHE STRING "AscendC compiler SoC")
set(ASCEND_CANN_PACKAGE_PATH "/usr/local/Ascend/cann-9.2.1" CACHE PATH "CANN root")
if(NOT SOC_VERSION MATCHES "^[Aa]scend310[Pp]3$")
  message(FATAL_ERROR "This experiment targets Ascend310P3")
endif()

find_file(ASCENDC_CMAKE ascendc.cmake
  PATHS "${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake"
        "${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/ascendc_kernel_cmake"
  NO_DEFAULT_PATH REQUIRED)
include("${ASCENDC_CMAKE}")
ascendc_library(qkv_matmul_310p_kernel SHARED qkv_matmul.cpp)

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
set(cann_arch_root "${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux")
set(cann_lib_paths "${ASCEND_CANN_PACKAGE_PATH}/lib64" "${cann_arch_root}/lib64")
find_library(ASCENDCL_LIB ascendcl PATHS ${cann_lib_paths} NO_DEFAULT_PATH REQUIRED)
find_library(PLATFORM_LIB platform PATHS ${cann_lib_paths} NO_DEFAULT_PATH REQUIRED)
find_library(TILING_API_LIB tiling_api PATHS ${cann_lib_paths} NO_DEFAULT_PATH REQUIRED)
separate_arguments(torch_cxx_flags NATIVE_COMMAND "${TORCH_CXX_FLAGS}")

add_library(qkv_matmul_310p_torch SHARED binding.cpp)
add_dependencies(qkv_matmul_310p_torch qkv_matmul_310p_kernel)
target_include_directories(qkv_matmul_310p_torch PRIVATE
  "${CMAKE_CURRENT_BINARY_DIR}/include"
  "${CMAKE_CURRENT_BINARY_DIR}/include/qkv_matmul_310p_kernel"
  "${ASCEND_CANN_PACKAGE_PATH}/include"
  "${cann_arch_root}/include"
  "${cann_arch_root}/ascendc/include"
  "${cann_arch_root}/ascendc/include/highlevel_api"
  "${cann_arch_root}/ascendc/include/highlevel_api/tiling"
  "${torch_npu_root}/include"
  "${torch_npu_root}/.."
  ${TORCH_INCLUDE_DIRS})
target_compile_options(qkv_matmul_310p_torch PRIVATE ${torch_cxx_flags})
target_link_libraries(qkv_matmul_310p_torch PRIVATE
  qkv_matmul_310p_kernel ${TORCH_LIBRARIES} "${TORCH_NPU_LIB}" "${ASCENDCL_LIB}"
  "${PLATFORM_LIB}" "${TILING_API_LIB}")
set_target_properties(qkv_matmul_310p_torch PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/$<0:>"
  BUILD_RPATH "$ORIGIN/lib"
  INSTALL_RPATH "$ORIGIN/lib"
  BUILD_RPATH_USE_ORIGIN TRUE
  SKIP_BUILD_RPATH FALSE
  BUILD_WITH_INSTALL_RPATH FALSE)
target_link_options(qkv_matmul_310p_torch PRIVATE
  "LINKER:--enable-new-dtags"
  "LINKER:-rpath,$ORIGIN/lib")
