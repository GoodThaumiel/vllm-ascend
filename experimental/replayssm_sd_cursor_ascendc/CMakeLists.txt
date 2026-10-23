cmake_minimum_required(VERSION 3.16)
project(replayssm_sd_cursor_ascendc LANGUAGES C CXX)

set(SOC_VERSION "Ascend910B4" CACHE STRING "AscendC compiler SoC")
set(ASCEND_CANN_PACKAGE_PATH "/usr/local/Ascend/cann-9.2.0" CACHE PATH "CANN root")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT SOC_VERSION MATCHES "^[Aa]scend910[Bb]")
    message(FATAL_ERROR "Replay SD cursor targets Ascend910B; got ${SOC_VERSION}")
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
ascendc_library(replayssm_sd_cursor_kernel SHARED ../../csrc/kernels/replayssm_sd_cursor.cpp)

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

add_library(replayssm_sd_cursor_torch SHARED csrc/torch_binding.cpp ../../csrc/replayssm_sd_cursor.cpp)
add_dependencies(replayssm_sd_cursor_torch replayssm_sd_cursor_kernel)
target_include_directories(replayssm_sd_cursor_torch PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../../csrc"
    "${CMAKE_CURRENT_BINARY_DIR}/include"
    "${torch_npu_root}/include"
    "${torch_npu_root}/.."
    "${ASCEND_CANN_PACKAGE_PATH}/include"
    ${TORCH_INCLUDE_DIRS})
separate_arguments(torch_cxx_flags NATIVE_COMMAND "${TORCH_CXX_FLAGS}")
target_compile_options(replayssm_sd_cursor_torch PRIVATE ${torch_cxx_flags} -O2 -Wall)
target_compile_definitions(replayssm_sd_cursor_torch PRIVATE VLLM_ASCEND_ENABLE_REPLAYSSM_SD_CURSOR)
target_link_libraries(replayssm_sd_cursor_torch PRIVATE
    replayssm_sd_cursor_kernel ${TORCH_LIBRARIES} "${TORCH_NPU_LIB}" "${ASCENDCL_LIB}")
set_target_properties(replayssm_sd_cursor_torch PROPERTIES
    BUILD_RPATH "$ORIGIN/lib"
    INSTALL_RPATH "$ORIGIN/lib"
    BUILD_RPATH_USE_ORIGIN TRUE
    SKIP_BUILD_RPATH FALSE
    BUILD_WITH_INSTALL_RPATH FALSE
)
target_link_options(replayssm_sd_cursor_torch PRIVATE
    "LINKER:-rpath,\$ORIGIN/lib"
    "LINKER:--enable-new-dtags"
)
