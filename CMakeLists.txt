# 文件说明：KuiperLLama 顶层构建入口，配置 CUDA/C++ 标准、第三方库和子目录。

cmake_minimum_required(VERSION 3.16)
set(CMAKE_CUDA_ARCHITECTURES 86)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Set the CUDA compiler path or name
set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")

project(llama_infer CXX CUDA)
include(cmake/cuda.cmake)

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)

option(LLAMA3_SUPPORT OFF)
if (LLAMA3_SUPPORT)
  message(STATUS "LLAMA3 SUPPORT")
endif()
option(QWEN2_SUPPORT OFF)
if (QWEN2_SUPPORT)
  message(STATUS "QWEN2 SUPPORT")
endif()

option(QWEN3_SUPPORT OFF)
if (QWEN3_SUPPORT)
  message(STATUS "QWEN3 SUPPORT")
endif()
# ---- Add dependencies via CPM ----
# see https://github.com/TheLartians/CPM.cmake for more info
option(USE_CPM "Use CPM for dependency management" OFF)

if(USE_CPM)
  # Use CPM to manage dependencies
  include(cmake/CPM.cmake)

  CPMAddPackage(
    NAME GTest
    GITHUB_REPOSITORY google/googletest
    VERSION 1.15.0
  )

  CPMAddPackage(
    NAME glog
    GITHUB_REPOSITORY google/glog
    VERSION 0.7.1
    OPTIONS "BUILD_TESTING Off"
  )

  CPMAddPackage(
    NAME Armadillo
    GITLAB_REPOSITORY conradsnicta/armadillo-code
    GIT_TAG 14.0.1
  )

  CPMAddPackage(
    NAME sentencepiece
    GITHUB_REPOSITORY google/sentencepiece
    VERSION 0.2.0
  )
  find_package(sentencepiece REQUIRED)

  if (LLAMA3_SUPPORT OR QWEN2_SUPPORT OR QWEN3_SUPPORT)
    CPMAddPackage(
        NAME absl
        GITHUB_REPOSITORY abseil/abseil-cpp
        GIT_TAG 20240722.0
        OPTIONS "BUILD_TESTING Off" "ABSL_PROPAGATE_CXX_STD ON" "ABSL_ENABLE_INSTALL ON"
    )
    CPMAddPackage(
        NAME re2
        GITHUB_REPOSITORY google/re2
        GIT_TAG 2024-07-02
    )
    CPMAddPackage(
        NAME nlohmann_json
        GITHUB_REPOSITORY nlohmann/json
        VERSION 3.11.3
    )
  endif()
endif()

# ---------------------------------
find_package(GTest REQUIRED)
find_package(glog REQUIRED)
find_package(Armadillo REQUIRED)

function(ensure_re2_target)
  if (TARGET re2::re2)
    return()
  endif ()

  find_package(re2 CONFIG QUIET)
  if (TARGET re2::re2)
    return()
  endif ()

  find_package(PkgConfig REQUIRED)
  pkg_check_modules(RE2 REQUIRED re2)
  add_library(re2::re2 INTERFACE IMPORTED)
  target_include_directories(re2::re2 INTERFACE ${RE2_INCLUDE_DIRS})
  target_link_directories(re2::re2 INTERFACE ${RE2_LIBRARY_DIRS})
  target_link_libraries(re2::re2 INTERFACE ${RE2_LIBRARIES})
  target_compile_options(re2::re2 INTERFACE ${RE2_CFLAGS_OTHER})
endfunction()

aux_source_directory(kuiper/source/tensor/ DIR_TENSOR)
aux_source_directory(kuiper/source/base/ DIR_BASE)
aux_source_directory(kuiper/source/op/ DIR_OP)
aux_source_directory(kuiper/source/model/ DIR_MODEL)
aux_source_directory(kuiper/source/op/kernels/cpu DIR_KERNEL_CPU)
aux_source_directory(kuiper/source/op/kernels/cuda DIR_KERNEL_CUDA)
aux_source_directory(kuiper/source/op/kernels/ DIR_KERNEL)
aux_source_directory(kuiper/source/sampler DIR_SAMPLE)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

function(configure_llama_target target_name)
  target_link_libraries(${target_name} PUBLIC sentencepiece glog::glog gtest gtest_main pthread cudart cublas armadillo)
  target_link_directories(${target_name} PUBLIC ${CMAKE_CUDA_COMPILER_LIBRARY_ROOT}/lib64)
  target_include_directories(${target_name} PUBLIC ${glog_INCLUDE_DIR})
  target_include_directories(${target_name} PUBLIC ${PROJECT_SOURCE_DIR}/kuiper/include)
  target_include_directories(${target_name} PUBLIC ${Armadillo_INCLUDE_DIR})
  target_include_directories(${target_name} PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
  if(USE_CPM)
    target_include_directories(${target_name} PUBLIC ${sentencepiece_SOURCE_DIR}/src)
  endif()
  set_target_properties(${target_name} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endfunction()

function(enable_json_regex_support target_name)
  find_package(absl REQUIRED)
  find_package(nlohmann_json REQUIRED)
  ensure_re2_target()
  target_link_libraries(${target_name} PUBLIC absl::base absl::strings re2::re2 nlohmann_json::nlohmann_json)
endfunction()

set(PRIMARY_MODEL_DEF "")
if (QWEN2_SUPPORT)
  set(PRIMARY_MODEL_DEF QWEN2_SUPPORT)
elseif (QWEN3_SUPPORT)
  set(PRIMARY_MODEL_DEF QWEN3_SUPPORT)
elseif (LLAMA3_SUPPORT)
  set(PRIMARY_MODEL_DEF LLAMA3_SUPPORT)
endif()

add_library(llama SHARED ${DIR_TENSOR} ${DIR_BASE} ${DIR_OP} ${DIR_KERNEL} ${DIR_MODEL} ${DIR_KERNEL_CPU} ${DIR_KERNEL_CUDA} ${DIR_KERNEL} ${DIR_SAMPLE})
configure_llama_target(llama)
if (PRIMARY_MODEL_DEF)
  target_compile_definitions(llama PUBLIC ${PRIMARY_MODEL_DEF})
endif()
if (LLAMA3_SUPPORT OR QWEN2_SUPPORT OR QWEN3_SUPPORT)
  enable_json_regex_support(llama)
endif()

if (LLAMA3_SUPPORT)
  add_library(llama_llama3 SHARED ${DIR_TENSOR} ${DIR_BASE} ${DIR_OP} ${DIR_KERNEL} ${DIR_MODEL} ${DIR_KERNEL_CPU} ${DIR_KERNEL_CUDA} ${DIR_KERNEL} ${DIR_SAMPLE})
  configure_llama_target(llama_llama3)
  target_compile_definitions(llama_llama3 PUBLIC LLAMA3_SUPPORT)
  enable_json_regex_support(llama_llama3)
endif()

if (QWEN2_SUPPORT)
  add_library(llama_qwen2 SHARED ${DIR_TENSOR} ${DIR_BASE} ${DIR_OP} ${DIR_KERNEL} ${DIR_MODEL} ${DIR_KERNEL_CPU} ${DIR_KERNEL_CUDA} ${DIR_KERNEL} ${DIR_SAMPLE})
  configure_llama_target(llama_qwen2)
  target_compile_definitions(llama_qwen2 PUBLIC QWEN2_SUPPORT)
  enable_json_regex_support(llama_qwen2)
endif()

if (QWEN3_SUPPORT)
  add_library(llama_qwen3 SHARED ${DIR_TENSOR} ${DIR_BASE} ${DIR_OP} ${DIR_KERNEL} ${DIR_MODEL} ${DIR_KERNEL_CPU} ${DIR_KERNEL_CUDA} ${DIR_KERNEL} ${DIR_SAMPLE})
  configure_llama_target(llama_qwen3)
  target_compile_definitions(llama_qwen3 PUBLIC QWEN3_SUPPORT)
  enable_json_regex_support(llama_qwen3)
endif()
add_subdirectory(test)
add_subdirectory(demo)
