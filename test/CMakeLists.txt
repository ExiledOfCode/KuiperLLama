# 文件说明：测试构建脚本，组织张量、算子、CUDA 和模型加载相关单元测试。

include(../cmake/cuda.cmake)
find_package(GTest REQUIRED)
find_package(glog REQUIRED)

set(link_ext_lib glog::glog GTest::gtest)
set(DIR_TEST
  ../test/test_main.cpp
  ../test/utils.cu
)

aux_source_directory(../test/test_cu DIR_TEST_CU)
aux_source_directory(../test/test_op DIR_TEST_OP)
aux_source_directory(../test/test_model DIR_TEST_MODEL)
aux_source_directory(../test/test_tensor DIR_TEST_TENSOR)
aux_source_directory(../test/optimized DIR_TEST_OPTIMIZED)

add_executable(test_llm ${DIR_TEST} ${DIR_TEST_CU} ${DIR_TEST_OP} ${DIR_TEST_OPTIMIZED} ${DIR_TEST_TENSOR} ${DIR_TEST_MODEL})

#set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g -G")
target_link_libraries(test_llm ${link_ext_lib})
target_include_directories(test_llm PUBLIC ${glog_INCLUDE_DIR})
target_include_directories(test_llm PUBLIC ${GTest_INCLUDE_DIR})

target_include_directories(test_llm PUBLIC ../src/include)
target_link_directories(test_llm PUBLIC ${PROJECT_SOURCE_DIR}/lib)
target_link_libraries(test_llm llama)

set_target_properties(test_llm PROPERTIES WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
set_target_properties(test_llm PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(bench_operator_micro bench_operator_micro.cpp)
target_link_libraries(bench_operator_micro glog::glog pthread cudart cublas)
target_include_directories(bench_operator_micro PUBLIC ${glog_INCLUDE_DIR})
target_include_directories(bench_operator_micro PUBLIC ../src/include)
target_link_directories(bench_operator_micro PUBLIC ${PROJECT_SOURCE_DIR}/lib)
target_link_libraries(bench_operator_micro llama)
set_target_properties(bench_operator_micro PROPERTIES WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
set_target_properties(bench_operator_micro PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
