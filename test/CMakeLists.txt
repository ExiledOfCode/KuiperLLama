include(../cmake/cuda.cmake)
find_package(GTest REQUIRED)
find_package(glog REQUIRED)

set(link_ext_lib glog::glog GTest::gtest)
aux_source_directory(../test DIR_TEST)

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

target_include_directories(test_llm PUBLIC ../kuiper/include)
target_link_directories(test_llm PUBLIC ${PROJECT_SOURCE_DIR}/lib)
if (LLAMA3_SUPPORT OR QWEN2_SUPPORT OR QWEN3_SUPPORT)
    message(STATUS "LINK LLM JSON/REGEX SUPPORT")
    find_package(absl REQUIRED)
    find_package(nlohmann_json REQUIRED)

    # Some distro packages (e.g. libre2-dev) do not ship re2Config.cmake.
    # Prefer CMake package, then fallback to pkg-config.
    if (NOT TARGET re2::re2)
        find_package(re2 CONFIG QUIET)
    endif ()
    if (NOT TARGET re2::re2)
        find_package(PkgConfig REQUIRED)
        pkg_check_modules(RE2 REQUIRED re2)
        add_library(re2::re2 INTERFACE IMPORTED)
        target_include_directories(re2::re2 INTERFACE ${RE2_INCLUDE_DIRS})
        target_link_directories(re2::re2 INTERFACE ${RE2_LIBRARY_DIRS})
        target_link_libraries(re2::re2 INTERFACE ${RE2_LIBRARIES})
        target_compile_options(re2::re2 INTERFACE ${RE2_CFLAGS_OTHER})
    endif ()

    target_link_libraries(llama absl::base absl::strings re2::re2 nlohmann_json::nlohmann_json)
endif ()
target_link_libraries(test_llm llama)

set_target_properties(test_llm PROPERTIES WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
set_target_properties(test_llm PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
