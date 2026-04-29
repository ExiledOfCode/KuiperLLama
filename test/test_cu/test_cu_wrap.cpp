// 文件说明：单元测试文件，验证 test_cu_wrap 相关模块的正确性。

#include <glog/logging.h>
#include <gtest/gtest.h>
#include <armadillo>
#include "../utils.cuh"

TEST(test_cu, test_function) {
  int32_t size = 32;
  float* ptr = new float[size];
  test_function(ptr, size);
  for (int32_t i = 0; i < size; ++i) {
    ASSERT_EQ(ptr[i], 1.f);
  }
  delete[] ptr;
}
