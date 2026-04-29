// 文件说明：CUDA 测试辅助声明，为各算子测试共享检查工具。

#ifndef TEST_CU_CUH
#define TEST_CU_CUH
void test_function(float* arr, int32_t size, float value = 1.f);

void set_value_cu(float* arr_cu, int32_t size, float value = 1.f);
#endif  // TEST_CU_CUH
