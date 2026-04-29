// 文件说明：单元测试文件，验证 test_main 相关模块的正确性。

#include <glog/logging.h>
#include <gtest/gtest.h>

int main(int argc, char* argv[]) {
  testing::InitGoogleTest(&argc, argv);
  google::InitGoogleLogging("Kuiper");
  FLAGS_log_dir = "./log/";
  FLAGS_alsologtostderr = true;

  LOG(INFO) << "Start Test...\n";
  return RUN_ALL_TESTS();
}