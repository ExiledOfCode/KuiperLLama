## 第三方依赖
1. google glog https://github.com/google/glog
2. google gtest https://github.com/google/googletest
3. sentencepiece https://github.com/google/sentencepiece
4. armadillo + openblas https://arma.sourceforge.net/download.html
5. Cuda Toolkit


## 模型下载地址




## 模型导出
```shell
python export.py llama2_7b.bin --meta-llama path/to/llama/model/7B
# 使用--hf标签从hugging face中加载模型， 指定--version3可以导出量化模型
# 其他使用方法请看export.py中的命令行参数实例
```


## 编译方法
```shell
  mkdir build 
  cd build
  # 需要安装上述的第三方依赖
  cmake ..
  # 或者开启 USE_CPM 选项，自动下载第三方依赖
  cmake -DUSE_CPM=ON ..
  make -j16
```

## 生成文本的方法
```shell
./llama_infer llama2_7b.bin tokenizer.model

```

# LLama3.2 推理

- 以 meta-llama/Llama-3.2-1B 为例，huggingface 上下载模型：
```shell
export HF_ENDPOINT=https://hf-mirror.com
pip3 install huggingface-cli
huggingface-cli download --resume-download meta-llama/Llama-3.2-1B --local-dir meta-llama/Llama-3.2-1B --local-dir-use-symlinks False
```
- 导出模型：
```shell
python3 tools/export.py Llama-3.2-1B.bin --hf=meta-llama/Llama-3.2-1B
```
- 编译：
```shell
mkdir build 
cd build
# 开启 USE_CPM 选项，自动下载第三方依赖，前提是需要网络畅通
cmake -DUSE_CPM=ON -DLLAMA3_SUPPORT=ON .. 
make -j16
```
- 运行：
```shell
./build/demo/llama_infer models/llama/stories110M.bin models/llama/tokenizer.model
# 和 huggingface 推理的结果进行对比
python3 hf_infer/llama3_infer.py
```

# Qwen2.5 推理

- 以 Qwen2.5-0.5B 为例，huggingface 上下载模型：
```shell
export HF_ENDPOINT=https://hf-mirror.com
pip3 install -U huggingface_hub
huggingface-cli download --resume-download Qwen/Qwen2.5-0.5B --local-dir Qwen/Qwen2.5-0.5B --local-dir-use-symlinks False

hf download Qwen/Qwen2.5-1.5B-Instruct --local-dir Qwen2.5-1.5B-Instruct
```
- 导出模型：
```shell
python3 tools/export_qwen2.py Qwen2.5-1.5B-Instruct/Qwen2.5-1.5B-Instruct.bin --hf=Qwen/Qwen2.5-1.5B-Instruct
```
- 编译：
```shell
mkdir build 
cd build
# 开启 USE_CPM 选项，自动下载第三方依赖，前提是需要网络畅通
cmake -DQWEN2_SUPPORT=ON -DQWEN3_SUPPORT=ON -DLLAMA3_SUPPORT=ON..
make -j$(nproc)


cmake -S W_InferEngine -B W_InferEngine/build -DUSE_CPM=OFF -DQWEN3_SUPPORT=ON -DQWEN2_SUPPORT=ON -DLLAMA3_SUPPORT=ON
cmake --build W_InferEngine/build -j$(nproc)

```
- 运行：
```shell
./build/demo/qwen_infer Qwen2.5-0.5B.bin Qwen/Qwen2.5-0.5B/tokenizer.json
# 和 huggingface 推理的结果进行对比
python3 hf_infer/qwen2_infer.py
```

## Qwen3推理

```shell
export HF_ENDPOINT=https://hf-mirror.com
pip3 install -U huggingface_hub

hf download Qwen/Qwen3-1.7B --local-dir Qwen3-1.7B
```
- 导出模型：
```shell
python3 tools/export_qwen3.py --model-dir Qwen3-1.7B --output Qwen3-1.7B/Qwen3-1.7B.bin --dtype fp32
python3 tools/export_qwen3.py --model-dir Qwen3-1.7B --output Qwen3-1.7B/Qwen3-1.7B-bf16.bin --dtype bf16
```
