```
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

**Needs Desktop development with C++ workload installed in Visual Studio**

```
set CMAKE_ARGS=-DGGML_CUDA=ON
pip install llama-cpp-python --force-reinstall --no-cache-dir
```

**Linux**
```
sudo apt install nvidia-driver-565
curl -fSsL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-drivers.gpg
sudo apt install cuda-toolkit-12-6
```

```
export PATH=/usr/local/cuda-12.6/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH
export CMAKE_ARGS=-DGGML_CUDA=ON
pip install llama-cpp-python --force-reinstall --no-cache-dir
```

**Models tested:**

https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF


**convert HuggingFace model to GGUF format**

download the model using:
```
from huggingface_hub import snapshot_download
model_id="repo/model"
snapshot_download(repo_id=model_id, local_dir="model_name",
                  local_dir_use_symlinks=False, revision="main")
```

```
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
pip install -r requirements.txt
```

```
python convert_hf_to_gguf model_folder --outfile model_name.gguf --outtype f16
```


**quantize fp16.gguf to 6bit**
```
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
mkdir build
cd build
cmake .. -G "Ninja" -DCMAKE_BUILD_TYPE=Release
cmake --build .
cd bin
./llama-quantize 3.1-8B.fp16.gguf 3.1-8B.q6_K.gguf Q6_K
```
