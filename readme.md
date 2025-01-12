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