from huggingface_hub import snapshot_download

model_id = "meta-llama/Llama-3.1-8B-Instruct"
snapshot_download(
    repo_id=model_id,
    local_dir="Llama-3.1-8B-Instruct",
    local_dir_use_symlinks=False,
    revision="main",
)
