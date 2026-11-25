# Ollama startup diagnosis

## Environment

- **platform**: Windows 11 AMD64
- **ollama_version**: <unavailable: FileNotFoundError>
- **cpu**: 64
- **ram_gb**: 15.9
- **nvidia_smi**: NVIDIA GeForce GTX 950, 2048 MiB, 581.29
- **disk_free_gb**: 198.4
- **disk_note**: models dir missing (measured home volume) - nothing pulled yet
- **log_path**: sample_logs\cuda_oom.log
- **port 11434**: closed

## Findings

### 1. [CRITICAL] GPU offload request exceeds available VRAM  (`CUDA_OOM`)

- Evidence: `time=2026-11-25T09:20:11.004+08:00 level=ERROR msg="cudaMalloc failed: out of memory"`
- What it means: The default num_ctx plus the requested layers do not fit. Lower context first (it is cheap), then cap layers. Do not assume the install is broken.
- Verify with: `OLLAMA_NUM_PARALLEL=1 ollama run <model> --verbose   # watch 'offloaded N/M layers'`

## Do not do this

Reinstalling before reading the log destroys the one piece of evidence that identifies the cause. Peeling the fix in the order above costs minutes; a blind reinstall costs hours and usually reproduces the same failure.
