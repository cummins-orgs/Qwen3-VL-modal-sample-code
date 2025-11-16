# Project Cleanup and uv Configuration

## Summary
Cleaned up the modal_qwen project to focus exclusively on `sgl_vlm.py` and configured it to work with `uv` package manager.

## Changes Made

### 1. Project Structure Simplified
- **Moved** `sgl_vlm.py` from `06_gpu_and_ml/llm-serving/` to project root
- **Removed** all unnecessary directories and files:
  - `internal/` - test and deployment scripts
  - `misc/` - various example scripts not related to sgl_vlm
  - `06_gpu_and_ml/` - other ML scripts (import_torch, torch_profiling, unsloth_finetune, vllm_inference, openai_compatible)
  - All `.DS_Store` files
  - `__pycache__` directories

### 2. pyproject.toml Rewritten
- **Before**: Only contained tool configurations (pytest, mypy, ruff)
- **After**: Full project configuration with dependencies
  - Added project metadata (name, version, description)
  - Defined all dependencies required by `sgl_vlm.py`:
    - modal (Modal platform SDK)
    - transformers 4.57.1 (latest, upgraded from 4.54.1)
    - torch 2.9.1
    - fastapi, pydantic, starlette (API framework)
    - accelerate (for device_map="auto")
    - pillow (image processing)
    - term-image (terminal image rendering)
    - huggingface-hub (model downloads)
    - And other required dependencies
  - Configured hatchling build system
  - Set minimum Python version to 3.11

### 3. uv Environment Setup
- Successfully ran `uv sync`
- Created `.venv` with Python 3.13.3
- Installed 82 packages including all dependencies
- Generated `uv.lock` for reproducible builds

## Verification
All key dependencies verified installed:
- ✅ transformers 4.57.1 (latest)
- ✅ torch 2.9.1
- ✅ modal 1.2.2
- ✅ fastapi 0.121.2
- ✅ pillow 10.4.0
- ✅ accelerate 1.11.0

## Files Remaining
- `sgl_vlm.py` - Main VLM inference script
- `pyproject.toml` - Project configuration
- `README.md` - Project documentation
- `LICENSE` - License file
- `.venv/` - Virtual environment (gitignored)
- `uv.lock` - Dependency lock file
