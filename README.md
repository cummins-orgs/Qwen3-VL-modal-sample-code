# Qwen3-VL Visual QA on Modal

A serverless Visual Question Answering (VQA) service using [Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) deployed on [Modal](https://modal.com). This service can answer questions about images through a simple HTTP API.

## What is this?

Vision-Language Models (VLMs) are like LLMs with eyes - they can generate text based not just on other text, but on images as well. This project deploys the Qwen3-VL model as a serverless API endpoint that:

- Accepts an image URL and a question
- Returns an AI-generated answer about the image
- Scales automatically from zero to handle any load
- Runs on GPU infrastructure without you managing servers

## Prerequisites

1. **Python 3.11+** - This project requires Python 3.11 or later
2. **Modal account** - Sign up for free at [modal.com](https://modal.com)
3. **Modal token** - After signing up, authenticate with:
   ```bash
   modal token new
   ```

## Installation

This project uses [uv](https://docs.astral.sh/uv/) for fast, reliable Python package management.

### 1. Install uv (if you don't have it)

```bash
# On macOS and Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 2. Clone and setup

```bash
git clone <your-repo-url>
cd modal_qwen
```

### 3. Create virtual environment and install dependencies

```bash
# Create a virtual environment
uv venv

# Activate it
source .venv/bin/activate  # On macOS/Linux
# or
.venv\Scripts\activate     # On Windows

# Install dependencies (just modal CLI)
uv pip install -e .
```

Note: The heavy dependencies (PyTorch, Transformers, etc.) are NOT installed locally. They're installed in the Modal container image automatically when you deploy.

## Usage

### Run the service locally (one-off execution)

This starts the service, sends a test request, and shuts it down:

```bash
modal run modal_app.py
```

You can customize the test query:

```bash
modal run modal_app.py --image-url "https://example.com/your-image.jpg" --question "What do you see?"
```

### Deploy as a persistent service

Deploy the service to run 24/7 on Modal's infrastructure:

```bash
modal deploy modal_app.py
```

After deployment, Modal will provide you with a web URL (something like `https://your-username--example-qwen3-vlm-model-generate.modal.run`). You can send POST requests to this endpoint:

```bash
curl -X POST "https://your-username--example-qwen3-vlm-model-generate.modal.run" \
  -H "Content-Type: application/json" \
  -d '{
    "image_url": "https://modal-public-assets.s3.amazonaws.com/golden-gate-bridge.jpg",
    "question": "What landmark is this?"
  }'
```

Interactive API documentation is available at the `/docs` route of your endpoint URL.

## Configuration

### GPU Configuration

You can customize the GPU used via environment variables:

```bash
# Use a different GPU type
GPU_TYPE=a100-80gb modal deploy modal_app.py

# Use multiple GPUs
GPU_COUNT=2 GPU_TYPE=h100 modal deploy modal_app.py
```

Available GPU types: `l40s`, `a10g`, `a100-40gb`, `a100-80gb`, `h100`

### Model Configuration

The model is defined in `modal_app.py`. To use a different Qwen3-VL variant, modify:

```python
MODEL_PATH = "Qwen/Qwen3-VL-4B-Instruct"  # Change to your preferred model
MODEL_REVISION = "d1883f3364402876f33f2ce7cfafe737d1e326c7"  # Update revision
```

## How it works

1. **Local setup**: You only need the Modal CLI installed locally
2. **Container image**: Modal builds a container with PyTorch, Transformers, and all required dependencies
3. **Model caching**: The Qwen3-VL model is downloaded once and cached in a Modal Volume
4. **Serverless execution**: The service scales from 0 to N replicas based on demand
5. **GPU acceleration**: Inference runs on NVIDIA GPUs for fast response times

## Architecture

- `modal_app.py` - Main application file defining the Modal service
- `pyproject.toml` - Project metadata and local dependencies (just Modal CLI)
- `LICENSE` - Apache 2.0 license

## Cost

Modal charges based on:
- GPU time (per second of inference)
- Container runtime (idle containers after `scaledown_window`)

The service automatically scales to zero when not in use, so you only pay for actual usage.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Model: [Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) by Alibaba
- Infrastructure: [Modal](https://modal.com)
- Based on Modal's VLM example
