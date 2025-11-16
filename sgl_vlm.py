# # Run Qwen3-VL with Transformers for Visual QA

# Vision-Language Models (VLMs) are like LLMs with eyes:
# they can generate text based not just on other text,
# but on images as well.

# This example shows how to run the Qwen3-VL model on Modal using the
# [Transformers](https://github.com/huggingface/transformers) library.

# Here's a sample inference, with the image rendered directly (and at low resolution) in the terminal:

# ![Sample output answering a question about a photo of the Statue of Liberty](https://modal-public-assets.s3.amazonaws.com/sgl_vlm_qa_sol.png)

# ## Setup

# First, we'll import the libraries we need locally
# and define some constants.

import os
import time
import warnings
from pathlib import Path
from typing import Optional
from uuid import uuid4

import modal

# VLMs are generally larger than LLMs with the same cognitive capability.
# LLMs are already hard to run effectively on CPUs, so we'll use a GPU here.
# We find that inference for a single input takes about 3-4 seconds on an A10G.

# You can customize the GPU type and count using the `GPU_TYPE` and `GPU_COUNT` environment variables.
# If you want to see the model really rip, try an `"a100-80gb"` or an `"h100"`
# on a large batch.

GPU_TYPE = os.environ.get("GPU_TYPE", "l40s")
GPU_COUNT = os.environ.get("GPU_COUNT", 1)

GPU_CONFIG = f"{GPU_TYPE}:{GPU_COUNT}"

MINUTES = 60  # seconds

# We use the [Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct)
# model by Alibaba, the latest version of their VLM series.

MODEL_PATH = "Qwen/Qwen3-VL-4B-Instruct"
MODEL_REVISION = "d1883f3364402876f33f2ce7cfafe737d1e326c7"
TOKENIZER_PATH = "Qwen/Qwen3-VL-4B-Instruct"

# We download it from the Hugging Face Hub using the Python function below.
# We'll store it in a [Modal Volume](https://modal.com/docs/guide/volumes)
# so that it's not downloaded every time the container starts.

MODEL_VOL_PATH = Path("/models")
MODEL_VOL = modal.Volume.from_name("sgl-cache", create_if_missing=True)
volumes = {MODEL_VOL_PATH: MODEL_VOL}


def download_model():
    from huggingface_hub import snapshot_download

    snapshot_download(
        MODEL_PATH,
        local_dir=str(MODEL_VOL_PATH / MODEL_PATH),
        revision=MODEL_REVISION,
        ignore_patterns=["*.pt", "*.bin"],
    )


# Modal runs Python functions on containers in the cloud.
# The environment those functions run in is defined by the container's `Image`.
# The block of code below defines our example's `Image`.
cuda_version = "12.8.0"  # should be no greater than host CUDA version
flavor = "devel"  #  includes full CUDA toolkit
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

vlm_image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.11")
    .entrypoint([])  # removes chatty prints on entry
    .uv_pip_install(  # add transformers and Python dependencies
        "transformers==4.54.1",
        "numpy<2",
        "fastapi[standard]==0.115.4",
        "pydantic==2.9.2",
        "requests==2.32.3",
        "starlette==0.41.2",
        "torch==2.7.1",
        "hf-xet==1.1.5",
        "accelerate",  # needed for device_map="auto"
        "pillow",  # needed for image processing
        pre=True,
    )
    .env(
        {
            "HF_HOME": str(MODEL_VOL_PATH),
            "HF_XET_HIGH_PERFORMANCE": "1",
        }
    )
    .run_function(  # download the model by running a Python function
        download_model, volumes=volumes
    )
    .uv_pip_install(  # add an optional extra that renders images in the terminal
        "term-image==0.7.1"
    )
)

# ## Defining a Visual QA service

# Running an inference service on Modal is as easy as writing inference in Python.

# The code below adds a modal `Cls` to an `App` that runs the VLM.

# We define a method `generate` that takes a URL for an image and a question
# about the image as inputs and returns the VLM's answer.

# By decorating it with `@modal.fastapi_endpoint`, we expose it as an HTTP endpoint,
# so it can be accessed over the public Internet from any client.

app = modal.App("example-qwen3-vlm")


@app.cls(
    gpu=GPU_CONFIG,
    timeout=20 * MINUTES,
    scaledown_window=20 * MINUTES,
    image=vlm_image,
    volumes=volumes,
)
@modal.concurrent(max_inputs=100)
class Model:
    @modal.enter()  # what should a container do after it starts but before it gets input?
    def start_runtime(self):
        """Loads the Qwen3 VLM model using transformers."""
        from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

        # Load the model with automatic device mapping
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            MODEL_PATH,
            dtype="auto",
            device_map="auto"
        )

        # Load the processor for handling images and text
        self.processor = AutoProcessor.from_pretrained(TOKENIZER_PATH)

    @modal.fastapi_endpoint(method="POST", docs=True)
    def generate(self, request: dict) -> str:
        from pathlib import Path

        import requests
        from term_image.image import from_file

        start = time.monotonic_ns()
        request_id = uuid4()
        print(f"Generating response to request {request_id}")

        image_url = request.get("image_url")
        if image_url is None:
            image_url = (
                "https://modal-public-assets.s3.amazonaws.com/golden-gate-bridge.jpg"
            )

        response = requests.get(image_url)
        response.raise_for_status()

        image_filename = image_url.split("/")[-1]
        image_path = Path(f"/tmp/{uuid4()}-{image_filename}")
        image_path.write_bytes(response.content)

        question = request.get("question")
        if question is None:
            question = "What is this?"

        # Prepare messages in the format expected by Qwen3-VL
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": str(image_path),
                    },
                    {"type": "text", "text": question},
                ],
            }
        ]

        # Prepare inputs using the processor
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )
        inputs = inputs.to(self.model.device)

        # Generate response
        generated_ids = self.model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        # show the question and image in the terminal for demonstration purposes
        print(Colors.BOLD, Colors.GRAY, "Question: ", question, Colors.END, sep="")
        terminal_image = from_file(image_path)
        terminal_image.draw()
        print(
            f"request {request_id} completed in {round((time.monotonic_ns() - start) / 1e9, 2)} seconds"
        )

        return output_text[0]

    @modal.exit()  # what should a container do before it shuts down?
    def shutdown_runtime(self):
        # Clean up resources if needed
        del self.model
        del self.processor


# ## Asking questions about images via POST

# Now, we can send this Modal Function a POST request with an image and a question
# and get back an answer.

# The code below will start up the inference service
# so that it can be run from the terminal as a one-off,
# like a local script would be, using `modal run`:

# ```bash
# modal run sgl_vlm.py
# ```

# By default, we hit the endpoint twice to demonstrate how much faster
# the inference is once the server is running.


@app.local_entrypoint()
def main(
    image_url: Optional[str] = None, question: Optional[str] = None, twice: bool = True
):
    import json
    import urllib.request

    model = Model()

    payload = json.dumps(
        {
            "image_url": image_url,
            "question": question,
        },
    )

    req = urllib.request.Request(
        model.generate.get_web_url(),
        data=payload.encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(req) as response:
        assert response.getcode() == 200, response.getcode()
        print(json.loads(response.read().decode()))

    if twice:
        # second response is faster, because the Function is already running
        with urllib.request.urlopen(req) as response:
            assert response.getcode() == 200, response.getcode()
            print(json.loads(response.read().decode()))


# ## Deployment

# To set this up as a long-running, but serverless, service, we can deploy it to Modal:

# ```bash
# modal deploy sgl_vlm.py
# ```

# And then send requests from anywhere. See the [docs](https://modal.com/docs/guide/webhook-urls)
# for details on the `web_url` of the function, which also appears in the terminal output
# when running `modal deploy`.

# You can also find interactive documentation for the endpoint at the `/docs` route of the web endpoint URL.

# ## Addenda

# The rest of the code in this example is just utility code.

warnings.filterwarnings(  # filter warning from the terminal image library
    "ignore",
    message="It seems this process is not running within a terminal. Hence, some features will behave differently or be disabled.",
    category=UserWarning,
)


class Colors:
    """ANSI color codes"""

    GREEN = "\033[0;32m"
    BLUE = "\033[0;34m"
    GRAY = "\033[0;90m"
    BOLD = "\033[1m"
    END = "\033[0m"
