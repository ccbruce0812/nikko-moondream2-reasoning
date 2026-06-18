#!/usr/bin/env python3
"""
Moondream2 HF inference with performance benchmarking.
Usage: python3 moondream2.py <image_path> <prompt>
"""
import sys
import time
import warnings
import logging

# ---- Suppress known-harmless noise ----
warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub")
warnings.filterwarnings("ignore", message=".*Failed to load image Python extension.*")
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

MODEL_ID = "vikhyatk/moondream2"
REVISION = "2024-04-02"

# ---- CLI args ----
if len(sys.argv) < 3:
    print("Usage: python3 moondream2.py <image_path> <prompt>")
    print("Example: python3 moondream2.py test.jpg 'Describe this image'")
    sys.exit(1)

image_path = sys.argv[1]
prompt = sys.argv[2]

# ---- Load model ----
print(f"Loading {MODEL_ID} (revision={REVISION}) ...", flush=True)
t0 = time.time()

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    device_map="cuda",
    torch_dtype=torch.float16,
    revision=REVISION,
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)

t_load = time.time() - t0
print(f"Model loaded in {t_load:.1f}s. Device: {model.device}", flush=True)

# ---- Encode image ----
image = Image.open(image_path)
print(f"Encoding image ({image.size}) ...", flush=True)
t0 = time.time()

with torch.no_grad():
    image_embeds = model.encode_image(image)

t_encode = time.time() - t0
print(f"Image encoded in {t_encode:.1f}s", flush=True)

# ---- Benchmark inference ----
print(f"Prompt: {prompt}")
print("Answer: ", end="", flush=True)

formatted_prompt = f"<image>\n\nQuestion: {prompt}\n\nAnswer:"
inputs_embeds = model.input_embeds(formatted_prompt, image_embeds, tokenizer)

# Track per-token timing
token_times = []
first_token_time = [None]  # list for mutability in closure

class TimedStreamer(TextStreamer):
    """Streamer that records per-token latency."""
    def on_finalized_text(self, text: str, stream_end: bool = False):
        now = time.time()
        token_times.append(now)
        if first_token_time[0] is None:
            first_token_time[0] = now
        print(text, end="", flush=True)

streamer = TimedStreamer(tokenizer, skip_prompt=True)

total_start = time.time()

with torch.no_grad():
    output_ids = model.text_model.generate(
        inputs_embeds=inputs_embeds,
        max_new_tokens=512,
        eos_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        pad_token_id=tokenizer.bos_token_id,
        streamer=streamer,
    )

total_time = time.time() - total_start

# Decode for accurate token count (streamer may batch tokens)
output_text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
output_tokens = tokenizer.encode(output_text)
num_tokens = len(output_tokens)

# ---- Report ----
print("\n")
print("=" * 50)
print("Performance Report")
print("=" * 50)
print(f"  Model load:        {t_load:6.1f}s")
print(f"  Image encode:      {t_encode:6.1f}s")
if first_token_time[0]:
    ftl = first_token_time[0] - total_start
    print(f"  First token (TTFT): {ftl:6.1f}s")
print(f"  Total inference:   {total_time:6.1f}s")
print(f"  Output tokens:     {num_tokens:6d}")
if num_tokens > 0 and total_time > 0:
    print(f"  Tokens/second:     {num_tokens/total_time:6.1f} tok/s")
print(f"  GPU memory:        {torch.cuda.max_memory_allocated()/1e9:.2f} GB peak")
print("=" * 50)
