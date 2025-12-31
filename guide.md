Right, for local Windows installation with full control:

```bash
# 1. Clone repo
git clone https://github.com/NVlabs/Sana.git
cd Sana

# 2. Create conda env
conda create -n sana python=3.10
conda activate sana

# 3. Install PyTorch with CUDA
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# 4. Install dependencies
pip install diffusers transformers accelerate peft einops omegaconf gradio pillow

# 5. Download a model (this happens automatically, but you can pre-download)
huggingface-cli download Efficient-Large-Model/Sana_1600M_1024px_BF16
```

## Quick Local Inference Script

Create `run_sana.py`:

```python
import torch
from diffusers import SanaPipeline

# Load model locally
pipe = SanaPipeline.from_pretrained(
    "Efficient-Large-Model/SANA1.5_1.6B_1024px_diffusers",
    torch_dtype=torch.bfloat16,
)
pipe.to("cuda")

# Generate
prompt = "a futuristic city at sunset, cyberpunk style"
image = pipe(
    prompt=prompt,
    height=1024,
    width=1024,
    guidance_scale=4.5,
    num_inference_steps=20,
    generator=torch.Generator(device="cuda").manual_seed(42),
).images[0]

image.save("output.png")
print("Saved to output.png")
```

Run it:
```bash
python run_sana.py
```

## Local Web UI (Gradio)

Create `sana_ui.py`:

```python
import torch
import gradio as gr
from diffusers import SanaPipeline

pipe = SanaPipeline.from_pretrained(
    "Efficient-Large-Model/SANA1.5_1.6B_1024px_diffusers",
    torch_dtype=torch.bfloat16,
)
pipe.to("cuda")

def generate(prompt, steps=20, guidance=4.5, seed=42):
    generator = torch.Generator(device="cuda").manual_seed(int(seed))
    image = pipe(
        prompt=prompt,
        height=1024,
        width=1024,
        guidance_scale=guidance,
        num_inference_steps=steps,
        generator=generator,
    ).images[0]
    return image

demo = gr.Interface(
    fn=generate,
    inputs=[
        gr.Textbox(label="Prompt"),
        gr.Slider(minimum=10, maximum=30, value=20, step=1, label="Steps"),
        gr.Slider(minimum=1, maximum=10, value=4.5, step=0.5, label="Guidance Scale"),
        gr.Number(value=42, label="Seed"),
    ],
    outputs="image",
    title="Sana Local",
)

demo.launch(server_name="127.0.0.1", server_port=7860)
```

Run it:
```bash
python sana_ui.py
```

Then open `http://127.0.0.1:7860`

## Storage Note

Models cache in `~/.cache/huggingface/hub/`. For your 4090, the 1.6B model (BF16) is perfect—generates 1024×1024 in ~1-2 seconds.