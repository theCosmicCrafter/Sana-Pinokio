import torch
import gradio as gr
from diffusers import SanaPipeline
import os

# Initialize pipeline
pipe = None

def load_model():
    """Load the Sana pipeline model"""
    global pipe
    if pipe is None:
        print("Loading Sana model...")
        pipe = SanaPipeline.from_pretrained(
            "Efficient-Large-Model/SANA1.5_1.6B_1024px_diffusers",
            torch_dtype=torch.bfloat16,
        )
        pipe.to("cuda")
        # Convert VAE and text_encoder to bfloat16 for better performance
        pipe.vae.to(torch.bfloat16)
        pipe.text_encoder.to(torch.bfloat16)
        print("Model loaded successfully!")
    return pipe

def generate_image(prompt, steps=20, guidance=4.5, seed=42, width=1024, height=1024):
    """Generate an image from a text prompt"""
    try:
        if pipe is None:
            load_model()
        
        generator = torch.Generator(device="cuda").manual_seed(int(seed))
        result = pipe(
            prompt=prompt,
            height=height,
            width=width,
            guidance_scale=guidance,
            num_inference_steps=steps,
            generator=generator,
        )
        # Handle both return formats: [images] or .images
        if isinstance(result, list):
            image = result[0][0] if isinstance(result[0], list) else result[0]
        else:
            image = result.images[0]
        
        return image
    except Exception as e:
        return f"Error generating image: {str(e)}"

# Load model on startup
print("Initializing Sana...")
load_model()

# Create Gradio interface
with gr.Blocks(title="Sana Image Generator", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🎨 Sana Image Generator
        Fast image generation using the Sana diffusion model.
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            prompt_input = gr.Textbox(
                label="Prompt",
                placeholder="a futuristic city at sunset, cyberpunk style",
                lines=3,
            )
            
            with gr.Row():
                width_slider = gr.Slider(
                    minimum=512,
                    maximum=2048,
                    value=1024,
                    step=64,
                    label="Width",
                )
                height_slider = gr.Slider(
                    minimum=512,
                    maximum=2048,
                    value=1024,
                    step=64,
                    label="Height",
                )
            
            steps_slider = gr.Slider(
                minimum=10,
                maximum=50,
                value=20,
                step=1,
                label="Inference Steps",
            )
            
            guidance_slider = gr.Slider(
                minimum=1.0,
                maximum=10.0,
                value=4.5,
                step=0.5,
                label="Guidance Scale",
            )
            
            seed_input = gr.Number(
                value=42,
                label="Seed",
                precision=0,
            )
            
            generate_btn = gr.Button("Generate", variant="primary", size="lg")
        
        with gr.Column(scale=1):
            output_image = gr.Image(
                label="Generated Image",
                type="pil",
            )
    
    gr.Markdown(
        """
        ### Tips:
        - Higher inference steps = better quality but slower generation
        - Guidance scale controls how closely the image follows the prompt
        - Change the seed for different variations
        """
    )
    
    generate_btn.click(
        fn=generate_image,
        inputs=[prompt_input, steps_slider, guidance_slider, seed_input, width_slider, height_slider],
        outputs=output_image,
    )
    
    # Example prompts
    gr.Examples(
        examples=[
            ["a futuristic city at sunset, cyberpunk style"],
            ["a serene mountain landscape with a lake, photorealistic"],
            ["a cute robot reading a book, digital art"],
            ["an abstract painting with vibrant colors"],
            ["a steampunk airship flying through clouds"],
        ],
        inputs=prompt_input,
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)

