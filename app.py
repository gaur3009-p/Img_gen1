import gradio as gr
import numpy as np
import random
from diffusers import DiffusionPipeline
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

if torch.cuda.is_available():
    torch.cuda.max_memory_allocated(device=device)
    pipe = DiffusionPipeline.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16", use_safetensors=True)
    pipe.enable_xformers_memory_efficient_attention()
    pipe = pipe.to(device)
else: 
    pipe = DiffusionPipeline.from_pretrained("stabilityai/sdxl-turbo", use_safetensors=True)
    pipe = pipe.to(device)

MAX_SEED = np.iinfo(np.int32).max
MAX_IMAGE_SIZE = 1024

def infer(prompt_part1, color, dress_type, front_design, back_design, prompt_part5, negative_prompt, seed, randomize_seed, width, height, guidance_scale, num_inference_steps):
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    
    generator = torch.Generator().manual_seed(seed)
    
    front_prompt = f"front view of {prompt_part1} {color} colored plain {dress_type} with {front_design} design, {prompt_part5}"
    front_image = pipe(
        prompt=front_prompt, 
        negative_prompt=negative_prompt,
        guidance_scale=guidance_scale, 
        num_inference_steps=num_inference_steps, 
        width=width, 
        height=height,
        generator=generator
    ).images[0]
    
    back_prompt = f"back view of {prompt_part1} {color} colored plain {dress_type} with {back_design} design, {prompt_part5}"
    back_image = pipe(
        prompt=back_prompt, 
        negative_prompt=negative_prompt,
        guidance_scale=guidance_scale, 
        num_inference_steps=num_inference_steps, 
        width=width, 
        height=height,
        generator=generator
    ).images[0]
    
    return front_image, back_image, seed

examples = [
    ["red", "t-shirt", "yellow stripes", "polka dots"],
    ["blue", "hoodie", "minimalist", "abstract art"],
    ["red", "sweat shirt", "geometric design", "plain"],
]

css = """
#col-container {
    margin: 0 auto;
    max-width: 520px;
}
"""

if torch.cuda.is_available():
    power_device = "GPU"
else:
    power_device = "CPU"

with gr.Blocks(css=css) as demo:
    
    with gr.Column(elem_id="col-container"):
        gr.Markdown(f"""
        # Text-to-Image Gradio Template
        Currently running on {power_device}.
        """)
        
        with gr.Row():
            
            prompt_part1 = gr.Textbox(
                value="a single", 
                label="Prompt Part 1",
                show_label=False,
                interactive=False,
                container=False,
                elem_id="prompt_part1",
                visible=False,
            )
            
            prompt_part2 = gr.Textbox(
                label="color",
                show_label=False,
                max_lines=1,
                placeholder="color (e.g., color category)",
                container=False,
            )
            
            prompt_part3 = gr.Textbox(
                label="dress_type",
                show_label=False,
                max_lines=1,
                placeholder="dress_type (e.g., t-shirt, sweatshirt, shirt, hoodie)",
                container=False,
            )
            
            prompt_part4_front = gr.Textbox(
                label="front design",
                show_label=False,
                max_lines=1,
                placeholder="front design",
                container=False,
            )

            prompt_part4_back = gr.Textbox(
                label="back design",
                show_label=False,
                max_lines=1,
                placeholder="back design",
                container=False,
            )
            
            prompt_part5 = gr.Textbox(
                value="hanging on the plain wall", 
                label="Prompt Part 5",
                show_label=False,
                interactive=False,
                container=False,
                elem_id="prompt_part5",
                visible=False,
            )

            negative_prompt = gr.Textbox(
                label="Negative prompt",
                max_lines=1,
                placeholder="Enter a negative prompt",
                visible=True,
            )
            
            
            run_button = gr.Button("Run", scale=0)
        
        front_result = gr.Image(label="Front View Result", show_label=False)
        back_result = gr.Image(label="Back View Result", show_label=False)
        seed_result = gr.Textbox(label="Seed Used", show_label=False, interactive=False)
        
        with gr.Accordion("Advanced Settings", open=False):
            
            seed = gr.Slider(
                label="Seed",
                minimum=0,
                maximum=MAX_SEED,
                step=1,
                value=0,
            )
            
            randomize_seed = gr.Checkbox(label="Randomize seed", value=True)
            
            with gr.Row():
                
                width = gr.Slider(
                    label="Width",
                    minimum=256,
                    maximum=MAX_IMAGE_SIZE,
                    step=32,
                    value=512,
                )
                
                height = gr.Slider(
                    label="Height",
                    minimum=256,
                    maximum=MAX_IMAGE_SIZE,
                    step=32,
                    value=512,
                )
            
            with gr.Row():
                
                guidance_scale = gr.Slider(
                    label="Guidance scale",
                    minimum=0.0,
                    maximum=10.0,
                    step=0.1,
                    value=0.0,
                )
                
                num_inference_steps = gr.Slider(
                    label="Number of inference steps",
                    minimum=1,
                    maximum=12,
                    step=1,
                    value=2,
                )
        
        gr.Examples(
            examples=examples,
            inputs=[prompt_part2, prompt_part3, prompt_part4_front, prompt_part4_back]
        )

    run_button.click(
        fn=infer,
        inputs=[prompt_part1, prompt_part2, prompt_part3, prompt_part4_front, prompt_part4_back, prompt_part5, negative_prompt, seed, randomize_seed, width, height, guidance_scale, num_inference_steps],
        outputs=[front_result, back_result, seed_result]
    )

demo.queue().launch()
