import torch
from diffusers import DiTPipeline, DPMSolverMultistepScheduler
from PIL import Image

def load_pipeline():
    """
    Loads the pre-trained DiT model from Hugging Face.
    Optimized for RTX 4060 using float16.
    """
    model_id = "facebook/DiT-XL-2-256"
    pipe = DiTPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("cuda")
    return pipe

def generate_image(class_id, steps=20, guidance_scale=4.0):
    """
    Generates an image using the DiT pipeline.
    """
    pipe = load_pipeline()
    
    # ImageNet class labels are required by this model
    image = pipe(
        class_labels=[class_id], 
        num_inference_steps=steps, 
        guidance_scale=guidance_scale
    ).images[0]
    
    return image

if __name__ == "__main__":
    # Example: Generate an Airplane (ImageNet class 404)
    img = generate_image(404)
    img.save("pretrained_test.png")
    print("Saved pretrained_test.png")