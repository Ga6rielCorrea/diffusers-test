from diffusers import StableDiffusionPipeline
import torch

model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

prompt = "A vibrant 3D cartoon-style illustration of a cool golden retriever dog in a futuristic sci-fi city. The dog is wearing glowing neon cyber-goggles and a sleek silver high-tech vest. It is sitting inside a floating hover-car, looking out with a happy, excited expression."
image = pipe(prompt).images[0]

image.save("generated_image.png")
image.show()
