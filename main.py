from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_ngrok import run_with_ngrok
from werkzeug.utils import secure_filename
from diffusers import AutoPipelineForInpainting, AutoencoderKL
from diffusers.utils import load_image
from SegBody import segment_body
from pyngrok import ngrok
import os
import torch
from io import BytesIO
import base64

# Initialize the Flask app and CORS
app = Flask(__name__)
CORS(app)
run_with_ngrok(app)

# Configure the upload folder
app.config['UPLOAD_FOLDER'] = 'uploads'
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

#Load the pretrained models
vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
pipeline = AutoPipelineForInpainting.from_pretrained(
    "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
    vae=vae,
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True
)
pipeline.load_ip_adapter(
    "h94/IP-Adapter",
    subfolder="sdxl_models",
    weight_name="ip-adapter_sdxl.bin",
    low_cpu_mem_usage=True
)

# Define the image processing function
def virtual_try_on(img, clothing, prompt, negative_prompt, ip_scale=1.0, strength=0.99, guidance_scale=7.5, steps=100):
    _, mask_img = segment_body(img, face=False)
    pipeline.set_ip_adapter_scale(ip_scale)
    images = pipeline(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=img,
        mask_image=mask_img,
        ip_adapter_image=clothing,
        strength=strength,
        guidance_scale=guidance_scale,
        num_inference_steps=steps,
    ).images
    return images[0]

@app.route("/")
def hello():
    return "Hello, world!"

# Define the route to receive images and return the processed image
@app.route("/submit-prompt", methods=["POST", "GET"])
def generate_image():
    person_image = request.files['person']
    clothing_image = request.files['clothing']
    print(person_image.name)
    print(clothing_image.name)

    if person_image and clothing_image:
        person_filename = secure_filename(person_image.filename)
        clothing_filename = secure_filename(clothing_image.filename)

        person_path = os.path.join(app.config['UPLOAD_FOLDER'], person_filename)
        clothing_path = os.path.join(app.config['UPLOAD_FOLDER'], clothing_filename)

        person_image.save(person_path)
        clothing_image.save(clothing_path)

        image = load_image(person_path).convert("RGB")
        ip_image = load_image(clothing_path).convert("RGB")

        result_image = virtual_try_on(
            image, ip_image,
            prompt="photorealistic, perfect body, beautiful skin, realistic skin, natural skin",
            negative_prompt="ugly, bad quality, bad anatomy, deformed body, deformed hands, deformed feet, deformed face, deformed clothing, deformed skin, bad skin, leggings, tights, stockings"
        )

        buffered = BytesIO()
        result_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue())
        img_str = "data:image/png;base64," + str(img_str)[2:-1]
        return jsonify({'generated_image': img_str})

# Start the Flask app using ngrok
if __name__ == '__main__':
    public_url = ngrok.connect(5000)
    app.run()
