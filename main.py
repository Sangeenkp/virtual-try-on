from flask import Flask, render_template, request, send_from_directory
from flask_ngrok import run_with_ngrok
from diffusers import AutoPipelineForInpainting, AutoencoderKL
from diffusers.utils import load_image
from SegBody import segment_body
import torch, base64
from io import BytesIO
from pyngrok import ngrok
import os
from werkzeug.utils import secure_filename

vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
pipeline = AutoPipelineForInpainting.from_pretrained("diffusers/stable-diffusion-xl-1.0-inpainting-0.1", vae=vae, torch_dtype=torch.float16, variant="fp16", use_safetensors=True).to("cuda")
# pipeline = AutoPipelineForInpainting.from_pretrained("diffusers/stable-diffusion-xl-1.0-inpainting-0.1", vae=vae, torch_dtype=torch.float16, variant="fp16", use_safetensors=True)
pipeline.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models", weight_name="ip-adapter_sdxl.bin", low_cpu_mem_usage=True)

def virtual_try_on(img, clothing, prompt, negative_prompt, ip_scale=1.0, strength=0.99, guidance_scale=7.5, steps=100):
    _, mask_img = segment_body(img, face=False)
    print("HERE 1")
    pipeline.set_ip_adapter_scale(ip_scale)
    print("HERE 2")
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
    print("HERE 3")
    return images[0]

# image = virtual_try_on(img,
#                clothing,
#                prompt="photorealistic, perfect body, beautiful skin, realistic skin, natural skin",
#                negative_prompt="ugly, bad quality, bad anatomy, deformed body, deformed hands, deformed feet, deformed face, deformed clothing, deformed skin, bad skin, leggings, tights, stockings")


app = Flask(__name__)
run_with_ngrok(app)

app.config['UPLOAD_FOLDER'] = 'uploads'

# Ensure the upload directory exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

@app.route("/")
def initial():
    return render_template("index.html")

@app.route("/submit-prompt", methods=["POST"])
def generate_image():
    print("inside 2nd api")

    person_image = request.files['person']
    clothing_image = request.files['clothing']

    if person_image and clothing_image:
        # Secure the filenames
        person_filename = secure_filename(person_image.filename)
        clothing_filename = secure_filename(clothing_image.filename)

        # Construct the full paths for the files
        person_path = os.path.join(app.config['UPLOAD_FOLDER'], person_filename)
        clothing_path = os.path.join(app.config['UPLOAD_FOLDER'], clothing_filename)

        # Save the uploaded files to the specified directory
        person_image.save(person_path)
        clothing_image.save(clothing_path)

        image = load_image(person_path).convert("RGB")
        ip_image = load_image(clothing_path).convert("RGB")

        result_image = virtual_try_on(image, ip_image, prompt="photorealistic, perfect body, beautiful skin, realistic skin, natural skin", negative_prompt="ugly, bad quality, bad anatomy, deformed body, deformed hands, deformed feet, deformed face, deformed clothing, deformed skin, bad skin, leggings, tights, stockings")

        # return result_image
        buffered = BytesIO()
        result_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue())
        img_str = "data:image/png;base64," + str(img_str)[2:-1]
        return render_template('index.html', generated_image=img_str)

if __name__ == '__main__':
    public_url = ngrok.connect(5000)
    app.run()