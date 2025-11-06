# =============================================================
# Diffusion Dashboard Server
# =============================================================


# =============================================================
# Global libraries
# =============================================================

# to paste processing before do_run()
# DO BASE REFINER ENSEMBLE OF EXPERT DENOISERS
# https://huggingface.co/docs/diffusers/using-diffusers/sdxl


# =============================================================
# =============================================================

import os
import sys
import uuid
import base64
import io
from typing import Optional


# =============================================================
# =============================================================

import torch
import numpy as np
import cv2
import matplotlib

from diffusers import DiffusionPipeline
from diffusers import StableDiffusionXLInpaintPipeline



from PIL import Image


# Add libs to path
libs_path = os.path.join(os.getcwd(), 'libs')
depth_anything_path = os.path.join(os.getcwd(), 'libs', 'Depth-Anything-V2')
if libs_path not in sys.path:
    sys.path.insert(0, libs_path)
if depth_anything_path not in sys.path:
    sys.path.insert(0, depth_anything_path)



# =============================================================
# =============================================================

from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles


# FastAPI app initialization
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def root():
    return FileResponse("static/index.html")



# =============================================================
# =============================================================



DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if (DEVICE == "cuda") else torch.float32

local_model_path = "models"

SDXL_PATH = "models/sd_xl_base_1.0.safetensors"
SAM_CKPT = "models/sam_vit_h_4b8939.pth"
HF_SDXL_MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"


# DIFFUSION MODEL BASE

diffusion_model_base = DiffusionPipeline.from_pretrained(
    HF_SDXL_MODEL_ID, 
    torch_dtype=torch.float16
).to(DEVICE)


# DIFFUSION MODEL REFINER

diffusion_model_refiner = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0",
    text_encoder_2=diffusion_model_base.text_encoder_2,
    vae=diffusion_model_base.vae,
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
).to(DEVICE)

# DIFFUSION MODEL INPAIN

inpaint_model = StableDiffusionXLInpaintPipeline(
    vae=diffusion_model_base.vae,
    text_encoder=diffusion_model_base.text_encoder,
    text_encoder_2=diffusion_model_base.text_encoder_2,
    tokenizer=diffusion_model_base.tokenizer,
    tokenizer_2=diffusion_model_base.tokenizer_2,
    unet=diffusion_model_base.unet,
    scheduler=diffusion_model_base.scheduler
).to(DEVICE)


# STABLE VIDEO
# from diffusers import StableVideoDiffusionPipeline
# from diffusers.utils import load_image, export_to_video
# stable_video_model = StableVideoDiffusionPipeline.from_pretrained(
#     "stabilityai/stable-video-diffusion-img2vid-xt", 
#     torch_dtype=torch.float16, 
#     variant="fp16"
# ).to(DEVICE)

# # Load the conditioning image
# image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/svd/rocket.png")
# image = image.resize((1024, 576))

# generator = torch.manual_seed(42)
# frames = stable_video_model(image, decode_chunk_size=8, generator=generator).frames[0]

# export_to_video(frames, "generated.mp4", fps=7)



# DEPTH-ANYTHING-V2 MODEL

from depth_anything_v2.dpt import DepthAnythingV2

model_configs = {
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
}

encoder = 'vitl'

depth_anything_v2_model = DepthAnythingV2(**model_configs[encoder])
depth_anything_v2_model.load_state_dict(torch.load(f'models/depth_anything_v2_{encoder}.pth', map_location='cpu'))
depth_anything_v2_model = depth_anything_v2_model.to(DEVICE).eval()


# ==========================================================
# STORAGE
# ==========================================================




storage_image_data = []


def pil_to_base64(image: Image.Image) -> str:
    """Convert PIL Image to base64 string for browser display."""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"


# Initialize with a black image as the first entry
black_image = Image.new('RGB', (512, 512), color=(0, 0, 0))
black_image_base64 = pil_to_base64(black_image)

initial_image_data = {
    "image_id": str(uuid.uuid4()),
    "image": black_image,
    "image_base64": black_image_base64,
    "prompt": "",
    "guidance_scale": 0.0,
    "num_inference_steps": 0,
    "seed": None,
    "name": None
}
storage_image_data.append(initial_image_data)




# ==========================================================
# API ENDPOINTS
# ==========================================================



# ==========================================================
# API STORAGE
# ==========================================================



@app.get("/api/storage")
async def get_storage_data():
    """
    Get list of stored images in memory (metadata only, no image data).
    Returns the metadata that was saved along with each image when it was generated.
    """
    try:
        # Return the metadata that was already saved with each image
        # Just exclude the large image data fields for the response
        storage_list = []
        for idx, img_data in enumerate(storage_image_data, start=1):
            # The metadata (prompt, guidance_scale, etc.) was saved when image was created
            # We just extract it and exclude the image/image_base64 fields
            metadata = {
                "index": idx,  # Added for frontend display only
                "image_id": img_data["image_id"],
                "prompt": img_data["prompt"],
                "guidance_scale": img_data["guidance_scale"],
                "num_inference_steps": img_data["num_inference_steps"],
                "seed": img_data["seed"],
                "name": img_data.get("name")  # Optional name field
            }
            storage_list.append(metadata)
        
        return JSONResponse({
            "success": True,
            "count": len(storage_image_data),
            "images": storage_list
        })
    except Exception as e:
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=500)




@app.get("/api/storage/{image_id}")
async def get_image_by_id(image_id: str):
    """
    Get a specific image by image_id, including base64 data.
    """
    try:
        # Find the image in storage
        for img_data in storage_image_data:
            if img_data["image_id"] == image_id:
                return JSONResponse({
                    "success": True,
                    "image_id": img_data["image_id"],
                    "image_base64": img_data["image_base64"],
                    "prompt": img_data["prompt"],
                    "guidance_scale": img_data["guidance_scale"],
                    "num_inference_steps": img_data["num_inference_steps"],
                    "seed": img_data["seed"]
                })
        
        return JSONResponse({
            "success": False,
            "error": "Image not found"
        }, status_code=404)
    except Exception as e:
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=500)




# ==========================================================
# API GENERATION
# ==========================================================



@app.post("/api/generate")
async def generate_txt_to_image_sdxl_fp16(
    prompt: str = Form(...),
    guidance_scale: Optional[float] = Form(None),
    num_inference_steps: Optional[int] = Form(None),
    seed: Optional[str] = Form(None)
):
    """
    Generate an image from a text prompt.
    Fast local endpoint for text-to-image generation.
    Stores images in memory and sends base64 to browser.
    """
    # Parse and validate form data
    guidance_scale = float(guidance_scale) if guidance_scale and str(guidance_scale).strip() else 7.0
    num_inference_steps = int(num_inference_steps) if num_inference_steps and str(num_inference_steps).strip() else 30
    seed_int = None
    if seed and str(seed).strip():
        try:
            seed_int = int(seed)
        except (ValueError, TypeError):
            seed_int = None
    try:
        # Generate image
        generator = None
        if seed_int is not None:
            generator = torch.Generator(device=DEVICE).manual_seed(seed_int)
        
        result = diffusion_model_base(
            prompt=prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator
        ).images[0]
        
        # Generate unique ID
        image_id = str(uuid.uuid4())
        
        # Convert image to base64 for browser
        image_base64 = pil_to_base64(result)
        
        # Store image data in memory (keep max 30)
        image_data = {
            "image_id": image_id,
            "image": result,  # Store PIL Image
            "image_base64": image_base64,
            "prompt": prompt,
            "guidance_scale": guidance_scale,
            "num_inference_steps": num_inference_steps,
            "seed": seed_int,
            "name": None
        }
        
        # Add to storage and keep max 30
        storage_image_data.append(image_data)
        if len(storage_image_data) > 30:
            storage_image_data.pop(0)  # Remove oldest
        

        response_data = {
            "success": True,
            "image_base64": image_base64,
            "image_id": image_id
        }
        
        return JSONResponse(response_data)
    except Exception as e:
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=500)





@app.post("/api/generate2")
async def generate_txt_to_image_sdxl_refined(
    prompt: str = Form(...),
    guidance_scale: Optional[float] = Form(None),
    num_inference_steps: Optional[int] = Form(None),
    seed: Optional[str] = Form(None)
):
    """
    Generate an image from a text prompt using standard SDXL base + refiner cascade.
    
    Standard ensemble of expert denoisers approach:
    - Base model: performs first 80% of denoising (denoising_end=0.8), outputs latent
    - Refiner model: performs last 20% of denoising (denoising_start=0.8), outputs PIL image
    - Both models use the same num_inference_steps, guidance_scale, and generator for consistency
    
    This is the standard SDXL cascade pattern - do not alter the denoising parameters.
    Stores images in memory and sends base64 to browser.
    """
    # Parse and validate form data
    guidance_scale = float(guidance_scale) if guidance_scale and str(guidance_scale).strip() else 7.0
    num_inference_steps = int(num_inference_steps) if num_inference_steps and str(num_inference_steps).strip() else 40
    seed_int = None
    if seed and str(seed).strip():
        try:
            seed_int = int(seed)
        except (ValueError, TypeError):
            seed_int = None
    try:
        # Generate image using base + refiner cascade (standard SDXL ensemble approach)
        generator = None
        if seed_int is not None:
            generator = torch.Generator(device=DEVICE).manual_seed(seed_int)
        
        # Base model: performs first 80% of denoising, outputs latent representation
        # denoising_end=0.8 means it stops at 80% through the noise schedule
        latent_image = diffusion_model_base(
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            denoising_end=0.8,
            output_type="latent",
            guidance_scale=guidance_scale,
            generator=generator
        ).images
        
        # Refiner model: performs last 20% of denoising, takes latent from base
        # denoising_start=0.8 means it starts at 80% through the noise schedule
        # This creates a seamless cascade where base does 0-80%, refiner does 80-100%
        result = diffusion_model_refiner(
            prompt=prompt,
            image=latent_image,
            num_inference_steps=num_inference_steps,
            denoising_start=0.8,
            guidance_scale=guidance_scale,
            generator=generator,
            output_type="pil"
        ).images[0]
        
        # Generate unique ID
        image_id = str(uuid.uuid4())
        
        # Convert image to base64 for browser
        image_base64 = pil_to_base64(result)
        
        # Store image data in memory (keep max 30)
        image_data = {
            "image_id": image_id,
            "image": result,  # Store PIL Image
            "image_base64": image_base64,
            "prompt": prompt,
            "guidance_scale": guidance_scale,
            "num_inference_steps": num_inference_steps,
            "seed": seed_int,
            "name": None
        }
        
        # Add to storage and keep max 30
        storage_image_data.append(image_data)
        if len(storage_image_data) > 30:
            storage_image_data.pop(0)  # Remove oldest
        

        response_data = {
            "success": True,
            "image_base64": image_base64,
            "image_id": image_id
        }
        
        return JSONResponse(response_data)
    except Exception as e:
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=500)




@app.post("/api/generate_img2img")
async def generate_image_to_image_refiner(
    image_id: str = Form(...),
    prompt: str = Form(...),
    guidance_scale: Optional[float] = Form(None),
    num_inference_steps: Optional[int] = Form(None),
    seed: Optional[str] = Form(None)
):
    """
    Image-to-image transformation using the refiner model with double refinement pass.
    Takes a selected image and applies refinement guided by a prompt.
    
    Process:
    1. First pass: 20% diffusion (denoising_start=0.8) - substantial refinement
    2. Second pass: 10% diffusion (denoising_start=0.9) - fine refinement for enhanced quality
    
    Uses the refiner model without inpainting - pure image-to-image transformation.
    The double pass improves image quality and detail preservation.
    """
    # Parse and validate form data
    guidance_scale = float(guidance_scale) if guidance_scale and str(guidance_scale).strip() else 7.0
    num_inference_steps = int(num_inference_steps) if num_inference_steps and str(num_inference_steps).strip() else 40
    seed_int = None
    if seed and str(seed).strip():
        try:
            seed_int = int(seed)
        except (ValueError, TypeError):
            seed_int = None
    
    try:
        # Find the image in storage
        source_image_data = None
        for img_data in storage_image_data:
            if img_data["image_id"] == image_id:
                source_image_data = img_data
                break
        
        if source_image_data is None:
            return JSONResponse({
                "success": False,
                "error": "Image not found in storage"
            }, status_code=404)
        
        # Get PIL Image from storage
        img_pil = source_image_data["image"]
        
        # Prepare generator
        generator = None
        if seed_int is not None:
            generator = torch.Generator(device=DEVICE).manual_seed(seed_int)
        
        # Encode the image to latent space using the VAE
        vae = diffusion_model_refiner.vae
        with torch.no_grad():
            # Convert PIL to tensor and normalize
            image_tensor = torch.from_numpy(np.array(img_pil)).float() / 255.0
            image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0).to(DEVICE, dtype=DTYPE)
            image_tensor = image_tensor * 2.0 - 1.0  # Normalize to [-1, 1]
            
            # Encode to latent space
            latent = vae.encode(image_tensor).latent_dist.sample()
            latent = latent * vae.config.scaling_factor
        
        # Add noise to the latent starting from timestep corresponding to 0.8
        # We'll use the scheduler to get the proper noise level
        scheduler = diffusion_model_refiner.scheduler
        scheduler.set_timesteps(num_inference_steps, device=DEVICE)
        
        # Calculate the starting timestep (80% through the schedule)
        # denoising_start=0.8 means we start at 80% of the noise schedule
        start_timestep_idx = int(num_inference_steps * 0.8)
        start_timestep_idx = min(start_timestep_idx, len(scheduler.timesteps) - 1)
        start_timestep = scheduler.timesteps[start_timestep_idx]
        
        # Add noise to the latent at the appropriate level
        if generator is not None:
            noise = torch.randn(latent.shape, generator=generator, device=latent.device, dtype=latent.dtype)
        else:
            noise = torch.randn_like(latent)
        # Convert scalar timestep to tensor with shape [1] for add_noise
        timesteps_tensor = start_timestep.unsqueeze(0) if start_timestep.dim() == 0 else start_timestep
        noisy_latent = scheduler.add_noise(latent, noise, timesteps_tensor)
        
        # First refinement pass: denoise from 0.8 to 1.0 (last 20%)
        # Pass the noisy latent tensor directly - the refiner will handle it
        first_pass_result = diffusion_model_refiner(
            prompt=prompt,
            image=noisy_latent,
            num_inference_steps=num_inference_steps,
            denoising_start=0.8,
            guidance_scale=guidance_scale,
            generator=generator,
            output_type="pil"
        ).images[0]
        
        # Second refinement pass: encode the refined image and apply another light refinement
        # This improves quality by doing a second pass with minimal noise (high denoising_start)
        with torch.no_grad():
            # Convert PIL result to tensor and normalize
            refined_tensor = torch.from_numpy(np.array(first_pass_result)).float() / 255.0
            refined_tensor = refined_tensor.permute(2, 0, 1).unsqueeze(0).to(DEVICE, dtype=DTYPE)
            refined_tensor = refined_tensor * 2.0 - 1.0  # Normalize to [-1, 1]
            
            # Encode refined image to latent space
            refined_latent = vae.encode(refined_tensor).latent_dist.sample()
            refined_latent = refined_latent * vae.config.scaling_factor
        
        # Add minimal noise for second refinement pass (90% through schedule for light refinement)
        scheduler.set_timesteps(num_inference_steps, device=DEVICE)
        second_start_timestep_idx = int(num_inference_steps * 0.9)
        second_start_timestep_idx = min(second_start_timestep_idx, len(scheduler.timesteps) - 1)
        second_start_timestep = scheduler.timesteps[second_start_timestep_idx]
        
        # Add minimal noise for fine refinement
        if generator is not None:
            second_noise = torch.randn(refined_latent.shape, generator=generator, device=refined_latent.device, dtype=refined_latent.dtype)
        else:
            second_noise = torch.randn_like(refined_latent)
        second_timesteps_tensor = second_start_timestep.unsqueeze(0) if second_start_timestep.dim() == 0 else second_start_timestep
        second_noisy_latent = scheduler.add_noise(refined_latent, second_noise, second_timesteps_tensor)
        
        # Second refinement pass: denoise from 0.9 to 1.0 (last 10% - fine refinement)
        result = diffusion_model_refiner(
            prompt=prompt,
            image=second_noisy_latent,
            num_inference_steps=num_inference_steps,
            denoising_start=0.9,
            guidance_scale=guidance_scale,
            generator=generator,
            output_type="pil"
        ).images[0]
        
        # Generate unique ID
        new_image_id = str(uuid.uuid4())
        
        # Convert image to base64 for browser
        image_base64 = pil_to_base64(result)
        
        # Store image data in memory (keep max 30)
        image_data = {
            "image_id": new_image_id,
            "image": result,  # Store PIL Image
            "image_base64": image_base64,
            "prompt": prompt,
            "guidance_scale": guidance_scale,
            "num_inference_steps": num_inference_steps,
            "seed": seed_int,
            "name": None
        }
        
        # Add to storage and keep max 30
        storage_image_data.append(image_data)
        if len(storage_image_data) > 30:
            storage_image_data.pop(0)  # Remove oldest
        
        response_data = {
            "success": True,
            "image_base64": image_base64,
            "image_id": new_image_id
        }
        
        return JSONResponse(response_data)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=500)








@app.post("/api/generate3")
async def generate_zoom_img2img(
    image_id: str = Form(...),
    prompt: Optional[str] = Form(None),
    guidance_scale: Optional[float] = Form(None),
    num_inference_steps: Optional[int] = Form(None),
    seed: Optional[str] = Form(None)
):
    """
    Zoom the selected image (crop center and resize) and apply img2img refiner with the same prompt.
    Uses the prompt from the selected image if not provided.
    
    Enhanced parameters for detail preservation:
    - Default: 60 inference steps (was 20) for finer refinement
    - denoising_start: 0.85 (was 0.9) - high value preserves detail by minimizing noise addition
    - guidance_scale: 7.5 (was 7.0) for better prompt adherence
    
    Key insight: Higher denoising_start + more steps = better detail preservation.
    Lower denoising_start adds too much noise and causes blur.
    """
    # Parse and validate form data
    guidance_scale = float(guidance_scale) if guidance_scale and str(guidance_scale).strip() else 7.5
    num_inference_steps = int(num_inference_steps) if num_inference_steps and str(num_inference_steps).strip() else 60
    seed_int = None
    if seed and str(seed).strip():
        try:
            seed_int = int(seed)
        except (ValueError, TypeError):
            seed_int = None
    
    try:
        # Find the image in storage
        source_image_data = None
        for img_data in storage_image_data:
            if img_data["image_id"] == image_id:
                source_image_data = img_data
                break
        
        if source_image_data is None:
            return JSONResponse({
                "success": False,
                "error": "Image not found in storage"
            }, status_code=404)
        
        # Get PIL Image from storage
        img_pil = source_image_data["image"]
        
        # Always use prompt from selected image for consistency
        prompt = source_image_data.get("prompt", "")
        if not prompt:
            return JSONResponse({
                "success": False,
                "error": "No prompt available for selected image"
            }, status_code=400)
        
        width, height = img_pil.size
        
        # Zoom: crop center 98% and resize back to original size
        zoom_factor = 0.98
        crop_width = int(width * zoom_factor)
        crop_height = int(height * zoom_factor)
        left = (width - crop_width) // 2
        top = (height - crop_height) // 2
        right = left + crop_width
        bottom = top + crop_height
        
        # Crop center portion
        zoomed_img = img_pil.crop((left, top, right, bottom))
        # Resize back to original size using high-quality resampling
        zoomed_img = zoomed_img.resize((width, height), Image.LANCZOS)
        
        # Prepare generator
        generator = None
        if seed_int is not None:
            generator = torch.Generator(device=DEVICE).manual_seed(seed_int)
        
        # Use high denoising_start (0.85) with many steps for detail preservation
        # Higher denoising_start = less noise added = preserves more detail
        # More steps = finer refinement even with small denoising range
        # This approach maintains sharpness while still refining the zoomed image
        result = diffusion_model_refiner(
            prompt=prompt,
            image=zoomed_img,
            num_inference_steps=num_inference_steps,
            denoising_start=0.85,
            guidance_scale=guidance_scale,
            generator=generator,
            output_type="pil"
        ).images[0]
        
        # Generate unique ID
        new_image_id = str(uuid.uuid4())
        
        # Convert image to base64 for browser
        image_base64 = pil_to_base64(result)
        
        # Store image data in memory (keep max 30)
        image_data = {
            "image_id": new_image_id,
            "image": result,  # Store PIL Image
            "image_base64": image_base64,
            "prompt": prompt,
            "guidance_scale": guidance_scale,
            "num_inference_steps": num_inference_steps,
            "seed": seed_int,
            "name": None
        }
        
        # Add to storage and keep max 30
        storage_image_data.append(image_data)
        if len(storage_image_data) > 30:
            storage_image_data.pop(0)  # Remove oldest
        
        response_data = {
            "success": True,
            "image_base64": image_base64,
            "image_id": new_image_id
        }
        
        return JSONResponse(response_data)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=500)






@app.post("/api/generate_depth")
async def generate_depth_map(
    image_id: str = Form(...)
):
    """
    Generate a depth map from the selected image using Depth-Anything-V2 model.
    """
    try:
        # Find the image in storage
        source_image_data = None
        for img_data in storage_image_data:
            if img_data["image_id"] == image_id:
                source_image_data = img_data
                break
        
        if source_image_data is None:
            return JSONResponse({
                "success": False,
                "error": "Image not found in storage"
            }, status_code=404)
        
        # Get PIL Image from storage
        img_pil = source_image_data["image"]
        
        # Convert PIL Image to numpy array (BGR format for cv2)
        img_np = np.array(img_pil)
        # PIL uses RGB, cv2 uses BGR
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        
        # Generate depth map
        depth = depth_anything_v2_model.infer_image(img_bgr, input_size=518)
        
        # Normalize depth to 0-1 range for colormap (colormap expects [0, 1])
        depth_min = depth.min()
        depth_max = depth.max()
        if depth_max > depth_min:
            depth_normalized = (depth - depth_min) / (depth_max - depth_min)
        else:
            # If all values are the same, create a uniform depth map
            depth_normalized = np.zeros_like(depth)
        
        # Apply colormap for visualization (Spectral_r colormap)
        cmap = matplotlib.colormaps.get_cmap('Spectral_r')
        depth_colored = (cmap(depth_normalized)[:, :, :3] * 255).astype(np.uint8)
        
        # Convert back to PIL Image (colormap returns RGB)
        depth_image = Image.fromarray(depth_colored)
        
        # Generate unique ID
        new_image_id = str(uuid.uuid4())
        
        # Convert image to base64 for browser
        image_base64 = pil_to_base64(depth_image)
        
        # Store image data in memory (keep max 30)
        image_data = {
            "image_id": new_image_id,
            "image": depth_image,  # Store PIL Image
            "image_base64": image_base64,
            "prompt": f"Depth map of {source_image_data.get('prompt', 'image')}",
            "guidance_scale": 0.0,
            "num_inference_steps": 0,
            "seed": None,
            "name": None
        }
        
        # Add to storage and keep max 30
        storage_image_data.append(image_data)
        if len(storage_image_data) > 30:
            storage_image_data.pop(0)  # Remove oldest
        
        response_data = {
            "success": True,
            "image_base64": image_base64,
            "image_id": new_image_id
        }
        
        return JSONResponse(response_data)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=500)




# ==========================================================
# API SEGMENTATION
# ==========================================================










# ==========================================================
# Server Startup
# ==========================================================

if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*60)
    print("FastAPI Server Starting")
    print("="*60)
    print(f"Server will be available at: http://localhost:8000")
    print(f"API documentation at: http://localhost:8000/docs")
    print("="*60 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")

