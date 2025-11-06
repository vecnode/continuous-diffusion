# =============================================================
# continuous-diffusion
# =============================================================

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

from PIL import Image, ImageDraw


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

@app.get("/favicon.ico")
async def favicon():
    # Return 204 No Content to suppress favicon 404 errors
    from fastapi.responses import Response
    return Response(status_code=204)



# =============================================================
# =============================================================

local_model_path = "models"
local_libs_path = "libs"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if (DEVICE == "cuda") else torch.float32

SDXL_PATH = "models/sd_xl_base_1.0.safetensors"
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
# STORAGE: In-memory image storage (max 30 images)
# ==========================================================

STORAGE_IMAGE_DATA = []


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
STORAGE_IMAGE_DATA.append(initial_image_data)






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
        for idx, img_data in enumerate(STORAGE_IMAGE_DATA, start=1):
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
            "count": len(STORAGE_IMAGE_DATA),
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
        for img_data in STORAGE_IMAGE_DATA:
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
        STORAGE_IMAGE_DATA.append(image_data)
        if len(STORAGE_IMAGE_DATA) > 30:
            STORAGE_IMAGE_DATA.pop(0)  # Remove oldest
        

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
        STORAGE_IMAGE_DATA.append(image_data)
        if len(STORAGE_IMAGE_DATA) > 30:
            STORAGE_IMAGE_DATA.pop(0)  # Remove oldest
        

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
        for img_data in STORAGE_IMAGE_DATA:
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
        STORAGE_IMAGE_DATA.append(image_data)
        if len(STORAGE_IMAGE_DATA) > 30:
            STORAGE_IMAGE_DATA.pop(0)  # Remove oldest
        
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
        for img_data in STORAGE_IMAGE_DATA:
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
        STORAGE_IMAGE_DATA.append(image_data)
        if len(STORAGE_IMAGE_DATA) > 30:
            STORAGE_IMAGE_DATA.pop(0)  # Remove oldest
        
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
        for img_data in STORAGE_IMAGE_DATA:
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
        STORAGE_IMAGE_DATA.append(image_data)
        if len(STORAGE_IMAGE_DATA) > 30:
            STORAGE_IMAGE_DATA.pop(0)  # Remove oldest
        
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
# API OUTPAINT
# ==========================================================

def outpaint_with_inpaint(
    img: Image.Image,
    side: str = "right",           # "left" | "right" | "top" | "bottom"
    add: int = 512,                # how many pixels to add
    prompt: str = "",
    guidance_scale: float = 5.0,
    steps: int = 30,
    overlap: int = 32              # small overlap strip to help blend
) -> Image.Image:
    assert side in {"left","right","top","bottom"}
    
    # Ensure image is in RGB mode
    if img.mode != "RGB":
        img = img.convert("RGB")
    
    w, h = img.size
    
    # Ensure add is a multiple of 8 for SDXL compatibility
    add = ((add + 7) // 8) * 8
    
    if side in {"left","right"}:
        W, H = w + add, h
    else:
        W, H = w, h + add

    # Ensure final dimensions are multiples of 8 (round up to avoid truncation issues)
    W = ((W + 7) // 8) * 8
    H = ((H + 7) // 8) * 8

    # 1) Build bigger canvas and paste original
    # Use white background - SDXL inpainting works better with white
    canvas = Image.new("RGB", (W, H), (255, 255, 255))
    if side == "left":
        paste_xy = (add, 0)
    elif side == "right":
        paste_xy = (0, 0)
    elif side == "top":
        paste_xy = (0, add)
    else: # bottom
        paste_xy = (0, 0)
    canvas.paste(img, paste_xy)
    
    # Verify the image was pasted correctly
    if canvas.size != (W, H):
        raise ValueError(f"Canvas size mismatch: expected {(W, H)}, got {canvas.size}")

    # 2) Build mask: WHITE (255) = area to inpaint; BLACK (0) = keep original
    # SDXL inpainting uses: white pixels = inpaint, black pixels = keep
    mask = Image.new("L", (W, H), 0)
    draw = ImageDraw.Draw(mask)

    # Mask the new area plus a small overlap into the original for blending
    if side == "left":
        # Mask from 0 to add+overlap (new area + overlap strip)
        draw.rectangle([0, 0, add + overlap, H], fill=255)
    elif side == "right":
        # Mask from w-overlap to W (overlap strip + new area)
        draw.rectangle([w - overlap, 0, W, H], fill=255)
    elif side == "top":
        # Mask from 0 to add+overlap (new area + overlap strip)
        draw.rectangle([0, 0, W, add + overlap], fill=255)
    else: # bottom
        # Mask from h-overlap to H (overlap strip + new area)
        draw.rectangle([0, h - overlap, W, H], fill=255)

    # 3) Ensure we have a valid prompt
    if not prompt or not prompt.strip():
        prompt = "continuation of the scene, seamless extension, consistent style and lighting"

    # 4) Verify inputs are valid before processing
    # Ensure canvas and mask are the same size
    if canvas.size != mask.size:
        raise ValueError(f"Canvas and mask size mismatch: canvas {canvas.size}, mask {mask.size}")
    
    # Verify mask has white pixels (areas to inpaint)
    mask_array = np.array(mask)
    white_pixels = np.sum(mask_array == 255)
    if white_pixels == 0:
        raise ValueError("Mask has no white pixels - nothing to inpaint!")
    
    # Verify canvas is valid (no NaN or inf values)
    canvas_array = np.array(canvas)
    if np.any(np.isnan(canvas_array)) or np.any(np.isinf(canvas_array)):
        raise ValueError("Canvas contains invalid values (NaN or inf)")
    
    # Ensure canvas values are in valid range [0, 255] and proper dtype
    canvas_array = np.clip(canvas_array, 0, 255).astype(np.uint8)
    canvas = Image.fromarray(canvas_array)
    
    # Ensure mask is also uint8
    mask_array = np.array(mask)
    mask_array = np.clip(mask_array, 0, 255).astype(np.uint8)
    mask = Image.fromarray(mask_array, mode='L')
    
    # 5) Run SDXL Inpaint with proper parameters
    # For outpainting, we want full synthesis in the masked area
    # Use strength=1.0 for complete regeneration of masked regions
    generator = torch.Generator(device=DEVICE)
    
    # Try without autocast first - autocast can cause black images in some cases
    # If CUDA is available, we'll use the model's dtype directly
    try:
        # Run without autocast to avoid potential issues
        out = inpaint_model(
            prompt=prompt,
            image=canvas,
            mask_image=mask,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            strength=1.0,  # Full strength for complete synthesis in masked area
            generator=generator,
        ).images[0]
    except Exception as e:
        # If that fails, try with autocast as fallback
        try:
            with torch.autocast("cuda"):
                out = inpaint_model(
                    prompt=prompt,
                    image=canvas,
                    mask_image=mask,
                    num_inference_steps=steps,
                    guidance_scale=guidance_scale,
                    strength=1.0,
                    generator=generator,
                ).images[0]
        except Exception as e2:
            raise RuntimeError(f"Inpainting failed: {str(e)} (fallback also failed: {str(e2)})")

    # Ensure output is RGB and has valid content
    if out.mode != "RGB":
        out = out.convert("RGB")
    
    # Verify output is not all black and has valid values
    out_array = np.array(out)
    
    # Check for invalid values
    if np.any(np.isnan(out_array)) or np.any(np.isinf(out_array)):
        raise RuntimeError("Inpainting produced invalid values (NaN or inf)")
    
    # Check if output is all black (or very dark)
    if np.all(out_array < 10):  # Allow some tolerance
        raise RuntimeError("Inpainting produced a black/dark image - check mask and prompt")
    
    # Ensure output values are in valid range
    out_array = np.clip(out_array, 0, 255).astype(np.uint8)
    out = Image.fromarray(out_array)

    return out


@app.post("/api/outpaint")
async def generate_outpaint(
    image_id: str = Form(...),
    side: str = Form("right"),           # "left" | "right" | "top" | "bottom"
    add: Optional[int] = Form(512),      # how many pixels to add
    prompt: Optional[str] = Form(""),     # prompt for outpaint
    guidance_scale: Optional[float] = Form(None),
    num_inference_steps: Optional[int] = Form(None),
    overlap: Optional[int] = Form(32)     # small overlap strip to help blend
):
    """
    Outpaint (extend) an image by adding pixels to one side using inpainting.
    Returns the outpainted image and stores it in memory.
    """
    # Parse and validate form data
    if side not in {"left", "right", "top", "bottom"}:
        return JSONResponse({
            "success": False,
            "error": f"Invalid side: {side}. Must be 'left', 'right', 'top', or 'bottom'"
        }, status_code=400)
    
    add_pixels = int(add) if add and str(add).strip() else 512
    overlap_pixels = int(overlap) if overlap and str(overlap).strip() else 32
    guidance_scale = float(guidance_scale) if guidance_scale and str(guidance_scale).strip() else 5.0
    num_inference_steps = int(num_inference_steps) if num_inference_steps and str(num_inference_steps).strip() else 30
    prompt_str = prompt if prompt and str(prompt).strip() else ""
    
    try:
        # Find the image in storage
        source_image_data = None
        for img_data in STORAGE_IMAGE_DATA:
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
        
        # Use prompt from source image if not provided, and enhance it for outpainting
        if not prompt_str:
            prompt_str = source_image_data.get("prompt", "")
            if not prompt_str:
                prompt_str = "wide cinematic scene, consistent style"
        
        # Enhance prompt for outpainting - add context about seamless extension
        # This helps the model understand it should continue the scene
        if prompt_str and len(prompt_str) > 0:
            # Add outpainting context to the prompt
            side_descriptions = {
                "right": "extending to the right",
                "left": "extending to the left", 
                "top": "extending upward",
                "bottom": "extending downward"
            }
            side_desc = side_descriptions.get(side, "extending")
            enhanced_prompt = f"{prompt_str}, seamless {side_desc}, continuation of scene, consistent lighting and style"
        else:
            enhanced_prompt = f"seamless extension, continuation of the scene, consistent style and lighting"
        
        # Perform outpaint
        result = outpaint_with_inpaint(
            img=img_pil,
            side=side,
            add=add_pixels,
            prompt=enhanced_prompt,
            guidance_scale=guidance_scale,
            steps=num_inference_steps,
            overlap=overlap_pixels
        )
        
        # Generate unique ID
        new_image_id = str(uuid.uuid4())
        
        # Convert image to base64 for browser
        image_base64 = pil_to_base64(result)
        
        # Store image data in memory (keep max 30)
        image_data = {
            "image_id": new_image_id,
            "image": result,  # Store PIL Image
            "image_base64": image_base64,
            "prompt": prompt_str,
            "guidance_scale": guidance_scale,
            "num_inference_steps": num_inference_steps,
            "seed": None,
            "name": None
        }
        
        # Add to storage and keep max 30
        STORAGE_IMAGE_DATA.append(image_data)
        if len(STORAGE_IMAGE_DATA) > 30:
            STORAGE_IMAGE_DATA.pop(0)  # Remove oldest
        
        response_data = {
            "success": True,
            "image_base64": image_base64,
            "image_id": new_image_id,
            "side": side,
            "add": add_pixels
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
# API 3D FIELD
# ==========================================================

@app.post("/api/generate_3d_field")
async def generate_3d_field(
    image_id: str = Form(...)
):
    """
    Generate a 3D point cloud field from the selected image using depth estimation.
    Returns point cloud data (points and colors) for visualization in Three.js.
    """
    try:
        # Find the image in storage
        source_image_data = None
        for img_data in STORAGE_IMAGE_DATA:
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
        W, H = img_pil.size
        img_np = np.array(img_pil)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        
        # Generate depth map
        with torch.no_grad():
            depth = depth_anything_v2_model.infer_image(img_bgr, input_size=518)
        depth = depth.astype(np.float32)
        
        # Normalize depth to a unit cube in Z
        # Map depth -> z in [-1, +1], with z=0 around the median of depth
        d_min = float(np.min(depth))
        d_max = float(np.max(depth))
        if d_max - d_min < 1e-6:
            return JSONResponse({
                "success": False,
                "error": "Depth map appears to be constant"
            }, status_code=400)
        
        d_med = float(np.median(depth))
        
        # Map to [-1, +1] with median ~ 0
        scale = max(d_max - d_med, d_med - d_min)
        z_norm = (depth - d_med) / (scale + 1e-8)
        z_norm = np.clip(z_norm, -1.0, 1.0)
        
        # Back-project to 3D (simple pinhole, fx=fy=max(W,H))
        # X and Y are scaled so the full image spans ~2 units across the shorter side
        fx = fy = float(max(W, H))
        cx, cy = (W - 1) * 0.5, (H - 1) * 0.5
        
        u = np.arange(W, dtype=np.float32)[None, :].repeat(H, axis=0)
        v = np.arange(H, dtype=np.float32)[:, None].repeat(W, axis=1)
        
        Z = z_norm  # [-1,1]
        X = (u - cx) / fx * 2.0  # roughly [-1,1] across width
        Y = -(v - cy) / fx * 2.0  # roughly [-aspect, aspect], flip Y for conventional view
        
        pts = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)
        
        # Color the point cloud from the RGB image
        colors = (img_np.reshape(-1, 3) / 255.0).astype(np.float32)
        
        # Downsample for performance (keep every Nth point)
        downsample_factor = max(1, int(np.sqrt(H * W) / 100))  # Adaptive downsampling
        if downsample_factor > 1:
            mask = np.arange(len(pts)) % downsample_factor == 0
            pts = pts[mask]
            colors = colors[mask]
        
        # Convert to lists for JSON serialization
        points_list = pts.tolist()
        colors_list = colors.tolist()
        
        response_data = {
            "success": True,
            "points": points_list,
            "colors": colors_list,
            "num_points": len(points_list)
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

