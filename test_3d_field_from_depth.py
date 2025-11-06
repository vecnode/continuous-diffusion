# test
# python test_3d_field_from_depth.py

import os
import sys
import torch
import numpy as np
import cv2
from PIL import Image
from io import BytesIO
import requests
import open3d as o3d

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------------------------------------------------------
# Paths and imports
# -----------------------------------------------------------------------------
libs_path = os.path.join(os.getcwd(), 'libs')
depth_anything_path = os.path.join(os.getcwd(), 'libs', 'Depth-Anything-V2')
if libs_path not in sys.path:
    sys.path.insert(0, libs_path)
if depth_anything_path not in sys.path:
    sys.path.insert(0, depth_anything_path)

from depth_anything_v2.dpt import DepthAnythingV2

# -----------------------------------------------------------------------------
# Load model
# -----------------------------------------------------------------------------
model_configs = {
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
}
encoder = 'vitl'

depth_anything_v2_model = DepthAnythingV2(**model_configs[encoder])
depth_anything_v2_model.load_state_dict(
    torch.load(f'models/depth_anything_v2_{encoder}.pth', map_location='cpu')
)
depth_anything_v2_model = depth_anything_v2_model.to(DEVICE).eval()
print("Depth-Anything-V2 model loaded successfully")

# -----------------------------------------------------------------------------
# Load an RGB image (local path or http/https URL)
# -----------------------------------------------------------------------------
img_src = r"path/to/your/image.jpg"

def load_pil_image(src: str) -> Image.Image:
    if src.startswith("http://") or src.startswith("https://"):
        r = requests.get(src, timeout=30)
        r.raise_for_status()
        return Image.open(BytesIO(r.content)).convert("RGB")
    else:
        return Image.open(src).convert("RGB")

img_pil = load_pil_image(img_src)
W, H = img_pil.size
img_np = np.array(img_pil)
img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

# -----------------------------------------------------------------------------
# Inference: get depth (float32, HxW)
# -----------------------------------------------------------------------------
print("Generating depth map...")
with torch.no_grad():
    depth = depth_anything_v2_model.infer_image(img_bgr, input_size=518)
depth = depth.astype(np.float32)
print(f"Depth range: {float(depth.min()):.4f} .. {float(depth.max()):.4f}")

# -----------------------------------------------------------------------------
# Normalize depth to a unit cube in Z
# Map depth -> z in [-1, +1], with z=0 around the median of depth
# -----------------------------------------------------------------------------
d_min = float(np.min(depth))
d_max = float(np.max(depth))
if d_max - d_min < 1e-6:
    raise RuntimeError("Depth map appears to be constant.")
d_med = float(np.median(depth))

# map to [-1, +1] with median ~ 0
# first shift so median=0 in [d_min, d_max], then scale by max(|d-d_med|)
scale = max(d_max - d_med, d_med - d_min)
z_norm = (depth - d_med) / (scale + 1e-8)  # roughly in [-1, +1]
z_norm = np.clip(z_norm, -1.0, 1.0)

# -----------------------------------------------------------------------------
# Back-project to 3D (simple pinhole, fx=fy=max(W,H))
# X and Y are scaled so the full image spans ~2 units across the shorter side
# -----------------------------------------------------------------------------
fx = fy = float(max(W, H))
cx, cy = (W - 1) * 0.5, (H - 1) * 0.5

u = np.arange(W, dtype=np.float32)[None, :].repeat(H, axis=0)
v = np.arange(H, dtype=np.float32)[:, None].repeat(W, axis=1)

Z = z_norm                                 # [-1,1]
X = (u - cx) / fx * 2.0                    # roughly [-1,1] across width
Y = -(v - cy) / fx * 2.0                   # roughly [-aspect, aspect], flip Y for conventional view

pts = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)
mask = np.ones((H * W,), dtype=bool)
pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts[mask]))
# Color the point cloud from the RGB image
colors = (img_np.reshape(-1, 3) / 255.0).astype(np.float32)
pcd.colors = o3d.utility.Vector3dVector(colors[mask])

# Optional: downsample for speed
pcd = pcd.voxel_down_sample(voxel_size=0.01)

# -----------------------------------------------------------------------------
# Build a wireframe unit cube around [-1,1]^3
# -----------------------------------------------------------------------------
def make_wire_cube(minb=(-1,-1,-1), maxb=(1,1,1), color=(0.2,0.2,0.2)):
    minb = np.array(minb, dtype=np.float32)
    maxb = np.array(maxb, dtype=np.float32)
    corners = np.array([
        [minb[0], minb[1], minb[2]],
        [maxb[0], minb[1], minb[2]],
        [maxb[0], maxb[1], minb[2]],
        [minb[0], maxb[1], minb[2]],
        [minb[0], minb[1], maxb[2]],
        [maxb[0], minb[1], maxb[2]],
        [maxb[0], maxb[1], maxb[2]],
        [minb[0], maxb[1], maxb[2]],
    ], dtype=np.float32)
    lines = np.array([
        [0,1],[1,2],[2,3],[3,0],
        [4,5],[5,6],[6,7],[7,4],
        [0,4],[1,5],[2,6],[3,7]
    ], dtype=np.int32)
    colors = np.tile(np.array(color, dtype=np.float32), (lines.shape[0],1))
    ls = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(corners),
        lines=o3d.utility.Vector2iVector(lines),
    )
    ls.colors = o3d.utility.Vector3dVector(colors)
    return ls

cube = make_wire_cube()

# -----------------------------------------------------------------------------
# Add an XY grid at z=0 (so you see a grid and the image “in the middle”)
# -----------------------------------------------------------------------------
def make_xy_grid(size=2.0, step=0.1, z=0.0, color=(0.6,0.6,0.6)):
    half = size * 0.5
    xs = np.arange(-half, half + 1e-6, step, dtype=np.float32)
    ys = np.arange(-half, half + 1e-6, step, dtype=np.float32)
    pts = []
    lines = []
    c = []
    idx = 0
    for x in xs:
        pts.append([x, -half, z]); pts.append([x, half, z])
        lines.append([idx, idx+1]); idx += 2
        c.append(color)
    for y in ys:
        pts.append([-half, y, z]); pts.append([half, y, z])
        lines.append([idx, idx+1]); idx += 2
        c.append(color)
    ls = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(np.array(pts, dtype=np.float32)),
        lines=o3d.utility.Vector2iVector(np.array(lines, dtype=np.int32)),
    )
    ls.colors = o3d.utility.Vector3dVector(np.array(c, dtype=np.float32))
    return ls

grid = make_xy_grid(size=2.0, step=0.1, z=0.0, color=(0.75,0.75,0.75))

# -----------------------------------------------------------------------------
# Make a textured quad (the image) at z=0, facing the camera at +Z
# -----------------------------------------------------------------------------
def make_textured_quad_from_image(img: np.ndarray, width=2.0):
    """Return (mesh, material) for Open3D rendering with the image as a texture."""
    h, w = img.shape[:2]
    aspect = h / float(w)
    # quad vertices (two triangles), centered at origin, z=0, normal +Z
    half_w = width * 0.5
    half_h = half_w * aspect
    vertices = np.array([
        [-half_w, -half_h, 0.0],  # 0 bottom-left
        [ half_w, -half_h, 0.0],  # 1 bottom-right
        [ half_w,  half_h, 0.0],  # 2 top-right
        [-half_w,  half_h, 0.0],  # 3 top-left
    ], dtype=np.float32)
    triangles = np.array([[0,1,2],[0,2,3]], dtype=np.int32)
    # UVs (Open3D expects per-corner of each triangle)
    uvs = np.array([
        [0.0, 1.0], [1.0, 1.0], [1.0, 0.0],   # tri 0: 0,1,2
        [0.0, 1.0], [1.0, 0.0], [0.0, 0.0],   # tri 1: 0,2,3
    ], dtype=np.float32)

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    mesh.triangle_uvs = o3d.utility.Vector2dVector(uvs.astype(np.float64))
    mesh.triangle_material_ids = o3d.utility.IntVector([0, 0])  # single material
    mesh.compute_vertex_normals()

    # Material with the image as base color map
    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "defaultLit"
    # Convert img (RGB) to Open3D Image
    if img.dtype != np.uint8:
        img_uint8 = np.clip(img, 0, 255).astype(np.uint8)
    else:
        img_uint8 = img
    o3d_tex = o3d.geometry.Image(img_uint8)
    mat.albedo_img = o3d_tex  # Set texture image

    return mesh, mat

img_quad, img_mat = make_textured_quad_from_image(img_np, width=2.0)

# -----------------------------------------------------------------------------
# Offscreen render and save with PIL
# -----------------------------------------------------------------------------
W_out, H_out = 1280, 720
renderer = o3d.visualization.rendering.OffscreenRenderer(W_out, H_out)

scene = renderer.scene
scene.set_background([1, 1, 1, 1])  # white
# Lighting
scene.scene.set_sun_light(np.array([1.0, 1.0, 1.0]),  # direction is ignored for sun off
                          np.array([1.0, 1.0, 1.0]),  # color
                          45000.0)                    # intensity
scene.scene.enable_sun_light(True)

# Add geometries
scene.add_geometry("cube", cube, o3d.visualization.rendering.MaterialRecord())
scene.add_geometry("grid", grid, o3d.visualization.rendering.MaterialRecord())
scene.add_geometry("image_quad", img_quad, img_mat)

# Draw a small, faint point cloud to visualize depth structure
pc_mat = o3d.visualization.rendering.MaterialRecord()
pc_mat.shader = "defaultUnlit"
pc_mat.point_size = 1.0
scene.add_geometry("pcd", pcd, pc_mat)

# Camera: place at (0,0,3) looking at origin, up +Y
cam = scene.camera
eye = np.array([0.0, 0.0, 3.0], dtype=np.float32)
center = np.array([0.0, 0.0, 0.0], dtype=np.float32)
up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
cam.look_at(center, eye, up)

# Field of view
cam.set_projection(60.0, W_out / H_out, 0.01, 100.0, o3d.visualization.rendering.Camera.FovType.Vertical)

# Render
img_o3d = renderer.render_to_image()
img_out = np.asarray(img_o3d)  # RGBA uint8

# Save via PIL
out_path = "depth_cube_projection.png"
Image.fromarray(img_out).save(out_path)
print(f"Saved visualization -> {out_path}")


