# continuous-diffusion

Under heavy development. 

### Models used

```
stabilityai/stable-diffusion-xl-base-1.0"
stabilityai/stable-diffusion-xl-refiner-1.0
depth_anything_v2_vitl
```

### Reproduce on Ubuntu 22.04
```
python3 -m venv venv 
source venv/bin/activate
pip3 install -r requirements

# Dependency of Depth-Anything-V2
cd libs/
git clone https://github.com/DepthAnything/Depth-Anything-V2
cd Depth-Anything-V2
pip3 install -r requirements.txt

# Run
python3 main.py
```

