// vecnode 2025

// Fast local communication for image generation
// Global variable to track current displayed image
let CURRENT_TEXTURE = null;

// Global Three.js objects for texture management
let scene3d = null;
let camera3d = null;
let renderer3d = null;
let controls3d = null;
let pointCloud3d = null; // Track the 3D point cloud object
let pointCloudMesh3d = null; // Track the mesh created from point cloud
let pointCloudData = null; // Store point cloud data (points and colors) for mesh creation

// Texture management system (max 50 textures)
const TEXTURE_MANAGER = {
    textures: [], // Array of { imageId, texture, mesh, plane }
    maxTextures: 50,
    textureGroup: null, // Group to hold all texture planes
    
    // Initialize texture group
    init: function(scene) {
        this.textureGroup = new THREE.Group();
        scene.add(this.textureGroup);
    },
    
    // Helper function to create texture and mesh from base64
    _createTextureMesh: function(imageBase64, position, planeSize) {
        // Create texture from base64 image
        const loader = new THREE.TextureLoader();
        const texture = loader.load(imageBase64, () => {
            // Texture loaded successfully
            texture.flipY = true; // Flip Y to show texture right-side up
            texture.generateMipmaps = true; // Generate mipmaps for better filtering
            texture.minFilter = THREE.LinearMipmapLinearFilter;
            texture.magFilter = THREE.LinearFilter; // Use LinearFilter for smooth but sharp rendering
            // Set maximum anisotropy for better quality when viewing at angles
            if (renderer3d && renderer3d.capabilities) {
                texture.anisotropy = renderer3d.capabilities.getMaxAnisotropy();
            } else {
                texture.anisotropy = 16; // Default high value
            }
            // Use sRGB encoding for better color representation (compatible with older Three.js)
            if (THREE.sRGBEncoding !== undefined) {
                texture.encoding = THREE.sRGBEncoding;
            }
            texture.needsUpdate = true;
        });
        
        // Create plane geometry
        const geometry = new THREE.PlaneGeometry(planeSize, planeSize);
        const material = new THREE.MeshBasicMaterial({ 
            map: texture,
            side: THREE.DoubleSide,
            transparent: false, // Fully opaque for better color and quality
            opacity: 1.0, // Full opacity - no transparency
            color: 0xffffff // Ensure full color range (white = no color tinting)
        });
        
        const mesh = new THREE.Mesh(geometry, material);
        
        // Set position
        mesh.position.set(position.x, position.y, position.z);
        
        // Rotate to face camera (or keep flat on grid)
        mesh.rotation.x = 0;
        mesh.rotation.y = 0;
        mesh.rotation.z = 0;
        
        return { texture, geometry, material, mesh };
    },
    
    // Load image as texture and create plane (normal texture, replaces previous)
    addTexture: function(imageId, imageBase64, scene) {
        // Check if texture already exists
        const existingIndex = this.textures.findIndex(t => t.imageId === imageId);
        if (existingIndex !== -1) {
            // Texture already exists, don't add duplicate
            return this.textures[existingIndex];
        }
        
        // Remove previous texture to show only one at a time on the grid
        // Keep only the most recent texture visible
        if (this.textures.length > 0) {
            const previous = this.textures[0];
            if (previous.mesh) {
                this.textureGroup.remove(previous.mesh);
                if (previous.texture) previous.texture.dispose();
                if (previous.material) previous.material.dispose();
                if (previous.geometry) previous.geometry.dispose();
            }
            this.textures = []; // Clear array to keep only one texture
        }
        
        // Remove oldest texture if at limit (for memory management)
        if (this.textures.length >= this.maxTextures) {
            const oldest = this.textures.shift();
            if (oldest.mesh) {
                this.textureGroup.remove(oldest.mesh);
                if (oldest.texture) oldest.texture.dispose();
                if (oldest.material) oldest.material.dispose();
                if (oldest.geometry) oldest.geometry.dispose();
            }
        }
        
        // Create plane geometry matching grid size (2.0 units = 512x512 equivalent)
        const planeSize = 2.0; // Match grid size to occupy whole length
        
        // Center the texture on XY, but place at back of cube (z=-1)
        const position = { x: 0, y: 0, z: -1 };
        
        const { texture, geometry, material, mesh } = this._createTextureMesh(imageBase64, position, planeSize);
        
        // Store texture data
        const textureData = {
            imageId: imageId,
            texture: texture,
            mesh: mesh,
            material: material,
            geometry: geometry,
            plane: mesh
        };
        
        this.textures.push(textureData);
        this.textureGroup.add(mesh);
        
        return textureData;
    },
    
    // Add outpainted texture positioned relative to the previous texture
    addOutpaintTexture: function(imageId, imageBase64, scene, side, previousImageId) {
        // Check if texture already exists
        const existingIndex = this.textures.findIndex(t => t.imageId === imageId);
        if (existingIndex !== -1) {
            // Texture already exists, don't add duplicate
            return this.textures[existingIndex];
        }
        
        // Find the previous texture to position relative to it
        let previousTexture = null;
        if (previousImageId) {
            previousTexture = this.textures.find(t => t.imageId === previousImageId);
        }
        
        // If no previous texture found, use the last texture in the array
        if (!previousTexture && this.textures.length > 0) {
            previousTexture = this.textures[this.textures.length - 1];
        }
        
        // Default position (center, back of cube)
        let position = { x: 0, y: 0, z: -1 };
        const planeSize = 2.0; // Standard plane size
        
        // If we have a previous texture, position relative to it
        if (previousTexture && previousTexture.mesh) {
            const prevPos = previousTexture.mesh.position;
            const offset = 2.0; // Offset by one plane size (2.0 units)
            
            switch (side) {
                case "right":
                    position = { x: prevPos.x + offset, y: prevPos.y, z: prevPos.z };
                    break;
                case "left":
                    position = { x: prevPos.x - offset, y: prevPos.y, z: prevPos.z };
                    break;
                case "top":
                    position = { x: prevPos.x, y: prevPos.y + offset, z: prevPos.z };
                    break;
                case "bottom":
                    position = { x: prevPos.x, y: prevPos.y - offset, z: prevPos.z };
                    break;
                default:
                    // Default to right if invalid side
                    position = { x: prevPos.x + offset, y: prevPos.y, z: prevPos.z };
            }
        }
        
        // Remove oldest texture if at limit (for memory management)
        if (this.textures.length >= this.maxTextures) {
            const oldest = this.textures.shift();
            if (oldest.mesh) {
                this.textureGroup.remove(oldest.mesh);
                if (oldest.texture) oldest.texture.dispose();
                if (oldest.material) oldest.material.dispose();
                if (oldest.geometry) oldest.geometry.dispose();
            }
        }
        
        const { texture, geometry, material, mesh } = this._createTextureMesh(imageBase64, position, planeSize);
        
        // Store texture data
        const textureData = {
            imageId: imageId,
            texture: texture,
            mesh: mesh,
            material: material,
            geometry: geometry,
            plane: mesh
        };
        
        this.textures.push(textureData);
        this.textureGroup.add(mesh);
        
        return textureData;
    },
    
    // Update all textures from storage
    async updateFromStorage() {
        if (!scene3d || !this.textureGroup) return;
        
        try {
            const response = await fetch("/api/storage");
            if (!response.ok) return;
            
            const data = await response.json();
            if (!data.success || !data.images) return;
            
            // Get current texture IDs
            const currentIds = new Set(this.textures.map(t => t.imageId));
            
            // Add new textures
            for (const img of data.images) {
                if (!currentIds.has(img.image_id)) {
                    // Fetch full image data
                    const imgResponse = await fetch(`/api/storage/${img.image_id}`);
                    if (imgResponse.ok) {
                        const imgData = await imgResponse.json();
                        if (imgData.success && imgData.image_base64) {
                            this.addTexture(img.image_id, imgData.image_base64, scene3d);
                        }
                    }
                }
            }
        } catch (error) {
            console.error("Error updating textures:", error);
        }
    },
    
    // Clear all textures
    clear: function(scene) {
        this.textures.forEach(item => {
            if (item.mesh) scene.remove(item.mesh);
            if (item.texture) item.texture.dispose();
            if (item.material) item.material.dispose();
            if (item.geometry) item.geometry.dispose();
        });
        this.textures = [];
        if (this.textureGroup) {
            scene.remove(this.textureGroup);
            this.textureGroup = new THREE.Group();
            scene.add(this.textureGroup);
        }
    }
};

document.addEventListener("DOMContentLoaded", () => {
    // Initialize 3D Field
    const canvas3d = document.getElementById("3d-field");
    if (canvas3d && typeof THREE !== 'undefined') {
        scene3d = new THREE.Scene();
        scene3d.background = new THREE.Color(0xffffff);
        
        camera3d = new THREE.PerspectiveCamera(60, 1, 0.01, 100);
        camera3d.position.set(0, 0, 3);
        camera3d.lookAt(0, 0, 0);
        
        renderer3d = new THREE.WebGLRenderer({ canvas: canvas3d, antialias: true });
        renderer3d.setSize(512, 512);
        renderer3d.setPixelRatio(window.devicePixelRatio);
        // Use sRGB encoding for better color representation (compatible with older Three.js)
        if (THREE.sRGBEncoding !== undefined) {
            renderer3d.outputEncoding = THREE.sRGBEncoding;
        }
        // Ensure proper color precision
        renderer3d.physicallyCorrectLights = false; // Keep false for MeshBasicMaterial
        renderer3d.toneMapping = THREE.NoToneMapping; // No tone mapping for accurate colors
        
        // Set border style explicitly (inline styles override CSS)
        canvas3d.style.border = '2px solid #ff0000';
        
        // Initialize texture manager
        TEXTURE_MANAGER.init(scene3d);
        
        // Create wireframe cube
        const cubeGeometry = new THREE.BoxGeometry(2, 2, 2);
        const cubeMaterial = new THREE.MeshBasicMaterial({ 
            color: 0x333333, 
            wireframe: true 
        });
        const cube = new THREE.Mesh(cubeGeometry, cubeMaterial);
        scene3d.add(cube);
        
        // Create XY grid at z=0
        const gridSize = 2.0;
        const gridDivisions = 20;
        const gridHelper = new THREE.GridHelper(gridSize, gridDivisions, 0xcccccc, 0xcccccc);
        gridHelper.position.z = 0;
        scene3d.add(gridHelper);
        
        // Add OrbitControls for mouse interaction
        if (typeof THREE.OrbitControls !== 'undefined') {
            controls3d = new THREE.OrbitControls(camera3d, renderer3d.domElement);
            controls3d.enableDamping = true;
            controls3d.dampingFactor = 0.05;
            controls3d.enableZoom = true;
            controls3d.enableRotate = true;
            controls3d.enablePan = true; // Enable panning for normal camera controls
            controls3d.target.set(0, 0, 0); // Set target to origin
            
            function animate() {
                requestAnimationFrame(animate);
                controls3d.update();
                renderer3d.render(scene3d, camera3d);
            }
            animate();
        } else {
            // Fallback to basic mouse controls if OrbitControls not available
            let isDragging = false;
            let previousMousePosition = { x: 0, y: 0 };
            let cameraDistance = 3;
            
            canvas3d.addEventListener('mousedown', (e) => {
                isDragging = true;
                previousMousePosition = { x: e.clientX, y: e.clientY };
            });
            
            canvas3d.addEventListener('mousemove', (e) => {
                if (isDragging) {
                    const deltaX = e.clientX - previousMousePosition.x;
                    const deltaY = e.clientY - previousMousePosition.y;
                    
                    const spherical = new THREE.Spherical();
                    spherical.setFromVector3(camera3d.position);
                    spherical.theta -= deltaX * 0.01;
                    spherical.phi += deltaY * 0.01;
                    spherical.phi = Math.max(0.1, Math.min(Math.PI - 0.1, spherical.phi));
                    
                    camera3d.position.setFromSpherical(spherical);
                    camera3d.lookAt(0, 0, 0);
                    
                    previousMousePosition = { x: e.clientX, y: e.clientY };
                }
            });
            
            canvas3d.addEventListener('mouseup', () => {
                isDragging = false;
            });
            
            canvas3d.addEventListener('wheel', (e) => {
                e.preventDefault();
                cameraDistance += e.deltaY * 0.01;
                cameraDistance = Math.max(1, Math.min(10, cameraDistance));
                camera3d.position.setLength(cameraDistance);
                camera3d.lookAt(0, 0, 0);
            });
            
            function animate() {
                requestAnimationFrame(animate);
                renderer3d.render(scene3d, camera3d);
            }
            animate();
        }
    }
    
    const form = document.getElementById("generate-form");
    const status = document.getElementById("status");
    const generationDashboard = document.getElementById("generation-dashboard");
    const generateBtn = document.getElementById("generate-btn");
    const generateBtnText = document.getElementById("generate-btn-text");
    const generate2Btn = document.getElementById("generate2-btn");
    const generate2BtnText = document.getElementById("generate2-btn-text");
    const generate3Btn = document.getElementById("generate3-btn");
    const generate3BtnText = document.getElementById("generate3-btn-text");
    const generate4Btn = document.getElementById("generate4-btn");
    const generate4BtnText = document.getElementById("generate4-btn-text");
    const generateDepthBtn = document.getElementById("generate-depth-btn");
    const generateDepthBtnText = document.getElementById("generate-depth-btn-text");
    const generate3dFieldBtn = document.getElementById("generate-3d-field-btn");
    const generate3dFieldBtnText = document.getElementById("generate-3d-field-btn-text");
    const generateControlnetBtn = document.getElementById("generate-controlnet-btn");
    const generateControlnetBtnText = document.getElementById("generate-controlnet-btn-text");
    const pointsToMeshBtn = document.getElementById("points-to-mesh-btn");
    const pointsToMeshBtnText = document.getElementById("points-to-mesh-btn-text");
    const outpaintBtn = document.getElementById("outpaint-btn");
    const outpaintBtnText = document.getElementById("outpaint-btn-text");
    const outpaintSideSelect = document.getElementById("outpaint-side-select");
    const storageDropdown = document.getElementById("storage-dropdown");
    const refreshStorageBtn = document.getElementById("refresh-storage-btn");
    const storageCount = document.getElementById("storage-count");
    const centerImageBtn = document.getElementById("center-image-btn");
    const centerCubeBtn = document.getElementById("center-cube-btn");

    
    
    // Function to load and display an image by ID
    async function loadImageById(imageId) {
        if (!imageId) return;
        
        try {
            const response = await fetch(`/api/storage/${imageId}`);
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
            }
            
            const data = await response.json();
            if (data.success) {
                CURRENT_TEXTURE = imageId;
                
                // Display the image
                generationDashboard.innerHTML = `
                    <img id="generated-image" src="${data.image_base64}" alt="Generated image" />
                `;
                
                // Add texture to 3D scene
                if (scene3d) {
                    TEXTURE_MANAGER.addTexture(imageId, data.image_base64, scene3d);
                }
            }
        } catch (error) {
            console.error("Error loading image:", error);
            status.className = "mt-3 small fw-semibold text-danger";
            status.textContent = `ERROR LOADING IMAGE: ${error.message}`;
        }
    }

    // Function to refresh storage dropdown
    async function refreshStorage() {
        try {
            const response = await fetch("/api/storage");
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
            }
            
            const data = await response.json();
            if (data.success) {
                // Update count
                storageCount.textContent = `${data.count} images in memory`;
                
                // Clear and populate dropdown
                storageDropdown.innerHTML = '<option value="">SELECT IMAGE</option>';
                
                data.images.forEach((img) => {
                    const option = document.createElement("option");
                    option.value = img.image_id;
                    // Use custom name if available, otherwise use "Image {index}"
                    option.textContent = img.name || `Image ${img.index}`;
                    storageDropdown.appendChild(option);
                });
                
                // If no current texture and images exist, load the first one (black image)
                if (!CURRENT_TEXTURE && data.images.length > 0) {
                    const firstImageId = data.images[0].image_id;
                    storageDropdown.value = firstImageId;
                    loadImageById(firstImageId);
                }
            }
        } catch (error) {
            console.error("Error refreshing storage:", error);
            storageCount.textContent = "Error loading storage";
        }
    }

    // Dropdown change handler
    storageDropdown.addEventListener("change", (e) => {
        const selectedImageId = e.target.value;
        if (selectedImageId) {
            loadImageById(selectedImageId);
        }
    });

    // Refresh button handler
    refreshStorageBtn.addEventListener("click", refreshStorage);
    
    // Center Image button handler - positions camera to fill view with 512x512 image
    centerImageBtn.addEventListener("click", () => {
        if (!camera3d || !controls3d) return;
        
        // Image plane is 2.0 units at z=-1 (back of cube)
        // To fill a 512x512 viewport with 60deg FOV, calculate optimal distance
        // For a plane of size 2.0, we want it to fill the viewport
        // Using FOV calculation: distance = (plane_size/2) / tan(FOV/2)
        const planeSize = 2.0;
        const planeZ = -1; // Texture is at back of cube
        const fovRad = (camera3d.fov * Math.PI) / 180;
        const distance = (planeSize / 2) / Math.tan(fovRad / 2);
        
        // Position camera slightly further to see the whole plane comfortably
        const targetDistance = distance * 1; // 10% margin
        
        // Animate camera to position looking at the image plane at z=-1
        // Camera positioned in front of the texture (positive z direction)
        const targetPosition = new THREE.Vector3(0, 0, planeZ + targetDistance);
        const targetLookAt = new THREE.Vector3(0, 0, planeZ);
        
        // Smooth transition
        const startPosition = camera3d.position.clone();
        const startLookAt = controls3d.target.clone();
        let progress = 0;
        const duration = 500; // 500ms animation
        const startTime = Date.now();
        
        function animateCamera() {
            const elapsed = Date.now() - startTime;
            progress = Math.min(elapsed / duration, 1);
            
            // Easing function (ease-in-out)
            const eased = progress < 0.5 
                ? 2 * progress * progress 
                : 1 - Math.pow(-2 * progress + 2, 2) / 2;
            
            camera3d.position.lerpVectors(startPosition, targetPosition, eased);
            controls3d.target.lerpVectors(startLookAt, targetLookAt, eased);
            controls3d.update();
            
            if (progress < 1) {
                requestAnimationFrame(animateCamera);
            }
        }
        
        animateCamera();
    });
    
    // Center Cube button handler - positions camera to view the front face of the cube
    centerCubeBtn.addEventListener("click", () => {
        if (!camera3d || !controls3d) return;
        
        // Cube is 2x2x2 centered at origin, front face is at z=1
        // Position camera further back to see the cube wall
        const targetPosition = new THREE.Vector3(0, 0, 5);
        const targetLookAt = new THREE.Vector3(0, 0, 1); // Look at front face of cube
        
        // Smooth transition
        const startPosition = camera3d.position.clone();
        const startLookAt = controls3d.target.clone();
        let progress = 0;
        const duration = 500; // 500ms animation
        const startTime = Date.now();
        
        function animateCamera() {
            const elapsed = Date.now() - startTime;
            progress = Math.min(elapsed / duration, 1);
            
            // Easing function (ease-in-out)
            const eased = progress < 0.5 
                ? 2 * progress * progress 
                : 1 - Math.pow(-2 * progress + 2, 2) / 2;
            
            camera3d.position.lerpVectors(startPosition, targetPosition, eased);
            controls3d.target.lerpVectors(startLookAt, targetLookAt, eased);
            controls3d.update();
            
            if (progress < 1) {
                requestAnimationFrame(animateCamera);
            }
        }
        
        animateCamera();
    });
    
   

    
    // Initial load
    refreshStorage().then(() => {
        // Load existing textures into 3D scene after initial storage load
        if (scene3d) {
            TEXTURE_MANAGER.updateFromStorage();
        }
    });
    
    form.addEventListener("submit", async (e) => {
        e.preventDefault();
        
        // Disable button and show loading
        generateBtn.disabled = true;
        generateBtnText.textContent = "GENERATING";

        status.className = "mt-3 small fw-semibold text-info";
        status.textContent = "GENERATING IMAGE PLEASE WAIT";

        try {
            const formData = new FormData(form);
            
            // Fast local POST request
            const startTime = Date.now();
            const response = await fetch("/api/generate", {
                method: "POST",
                body: formData
            });

            if (!response.ok) {
                const errorData = await response.json().catch(() => ({ detail: "Unknown error" }));
                throw new Error(errorData.detail || errorData.error || `HTTP ${response.status}`);
            }
            
            const data = await response.json();
            const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);

            if (data.success) {
                status.className = "mt-3 small fw-semibold text-success";
                status.textContent = `SUCCESS GENERATED IN ${elapsed}S`;
                
                generationDashboard.innerHTML = `
                    <img id="generated-image" src="${data.image_base64}" alt="Generated image" />
                `;
                
                // Update CURRENT_TEXTURE after generation
                CURRENT_TEXTURE = data.image_id;
                
                // Add texture to 3D scene
                if (scene3d) {
                    TEXTURE_MANAGER.addTexture(data.image_id, data.image_base64, scene3d);
                }
                
                // Update dropdown to select the newly generated image
                storageDropdown.value = data.image_id;
                
                // Refresh storage to update count
                refreshStorage();
            } else {
                status.className = "mt-3 small fw-semibold text-danger";
                status.textContent = `ERROR: ${data.error}`;
            }
        } catch (error) {
            status.className = "mt-3 small fw-semibold text-danger";
            status.textContent = `ERROR: ${error.message}`;
        } finally {
            // Re-enable button
            generateBtn.disabled = false;
            generateBtnText.textContent = "GENERATE IMAGE";

        }
    });

    // Generate 2 button handler (base + refiner cascade)
    generate2Btn.addEventListener("click", async (e) => {
        e.preventDefault();
        
        // Disable button and show loading
        generate2Btn.disabled = true;
        generate2BtnText.textContent = "GENERATING";

        status.className = "mt-3 small fw-semibold text-info";
        status.textContent = "GENERATING IMAGE WITH REFINER (BASE + REFINER) PLEASE WAIT";

        try {
            const formData = new FormData(form);
            
            // Fast local POST request to generate2 endpoint
            const startTime = Date.now();
            const response = await fetch("/api/generate2", {
                method: "POST",
                body: formData
            });

            if (!response.ok) {
                const errorData = await response.json().catch(() => ({ detail: "Unknown error" }));
                throw new Error(errorData.detail || errorData.error || `HTTP ${response.status}`);
            }
            
            const data = await response.json();
            const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);

            if (data.success) {
                status.className = "mt-3 small fw-semibold text-success";
                status.textContent = `SUCCESS GENERATED WITH REFINER IN ${elapsed}S`;
                
                generationDashboard.innerHTML = `
                    <img id="generated-image" src="${data.image_base64}" alt="Generated image" />
                `;
                
                // Update CURRENT_TEXTURE after generation
                CURRENT_TEXTURE = data.image_id;
                
                // Add texture to 3D scene
                if (scene3d) {
                    TEXTURE_MANAGER.addTexture(data.image_id, data.image_base64, scene3d);
                }
                
                // Update dropdown to select the newly generated image
                storageDropdown.value = data.image_id;
                
                // Refresh storage to update count
                refreshStorage();
            } else {
                status.className = "mt-3 small fw-semibold text-danger";
                status.textContent = `ERROR: ${data.error}`;
            }
        } catch (error) {
            status.className = "mt-3 small fw-semibold text-danger";
            status.textContent = `ERROR: ${error.message}`;
        } finally {
            // Re-enable button
            generate2Btn.disabled = false;
            generate2BtnText.textContent = "GENERATE 2";
        }
    });
    
    // Generate 3 button handler (zoom and img2img with same prompt)
    generate3Btn.addEventListener("click", async (e) => {
        e.preventDefault();
        
        // Check if an image is selected
        if (!CURRENT_TEXTURE) {
            status.className = "mt-3 small fw-semibold text-danger";
            status.textContent = "ERROR: Please select an image first";
            return;
        }
        
        // Disable button and show loading
        generate3Btn.disabled = true;
        generate3BtnText.textContent = "GENERATING";

        status.className = "mt-3 small fw-semibold text-info";
        status.textContent = "ZOOMING IMAGE AND APPLYING IMG2IMG WITH SAME PROMPT PLEASE WAIT";

        try {
            // Get the prompt from the selected image
            let imagePrompt = "";
            try {
                const storageResponse = await fetch("/api/storage");
                if (storageResponse.ok) {
                    const storageData = await storageResponse.json();
                    if (storageData.success) {
                        const selectedImage = storageData.images.find(img => img.image_id === CURRENT_TEXTURE);
                        if (selectedImage && selectedImage.prompt) {
                            imagePrompt = selectedImage.prompt;
                        }
                    }
                }
            } catch (err) {
                console.warn("Could not fetch prompt from storage:", err);
            }
            
            const formData = new FormData(form);
            formData.append("image_id", CURRENT_TEXTURE);
            // Use the prompt from the selected image (backend will use it if not provided)
            if (imagePrompt) {
                formData.set("prompt", imagePrompt);
            }
            
            // Fast local POST request to generate3 endpoint
            const startTime = Date.now();
            const response = await fetch("/api/generate3", {
                method: "POST",
                body: formData
            });

            if (!response.ok) {
                const errorData = await response.json().catch(() => ({ detail: "Unknown error" }));
                throw new Error(errorData.detail || errorData.error || `HTTP ${response.status}`);
            }
            
            const data = await response.json();
            const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);

            if (data.success) {
                status.className = "mt-3 small fw-semibold text-success";
                status.textContent = `SUCCESS GENERATED IN ${elapsed}S`;
                
                generationDashboard.innerHTML = `
                    <img id="generated-image" src="${data.image_base64}" alt="Generated image" />
                `;
                
                // Update CURRENT_TEXTURE after generation
                CURRENT_TEXTURE = data.image_id;
                
                // Add texture to 3D scene
                if (scene3d) {
                    TEXTURE_MANAGER.addTexture(data.image_id, data.image_base64, scene3d);
                }
                
                // Update dropdown to select the newly generated image
                storageDropdown.value = data.image_id;
                
                // Refresh storage to update count
                refreshStorage();
            } else {
                status.className = "mt-3 small fw-semibold text-danger";
                status.textContent = `ERROR: ${data.error}`;
            }
        } catch (error) {
            status.className = "mt-3 small fw-semibold text-danger";
            status.textContent = `ERROR: ${error.message}`;
        } finally {
            // Re-enable button
            generate3Btn.disabled = false;
            generate3BtnText.textContent = "GENERATE 3";
        }
    });
    
    // Generate ControlNet button handler
    generateControlnetBtn.addEventListener("click", async (e) => {
        e.preventDefault();
        
        // Check if an image is selected
        if (!CURRENT_TEXTURE) {
            status.className = "mt-3 small fw-semibold text-danger";
            status.textContent = "ERROR: Please select an image first";
            return;
        }
        
        // Disable button and show loading
        generateControlnetBtn.disabled = true;
        generateControlnetBtnText.textContent = "GENERATING";

        status.className = "mt-3 small fw-semibold text-info";
        status.textContent = "GENERATING CONTROLNET WITH 0.9X ZOOM PLEASE WAIT";

        try {
            const formData = new FormData(form);
            formData.append("image_id", CURRENT_TEXTURE);
            
            // Fast local POST request to generate_controlnet endpoint
            const startTime = Date.now();
            const response = await fetch("/api/generate_controlnet", {
                method: "POST",
                body: formData
            });

            if (!response.ok) {
                const errorData = await response.json().catch(() => ({ detail: "Unknown error" }));
                throw new Error(errorData.detail || errorData.error || `HTTP ${response.status}`);
            }
            
            const data = await response.json();
            const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);

            if (data.success) {
                status.className = "mt-3 small fw-semibold text-success";
                status.textContent = `SUCCESS GENERATED IN ${elapsed}S`;
                
                generationDashboard.innerHTML = `
                    <img id="generated-image" src="${data.image_base64}" alt="Generated image" />
                `;
                
                // Update CURRENT_TEXTURE after generation
                CURRENT_TEXTURE = data.image_id;
                
                // Add texture to 3D scene
                if (scene3d) {
                    TEXTURE_MANAGER.addTexture(data.image_id, data.image_base64, scene3d);
                }
                
                // Update dropdown to select the newly generated image
                storageDropdown.value = data.image_id;
                
                // Refresh storage to update count
                refreshStorage();
            } else {
                status.className = "mt-3 small fw-semibold text-danger";
                status.textContent = `ERROR: ${data.error}`;
            }
        } catch (error) {
            status.className = "mt-3 small fw-semibold text-danger";
            status.textContent = `ERROR: ${error.message}`;
        } finally {
            // Re-enable button
            generateControlnetBtn.disabled = false;
            generateControlnetBtnText.textContent = "Generate ControlNet";
        }
    });
    
    // Generate 4 button handler (image-to-image refiner)
    if (generate4Btn) {
        generate4Btn.addEventListener("click", async (e) => {
        e.preventDefault();
        
        // Check if an image is selected
        if (!CURRENT_TEXTURE) {
            status.className = "mt-3 small fw-semibold text-danger";
            status.textContent = "ERROR: Please select an image first";
            return;
        }
        
        // Disable button and show loading
        generate4Btn.disabled = true;
        generate4BtnText.textContent = "GENERATING";

        status.className = "mt-3 small fw-semibold text-info";
        status.textContent = "REFINING IMAGE WITH 20% DIFFUSION PLEASE WAIT";

        try {
            const formData = new FormData(form);
            formData.append("image_id", CURRENT_TEXTURE);
            
            // Fast local POST request to generate_img2img endpoint
            const startTime = Date.now();
            const response = await fetch("/api/generate_img2img", {
                method: "POST",
                body: formData
            });

            if (!response.ok) {
                const errorData = await response.json().catch(() => ({ detail: "Unknown error" }));
                throw new Error(errorData.detail || errorData.error || `HTTP ${response.status}`);
            }
            
            const data = await response.json();
            const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);

            if (data.success) {
                status.className = "mt-3 small fw-semibold text-success";
                status.textContent = `SUCCESS REFINED IN ${elapsed}S`;
                
                generationDashboard.innerHTML = `
                    <img id="generated-image" src="${data.image_base64}" alt="Generated image" />
                `;
                
                // Update CURRENT_TEXTURE after generation
                CURRENT_TEXTURE = data.image_id;
                
                // Add texture to 3D scene
                if (scene3d) {
                    TEXTURE_MANAGER.addTexture(data.image_id, data.image_base64, scene3d);
                }
                
                // Update dropdown to select the newly generated image
                storageDropdown.value = data.image_id;
                
                // Refresh storage to update count
                refreshStorage();
            } else {
                status.className = "mt-3 small fw-semibold text-danger";
                status.textContent = `ERROR: ${data.error}`;
            }
        } catch (error) {
            status.className = "mt-3 small fw-semibold text-danger";
            status.textContent = `ERROR: ${error.message}`;
        } finally {
            // Re-enable button
            generate4Btn.disabled = false;
            generate4BtnText.textContent = "GENERATE 4";
        }
        });
    }
    
    // Generate Depth button handler
    generateDepthBtn.addEventListener("click", async (e) => {
        e.preventDefault();
        
        // Check if an image is selected
        if (!CURRENT_TEXTURE) {
            status.className = "mt-3 small fw-semibold text-danger";
            status.textContent = "ERROR: Please select an image first";
            return;
        }
        
        // Disable button and show loading
        generateDepthBtn.disabled = true;
        generateDepthBtnText.textContent = "GENERATING";

        status.className = "mt-3 small fw-semibold text-info";
        status.textContent = "GENERATING DEPTH MAP PLEASE WAIT";

        try {
            const formData = new FormData();
            formData.append("image_id", CURRENT_TEXTURE);
            
            // Fast local POST request to generate_depth endpoint
            const startTime = Date.now();
            const response = await fetch("/api/generate_depth", {
                method: "POST",
                body: formData
            });

            if (!response.ok) {
                const errorData = await response.json().catch(() => ({ detail: "Unknown error" }));
                throw new Error(errorData.detail || errorData.error || `HTTP ${response.status}`);
            }
            
            const data = await response.json();
            const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);

            if (data.success) {
                status.className = "mt-3 small fw-semibold text-success";
                status.textContent = `SUCCESS GENERATED DEPTH MAP IN ${elapsed}S`;
                
                generationDashboard.innerHTML = `
                    <img id="generated-image" src="${data.image_base64}" alt="Generated depth map" />
                `;
                
                // Update CURRENT_TEXTURE after generation
                CURRENT_TEXTURE = data.image_id;
                
                // Add texture to 3D scene
                if (scene3d) {
                    TEXTURE_MANAGER.addTexture(data.image_id, data.image_base64, scene3d);
                }
                
                // Update dropdown to select the newly generated image
                storageDropdown.value = data.image_id;
                
                // Refresh storage to update count
                refreshStorage();
            } else {
                status.className = "mt-3 small fw-semibold text-danger";
                status.textContent = `ERROR: ${data.error}`;
            }
        } catch (error) {
            status.className = "mt-3 small fw-semibold text-danger";
            status.textContent = `ERROR: ${error.message}`;
        } finally {
            // Re-enable button
            generateDepthBtn.disabled = false;
            generateDepthBtnText.textContent = "GENERATE DEPTH";
        }
    });
    
    // Generate 3D Field button handler
    generate3dFieldBtn.addEventListener("click", async (e) => {
        e.preventDefault();
        
        // Check if an image is selected
        if (!CURRENT_TEXTURE) {
            status.className = "mt-3 small fw-semibold text-danger";
            status.textContent = "ERROR: Please select an image first";
            return;
        }
        
        // Disable button and show loading
        generate3dFieldBtn.disabled = true;
        generate3dFieldBtnText.textContent = "GENERATING";

        status.className = "mt-3 small fw-semibold text-info";
        status.textContent = "GENERATING 3D FIELD FROM DEPTH PLEASE WAIT";

        try {
            const formData = new FormData();
            formData.append("image_id", CURRENT_TEXTURE);
            
            // Fast local POST request to generate_3d_field endpoint
            const startTime = Date.now();
            const response = await fetch("/api/generate_3d_field", {
                method: "POST",
                body: formData
            });

            if (!response.ok) {
                const errorData = await response.json().catch(() => ({ detail: "Unknown error" }));
                throw new Error(errorData.detail || errorData.error || `HTTP ${response.status}`);
            }
            
            const data = await response.json();
            const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);

            if (data.success) {
                status.className = "mt-3 small fw-semibold text-success";
                status.textContent = `SUCCESS GENERATED 3D FIELD (${data.num_points} points) IN ${elapsed}S`;
                
                // Remove previous point cloud if exists
                if (pointCloud3d && scene3d) {
                    scene3d.remove(pointCloud3d);
                    if (pointCloud3d.geometry) pointCloud3d.geometry.dispose();
                    if (pointCloud3d.material) pointCloud3d.material.dispose();
                    pointCloud3d = null;
                }
                
                // Create Three.js point cloud
                if (scene3d && data.points && data.colors) {
                    // Store point cloud data for mesh creation
                    pointCloudData = {
                        points: data.points,
                        colors: data.colors
                    };
                    
                    const geometry = new THREE.BufferGeometry();
                    
                    // Convert points and colors to Float32Arrays
                    const positions = new Float32Array(data.points.length * 3);
                    const colors = new Float32Array(data.colors.length * 3);
                    
                    for (let i = 0; i < data.points.length; i++) {
                        positions[i * 3] = data.points[i][0];
                        positions[i * 3 + 1] = data.points[i][1];
                        positions[i * 3 + 2] = data.points[i][2];
                        
                        colors[i * 3] = data.colors[i][0];
                        colors[i * 3 + 1] = data.colors[i][1];
                        colors[i * 3 + 2] = data.colors[i][2];
                    }
                    
                    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
                    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
                    
                    const material = new THREE.PointsMaterial({
                        size: 0.01,
                        vertexColors: true,
                        transparent: false,
                        opacity: 1.0
                    });
                    
                    pointCloud3d = new THREE.Points(geometry, material);
                    scene3d.add(pointCloud3d);
                    
                    // Remove any existing mesh when new point cloud is created
                    if (pointCloudMesh3d && scene3d) {
                        scene3d.remove(pointCloudMesh3d);
                        if (pointCloudMesh3d.geometry) pointCloudMesh3d.geometry.dispose();
                        if (pointCloudMesh3d.material) pointCloudMesh3d.material.dispose();
                        pointCloudMesh3d = null;
                    }
                }
            } else {
                status.className = "mt-3 small fw-semibold text-danger";
                status.textContent = `ERROR: ${data.error}`;
            }
        } catch (error) {
            status.className = "mt-3 small fw-semibold text-danger";
            status.textContent = `ERROR: ${error.message}`;
        } finally {
            // Re-enable button
            generate3dFieldBtn.disabled = false;
            generate3dFieldBtnText.textContent = "GENERATE 3D FIELD";
        }
    });
    
    // Outpaint button handler
    if (outpaintBtn) {
        outpaintBtn.addEventListener("click", async (e) => {
            e.preventDefault();
            
            // Check if an image is selected
            if (!CURRENT_TEXTURE) {
                status.className = "mt-3 small fw-semibold text-danger";
                status.textContent = "ERROR: Please select an image first";
                return;
            }
            
            // Get selected side
            const side = outpaintSideSelect ? outpaintSideSelect.value : "right";
            
            // Disable button and show loading
            outpaintBtn.disabled = true;
            outpaintBtnText.textContent = "OUTPAINTING";
            
            status.className = "mt-3 small fw-semibold text-info";
            status.textContent = `OUTPAINTING IMAGE TO THE ${side.toUpperCase()} PLEASE WAIT`;
            
            try {
                const formData = new FormData();
                formData.append("image_id", CURRENT_TEXTURE);
                formData.append("side", side);
                formData.append("add", "512"); // Default add 512 pixels
                
                // Fast local POST request to outpaint endpoint
                const startTime = Date.now();
                const response = await fetch("/api/outpaint", {
                    method: "POST",
                    body: formData
                });
                
                if (!response.ok) {
                    const errorData = await response.json().catch(() => ({ detail: "Unknown error" }));
                    throw new Error(errorData.detail || errorData.error || `HTTP ${response.status}`);
                }
                
                const data = await response.json();
                const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
                
                if (data.success) {
                    status.className = "mt-3 small fw-semibold text-success";
                    status.textContent = `SUCCESS OUTPAINTED TO THE ${side.toUpperCase()} IN ${elapsed}S`;
                    
                    generationDashboard.innerHTML = `
                        <img id="generated-image" src="${data.image_base64}" alt="Outpainted image" />
                    `;
                    
                    // Update CURRENT_TEXTURE after outpaint
                    const previousImageId = CURRENT_TEXTURE;
                    CURRENT_TEXTURE = data.image_id;
                    
                    // Add outpainted texture to 3D scene positioned relative to previous texture
                    if (scene3d) {
                        TEXTURE_MANAGER.addOutpaintTexture(data.image_id, data.image_base64, scene3d, side, previousImageId);
                    }
                    
                    // Update dropdown to select the newly outpainted image
                    storageDropdown.value = data.image_id;
                    
                    // Refresh storage to update count
                    refreshStorage();
                } else {
                    status.className = "mt-3 small fw-semibold text-danger";
                    status.textContent = `ERROR: ${data.error}`;
                }
            } catch (error) {
                status.className = "mt-3 small fw-semibold text-danger";
                status.textContent = `ERROR: ${error.message}`;
            } finally {
                // Re-enable button
                outpaintBtn.disabled = false;
                outpaintBtnText.textContent = "OUTPAINT";
            }
        });
    }
    
    // Points-to-Mesh button handler - toggle mesh visibility
    if (pointsToMeshBtn) {
        // Function to create mesh from point cloud
        function createMeshFromPointCloud() {
            if (!pointCloudData || !scene3d) {
                status.className = "mt-3 small fw-semibold text-danger";
                status.textContent = "ERROR: No point cloud data available. Generate 3D Field first.";
                return;
            }
            
            const points = pointCloudData.points;
            const colors = pointCloudData.colors;
            
            if (!points || points.length === 0) {
                status.className = "mt-3 small fw-semibold text-danger";
                status.textContent = "ERROR: Point cloud data is empty.";
                return;
            }
            
            // Create geometry from points
            const geometry = new THREE.BufferGeometry();
            
            // Create mesh from point cloud
            // Points come from a depth map (2D image), so we need to reconstruct the grid structure
            // by sorting points by their Y (vertical) then X (horizontal) coordinates
            try {
                const numPoints = points.length;
                
                // Create array with indices for sorting
                const indexedPoints = points.map((p, idx) => ({
                    point: p,
                    color: colors[idx],
                    index: idx
                }));
                
                // Sort points by Y (rows) then X (columns) to reconstruct grid
                // Y is typically the second coordinate, but we need to check the coordinate system
                // In Three.js, Y is usually up, so we sort by Y first (rows), then X (columns)
                indexedPoints.sort((a, b) => {
                    // Sort by Y first (vertical position - rows)
                    const yDiff = a.point[1] - b.point[1];
                    if (Math.abs(yDiff) > 0.001) {
                        return yDiff;
                    }
                    // Then by X (horizontal position - columns)
                    return a.point[0] - b.point[0];
                });
                
                // Rebuild positions and colors arrays in sorted order
                const sortedPositions = new Float32Array(numPoints * 3);
                const sortedColors = new Float32Array(numPoints * 3);
                const indexMap = new Map(); // Map old index to new index
                
                for (let i = 0; i < numPoints; i++) {
                    const item = indexedPoints[i];
                    sortedPositions[i * 3] = item.point[0];
                    sortedPositions[i * 3 + 1] = item.point[1];
                    sortedPositions[i * 3 + 2] = item.point[2];
                    
                    sortedColors[i * 3] = item.color[0];
                    sortedColors[i * 3 + 1] = item.color[1];
                    sortedColors[i * 3 + 2] = item.color[2];
                    
                    indexMap.set(item.index, i);
                }
                
                geometry.setAttribute('position', new THREE.BufferAttribute(sortedPositions, 3));
                geometry.setAttribute('color', new THREE.BufferAttribute(sortedColors, 3));
                
                // Detect grid dimensions by finding unique Y values (rows)
                const uniqueYs = [];
                let lastY = null;
                for (let i = 0; i < numPoints; i++) {
                    const y = sortedPositions[i * 3 + 1];
                    if (lastY === null || Math.abs(y - lastY) > 0.001) {
                        uniqueYs.push(y);
                        lastY = y;
                    }
                }
                
                const numRows = uniqueYs.length;
                const numCols = Math.floor(numPoints / numRows);
                
                // Create grid-based triangulation
                const indices = [];
                
                // Create a 2D grid mapping: grid[row][col] = point index
                // First, group points by Y (rows)
                const rowsMap = new Map();
                for (let i = 0; i < numPoints; i++) {
                    const y = sortedPositions[i * 3 + 1];
                    // Round Y to group nearby points (handles floating point precision)
                    const yKey = Math.round(y * 1000) / 1000;
                    
                    if (!rowsMap.has(yKey)) {
                        rowsMap.set(yKey, []);
                    }
                    rowsMap.get(yKey).push(i);
                }
                
                // Convert to array and sort each row by X coordinate
                const grid = [];
                const sortedYKeys = Array.from(rowsMap.keys()).sort((a, b) => a - b);
                
                for (const yKey of sortedYKeys) {
                    const rowIndices = rowsMap.get(yKey);
                    // Sort row by X coordinate
                    rowIndices.sort((a, b) => {
                        return sortedPositions[a * 3] - sortedPositions[b * 3];
                    });
                    grid.push(rowIndices);
                }
                
                // Create triangles connecting adjacent points in the grid
                // Connect each point to its neighbors in the same row and next row
                for (let row = 0; row < grid.length - 1; row++) {
                    if (!grid[row] || !grid[row + 1]) continue;
                    
                    const rowPoints = grid[row];
                    const nextRowPoints = grid[row + 1];
                    
                    // For each point in current row, connect to adjacent points
                    for (let col = 0; col < rowPoints.length; col++) {
                        const idx = rowPoints[col];
                        
                        // Connect to point to the right in same row
                        if (col + 1 < rowPoints.length) {
                            const rightIdx = rowPoints[col + 1];
                            
                            // Find corresponding points in next row
                            // Match by column position (assuming similar grid structure)
                            const nextCol = Math.min(col, nextRowPoints.length - 1);
                            const nextColRight = Math.min(col + 1, nextRowPoints.length - 1);
                            
                            if (nextCol < nextRowPoints.length && nextColRight < nextRowPoints.length) {
                                const nextIdx = nextRowPoints[nextCol];
                                const nextRightIdx = nextRowPoints[nextColRight];
                                
                                // Create two triangles forming a quad
                                // Triangle 1: current, next, right
                                indices.push(idx, nextIdx, rightIdx);
                                // Triangle 2: right, next, nextRight
                                indices.push(rightIdx, nextIdx, nextRightIdx);
                            }
                        }
                    }
                }
                
                if (indices.length === 0) {
                    // Fallback: Simple triangulation if grid detection fails
                    for (let i = 0; i < numPoints - 2; i += 1) {
                        if (i + 2 < numPoints) {
                            indices.push(i, i + 1, i + 2);
                        }
                    }
                }
                
                if (indices.length === 0) {
                    throw new Error("Could not create mesh indices from point cloud");
                }
                
                geometry.setIndex(indices);
                geometry.computeVertexNormals();
                
                const material = new THREE.MeshBasicMaterial({
                    vertexColors: true,
                    side: THREE.DoubleSide,
                    transparent: false,
                    opacity: 1.0,
                    wireframe: false,
                    flatShading: false
                });
                
                pointCloudMesh3d = new THREE.Mesh(geometry, material);
                
                scene3d.add(pointCloudMesh3d);
                pointCloudMesh3d.visible = true;
                pointsToMeshBtnText.textContent = "POINTS-TO-MESH (ON)";
                status.className = "mt-3 small fw-semibold text-success";
                status.textContent = "Mesh created and visible";
            } catch (error) {
                console.error("Error creating mesh:", error);
                status.className = "mt-3 small fw-semibold text-danger";
                status.textContent = `ERROR: Failed to create mesh: ${error.message}`;
            }
        }
        
        pointsToMeshBtn.addEventListener("click", (e) => {
            e.preventDefault();
            
            if (!pointCloudData) {
                status.className = "mt-3 small fw-semibold text-danger";
                status.textContent = "ERROR: No point cloud data. Generate 3D Field first.";
                return;
            }
            
            // Check current mesh visibility state
            const isCurrentlyVisible = pointCloudMesh3d && pointCloudMesh3d.visible;
            
            if (!isCurrentlyVisible) {
                // Create and show mesh
                if (!pointCloudMesh3d) {
                    createMeshFromPointCloud();
                } else {
                    // Mesh exists, just show it
                    pointCloudMesh3d.visible = true;
                    pointsToMeshBtnText.textContent = "POINTS-TO-MESH (ON)";
                    status.className = "mt-3 small fw-semibold text-success";
                    status.textContent = "Mesh visible";
                }
            } else {
                // Hide mesh
                if (pointCloudMesh3d) {
                    pointCloudMesh3d.visible = false;
                    pointsToMeshBtnText.textContent = "POINTS-TO-MESH (OFF)";
                    status.className = "mt-3 small fw-semibold text-info";
                    status.textContent = "Mesh hidden";
                }
            }
        });
    }
});


