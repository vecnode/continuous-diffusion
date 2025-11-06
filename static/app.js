// Fast local communication for image generation
// Global variable to track current displayed image
let CURRENT_TEXTURE = null;

document.addEventListener("DOMContentLoaded", () => {
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
    const storageDropdown = document.getElementById("storage-dropdown");
    const refreshStorageBtn = document.getElementById("refresh-storage-btn");
    const storageCount = document.getElementById("storage-count");

    
    
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
    
   

    
    // Initial load
    refreshStorage();
    
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
    
    // Generate 4 button handler (image-to-image refiner)
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
});


