#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings


warnings.filterwarnings('ignore')

# ==========================================
# 1. METRIC FUNCTIONS (PSNR & SSIM)
# ==========================================
def calculate_psnr(original, compressed):
    """Calculate Peak Signal-to-Noise Ratio (PSNR)."""
    mse = np.mean((original.astype(float) - compressed.astype(float)) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

def calculate_ssim(img1, img2):
    """Calculate Structural Similarity Index (SSIM)."""
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    
    img1 = img1.astype(float)
    img2 = img2.astype(float)
    
    mu1 = np.mean(img1)
    mu2 = np.mean(img2)
    
    sigma1_sq = np.var(img1)
    sigma2_sq = np.var(img2)
    sigma12 = np.cov(img1.flatten(), img2.flatten())[0, 1]
    
    numerator = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2)
    
    return numerator / denominator

# ==========================================
# 2. FILTER CREATION FUNCTIONS
# ==========================================
def create_lowpass_filter(shape, radius):
    """Creates a Low-Pass Filter (keeps center frequencies)."""
    rows, cols = shape
   
    crow, ccol = rows // 2, cols // 2
    mask = np.zeros((rows, cols), np.uint8)
    cv2.circle(mask, (ccol, crow), radius, 1, thickness=-1)
    return mask

def create_highpass_filter(shape, radius):
    """Creates a High-Pass Filter (removes center frequencies)."""
    return 1 - create_lowpass_filter(shape, radius)

def create_bandpass_filter(shape, radius_in, radius_out):
    """Creates a Band-Pass Filter (keeps a ring of frequencies)."""
    rows, cols = shape
  
    crow, ccol = rows // 2, cols // 2
    mask = np.zeros((rows, cols), np.uint8)
    cv2.circle(mask, (ccol, crow), radius_out, 1, thickness=-1)
    cv2.circle(mask, (ccol, crow), radius_in, 0, thickness=-1)
    return mask

def apply_filter(image, mask):
    """Applies a frequency domain mask to an image."""
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    fshift_filtered = fshift * mask
    f_ishift = np.fft.ifftshift(fshift_filtered)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    return np.clip(img_back, 0, 255).astype(np.uint8)

# ==========================================
# 3. VISUALIZATION FUNCTIONS
# ==========================================
def visualize_spectrum(image, title):
    """Visualizes the Magnitude and Phase Spectrum."""
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    
    # Magnitude Spectrum (Log scale)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
    # Phase Spectrum
    phase_spectrum = np.angle(fshift)
    
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1); plt.imshow(image, cmap='gray'); plt.title(f'Original: {title}')
    plt.axis('off')
    plt.subplot(1, 3, 2); plt.imshow(magnitude_spectrum, cmap='gray'); plt.title('Magnitude Spectrum')
    plt.axis('off')
    plt.subplot(1, 3, 3); plt.imshow(phase_spectrum, cmap='gray'); plt.title('Phase Spectrum')
    plt.axis('off')
    plt.show()

# ==========================================
# 4. MAIN PROJECT EXECUTION
# ==========================================

# A. LOAD IMAGES

image_files = {
    "Urban": r"C:\Users\Rana\Desktop\urban.jpg", 
    "Forest": r"C:\Users\Rana\Desktop\forset.jpg", 
    "Flowers": r"C:\Users\Rana\Desktop\flowers.jpg"
}

project_images = {}

print("--- 1. LOADING IMAGES ---")
for title, filename in image_files.items():
    if os.path.exists(filename):
        img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            project_images[title] = img
            print(f"Loaded {title}")
        else:
            print(f"Error reading {filename}")
    else:
        print(f"File not found: {filename} ")

if not project_images:
    print("No images found. Stopping.")
else:
    # B. VISUALIZE SPECTRUMS 
    print("\n--- 2. VISUALIZING SPECTRUMS ---")
    for title, image in project_images.items():
        visualize_spectrum(image, title)

    # C. DEMONSTRATE FILTERS & APPLICATIONS
    print("\n--- 3. PRACTICAL APPLICATIONS & COMPARISONS ---")
    
    for title, image in project_images.items():
        print(f"\nProcessing {title}...")
        
        # --- Task 1: NOISE REDUCTION (LPF vs Gaussian) ---
        
        image_float = image.astype(np.float32)
        noise = np.random.normal(0, 20, image.shape).astype(np.float32)
        noisy_image_float = image_float + noise
        
        
        noisy_image = np.clip(noisy_image_float, 0, 255).astype(np.uint8)
        
        # Freq Domain: Low Pass 
        lpf_mask = create_lowpass_filter(image.shape, radius=80)
        freq_denoised = apply_filter(noisy_image, lpf_mask)
        
        # Spatial Domain: Gaussian
        spatial_denoised = cv2.GaussianBlur(noisy_image, (5, 5), 0)
        
        # Metrics
        psnr_freq = calculate_psnr(image, freq_denoised)
        ssim_freq = calculate_ssim(image, freq_denoised)
        psnr_spatial = calculate_psnr(image, spatial_denoised)
        ssim_spatial = calculate_ssim(image, spatial_denoised)
        
        print(f"  [Denoising] Freq LPF PSNR: {psnr_freq:.2f} dB, SSIM: {ssim_freq:.4f}")
        print(f"  [Denoising] Spatial PSNR:  {psnr_spatial:.2f} dB, SSIM: {ssim_spatial:.4f}")

        # Plot Denoising
        plt.figure(figsize=(16, 4))
        plt.subplot(1, 4, 1); plt.imshow(noisy_image, cmap='gray'); plt.title('Noisy Input')
        plt.subplot(1, 4, 2); plt.imshow(freq_denoised, cmap='gray'); plt.title(f'Freq LPF\nPSNR:{psnr_freq:.2f}')
        plt.subplot(1, 4, 3); plt.imshow(spatial_denoised, cmap='gray'); plt.title(f'Spatial Gaussian\nPSNR:{psnr_spatial:.2f}')
        plt.subplot(1, 4, 4); plt.imshow(image, cmap='gray'); plt.title('Ground Truth')
        plt.show()

        # --- Task 2: EDGE ENHANCEMENT (High Pass) ---
        hpf_mask = create_highpass_filter(image.shape, radius=30)
        edges = apply_filter(image, hpf_mask)
        
        # --- Task 3: TEXTURE ISOLATION (Band Pass) ---
        bpf_mask = create_bandpass_filter(image.shape, 30, 90)
        bandpass_output = apply_filter(image, bpf_mask)

        # Plot Filters
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1); plt.imshow(image, cmap='gray'); plt.title('Original')
        plt.subplot(1, 3, 2); plt.imshow(edges, cmap='gray'); plt.title('High Pass (Edges)')
        plt.subplot(1, 3, 3); plt.imshow(bandpass_output, cmap='gray'); plt.title('Band Pass (Texture)')
        plt.show()
        
        print("-" * 50)


# In[ ]:




