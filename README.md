# AI Inpainting: Generative Models vs. Traditional Methods

## Overview

This project explores how modern generative AI transforms image inpainting compared to classic techniques. By benchmarking two state‐of‐the‐art generative models—**LaMa** and **Stable Diffusion**—against the traditional **Telea** method, the study reveals significant improvements in both performance and visual quality.

![Inpainting Results](Vis/Collage/viswithtitle1.png)

## Technologies Used

- Python  
- PyTorch  
- Stable Diffusion  
- LaMa  
- OpenCV  
- Pandas  

## Challenges Addressed

- Image preparation  
- Mask improvement  
- Model evaluation  
- Bootstrapping  
- Significance testing  

## Models

- **LaMa**  
- **Stable Diffusion** (Hugging Face)  
- **Telea** (OpenCV)  

## Key Findings

- **Superior Performance:** AI-based models (LaMa & Stable Diffusion) outperform Telea by approximately 14%.  
- **Mask Size is Critical:** The missing area's size is the most influential factor affecting inpainting quality.  
- **Model Suitability:**  
  - **Stable Diffusion** excels with large masks (though it may sometimes generate "hallucinations").  
  - **LaMa** is optimal for small-to-medium masks but struggles with larger missing areas.  
- **Future Directions:** Enhancing model efficiency and reducing unwanted object generation will further improve inpainting quality.

## What is Image Inpainting?

Image inpainting is the process of restoring missing, damaged, or removed parts of an image. It is widely used for image restoration, enhancement, and object removal.

## Results and Model Comparison

### Small Mask

| Model             | Improvement to Baseline (%) | Ranking |
|-------------------|-----------------------------|---------|
| LaMa              | +0.17%                      | 1       |
| Telea (Baseline)  | 0%                          | 2       |
| Stable Diffusion  | -21.48%                     | 3*      |

### Medium Mask

| Model             | Improvement to Baseline (%) | Ranking |
|-------------------|-----------------------------|---------|
| LaMa              | +11.67%                     | 1*      |
| Telea (Baseline)  | 0%                          | 3*      |
| Stable Diffusion  | +3.83%                      | 2*      |

### Large Mask

| Model             | Improvement to Baseline (%) | Ranking |
|-------------------|-----------------------------|---------|
| LaMa              | +14.27%                     | 2*      |
| Telea (Baseline)  | 0%                          | 3*      |
| Stable Diffusion  | +18.78%                     | 1*      |

### Overall

| Model             | Improvement to Baseline (%) | Ranking |
|-------------------|-----------------------------|---------|
| LaMa              | +14.64%                     | 1*      |
| Telea (Baseline)  | 0%                          | 3*      |
| Stable Diffusion  | +13.40%                     | 2*      |

---



