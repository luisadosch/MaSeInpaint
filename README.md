# AI Inpainting: Generative Models vs. Traditional Methods

![Inpainting Results](Vis/Collage/viswithtitle1.png)

## Description
Developed a study to explore how modern generative AI transforms image inpainting compared to classic techniques. By benchmarking **LaMa** and **Stable Diffusion** against the traditional **Telea** method, the research reveals striking improvements in both performance and visual quality.

## Technologies Used
- Python
- scikit-learn
- Stable Diffusion
- PyTorch
- LaMa
- OpenCV

## Challenges
- Image preparation
- Mask improvement
- Model evaluation
- Bootstrapping
- Significance testing

## Models Used
- **LaMa**
- **Stable Diffusion** (Hugging Face)
- **Telea** (OpenCV)

## Findings
- **AI inpainting** outperforms traditional models significantly.
- **Mask size matters:** Different models excel depending on the size of the missing area.
- Tailored solutions for varied inpainting challenges are possible by choosing the right model.

## What is Image Inpainting?
- **Definition:** Restoring missing, damaged, or removed parts of an image.
- **Applications:** Restoration, enhancement, and object removal.

## Results and Model Comparison

| Mask Size     | Model                 | FID    | Improvement to Baseline (%) | Ranking |
|---------------|-----------------------|--------|-----------------------------|---------|
| **Small Mask**    | LaMa                | 30.19  | +0.17%                      | 1       |
|               | Telea (Baseline)      | 30.24  | 0%                          | 2       |
|               | Stable Diffusion      | 36.74  | -21.48%                     | 3*      |
| **Medium Mask**   | LaMa                | 45.20  | +11.67%                     | 1*      |
|               | Telea (Baseline)      | 51.17  | 0%                          | 3*      |
|               | Stable Diffusion      | 49.21  | +3.83%                      | 2*      |
| **Large Mask**    | LaMa                | 93.08  | +14.27%                     | 2*      |
|               | Telea (Baseline)      | 108.59 | 0%                          | 3*      |
|               | Stable Diffusion      | 88.18  | +18.78%                     | 1*      |
| **Overall**       | LaMa                | 31.67  | +14.64%                     | 1*      |
|               | Telea (Baseline)      | 37.10  | 0%                          | 3*      |
|               | Stable Diffusion      | 32.13  | +13.40%                     | 2*      |

*Improvement to baseline is computed as:  
`((Telea Mean FID - Model Mean FID) / Telea Mean FID) × 100%`  
A positive value indicates better performance than Telea. An asterisk (*) denotes statistically significant ranking.

## Key Findings
- **Superior Performance:** AI-based models (LaMa & Stable Diffusion) outperform Telea by approximately 14%.
- **Mask Size is Critical:** The size of the missing area is the most influential factor affecting inpainting quality.
- **Model Suitability:**  
  - **Stable Diffusion** excels with large masks, though it may sometimes generate "hallucinations."  
  - **LaMa** is optimal for small-to-medium masks, but struggles with larger missing areas.
- **Future Directions:** Enhancing model efficiency and reducing unwanted object generation will further improve inpainting quality.

---

Explore the evolving landscape of image inpainting—where AI meets creativity and precision!
