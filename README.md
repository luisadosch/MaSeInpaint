# Comparative Study on Inpainting Methods: Generative AI vs. Traditional Techniques

This study investigates the effectiveness of modern generative AI inpainting methods compared to traditional techniques for object removal in images. In particular, we compare **Large Mask Inpainting (LaMa)** and **Stable Diffusion** with the classic **Telea** algorithm.

## Overview

- **Objective:**  
  Evaluate how well generative AI models reconstruct missing image regions versus traditional inpainting methods.
  
- **Key Findings:**  
  - **LaMa** reduces the Frechet Inception Distance (FID) by **14.64%** relative to Telea, indicating more natural image reconstructions.  
  - **Stable Diffusion** shows a **13.40%** improvement in FID over Telea, effectively filling large missing areas with realistic details.

These results demonstrate that AI-based methods produce more perceptually convincing outputs than classical approaches.

## Model Descriptions

### Telea Inpainting
- **Methodology:**  
  Uses the Fast Marching Method (FMM) to propagate pixel information from the edges of missing regions, computing replacement values as weighted averages of neighboring pixels.
- **Strengths:**  
  - Efficient and simple for small to medium repairs.
  - Provides smooth transitions in less complex scenarios.
- **Limitations:**  
  - Struggles with large gaps and complex textures due to its localized approach.

### Large Mask Inpainting (LaMa)
- **Approach:**  
  Integrates traditional convolutional layers with Fourier convolutions to capture both local details and global patterns.
- **Strengths:**  
  - Adversarial training on diverse masks enables robust reconstruction of large missing areas.
  - Preserves the overall structure of images effectively.
- **Trade-Off:**  
  - Tends to produce blurrier results as the size of the missing area increases.

### Stable Diffusion
- **Mechanism:**  
  Operates in a compressed latent space using a U-Net architecture, combining mask-based editing with text-guided refinements (e.g., using prompts like "background").
- **Strengths:**  
  - Excels at filling extensive missing regions with plausible and detailed content.
  - Adapts well to both visual context and semantic cues.
- **Trade-Off:**  
  - May introduce inaccuracies or extraneous details, particularly when precise reconstruction is needed.

## Evaluation Methodology

### Frechet Inception Distance (FID)
- **Purpose:**  
  Measures the difference between feature distributions (extracted via the Inception v3 network) of real versus inpainted images.  
- **Interpretation:**  
  Lower FID values correspond to reconstructions that more closely resemble the original images.

### Bootstrapping & Statistical Analysis
- **Bootstrapping:**  
  Repeated random sampling (with replacement) is used to generate distributions of FID scores, providing a measure of variability.
- **Statistical Tests:**  
  - **Shapiro-Wilk Test:** Checks for normality of FID distributions.  
  - **Two-sided T-Test:** Compares mean FID values when normality holds.  
  - **Wilcoxon Signed-Rank Test:** Used when data deviates from a normal distribution.
  
These methods ensure that the observed performance differences between models are statistically significant.

## Experimental Results and Analysis

- **Small to Medium Masks:**  
  - **LaMa** outperforms Telea by maintaining structural coherence and achieving lower FID scores.  
  - **Stable Diffusion** also improves over Telea, though its advantages become more pronounced for larger gaps.
  
- **Large Masks:**  
  - **Stable Diffusion** excels by generating coherent textures in extensive missing regions, albeit with occasional detail inaccuracies.
  - **LaMa** still preserves overall structure but may yield blurrier outcomes compared to Stable Diffusion.

## Discussion

The results underscore that:
- **Generative AI Methods:** Both LaMa and Stable Diffusion significantly outperform traditional inpainting (Telea), especially in handling complex or extensive missing regions.
- **Model Suitability:**  
  - **LaMa** is ideal for applications requiring precise object removal in small to medium areas.
  - **Stable Diffusion** is more effective for large-area repairs where the synthesis of realistic textures is paramount.
- **Challenges:**  
  The trade-offs between maintaining structural integrity and achieving texture realism highlight the importance of aligning the inpainting method with the specific requirements of the task.

## Conclusion

This study demonstrates that modern generative AI models, namely LaMa and Stable Diffusion, offer substantial improvements over classical inpainting techniques like Telea. While each model has its own strengths and weaknesses, their ability to produce context-aware and visually natural reconstructions marks a significant advancement in the field of image restoration. Future research may explore hybrid approaches and fine-tuning strategies to further enhance performance and address current limitations.

---

This summary provides an integrated view of the models, evaluation methods, experimental results, and conclusions drawn from the study, offering valuable insights for selecting the appropriate inpainting technique based on task-specific requirements.
