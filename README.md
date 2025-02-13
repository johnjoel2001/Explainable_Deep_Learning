# Explainable Deep Learning: Investigating The Role of Facial Features in the Classification of Dogs Using Integrated Gradients Explainability Technique

This repository contains a Colab/Jupyter Notebook that demonstrates how to use Integrated Gradients (IG) to explain a pretrained VGG16 model's decision when classifying dog images. The analysis focuses on comparing the importance of the dog’s facial region versus non-facial regions.

## Overview

The notebook performs the following steps:

1. **Model & Preprocessing:**
   - Loads a pretrained VGG16 model with ImageNet weights.
   - Preprocesses input images (resizing, center cropping, and normalization) as expected by the model.

2. **Integrated Gradients:**
   - Uses Integrated Gradients to compute attribution maps for a chosen target class (e.g., index `207` for "golden retriever").
   - Aggregates attributions across channels and normalizes them for visualization.

3. **Heuristic ROI (Region of Interest):**
   - This heuristic mask is used to compute average attribution scores for the face and non-face regions.

4. **Visualization:**
   - For each image in the `dog_images` folder, the notebook displays:
     - The **original image**
     - The **heuristic mask** (binary image showing the central region)
     - The **IG attribution map** (displayed with a "jet" colormap)
   - All images are displayed in a grid so that you can easily compare the results across multiple images.

5. **Statistical Analysis :**
   - The notebook computes the average attribution values for the face and non-face regions across images.
   - A paired t-test is performed to assess whether the difference in attribution is statistically significant.
  
## Observations
  
**Quantitative Analysis:**

* For the images that we tested, the mean integrated gradients attribution over the dog's face area was considerably more elevated than in non-facial areas. 

* The hypothesis testing using a paired t-test across images yielded a p-value less than 0.05, and hence we reject the null hypothesis (H₀).

**Qualitative Observations:** 

* From the visualizations we see that the visual (in all the four images) overlays  point to the dog face as an important area for the classification decision.

## Conclusion

* Overall, the results from the above quantitative and qualitative studies support the alternative hypothesis (H₁), thereby rejecting the null hypothesis (H₀).

* It suggests that the VGG16 model classifies images of dogs, based largely upon facial features.

## Installation and Setup

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/johnjoel2001/Explainable_Deep_Learning.git
   cd Explainable_Deep_Learning
   ```
2. Run the Explainable_Deep_Learning.ipynb
3. The requirements are listed in the first cell.

# **Notes & References**

1) 
```
half_width, half_height = 50, 50  
x_start, y_start = center_x - half_width, center_y - half_height
x_end, y_end = center_x + half_width, center_y + half_height
```

Three three lines of code were generated using Chatgpt o3-mini-high on 02/12/24 at 11:50 pm.

2) The dog images in the folder `Dog_Images` were generated using Grok on 02/12/24 at 10:30 pm. This was solely used for testing purpose.

3) Apart from these, AI was not used in any way.

4) https://www.geeksforgeeks.org/vgg-net-architecture-explained/
  
