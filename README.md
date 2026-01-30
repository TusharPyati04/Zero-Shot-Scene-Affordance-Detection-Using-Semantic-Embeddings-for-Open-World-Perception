# Zero-Shot-Scene-Affordance-Detection-Using-Semantic-Embeddings-for-Open-World-Perception


This repository implements a **zero-shot vision–language-based framework for drivable affordance detection** in road scenes. The proposed approach leverages semantic embeddings and prompt-driven segmentation to localize navigable regions without task-specific training, enabling scalable and open-world perception for autonomous driving scenarios.

---

## Introduction

Understanding road scenes in terms of *actionable regions* is critical for autonomous and assisted driving systems. Unlike traditional object-centric perception approaches, this project focuses on **affordance-centric scene understanding**, specifically identifying **drivable regions** where a vehicle can safely operate. The framework utilizes pretrained vision–language models to perform **zero-shot affordance localization**, eliminating the need for costly pixel-level annotations and retraining.

---

## Core Components

- **CLIPSeg**: A vision–language segmentation model used to generate pixel-level affordance masks from textual prompts.
- **Affordance-Focused Prompting**: Multiple semantically related prompts are used and aggregated to improve spatial consistency.
- **Fourier Domain Adaptation (FDA)**: Applied as a preprocessing step to evaluate robustness under adverse weather and lighting conditions without retraining.
- **Aggregation & Thresholding**: Combines prompt-wise outputs to produce a final binary drivable affordance mask.

---

## Methodology Pipeline

1. Input RGB road scene images  
2. Optional FDA preprocessing for adverse weather or lighting conditions  
3. Encoding of affordance-specific textual prompts  
4. Prompt-driven segmentation using CLIPSeg  
5. Aggregation and thresholding of segmentation outputs  
6. Generation of final drivable affordance mask  

---

## Datasets

- **BDD100K**  
  Used for quantitative evaluation. Provides diverse driving scenes with ground-truth annotations for drivable regions.

- **DAWN**  
  Used for qualitative robustness analysis. Contains images captured under adverse weather conditions such as fog, rain, and haze. No ground-truth annotations are provided.

---

## Evaluation Metric

- **Intersection over Union (IoU)**  
  Used to measure the overlap between predicted drivable regions and ground-truth annotations on BDD100K.

---

## Results Summary

The proposed framework achieves a mean IoU of **0.619** for drivable affordance localization on the BDD100K dataset. Qualitative results demonstrate stable and coherent drivable region detection under adverse weather conditions, indicating robustness and strong generalization in open-world settings.

---

## Requirements

- Python 3.x  
- PyTorch  
- CLIPSeg  
- NumPy  
- OpenCV  
- Matplotlib  

(Exact versions can be specified in a `requirements.txt` file.)

---

## Repository Structure

├── data/ 
├── models/ 
├── fda/ 
├── inference/ 
├── results/ 
├── utils/ 
└── README.md

