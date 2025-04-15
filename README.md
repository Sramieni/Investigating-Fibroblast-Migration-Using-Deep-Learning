# Investigating-Fibroblast-Migration-Using-Deep-Learning
Investigating Fibroblast Migration in a Wound Healing Assay Using AI-Based Image Segmentation and Deep Learning
# Fibroblast Wound Healing Segmentation

This project uses deep learning to automatically detect wound edges in microscopy images and track fibroblast migration over time. We built a complete pipeline using both a U-Net and a basic CNN model to compare their performance on this task.

---

## 🧠 Project Summary

- **Objective:** Segment wound regions in grayscale bright-field microscopy images using AI models.
- **Models Used:** U-Net and a lightweight CNN.
- **Metric:** Dice Score to evaluate segmentation accuracy.
- **Outcome:** U-Net performs slightly better in both accuracy and visual output.

---

## 🔬 Background

Fibroblast migration plays a key role in wound healing. Traditional segmentation methods are manual and time-consuming. This project aims to automate the process using deep learning so that wound area can be quantified quickly and accurately across time-lapse image sequences.

---

## 🛠 What This Repository Includes

- Preprocessing script for converting `.tif` stacks into clean images and masks
- Training code for both U-Net and a basic CNN model using PyTorch
- Dice score plots to monitor performance
- Overlay visualizations to inspect predicted wound edges
- Model weights and structure ready to deploy or retrain

---

## 🧪 Model Comparison

| Model     | Final Validation Dice Score |
|-----------|-----------------------------|
| U-Net     | **0.914**                   |
| Basic CNN| **0.907**                   |

### Training Curves

U-Net:
![Screenshot 2025-04-14 at 11 26 27 PM](https://github.com/user-attachments/assets/3cdb5154-4343-4a0e-b5cc-be3bb42c005f)



Basic CNN:

![Screenshot 2025-04-14 at 11 26 52 PM](https://github.com/user-attachments/assets/eca3afc9-675e-4275-a8d8-4be28b35fa17)

---

## 🔍 Visual Results

U-Net Prediction:
![Screenshot 2025-04-14 at 11 27 13 PM](https://github.com/user-attachments/assets/179ef56f-fb85-4492-b223-fac041b19b6f)


BasicCNN Prediction:

![Screenshot 2025-04-14 at 11 27 30 PM](https://github.com/user-attachments/assets/a3d30e7e-e365-4031-a6cc-1dc4c6ba9aac)


---

## 💾 Dataset Access

The dataset used in this project is currently **private** and will remain unpublished until the associated research paper is released.

However, we are happy to **share the dataset for evaluation** purposes. Please reach out via email or GitHub if you're interested.


## 🧰 How to Use This Repository

### 1. Clone the Repo

```bash
git clone https://github.com/your-username/Investigating-Fibroblast-Migration-Using-Deep-Learning.git
cd Investigating-Fibroblast-Migration-Using-Deep-Learning
```

---

### 2. Run Preprocessing

Converts the original `.tif` stack into enhanced `.png` images and binary wound masks.

```python
# Run this in the Jupyter notebook or Python script
python preprocess_tiff_stack.py
```

This script performs:
- 16-bit to 8-bit conversion with contrast stretching
- CLAHE enhancement for better contrast
- Otsu thresholding + morphological operations
- Binary mask cleanup and saving

Processed images and masks will be saved to:
```
data/images/
data/masks/
```

---

### 3. Train the U-Net Model

Launch training for the U-Net:

```python
python train_unet.py
```

- Trains the model on preprocessed images
- Uses BCE + Dice Loss
- Saves best model as `small_unet_cpu.pth`
- Logs Dice scores for each epoch

---

### 4. Train the Basic CNN

To train the simpler baseline CNN model:

```python
python train_cnn.py
```

- Lighter model to compare performance against U-Net
- Also logs training and validation Dice scores

---

### 5. Visualize Predictions

After training, you can run:

```python
python visualize_results.py
```

- Loads trained model and picks a random validation image
- Overlays predicted wound edge as a contour
- Useful for qualitative assessment

---

## 🧠 Requirements

Install all required packages using pip:

```bash
pip install -r requirements.txt
```

Main dependencies:
- `numpy`
- `opencv-python`
- `tifffile`
- `matplotlib`
- `torch`
- `scipy`
- `tqdm`

---

## 📬 Contact

For dataset requests or questions, feel free to contact:

**Sesha Sai Ramineni**  
📧 [ramineniseshasai@gamil.com]

---

## 🙏 Acknowledgments

This project was conducted as part of an academic research effort to explore AI-based analysis in biomedical imaging. Special thanks to our advisor for supporting this initiative.

