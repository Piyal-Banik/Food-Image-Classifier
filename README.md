# Food Image Classification with PyTorch  

##  Project Overview  
This project is a **Deep Learning-based Food Image Classifier** built using **PyTorch**. The model is trained on a custom dataset of food images and can classify different food items. It uses **transfer learning**, **data augmentation**, and **PyTorch’s training pipeline** for efficient model training and evaluation.

---

## Project Structure  
```bash
📂 dataset/
├── 📂 train/
│   ├── 📂 class_1/
│   │   ├── img1.jpg
│   │   ├── img2.jpg
│   │   └── ...
│   ├── 📂 class_2/
│   ├── 📂 class_3/
│   └── ...
│
├── 📂 test/
│   ├── 📂 class_1/
│   ├── 📂 class_2/
│   ├── 📂 class_3/
│   └── ...
📂 models/
├── model.pth
📂 notebooks/
├── run_pipeline.py
📂 src/
├── data_download.py
├── data_setup.py
├── model.py
├── train.py
├── training_pipeline.py
├── utils.py 

```

| File                  | Description |
|----------------------|-------------|
| `data_setup.py`      | Handles dataset loading, transformations, and creating train/test dataloaders. |
| `model.py`          | Defines the deep learning model architecture used for image classification. |
| `train.py`          | Main script for training the model, including argument parsing, data loading, and model initialization. |
| `training_pipeline.py` | Contains functions for training and evaluating the model, including loss calculation and performance metrics. |
| `utils.py`          | Helper function for model saving/loading. |

Each file plays a crucial role in ensuring the model is trained efficiently and can be used for inference later.  


## Features  
✅ **Custom Dataset Handling** – Loads and preprocesses images.  
✅ **Data Augmentation** – Uses transformations for better generalization.  
✅ **Transfer Learning** – Fine-tunes a pre-trained model.  
✅ **GPU Support** – Trains efficiently using CUDA if available.  

---

## Installation  

Clone the repository:  
```bash
https://github.com/Piyal-Banik/Food-Image-Classifier.git

```
Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage
### Train the Model
```bash
python train.py --train_dir path/to/train --test_dir path/to/test --num_epochs 10 --batch_size 32 --learning_rate 0.001
```

| Argument        | Description                                      | Default |
|----------------|--------------------------------------------------|---------|
| `--train_dir`  | Path to the training dataset                     | Required |
| `--test_dir`   | Path to the test dataset                         | Required |
| `--num_epochs` | Number of epochs for training                    | `5` |
| `--batch_size` | Batch size for training                          | `32` |
| `--learning_rate` | Learning rate for optimizer                  | `0.001` |
| `--img_size`   | Image size for resizing (assumes square images)  | `224` |
| `--device`     | Device to train on (`auto`, `cpu`, `cuda`)       | `auto` |

### Evaluate the Model
After training, the model is automatically evaluated on the test dataset. To manually test, modify training_pipeline.py to evaluate on custom images.

### Save & Load Model
The trained model is saved as model.pth.

To load the trained model for inference:
```bash
import torch
from model import FoodImageClassifier

# Load model
model = FoodImageClassifier(fine_tune=False, num_classes=3)
model.load_state_dict(torch.load("model.pth"))
model.eval()

```

### Run Inference on New Images
Modify utils.py to load an image and predict its class. Example usage:
```bash
from utils import predict_image

image_path = "path/to/image.jpg"
prediction = predict_image(model, image_path)
print(f"Predicted class: {prediction}")
```