# U-Net Semantic Image Segmentation

A deep learning project that performs semantic image segmentation using a U-Net architecture implemented with TensorFlow and Keras.

The model learns to identify and segment objects or regions of interest at the pixel level by generating segmentation masks from input images.

## Features

- U-Net encoder-decoder architecture
- Pixel-level semantic segmentation
- Image and mask preprocessing with OpenCV
- Training and validation dataset splitting
- Binary cross-entropy loss
- Model checkpointing
- Learning rate reduction
- Early stopping
- Training and validation loss visualization
- Predicted mask visualization

## Technologies Used

- Python
- TensorFlow
- Keras
- OpenCV
- NumPy
- Matplotlib
- Scikit-learn

## Project Structure

```text
unet-semantic-segmentation/
│
├── main.py
├── Semantic segmentation dataset/
│   ├── tile_1/
│   │   ├── images/
│   │   └── masks/
│   ├── tile_2/
│   └── ...
│
├── unet_model.keras
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/unet-semantic-segmentation.git
cd unet-semantic-segmentation
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

Create a `requirements.txt` file:

```text
tensorflow
numpy
opencv-python
matplotlib
scikit-learn
```

Install the dependencies:

```bash
python -m pip install -r requirements.txt
```

## Dataset

The project expects a dataset organized into tile directories. Each tile contains separate image and mask folders.

```text
Semantic segmentation dataset/
├── tile_1/
│   ├── images/
│   └── masks/
├── tile_2/
│   ├── images/
│   └── masks/
└── ...
```

Each input image must have a corresponding PNG segmentation mask with the same filename.

Images and masks are resized to 128 × 128 pixels before training.

## U-Net Architecture

The model uses an encoder-decoder architecture with skip connections.

### Encoder

- Extracts image features using convolutional layers.
- Applies ReLU activation.
- Uses max pooling to reduce spatial dimensions.
- Progressively increases the number of feature channels.

### Bottleneck

- Captures deeper and more abstract image features.
- Uses convolutional layers with 256 filters.

### Decoder

- Uses transposed convolutions for upsampling.
- Combines encoder features through skip connections.
- Reconstructs spatial information.
- Generates a pixel-level segmentation mask using sigmoid activation.

## Model Training

The model is compiled with:

| Parameter | Configuration |
|---|---|
| Architecture | U-Net |
| Optimizer | Adam |
| Loss Function | Binary Crossentropy |
| Batch Size | 16 |
| Epochs | 10 |
| Input Size | 128 × 128 × 3 |
| Output Size | 128 × 128 × 1 |

### Callbacks

- ModelCheckpoint: Saves the best model.
- ReduceLROnPlateau: Reduces the learning rate when validation loss stops improving.
- EarlyStopping: Stops training when validation loss does not improve.

## Visualization

The project visualizes three outputs side by side:

1. Input Image
2. Ground Truth Mask
3. Predicted Segmentation Mask

It also plots training and validation loss to monitor model performance.

## How to Run

After preparing the dataset and installing the dependencies, run:

```bash
python main.py
```

The model will train on the dataset, display the loss curves, and visualize a sample segmentation prediction.

## Potential Applications

- Medical image segmentation
- Satellite image analysis
- Land cover classification
- Object segmentation
- Autonomous driving
- Industrial image analysis

These are potential applications of U-Net-based segmentation techniques.

## Future Improvements

- Add Dice coefficient and Intersection over Union (IoU) metrics.
- Implement data augmentation.
- Experiment with different U-Net configurations.
- Increase image resolution.
- Add multiclass segmentation support.
- Evaluate model performance on unseen images.

## Acknowledgments

The project is based on the U-Net encoder-decoder architecture, widely used in biomedical image segmentation and other computer vision applications.

---

Developed as a deep learning project exploring semantic image segmentation with TensorFlow and Keras.
