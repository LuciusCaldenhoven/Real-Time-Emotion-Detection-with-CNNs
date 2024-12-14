# Real-Time-Emotion-Detection-with-CNNs

## Description
This project uses a Convolutional Neural Network (CNN) implemented in PyTorch to detect facial emotions in real time. The system consists of two main components:

1. **Model Training**: Trains a CNN using a labeled dataset of facial images.
2. **Real-Time Detection**: Utilizes the trained model to detect emotions in faces captured from a webcam.

## Project Structure
```
emotion_recognition/
├── models/
│   ├── emotion_model.py          # Defines the EmotionRecognitionCNN model
│   ├── __init__.py               # Indicates models is a package
├── scripts/
│   ├── train.py                  # Code for training the model
│   ├── detect.py                 # Code for real-time detection
├── checkpoints/
│   ├── emotion_recognition_model.pth  # Trained model weights
├── data/                         # Folder for training and test data
│   ├── train/                    # Training data
│   ├── test/                     # Test data
├── utils/                        # Helper functions
│   ├── data_transforms.py        # Data augmentations and preprocessing
│   ├── __init__.py               # Indicates utils is a package
├── requirements.txt              # Project dependencies
├── README.md                     # Project documentation
```

## Requirements
To run this project, you need:

- Python 3.8 or higher
- PyTorch
- torchvision
- OpenCV
- Other packages listed in `requirements.txt`

Install dependencies by running:
```bash
pip install -r requirements.txt
```

## Model Training

1. Place your training and test data in the `data/train` and `data/test` folders.
   - Each subfolder should represent a class (e.g., `angry`, `happy`, etc.).

2. Run the training script:
   ```bash
   python scripts/train.py
   ```

   This will train the model and save the weights in `checkpoints/emotion_recognition_model.pth`.

## Real-Time Detection

1. Ensure your webcam is connected.

2. Run the detection script:
   ```bash
   python scripts/detect.py
   ```

   This will open a window showing the real-time video feed with the detected emotion for each identified face.

## Data Transformations
The project uses data augmentations to improve model generalization. Transformations are defined in `utils/data_transforms.py` and include:

- Image resizing
- Random rotations
- Random cropping
- Brightness and contrast adjustments
- Gaussian blur

## Demo

See the emotion recognition system in action:

![Emotion Recognition Demo](demo.gif)

