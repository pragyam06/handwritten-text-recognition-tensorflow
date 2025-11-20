Handwritten Line Text Recognition using Deep Learning (TensorFlow)

This project, developed by Pragya Mundra, implements a full pipeline for handwritten line-level text recognition using deep learning techniques. It uses TensorFlow-based neural network models to read, interpret, and convert handwritten text images into machine-readable text. The repository includes model training scripts, inference utilities, a web interface, sample datasets, and pre-trained checkpoints for quick experimentation.

Overview

The system is designed to recognize handwritten words and full text lines rather than isolated characters.
It includes:

A trained deep-learning model for handwriting recognition

Data preprocessing utilities

Scripts for inference with custom input images

Web UI for uploading images and getting recognized text

Multiple TensorFlow versions (TF1.x & TF2.x compatible code)

Sample dataset and character lists

Pre-trained model files

📁 Project Structure
Handwritten-Line-Text-Recognition-using-Deep-Learning-with-Tensorflow/
│
├── data/                       # Character lists, corpus, sample test images
├── images/                     # Additional reference images
├── model/                      # Pretrained model files & checkpoints
│
├── src/                        # TensorFlow main implementation (training + inference)
│   ├── DataLoader.py
│   ├── Model.py
│   ├── SamplePreprocessor.py
│   ├── main.py                 # Run inference
│   ├── train.py                # Train the model
│   └── WordSegmentation.py
│
├── src_tensorflow1/            # TF1.x compatible version (training & prediction)
├── src_tensorflow2/            # TF2.x compatible version + Web UI
│   ├── app.py                  # Flask Web Demo
│   ├── static/                 # CSS, JS, images
│   └── templates/              # HTML templates for the web UI
│
├── LICENSE
└── README.md (new version you will replace with this)

Features

Line-level handwritten text recognition

Works with grayscale, thresholded, and raw scanned images

Includes pretrained models for quick testing

Supports TensorFlow 1.x & 2.x

Flask-based web application for browser-based testing

Dataset-ready structure for retraining

Word segmentation module for multi-word input

Clean modular code for easy customization

Model Architecture (Summary)

The project uses a deep neural network consisting of:

CNN layers for visual feature extraction

Bi-directional LSTM layers for sequence learning

CTC loss for aligning predicted text with variable-length handwriting

Beam search decoding for improved text prediction accuracy

This architecture is commonly used in state-of-the-art handwriting recognition systems.

Installation
1. Clone the repository
git clone https://github.com/your-repo-name.git
cd Handwritten-Line-Text-Recognition-using-Deep-Learning-with-Tensorflow

2. Install dependencies

For TensorFlow 2.x version:

pip install -r requirements.txt


(If the file doesn’t exist, simply install the basics:)

pip install tensorflow numpy opencv-python flask pillow matplotlib

Running Inference
TensorFlow 2 version
cd src_tensorflow2
python app.py


This launches a web interface where you can upload images and view predictions.

TensorFlow main version

Run inference on a sample image:

cd src
python main.py

Train the model
cd src
python train.py

Sample Data

The data/ directory includes:

charList.txt — character set used by the model

wordCharList.txt — alternative character mapping

corpus.txt — dataset corpus

Several test images (self.png, testImage.png, etc.)

Use these for quick testing or create your own dataset.

Web Demo (Flask App)

The folder src_tensorflow2/ contains a ready-to-run web-based interface:

Upload handwriting images

Model returns recognized text

Clean Bootstrap-based UI

Suitable for demos or deployments

Start the server:

python app.py


Open browser:

http://127.0.0.1:5000/

Future Improvements

Integrate attention-based transformers

Improve accuracy on noisy scanned documents

Add multilingual handwriting recognition

Build a full OCR pipeline with page layout detection
Author

Pragya Mundra

Feel free to open issues, suggest improvements, or submit pull requests.
