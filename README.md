
# Hand Sign Detection

This repository contains the implementation of a hand sign detection system using Convolutional Neural Networks (CNNs).

## Overview

The Hand Sign Detection project aims to recognize different hand signs using a deep learning approach. The system is trained to identify various hand gestures, making it useful for applications in sign language recognition, human-computer interaction, and more.

## Features

- Hand sign detection using CNNs
- Pre-trained models included
- Easy-to-use API for inference

## Getting Started

### Prerequisites

- Python 3.x
- pip

### Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/MatijaMikulic/HandSignDetection.git
   ```
2. Navigate to the project directory:
   ```sh
   cd HandSignDetection
   ```
3. Install the required dependencies:
   ```sh
   pip install -r requirements.txt
   ```

### Usage

1. Navigate to the `src` directory:
   ```sh
   cd src
   ```
2. Run the detection script:
   ```sh
   python detect.py --image path_to_image
   ```

### Training the Model

1. Prepare your dataset and place it in the `data/` directory.
2. Navigate to the `cnn_model` directory:
   ```sh
   cd cnn_model
   ```
3. Train the model:
   ```sh
   python train.py --epochs 50 --batch-size 32
   ```

### Directory Structure

- `cnn_model/`: Contains the CNN model definition and training scripts.
- `sign_rec_model/`: Contains the sign recognition model.
- `src/`: Contains the source code for detection and utilities.
- `data/`: Place your training and testing data here.
- `requirements.txt`: Python dependencies.

## Contributing

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes.
4. Commit your changes (`git commit -m 'Add new feature'`).
5. Push to the branch (`git push origin feature-branch`).
6. Open a Pull Request.

## License

This project is licensed under the MIT License.

## Acknowledgements

- Open-source community for their support and contributions.

