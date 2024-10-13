# Emotion Detection from Video Stream

## Overview
This project implements real-time emotion detection using a webcam feed. It utilizes deep learning models to identify emotions from facial expressions and visually represents the predictions on the video stream. The primary goal is to create an interactive application that can analyze and classify emotions in a user-friendly manner.

## Features
- **Real-time Emotion Detection**: Detects emotions such as happiness, sadness, anger, surprise, disgust, and fear from live video.
- **Face Detection**: Utilizes Haar Cascade Classifier to identify faces within the video stream.
- **Smoothing Predictions**: Implements a rolling window mechanism to smooth the emotion predictions over a defined number of frames, improving stability and reducing noise in the output.
- **Dynamic Visualization**: Displays emotion probabilities as horizontal bars alongside the detected face in the video feed for easy interpretation.

## Technologies Used
- **Python**: The primary programming language used for implementation.
- **OpenCV**: Used for video capture and image processing.
- **Keras**: Used for loading and running the pre-trained emotion classification model.
- **TensorFlow**: Backend for Keras, handling model inference and computations.

## Getting Started
1. **Clone the repository**:
   ```bash
   git clone https://github.com/jaikisan157/Facial-Emotion-detection.git
   cd src
   ```

2. **Install required packages**: Make sure to install the necessary libraries:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download Models**: Ensure you have the pre-trained Haar Cascade model for face detection and the FER2013 emotion classification model. Place them in the appropriate directories as specified in the code.

4. **Run the Application**: Start the video stream and emotion detection:
   ```bash
   python emotion_detection.py
   ```

5. **Stop the Application**: Press 'q' to exit the video stream.

## License
This project is licensed under the MIT License - see the LICENSE file for details.
