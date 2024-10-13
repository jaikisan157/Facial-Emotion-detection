import cv2
import numpy as np
from keras.models import load_model
from utils.datasets import get_labels
from utils.inference import detect_faces
from utils.inference import draw_text
from utils.inference import draw_bounding_box
from utils.inference import apply_offsets
from utils.inference import load_detection_model
from utils.preprocessor import preprocess_input

# Parameters for loading data and images
detection_model_path = '../trained_models/detection_models/haarcascade_frontalface_default.xml'
emotion_model_path = '../trained_models/emotion_models/fer2013_mini_XCEPTION.102-0.66.hdf5'
emotion_labels = get_labels('fer2013')

# Hyper-parameters for bounding boxes shape
emotion_offsets = (20, 40)
frame_window = 10  # For smoothing predictions

# Load models
face_detection = load_detection_model(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)

# Getting input model shapes for inference
emotion_target_size = emotion_classifier.input_shape[1:3]

# Starting video streaming
cv2.namedWindow('window_frame')
video_capture = cv2.VideoCapture(0)

# Define thresholds for adjusting emotions
THRESHOLD = 0.6  # Threshold to boost emotions
BOOST_FACTOR = 0.1  # Factor to boost probabilities

# Initialize a list to store predictions for smoothing
rolling_predictions = []

while True:
    # Read a frame from the video capture
    bgr_image = video_capture.read()[1]
    bgr_image = cv2.flip(bgr_image, 1)  # Mirror the image horizontally

    # Convert to grayscale for display
    gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)

    # Histogram equalization for better contrast
    gray_image = cv2.equalizeHist(gray_image)

    # Detect faces
    faces = detect_faces(face_detection, gray_image)

    for face_coordinates in faces:
        x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
        gray_face = gray_image[y1:y2, x1:x2]

        try:
            gray_face = cv2.resize(gray_face, emotion_target_size)
        except Exception as e:
            print(f"Error resizing face: {e}")
            continue

        gray_face = preprocess_input(gray_face, True)
        gray_face = np.expand_dims(gray_face, 0)
        gray_face = np.expand_dims(gray_face, -1)

        # Predict emotions
        emotion_prediction = emotion_classifier.predict(gray_face)[0]

        # Apply adjustments based on the threshold
        adjusted_prediction = np.copy(emotion_prediction)
        for i, prob in enumerate(emotion_prediction):
            if prob < THRESHOLD:
                adjusted_prediction[i] += BOOST_FACTOR

        # Normalize the adjusted predictions to ensure they sum to 1
        adjusted_prediction = adjusted_prediction / np.sum(adjusted_prediction)

        # Append the adjusted prediction for smoothing
        rolling_predictions.append(adjusted_prediction)
        if len(rolling_predictions) > frame_window:
            rolling_predictions.pop(0)

        # Average the predictions over the window
        smoothed_prediction = np.mean(rolling_predictions, axis=0)

        # Draw bounding box for the detected face on the grayscale image
        color = (255)  # White for bounding box in grayscale
        draw_bounding_box(face_coordinates, gray_image, color)

        # Draw adjusted emotion probabilities and text
        start_y = y1  # Start from the top of the bounding box
        for i, prob in enumerate(smoothed_prediction):
            # Scale the probability to display a max of 100% for a max of 60%
            scaled_prob = min(prob / 0.9, 1.0)  # Scale the probability
            display_prob = scaled_prob * 100  # Convert to percentage

            # Determine color based on predicted emotion (white text for visibility)
            text_color = (255)  # White for text

            # Draw a filled rectangle behind the text for better visibility
            text_background_color = (0)  # Black background
            text_width = cv2.getTextSize(f"{emotion_labels[i]}: {display_prob:.2f}", cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0][0]
            text_height = cv2.getTextSize(f"{emotion_labels[i]}: {display_prob:.2f}", cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0][1]
            cv2.rectangle(gray_image, (x1 - 105, start_y), (x1 - 105 + text_width, start_y + text_height + 5), text_background_color, -1)

            # Draw the probability bar based on the adjusted values
            bar_length = int(scaled_prob * 100)  # Convert scaled probability to length
            cv2.rectangle(gray_image, (x1 - 105, start_y), (x1 - 105 + bar_length, start_y + 20), color=color, thickness=-1)
            draw_text((x1 - 105 + bar_length + 5, start_y + 10), gray_image, f"{emotion_labels[i]}: {display_prob:.2f}", text_color, 0, 0, 0.6, 1)

            # Increment start_y for the next emotion bar
            start_y += 25  # Increase the space for the next emotion bar

    # Show the grayscale image with detected results
    cv2.imshow('window_frame', gray_image)  # Display the grayscale image

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
