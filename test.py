import cv2
import numpy as np
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model('digit_recognizer.keras')

# Function to preprocess a frame for prediction
def preprocess_frame(frame):
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Resize to 28x28
    resized = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)
    # Invert colors (white digits on black background)
    inverted = cv2.bitwise_not(resized)
    # Normalize to [0, 1]
    normalized = inverted / 255.0
    # Add batch and channel dimensions
    processed = normalized.reshape(1, 28, 28, 1)
    return processed

# Initialize webcam
cap = cv2.VideoCapture(0)  # Use 0 for default webcam
if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

# Set webcam resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame")
        break

    # Crop a central 200x200 region (adjust as needed)
    h, w = frame.shape[:2]
    crop_size = 200
    start_x = (w - crop_size) // 2
    start_y = (h - crop_size) // 2
    cropped = frame[start_y:start_y+crop_size, start_x:start_x+crop_size]

    # Preprocess the cropped frame
    processed = preprocess_frame(cropped)

    # Predict digit
    prediction = model.predict(processed, verbose=0)
    predicted_digit = np.argmax(prediction)
    confidence = np.max(prediction)

    # Display prediction on the frame
    text = f"Digit: {predicted_digit} ({confidence:.2f})"
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Draw a rectangle to show the cropped region
    cv2.rectangle(frame, (start_x, start_y), (start_x+crop_size, start_y+crop_size), (0, 255, 0), 2)

    # Show the frame
    cv2.imshow("Handwritten Digit Recognition", frame)

    # Quit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()