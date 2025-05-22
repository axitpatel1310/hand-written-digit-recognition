import cv2
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model('digit_recognizer.keras')

def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)
    inverted = cv2.bitwise_not(resized)
    normalized = inverted / 255.0
    processed = normalized.reshape(1, 28, 28, 1)
    return processed

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame")
        break

    h, w = frame.shape[:2]
    crop_size = 200
    start_x = (w - crop_size) // 2
    start_y = (h - crop_size) // 2
    cropped = frame[start_y:start_y+crop_size, start_x:start_x+crop_size]

    processed = preprocess_frame(cropped)

    prediction = model.predict(processed, verbose=0)
    predicted_digit = np.argmax(prediction)
    confidence = np.max(prediction)

    text = f"Digit: {predicted_digit} ({confidence:.2f})"
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.rectangle(frame, (start_x, start_y), (start_x+crop_size, start_y+crop_size), (0, 255, 0), 2)

    cv2.imshow("Handwritten Digit Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
