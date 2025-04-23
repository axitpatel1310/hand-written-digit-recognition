import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Custom preprocessing function for ImageDataGenerator
def custom_preprocess(img):
    # Ensure image is grayscale
    if len(img.shape) == 3 and img.shape[-1] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Invert colors (white digits on black background)
    img = cv2.bitwise_not(img.astype(np.uint8))
    # Normalize to [0, 1]
    img = img / 255.0
    # Add channel dimension
    img = img[..., np.newaxis]
    return img

# Function to preprocess a single image for prediction
def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")
    img = cv2.resize(img, (28, 28))
    img = cv2.bitwise_not(img)  # Invert: white digits on black background
    img = img / 255.0  # Normalize to [0, 1]
    img = img[..., np.newaxis]  # Add channel dimension
    return img

# Build CNN model
def build_model():
    model = Sequential([
        Input(shape=(28, 28, 1)),  # Explicit input layer
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(10, activation='softmax')  # 10 classes (0-9)
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Main function
def main():
    # Dataset directory
    data_dir = "digits"
    
    # Verify dataset structure
    print("Verifying dataset structure...")
    total_images = 0
    for digit in range(10):
        folder = os.path.join(data_dir, str(digit))
        if not os.path.exists(folder):
            raise FileNotFoundError(f"Folder {folder} not found. Ensure digits are in folders 0-9.")
        images = [f for f in os.listdir(folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
        if not images:
            raise ValueError(f"No images found in {folder}. Add images to train the model.")
        print(f"Found {len(images)} images in {folder}")
        total_images += len(images)
    print(f"Total images found: {total_images}")

    # Set up data generators
    print("Setting up data generators...")
    datagen = ImageDataGenerator(
        validation_split=0.2,  # 20% for validation
        preprocessing_function=custom_preprocess  # Custom preprocessing
    )

    train_generator = datagen.flow_from_directory(
        data_dir,
        target_size=(28, 28),
        color_mode='grayscale',
        batch_size=32,
        class_mode='sparse',
        subset='training',
        shuffle=True
    )

    val_generator = datagen.flow_from_directory(
        data_dir,
        target_size=(28, 28),
        color_mode='grayscale',
        batch_size=32,
        class_mode='sparse',
        subset='validation',
        shuffle=True
    )

    # Build model
    print("Building model...")
    model = build_model()
    model.summary()

    # Train model
    print("Training model...")
    history = model.fit(
        train_generator,
        epochs=5,  # Increase to 10-15 if accuracy is low
        validation_data=val_generator,
        verbose=1
    )

    # Evaluate model
    print("Evaluating model...")
    val_loss, val_accuracy = model.evaluate(val_generator, verbose=0)
    print(f"Validation Accuracy: {val_accuracy:.4f}")

    # Save model
    model.save('digit_recognizer.h5')
    print("Model saved as 'digit_recognizer.h5'")

    # Plot training history
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('training_history.png')
    plt.close()
    print("Training history plot saved as 'training_history.png'")

    # Function to predict a single digit
    def predict_digit(image_path):
        img = preprocess_image(image_path)
        img = img.reshape(1, 28, 28, 1)
        prediction = model.predict(img)
        return np.argmax(prediction)

    # Example: Predict a single digit (uncomment and replace with your image path)
    # test_image_path = "test_digit.png"
    # predicted_digit = predict_digit(test_image_path)
    # print(f"Predicted Digit for {test_image_path}: {predicted_digit}")

if __name__ == "__main__":
    main()