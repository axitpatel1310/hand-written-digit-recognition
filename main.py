import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

def custom_preprocess(img):
    if len(img.shape) == 3 and img.shape[-1] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = cv2.bitwise_not(img.astype(np.uint8))
    img = img / 255.0
    img = img[..., np.newaxis]
    return img

def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")
    img = cv2.resize(img, (28, 28))
    img = cv2.bitwise_not(img) 
    img = img / 255.0  
    img = img[..., np.newaxis]  
    return img

def build_model():
    model = Sequential([
        Input(shape=(28, 28, 1)),  
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def main():
    data_dir = "digits"
    
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

    print("Setting up data generators...")
    datagen = ImageDataGenerator(
        validation_split=0.2,  
        preprocessing_function=custom_preprocess  
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

    print("Building model...")
    model = build_model()
    model.summary()

    print("Training model...")
    history = model.fit(
        train_generator,
        epochs=5,  
        validation_data=val_generator,
        verbose=1
    )

    print("Evaluating model...")
    val_loss, val_accuracy = model.evaluate(val_generator, verbose=0)
    print(f"Validation Accuracy: {val_accuracy:.4f}")

    model.save('digit_recognizer.h5')
    print("Model saved as 'digit_recognizer.h5'")

    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('training_history.png')
    plt.close()
    print("Training history plot saved as 'training_history.png'")

    def predict_digit(image_path):
        img = preprocess_image(image_path)
        img = img.reshape(1, 28, 28, 1)
        prediction = model.predict(img)
        return np.argmax(prediction)
if __name__ == "__main__":
    main()
