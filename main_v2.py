import os
import numpy as np
import cv2
import random
import shutil
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, MaxPooling2D, Flatten, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Define dataset paths
input_path = 'D:\\dataset\\asl_alphabet_train\\asl_alphabet_train'
output_path = 'D:\\dataset\\asl_alphabet_train\\augmented_images'

# Image size for CNN model
IMG_SIZE = (224, 224)  # Resize images to 224x224
data = []
labels = []

# Define advanced augmentation function
def augment_data(img):
    transformations = [
        lambda x: cv2.rotate(x, cv2.ROTATE_90_CLOCKWISE),
        lambda x: cv2.rotate(x, cv2.ROTATE_90_COUNTERCLOCKWISE),
        lambda x: cv2.flip(x, 1),  # Horizontal Flip
        lambda x: cv2.flip(x, 0),  # Vertical Flip
        lambda x: cv2.convertScaleAbs(x, alpha=1.2, beta=10),  # Brightness Adjust
        lambda x: cv2.GaussianBlur(x, (5, 5), 0)  # Gaussian Blur (Noise)
    ]
    return random.choice(transformations)(img)

# Data Preprocessing and Augmentation
def data_preprocess(input_path, output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    for class_name in os.listdir(input_path):
        class_path = os.path.join(input_path, class_name)
        if os.path.isdir(class_path):
            output_class_path = os.path.join(output_path, class_name)
            if os.path.exists(output_class_path):
                shutil.rmtree(output_class_path)
            os.makedirs(output_class_path)

            for img_name in os.listdir(class_path):
                image_path = os.path.join(class_path, img_name)
                img = cv2.imread(image_path)

                if img is not None:
                    img = cv2.resize(img, IMG_SIZE)  # Resize Image
                    img = img / 255.0  # Normalize Pixel Values
                    
                    # Store original image
                    data.append(img)
                    labels.append(class_name)

                    # Generate Augmented Images
                    for i in range(3):
                        aug_img = augment_data(img)
                        data.append(aug_img)
                        labels.append(class_name)

    return np.array(data), np.array(labels)

# Preprocess dataset
data, labels = data_preprocess(input_path, output_path)

# Encode labels
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

# Split dataset into train/validation/test sets
x_train, x_temp, y_train, y_temp = train_test_split(
    data, encoded_labels, test_size=0.2, random_state=42, stratify=encoded_labels
)
x_val, x_test, y_val, y_test = train_test_split(
    x_temp, y_temp, test_size=0.1, random_state=42, stratify=y_temp
)

# Convert labels to categorical format
num_classes = len(np.unique(y_train))
y_train = to_categorical(y_train, num_classes)
y_val = to_categorical(y_val, num_classes)
y_test = to_categorical(y_test, num_classes)

# Define CNN Model (Improved)
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    BatchNormalization(),
    MaxPooling2D((2, 2)),

    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),

    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),

    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')  # Output Layer
])

# Compile the Model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train Model
model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=15, batch_size=32)

# Evaluate Model
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {test_acc:.4f}")
