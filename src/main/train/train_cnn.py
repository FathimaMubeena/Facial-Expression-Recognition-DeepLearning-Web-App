import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import os

class EmotionClassifier:
    def __init__(self):
        self.model = None

    def train(self):
        # resources path
        file_path = os.path.join(os.path.dirname(__file__), '..', 'resources', 'train.csv')

        # Load data
        df = pd.read_csv(file_path)

        # Convert the pixels to arrays and normalize the values
        df['pixels'] = df['pixels'].apply(lambda x: np.array(x.split(), dtype='float32'))

        # Normalize pixel values to be between 0 and 1
        df['pixels'] = df['pixels'] / 255.0

        # Reshape the pixel data to 48x48x1
        X = np.array(df['pixels'].tolist()).reshape(-1, 48, 48, 1)

        # Get the emotion labels and convert to categorical
        y = to_categorical(df['emotion'], num_classes=7)

        # Split the data into training and testing sets
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

        # Data Augmentation only for the training set
        datagen = ImageDataGenerator(
            rotation_range=10,  # Rotate images randomly by 10 degrees
            width_shift_range=0.1,  # Shift the image horizontally by 10%
            height_shift_range=0.1,  # Shift the image vertically by 10%
            shear_range=0.1,  # Shear transformations
            zoom_range=0.1,  # Zoom into the image
            horizontal_flip=True,  # Flip images horizontally
            fill_mode='nearest'  # Fill missing pixels after transformations
        )

        # Create the model
        self.model = Sequential()
        # Add convolutional layers
        self.model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)))
        self.model.add(MaxPooling2D((2, 2)))
        self.model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        # Flatten the output
        self.model.add(Flatten())

        # Add dense layers
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(7, activation='softmax'))

        # Compile the model
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        # Fit the model using data augmentation for training and no augmentation for validation
        history = self.model.fit(
            datagen.flow(X_train, y_train, batch_size=64),  # Augment training data
            validation_data=(X_val, y_val),  # No augmentation for validation data
            steps_per_epoch=len(X_train) // 64,
            epochs=30  # Train for 30 epochs
        )

        # Plot accuracy for training and validation sets
        plt.figure(figsize=(10, 5))
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(loc='upper left')
        plt.grid(True)
        plt.show()

        # Save the model
        self.model.save(os.path.join(os.path.dirname(__file__), '..', 'deploy', 'emotion_model.keras'))

if __name__ == '__main__':
    model = EmotionClassifier()
    model.train()
