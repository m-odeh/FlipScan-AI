import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers, models
from sklearn.metrics import accuracy_score, f1_score

#Data Dir
training_dir='C:/Users/M-ODE/Desktop/Apziva/projects/4th Project/data/images/training/'
testing_dir='C:/Users/M-ODE/Desktop/Apziva/projects/4th Project/data/images/testing/'


#Function for f1 score for keras
from keras import backend as K
def get_f1(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val


# Create an ImageDataGenerator for training data with data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,

)

# Create an ImageDataGenerator for testing data (no augmentation)
test_datagen = ImageDataGenerator(rescale=1./255)

# Load and preprocess training data using the generator
train_generator = train_datagen.flow_from_directory(
    training_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)

# Load and preprocess testing data using the generator
test_generator = test_datagen.flow_from_directory(
    testing_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary',
    shuffle=False  # To ensure the order of predictions matches the order of images
)

# Load the pre-trained MobileNetV2 model
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(150, 150, 3))

# Freeze the convolutional layers
for layer in base_model.layers:
    layer.trainable = False

# Create a custom model for binary classification
model = models.Sequential()
model.add(base_model)
model.add(layers.Flatten())
model.add(layers.Dense(98, activation='relu'))
#model.add(layers.Dropout(0.5))  # Adding dropout for regularization
#model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation='sigmoid'))


# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['Accuracy', get_f1])

# Train the model using the generator
model.fit(
    train_generator,
    epochs=10,
    validation_data=test_generator
)

# Save the model
model.save('model-mobilenet.h5')

# Evaluate the model on the test set
y_pred_proba = model.predict(test_generator)
y_pred_classes = (y_pred_proba > 0.5).astype(int)

# Extract true labels from the generator
y_true = test_generator.classes

accuracy = accuracy_score(y_true, y_pred_classes)
f1 = f1_score(y_true, y_pred_classes)

print(f"F1 Score: {f1}")
print(f"Test Accuracy: {accuracy}")

