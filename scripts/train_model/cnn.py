from PIL import Image
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
import cv2
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import matplotlib.pyplot as plt
import io

# Fungsi untuk memuat gambar
def load_images(directory):
    images = []
    for filename in os.listdir(directory):
        img = Image.open(os.path.join(directory, filename))
        img = img.resize((224, 224))
        img = img.convert('RGB')
        img = np.array(img) / 255.0
        images.append(img)
    return images

# Memuat gambar dari direktori masing-masing
normal = load_images('../../static/train/normal')
benign = load_images('../../static/train/benign')
malignant = load_images('../../static/train/malignant')

# Fungsi untuk menetapkan label
def assign_labels(normal_images, benign_images, malignant_images):
    normal_labels = np.zeros(len(normal_images))  # Kelas 0
    benign_labels = np.ones(len(benign_images))   # Kelas 1
    malignant_labels = np.full(len(malignant_images), 2)  # Kelas 2
    return normal_labels, benign_labels, malignant_labels

# Menetapkan label untuk tiga kelas
normal_labels, benign_labels, malignant_labels = assign_labels(normal, benign, malignant)

# Menggabungkan data dan label
data = normal + benign + malignant
labels = np.concatenate((normal_labels, benign_labels, malignant_labels), axis=0)

X = np.array(data)
y = np.array(labels)

# Mengonversi label ke format kategorikal
y = tf.keras.utils.to_categorical(y, num_classes=3)

from sklearn.model_selection import train_test_split

# Membagi dataset menjadi set pelatihan dan pengujian
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Membuat dataset TensorFlow
train = tf.data.Dataset.from_tensor_slices((X_train, y_train))
test = tf.data.Dataset.from_tensor_slices((X_test, y_test))

# Menetapkan ukuran validasi
validation_size = int(0.1 * len(X_train))
train = train.skip(validation_size)
val = train.take(validation_size)

BATCH_SIZE = 32
train = train.batch(BATCH_SIZE)
test = test.batch(BATCH_SIZE)
val = val.batch(BATCH_SIZE)

# Membangun model CNN
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3), padding='valid'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(3, activation='softmax'))  # Mengubah menjadi 3 kelas

# Mengompilasi model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',  # Mengubah menjadi categorical crossentropy
              metrics=['accuracy'])

# Melatih model
history = model.fit(train, validation_data=val, epochs=50, verbose=1)

# Mengevaluasi model
evaluation = model.evaluate(test)

# Plot akurasi
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['Akurasi', 'Akurasi Validasi'], loc='upper right')
plt.title('Akurasi')
plt.xlabel('Epochs')
plt.ylabel('Akurasi')
plt.show()

# Menyimpan model
model.save("../../model/model_cnn.h5")

# Fungsi untuk memuat gambar yang diunggah
def load_uploaded_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes))
    img = img.resize((224, 224))
    img_array = np.array(img)
    img_array = img_array / 255.0
    return img_array

# Fungsi untuk memprediksi gambar
def predict_image(image_bytes):
    img_array = load_uploaded_image(image_bytes)
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    class_index = np.argmax(prediction[0])  
    if class_index == 0:
        return "Normal"
    elif class_index == 1:
        return "Benign"
    else:
        return "Malignant"
