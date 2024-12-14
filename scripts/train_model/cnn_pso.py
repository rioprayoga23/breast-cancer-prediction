import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from sklearn.model_selection import train_test_split
from pyswarm import pso
import matplotlib.pyplot as plt
from PIL import Image
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
print("===================================")
print("         Memuat Gambar            ")
print("===================================")

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

# Mencetak jumlah total gambar untuk setiap kategori
print("===================================")
print("       Total Gambar per Kategori   ")
print("===================================")
print(f"Total gambar normal: {len(normal)}")
print(f"Total gambar jinak: {len(benign)}")
print(f"Total gambar ganas: {len(malignant)}")

# Mengonversi label ke format kategorikal
y = tf.keras.utils.to_categorical(y, num_classes=3)

# Membagi dataset menjadi set pelatihan dan pengujian
print("===================================")
print("   Membagi Dataset menjadi Set     ")
print("         Pelatihan dan Pengujian   ")
print("===================================")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Jumlah gambar pelatihan: {len(X_train)}")
print(f"Jumlah gambar pengujian: {len(X_test)}")

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

# Fungsi untuk membuat model CNN
def create_model(num_filters, kernel_size, dropout_rate, learning_rate):
    model = Sequential()
    model.add(Conv2D(int(num_filters), (int(kernel_size), int(kernel_size)), activation='relu', input_shape=(224, 224, 3), padding='valid'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(int(num_filters * 2), (int(kernel_size), int(kernel_size)), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(3, activation='softmax'))

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Fungsi untuk mengevaluasi model
def evaluate_model(params):
    num_filters, kernel_size, dropout_rate, learning_rate = params
    print("===================================")
    print(" Mengevaluasi Model dengan Param: ")
    print(f" num_filters={num_filters}, kernel_size={kernel_size}, dropout_rate={dropout_rate}, learning_rate={learning_rate}")
    
    model = create_model(num_filters, kernel_size, dropout_rate, learning_rate)
    
    # Melatih model
    print("Memulai pelatihan model...")
    history = model.fit(train, validation_data=val, epochs=5, verbose=1)
    print("Pelatihan model selesai.")
    
    val_accuracy = history.history['val_accuracy'][-1]  # Mendapatkan akurasi validasi terakhir
    print(f" Akurasi Validasi: {val_accuracy:.4f}")
    return -val_accuracy  # Mengembalikan negatif untuk minimisasi

# Mendefinisikan batas untuk hiperparameter
lb = [16, 3, 0.1, 1e-5]  # Batas bawah: num_filters, kernel_size, dropout_rate, learning_rate
ub = [64, 7, 0.5, 1e-2]  # Batas atas: num_filters, kernel_size, dropout_rate, learning_rate

# Menjalankan PSO
print("===================================")
print("   Menjalankan Particle Swarm Optimization   ")
print("===================================")
print("Memulai Particle Swarm Optimization...")

best_params, best_val_accuracy = pso(evaluate_model, lb, ub, swarmsize=10, maxiter=2)

print("Particle Swarm Optimization selesai.")

print("===================================")
print("          Parameter Terbaik:       ")
print("===================================")
print("Parameter Terbaik: ", best_params)
print("Akurasi Validasi Terbaik: ", -best_val_accuracy)

# Membongkar parameter terbaik
num_filters, kernel_size, dropout_rate, learning_rate = best_params

# Membuat dan melatih model akhir
print("===================================")
print("   Membuat dan Melatih Model Akhir   ")
print("===================================")

final_model = create_model(num_filters, kernel_size, dropout_rate, learning_rate)
final_history = final_model.fit(train, validation_data=val, epochs=50, verbose=1)

# Mengevaluasi model akhir
print("===================================")
print("       Mengevaluasi Model Akhir       ")
print("===================================")

final_evaluation = final_model.evaluate(test)
print(f"Evaluasi model akhir: {final_evaluation}")

# Plot akurasi untuk model akhir
plt.plot(final_history.history['accuracy'])
plt.plot(final_history.history['val_accuracy'])
plt.legend(['Akurasi', 'Akurasi Validasi'], loc='upper right')
plt.title('Akurasi Model Akhir')
plt.xlabel('Epoch')
plt.ylabel('Akurasi')
plt.show()

# Menyimpan model akhir
print("===================================")
print("         Menyimpan Model Akhir         ")
print("===================================")

final_model.save("../../model/model_cnn_pso.h5")
print("Model berhasil disimpan.")
