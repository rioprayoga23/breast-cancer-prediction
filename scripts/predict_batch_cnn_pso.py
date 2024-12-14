import os
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

def load_test_images(directory):
    images = []
    labels = []
    for label in ['normal', 'benign', 'malignant']:
        label_dir = os.path.join(directory, label)
        for filename in os.listdir(label_dir):
            img = Image.open(os.path.join(label_dir, filename))
            img = img.resize((224, 224))
            img = img.convert('RGB')
            img = np.array(img) / 255.0
            images.append(img)
            labels.append(label)  # Append the label based on the folder name
    return np.array(images), np.array(labels)

# Load test images
X_test, y_true = load_test_images('../static/test')

# Convert true labels to numerical format
label_map = {'normal': 0, 'benign': 1, 'malignant': 2}
y_true_numeric = np.array([label_map[label] for label in y_true])

# Load the model for prediction model
model = tf.keras.models.load_model("../model/model_cnn_pso.h5")

# Make predictions
y_pred_prob = model.predict(X_test)
y_pred = np.argmax(y_pred_prob, axis=1)

# Calculate the number of correct predictions
correct_predictions = np.sum(y_pred == y_true_numeric)

# Calculate the total number of predictions
total_predictions = len(y_true_numeric)

# Calculate accuracy
accuracy = correct_predictions / total_predictions

# Convert accuracy to percentage
accuracy_percentage = accuracy * 100

# Print the average accuracy
print(f'Average Accuracy: {accuracy_percentage:.2f}%')

# Generate confusion matrix
cm = confusion_matrix(y_true_numeric, y_pred)

# Display confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Normal', 'Benign', 'Malignant'])
disp.plot(cmap=plt.cm.Blues)

# Set title for CNN PSO
plt.title('Confusion Matrix CNN PSO')
plt.show()

# Print classification report
print(classification_report(y_true_numeric, y_pred, target_names=['Normal', 'Benign', 'Malignant']))
