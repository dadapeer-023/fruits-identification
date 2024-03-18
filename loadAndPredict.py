from keras.models import load_model
from keras.preprocessing import image
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from sklearn.metrics import confusion_matrix

train_path = "newFruitsDataset/Training/" 
test_path = "newFruitsDataset/Test - Copy - False/"

batch_size = 32 

loaded_model = load_model('fruits-cnn-model-compressed.keras')

img = load_img(test_path + "Cocos/r_76_100.jpg")
x = img_to_array(img) 

train_datagen = ImageDataGenerator(rescale= 1./255,
                   shear_range = 0.3, 
                   horizontal_flip=True, 
                   zoom_range = 0.3) 


test_datagen = ImageDataGenerator(rescale= 1./255) 

train_generator = train_datagen.flow_from_directory(
        train_path,
        target_size = x.shape[:2], 
        color_mode= "rgb",
        class_mode= "categorical")

test_generator = test_datagen.flow_from_directory(
        test_path, 
        target_size=x.shape[:2],
        batch_size = batch_size,
        color_mode= "rgb",
        class_mode= "categorical")


true_classes = test_generator.classes
class_labels = list(test_generator.class_indices.keys())
print("class_labels", test_generator.class_indices)

#--------------------------------- Below code for Single prediction---------------

# # Function to predict class labels for an image
# def predict_image_class(image_path):
#     img = image.load_img(image_path, target_size= (100, 100,3))
#     x = image.img_to_array(img)
#     x = np.expand_dims(x, axis=0)
#     x = x / 255.0  # Rescale pixel values to [0, 1]
#     predicted_class = loaded_model.predict(x)
#     predicted_label = class_labels[np.argmax(predicted_class)]
#     return predicted_label

# image_path = test_path +"Cocos/14_100.jpg"

# predicted_label = predict_image_class(image_path)
# print("Predicted label:", predicted_label)



#--------------------------------- Below code for bulk prediction---------------

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os 

def predict_image_classes_in_bulk(directory):
    predicted_labels = []
    true_labels = []
    for subdir in os.listdir(directory):
        subdir_path = os.path.join(directory, subdir)
        if os.path.isdir(subdir_path):
            for filename in os.listdir(subdir_path):
                if filename.endswith(".jpg") or filename.endswith(".png"):
                    print(os.path.join(subdir_path, filename))
                    img = image.load_img(os.path.join(subdir_path, filename), target_size=(100, 100, 3))
                    x = image.img_to_array(img)
                    x = np.expand_dims(x, axis=0)
                    x = x / 255.0  # Rescale pixel values to [0, 1]
                    predicted_class = loaded_model.predict(x)
                    predicted_label = class_labels[np.argmax(predicted_class)]
                    predicted_labels.append(predicted_label)
                    true_labels.append(subdir)
    return true_labels, predicted_labels

true_labels, predicted_labels = predict_image_classes_in_bulk(test_path)

# print(true_labels, predicted_labels)
# Compute confusion matrix
conf_matrix = confusion_matrix(true_labels, predicted_labels)

# Print confusion matrix
print("Confusion Matrix:")
print(conf_matrix)

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, cmap="YlGnBu", fmt="d", xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()
