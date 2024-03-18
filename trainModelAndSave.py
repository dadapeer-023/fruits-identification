from keras.models import Sequential 
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix    # Import Confusion matrix
from glob import glob
import numpy as np


train_path = "newFruitsDataset/Training/" 
test_path = "newFruitsDataset/Test/"


img = load_img(train_path + "Cocos/15_100.jpg")
plt.imshow(img)
plt.axis("off")
plt.show()

x = img_to_array(img) 
print(x.shape) 

className = glob(train_path + '/*') 
numberOfClass = len(className)
print("NumberOfClass: ",numberOfClass)
print("x.shape: ",x.shape)

# eng : 32 => number of filters
# tr : 32 => filtre sayısı
model = Sequential()
model.add(Conv2D(32,(3, 3),input_shape = x.shape))
model.add(Activation("relu"))
model.add(MaxPooling2D())

model.add(Conv2D(32,(3, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D())

model.add(Conv2D(32,(3, 3))) 
model.add(Activation("relu"))
model.add(MaxPooling2D())

model.add(Flatten())
model.add(Dense(1024)) 
model.add(Activation("relu"))
model.add(Dropout(0.5))
model.add(Dense(numberOfClass)) # output 
model.add(Activation("softmax"))

model.compile(loss = "categorical_crossentropy",
              optimizer = "rmsprop",
              metrics = ["accuracy"])

batch_size = 32 


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

hist = model.fit_generator(
        generator = train_generator,
        steps_per_epoch = 1600 // batch_size,
        epochs=100,
        validation_data = test_generator,
        validation_steps = 800 // batch_size) 

model.save('fruits-cnn-model-compressed.keras')    # Save the model for easier access.

print(hist.history.keys())
plt.plot(hist.history["loss"], label = "Train Loss")
plt.plot(hist.history["val_loss"], label = "Validation Loss")
plt.legend()
plt.show()
plt.figure()
plt.plot(hist.history["accuracy"], label = "Train acc")
plt.plot(hist.history["val_accuracy"], label = "Validation acc")
plt.legend()
plt.show()



predictions = model.predict_generator(test_generator, steps=len(test_generator), verbose=1)
predicted_classes = np.argmax(predictions, axis=1)

true_classes = test_generator.classes
class_labels = list(test_generator.class_indices.keys())

confusion_matrix = confusion_matrix(true_classes, predicted_classes)

plt.figure(figsize=(8, 6))
plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(len(class_labels))
plt.xticks(tick_marks, class_labels, rotation=90)
plt.yticks(tick_marks, class_labels)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()
