
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import numpy as np
import scipy


model = keras.Sequential([
    keras.layers.Input(shape=(64, 64, 3)),
    keras.layers.Conv2D(32, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


train_datagen = keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
    "C:\\Users\\Administrator\\Desktop\\dataset\\train_set",
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary'
)

test_set = test_datagen.flow_from_directory(
    "C:\\Users\\Administrator\\Desktop\\dataset\\test_set",
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary'
)

model.fit(
    training_set,
    steps_per_epoch=80,
    epochs=10,
    validation_data=test_set,
    validation_steps=46
)

def predict_image(image_path):
    img = Image.open(image_path)
    img = img.resize((64, 64))
    img = np.array(img)
    img = np.expand_dims(img, axis=0)
    result = model.predict(img)
    return "plane" if result[0][0] == 1 else "non-plane"


def draw_rectangle(type):
    if type == "plane":
        pass
    else :
        pass

print(predict_image("C:/Users/Administrator/Desktop/tahmin/ucak.jpg"))
print(predict_image("C:/Users/Administrator/Desktop/tahmin/cat.jpg"))
print(predict_image("C:/Users/Administrator/Desktop/tahmin/ucak2.jpg"))
print(predict_image("C:/Users/Administrator/Desktop/tahmin/dog.jpg"))
print(predict_image("C:/Users/Administrator/Desktop/tahmin/car.jpg"))
print(predict_image("C:/Users/Administrator/Desktop/tahmin/motor.jpg"))
print(predict_image("C:/Users/Administrator/Desktop/tahmin/ucak3.jpg"))
print(predict_image("C:/Users/Administrator/Desktop/tahmin/ucak3.jpg"))
