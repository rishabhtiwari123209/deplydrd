
import numpy as np
import cv2
import os
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
import pathlib
data_dir = pathlib.Path(r"C:\Users\SIRT\Desktop\practice\eye all dataset\347mb\diabetic-retinopathy-dataset")
# data_dir
i_images_dict = {
    'h': list(data_dir.glob('Healthy/*')),
    'mi': list(data_dir.glob('Mild DR/*')),
    'mo': list(data_dir.glob('Moderate DR/*')),
    'p': list(data_dir.glob('Proliferate/*')),
    's': list(data_dir.glob('Severe DR/*')),
}
i_labels_dict = {
    'h': 0,
    'mi': 1,
    'mo': 2,
    'p': 3,
    's': 4,
}
x,y=[],[]
for i_name,images in i_images_dict.items():
  for image in images:
    img=cv2.imread(str(image))
    resize_img=cv2.resize(img,(180,180))
    x.append(resize_img)
    y.append(i_labels_dict[i_name])
x=np.array(x)
y=np.array(y)
print(x.shape,y.shape)
# Train test split

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2,train_size=0.8, random_state=0)
# Preprocessing: scale images
X_train_scaled = X_train / 255
X_test_scaled = X_test / 255
num_classes = 5

model = Sequential([
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

modelsave=model.fit(X_train_scaled, y_train, epochs=30)
model.evaluate(X_test_scaled,y_test)
predictions = model.predict(X_test_scaled)
print(predictions)
model.save("modelsave.h5")
