
from django.shortcuts import render,redirect
from .models import *
from PIL import Image
import os
from datetime import datetime
from django.conf import settings
import pickle
import tensorflow
from django.contrib.auth.decorators import login_required
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import PIL
import matplotlib.pyplot as plt
import tensorflow as tf
from django.core.files.storage import FileSystemStorage
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import BatchNormalization
class CustomBatchNormalization(BatchNormalization):
    def call(self, inputs, training=None):
        return super().call(inputs, training=training)

# Define custom objects if needed
custom_objects = {
    'BatchNormalization': CustomBatchNormalization,
}

# from tensorflow.python.keras.saving import hdf5_format
# from keras.saving.hdf5_format import save_attributes_to_hdf5_group 
# tensorflow.__version__
# Load the pre-trained model from the Pickle file
def model_fun():    
    model_path=os.path.join(settings.MEDIA_ROOT,'drdm/densenet_bestqwk.h5')
    model = tf.keras.models.load_model(model_path,custom_objects=custom_objects)
    return model
# model_path="C://Users/SIRT/Desktop/natlv/practice/core/public/static/model1.pkl"
# with open(model_path,'rb')as f:
#     model=pickle.load(f)

# def save_into_db(prediction):
    
def pretrain_modalPridict(request):
    if request.method == 'POST':
        image = request.FILES['image']
        fs=FileSystemStorage()
        filename=fs.save(image.name,image)
        uploaded_file_url=fs.url(filename)
        image_path=os.path.join(settings.MEDIA_ROOT,filename)
       
        ratina_image_url=fs.url(filename)
        print(image_path)
        img = cv2.imread(image_path)
        # img=cv2.imdecode(np.fromstring(image.read(),np.uint8),cv2.IMREAD_COLOR)
        img=cv2.resize(img,(300,300))
        img = np.expand_dims(img, axis=0)
        img = img / 255.0 
        model=model_fun()
        prediction = model.predict(img)
        prediction=prediction.flatten()
        max_value_index = np.argmax(prediction)
        max_value = prediction[max_value_index]
        disease={0:"healthy",1:"mild",2:"moderate",3:"proliferate",4:"severe"}
        final_disease=''
        for i in disease:
            if max_value_index==i:
                final_disease=disease[i]
        
        prediction=prediction.tolist()
        try:
            ImagePrediction.objects.create(
            user=request.user,image=ratina_image_url,
            healthy=prediction[0],mild=prediction[1],
            moderate=prediction[2],poliferate=prediction[3],
            severe=prediction[4]
            )
        except:
            pass
        print(type(prediction))
        return render(request,'pretrain_modalPridict.html',{
            'prediction':prediction,
            "final_disease":final_disease,
            'max_value':max_value,
            'max_value_index':max_value_index,
            'image_path':ratina_image_url

        })
        # metadata = extract_metadata(image_path)
        # image = Image.open(image_path)
    model=model_fun()
    return render(request,'pretrain_modalPridict.html')
