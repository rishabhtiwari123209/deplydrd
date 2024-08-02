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
from django.http import JsonResponse
import time
# Define custom objects if needed
custom_objects = {
    'BatchNormalization': CustomBatchNormalization,
}


# def load_model_view(request):
    # time.sleep(95)    
print("start")
model_path=os.path.join(settings.MEDIA_ROOT,'drdm/densenet_bestqwk.h5')
model_path2=os.path.join(settings.MEDIA_ROOT,'drdm/diabetic_retinopathy_model_with_pretrained.h5')
model_path3=os.path.join(settings.MEDIA_ROOT,'drdm/Diabetic Retinopathy.h5')
model = tf.keras.models.load_model(model_path)
model2 = tf.keras.models.load_model(model_path2)
model3= tf.keras.models.load_model(model_path3)
    # return JsonResponse({'message': 'Model loaded successfully'})
# model_path="C://Users/SIRT/Desktop/natlv/practice/core/public/static/model1.pkl"
# with open(model_path,'rb')as f:
#     model=pickle.load(f)

# def save_into_db(prediction):
    
def ensemble_model_pridiction(request):
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


        
       
        img2=cv2.resize(img,(224,224))#noraml
        img3=cv2.resize(img,(224,224))#mymodel
        img1=cv2.resize(img,(300, 300))#densemodel

        img1 = np.expand_dims(img1, axis=0)
        img1 = img1 / 255.0 
        img2 = np.expand_dims(img2, axis=0)
        img2 = img2 / 255.0
        img3 = np.expand_dims(img3, axis=0)
        img3 = img3 / 255.0
        prediction = model.predict(img1)
        prediction2 = model2.predict(img2)
        prediction3 = model3.predict(img3)

        prediction=prediction.flatten()
        prediction2=prediction2.flatten()
        prediction3=prediction3.flatten()


        
        prediction=prediction.tolist()
        prediction2=prediction2.tolist()
        prediction3=prediction3.tolist()
        try:
            ImagePrediction.objects.create(
            user=request.user,image=ratina_image_url,
            healthy=prediction[0],mild=prediction[1],
            moderate=prediction[2],poliferate=prediction[3],
            severe=prediction[4]
            )
        except:
            pass
        prediction1=prediction
        # Simple averaging ensemble
        for i in range(5):
            prediction[i]=(prediction[i]+prediction2[i]+prediction3[i])/3
        
        max_value_index = np.argmax(prediction)
        max_value = prediction[max_value_index]
        disease={0:"healthy",1:"mild",2:"moderate",3:"proliferate",4:"severe"}
        final_disease=''
        for i in disease:
            if max_value_index==i:
                final_disease=disease[i]
        return render(request,'esemblemodalPridict.html',{
            'prediction1':prediction1,'prediction2':prediction2,'prediction3':prediction3,
            "final_disease":final_disease,'prediction':prediction,
            'max_value':max_value,
            'max_value_index':max_value_index,
            'image_path':ratina_image_url

        })
        # metadata = extract_metadata(image_path)
        # image = Image.open(image_path)
    # model,model2,model3=model_fun()
    
    return render(request,'esemblemodalPridict.html')
