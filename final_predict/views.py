
from django.shortcuts import render,redirect
from .models import *
from vege.models import Profile 
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
from django.shortcuts import redirect, get_object_or_404
# from .scientific_notation import scientific_notation
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
@login_required(login_url='/loginp/')
def profile(request):
    user_profile = Profile.objects.get(user=request.user)
    predictionsdata = Image_Prediction.objects.filter(user=request.user).order_by('-timestamp')
    return render(request, 'profile.html', {'profile': user_profile, 'predictionsdata': predictionsdata})
# def save_into_db(prediction):
@login_required(login_url='/loginp/')
def delete_prediction(request,id):
    prediction = get_object_or_404(Image_Prediction, id=id)
    # prediction.delete()
    # Ensure the user requesting the delete owns the prediction
    if request.user == prediction.user:
        prediction.delete()
    
    return redirect('automated_grading') 
@login_required(login_url='/loginp/')
def automated_grading(request):
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
        # try:
        Image_Prediction.objects.create(
            user=request.user,image=ratina_image_url,
            healthy=prediction[0],mild=prediction[1],
            moderate=prediction[2],proliferate=prediction[3],
            severe=prediction[4]
            )
        # except Exception as e:
        #     # Handle exceptions appropriately
        #         print(f"Error uploading image: {e}")
        print(type(prediction))
        # predictionsdata = Image_Prediction.objects.all().order_by('-timestamp')
        predictionsdata = Image_Prediction.objects.filter(user=request.user).order_by('-timestamp')
        return render(request,'automated_grading.html',{
            'prediction':prediction,"predictionsdata":predictionsdata,
            "final_disease":final_disease,
            'max_value':max_value,
            'max_value_index':max_value_index,
            'image_path':ratina_image_url

        })
        # metadata = extract_metadata(image_path)
        # image = Image.open(image_path)
    model=model_fun()
    predictionsdata = Image_Prediction.objects.filter(user=request.user).order_by('-timestamp')
    # print(predictionsdata.)
    return render(request,'automated_grading.html',{"predictionsdata":predictionsdata,})
