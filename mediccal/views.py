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
import os
from tensorflow.keras.models import load_model
import tensorflow as tf
from tensorflow.keras.regularizers import Regularizer

class DiffL1(Regularizer):
    def __init__(self, l1=0.01, diff=10.0):
        self.l1 = l1
        self.diff = diff

    def __call__(self, x):
        regularization = 0.0
        regularization += tf.reduce_sum(self.l1 * tf.abs(x))
        return regularization

    def get_config(self):
        return {'l1': float(self.l1), 'diff': float(self.diff)}

# Register the custom regularizer
custom_objects = {'DiffL1': DiffL1}

# Load the model within the custom object scope
# model_path = os.path.join(settings.MEDIA_ROOT, 'export.pkl')

# model = load_model(model_path, custom_objects=custom_objects)

def model_fun():  

    model_path=os.path.join(settings.MEDIA_ROOT,'drdm/densenet_bestqwk.h5')
    model = tf.keras.models.load_model(model_path,custom_objects=custom_objects)
    return model
# model_path="C://Users/SIRT/Desktop/natlv/practice/core/public/static/model1.pkl"
# with open(model_path,'rb')as f:
#     model=pickle.load(f)

# def save_into_db(prediction):
    
def model_pridiction(request):
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
        # img=cv2.resize(img,(224,224))#noraml
        # img=cv2.resize(img,(728,728))#mymodel
        img=cv2.resize(img,(300,300))#densemodel
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
        return render(request,'modalPridict.html',{
            'prediction':prediction,
            "final_disease":final_disease,
            'max_value':max_value,
            'max_value_index':max_value_index,
            'image_path':ratina_image_url

        })
        # metadata = extract_metadata(image_path)
        # image = Image.open(image_path)
    model=model_fun()
    return render(request,'modalPridict.html')


def augmented_images_fun(image,filename):
    # convert imag to array
    fs=FileSystemStorage()
    augumented_images=[]
    image_array=np.expand_dims(image,axis=0)
    datagen=ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
    )
    i=0
    for batch in datagen.flow(image_array,batch_size=1):
        aug_image_path=os.path.join(settings.MEDIA_ROOT,f'augumented/augumented_{i}_{filename}')
        cv2.imwrite(aug_image_path,batch[0].astype(np.uint8))
        augumented_images.append(fs.url(f'augumented/augumented_{i}_{filename}'))
        i+=1
        if i>=10:
            break
    return augumented_images
# @login_required(login_url ="/loginp/")
def uploadImg(request):
    if request.method=='POST' and request.FILES['image']:
        image=request.FILES['image']
        fs=FileSystemStorage()
        filename=fs.save(image.name,image)
        uploaded_file_url=fs.url(filename)

        # apply gaussian filter
        image_path=os.path.join(settings.MEDIA_ROOT,filename)
        image=cv2.imread(image_path,cv2.IMREAD_COLOR)

        gaussian_image=cv2.GaussianBlur(image,(5,5),1.0)
        median_image=cv2.medianBlur(image,15)
        bilateral_image=cv2.bilateralFilter(image,9,75,75)
        # CLathe
        gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        clahe=cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
        clahe_image =clahe.apply(gray)
        clahe_image=cv2.cvtColor(clahe_image,cv2.COLOR_GRAY2BGR)

        # save data after apply filter 
        gaussian_image_path=os.path.join(settings.MEDIA_ROOT,'gaussian/gaussian_'+filename)
        cv2.imwrite(gaussian_image_path,gaussian_image)
        median_image_path=os.path.join(settings.MEDIA_ROOT,'median/median_'+filename)
        cv2.imwrite(median_image_path,median_image)
        bilateral_image_path=os.path.join(settings.MEDIA_ROOT,'bilateral/bilateral_'+filename)
        cv2.imwrite(bilateral_image_path,bilateral_image)
        clahe_image_path=os.path.join(settings.MEDIA_ROOT,'clahe/clahe_'+filename)
        cv2.imwrite(clahe_image_path,clahe_image)


        gaussian_image_url=fs.url('gaussian/gaussian_'+filename)
        median_image_url=fs.url('median/median_'+filename)
        bilateral_image_url=fs.url('bilateral/bilateral_'+filename)
        clahe_image_url=fs.url('clahe/clahe_'+filename)

        augmented_images=augmented_images_fun(image,filename);

        return render(request,'formImgUpload.html',{
            'gaussian_image_url':gaussian_image_url,
            'median_image_url':median_image_url,
            'bilateral_image_url':bilateral_image_url,
            'clahe_image_url':clahe_image_url,
            'augmented_images':augmented_images

        })
    return render(request, 'formImgUpload.html')


def previous3Checkup(request):
    return render(request, 'previous3Checkup.html')
def register_patient(request):
    return render(request, 'register_patient.html')
def results(request):
    return render(request, 'results.html')
def routes(request):
    return render(request,'routes.html')
# def upload_image(request):
#     queryset=Medical.objects
#     if request.method == 'POST':
#         image = request.FILES['image']
#         queryset.create(Eimage=image)
#         queryset.save()
        
#         # Extract metadata
#         metadata = extract_metadata(image_path)
        
#         return render(request, 'image_upload.html', {'image_path': image_path, 'metadata': metadata})
    
#     return render(request, 'image_upload.html')

# Create your views here.
