"""
URL configuration for core project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from home.views import *
from mediccal.views import *
from vege.views import*
from oct.views import*
from ensemble_learn.views import*
from pretrain_model.views import*
from final_predict.views import*
from django.conf.urls.static import static 
from django.conf import settings 
urlpatterns = [
    path('admin/', admin.site.urls),
    path('',routes,name="routes"),
    path('home/',home,name="home"),
    path('loginp/',loginp,name="login"),
    path('logoutp/',logoutp,name="logout"),
    path('uploadImg/',uploadImg,name="preprocess_image"),
    path('register/',register,name="register"),
    path('pretrain_modalPridict/',pretrain_modalPridict,name="pretrain_modal"),
    path('automated_grading_system/',automated_grading,name="automated_grading"),
    path('profile/', profile, name='profile'),
    path('delete_prediction/<int:id>/', delete_prediction, name='delete_prediction'),
    path('oct_modalPridict/',oct_modalPridict,name="oct_modalPridict"),
    path('register_patient/',register_patient,name="rf"),
    path('previous3Checkup/',previous3Checkup,name="rw"),
     path('modalPridict/',model_pridiction,name="model_pridiction"),
     path('ensemblemodalPridict/',ensemble_model_pridiction,name="ensemble_model_pridiction")
]+ static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT) 
