from django.db import models
from django.contrib.auth.models import User
class ImagePrediction(models.Model):
    user=models.ForeignKey(User,on_delete=models.SET_NULL,null=True,blank=True)
    # email=models.EmailField()
    healthy=models.FloatField()
    mild =models.FloatField()
    moderate=models.FloatField()
    poliferate=models.FloatField()
    severe=models.FloatField()
    image=models.ImageField(upload_to='medical')
    timestamp=models.DateTimeField(auto_now_add=True)
# class User
# Create your models here.
