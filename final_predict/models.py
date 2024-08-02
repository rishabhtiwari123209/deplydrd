from django.db import models
from django.contrib.auth.models import User
class Image_Prediction(models.Model):
    user=models.ForeignKey(User,on_delete=models.SET_NULL,null=True,blank=True)
    email = models.EmailField()
    mobile = models.CharField(max_length=15)
    healthy = models.FloatField()
    mild = models.FloatField()
    moderate = models.FloatField()
    proliferate = models.FloatField()
    severe = models.FloatField()
    image = models.ImageField(upload_to='medical')
    timestamp = models.DateTimeField(auto_now_add=True)
