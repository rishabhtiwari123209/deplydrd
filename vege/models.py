from django.db import models
from django.contrib.auth.models import User

# Create your models here.
class Receipe(models.Model):
    user=models.ForeignKey(User,on_delete=models.SET_NULL,null=True,blank=True)
    receipe_name=models.CharField(max_length=100)
    receipe_description=models.TextField()
    receipe_image=models.ImageField(upload_to='receipe')
# class User



class Profile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    date_of_birth = models.DateField(null=True, blank=True)
    gender = models.CharField(max_length=10, choices=[('Male', 'Male'), ('Female', 'Female')], null=True, blank=True)
    contact_number = models.CharField(max_length=15, null=True, blank=True)
    address = models.TextField(null=True, blank=True)
    diabetic_status = models.CharField(max_length=50, choices=[('Type 1', 'Type 1'), ('Type 2', 'Type 2'), ('Pre-diabetes', 'Pre-diabetes'), ('None', 'None')], null=True, blank=True)
    additional_info = models.TextField(null=True, blank=True)

    def __str__(self):
        return self.user.username

