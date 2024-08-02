from django.shortcuts import render,redirect
from .models import *
from django.http import HttpResponse
from django .contrib.auth.models import User
from django.contrib import messages
from django.contrib.auth import authenticate,login,logout
from django.contrib.auth.decorators import login_required


def home(request):
    return render(request,'drd1.html')
def logoutp(request):
    logout(request)
    return redirect('/loginp/')
def loginp(request):
    if request.method=='POST':
        
        username=request.POST.get('username')
        password=request.POST.get('password')
        if not User.objects.filter(username=username).exists():
            messages.error(request,'Invalid Username')
            return redirect('/loginp/')
        user=authenticate(username=username,password=password)
        if user is None:
            messages.error(request,'Invakid Password')
            return redirect('/loginp/')
        else:
            login(request,user)
            return redirect('/uploadImg/')
    return render(request,'loginp.html')

def register(request):
    if request.method == 'POST':
        first_name = request.POST.get('first_name')
        last_name = request.POST.get('last_name')
        username = request.POST.get('username')
        password = request.POST.get('password')
        date_of_birth = request.POST.get('date_of_birth')
        gender = request.POST.get('gender')
        contact_number = request.POST.get('contact_number')
        address = request.POST.get('address')
        diabetic_status = request.POST.get('diabetic_status')
        additional_info = request.POST.get('additional_info')

        user = User.objects.filter(username=username)
        if user.exists():
            messages.info(request, 'Username already taken')
            return redirect('/register/')

        user = User.objects.create(
            first_name=first_name,
            last_name=last_name,
            username=username
        )
        user.set_password(password)
        user.save()

        profile = Profile.objects.create(
            user=user,
            date_of_birth=date_of_birth,
            gender=gender,
            contact_number=contact_number,
            address=address,
            diabetic_status=diabetic_status,
            additional_info=additional_info
        )
        profile.save()

        messages.info(request, 'Account created')
        return redirect('/register/')
    return render(request, 'register.html')