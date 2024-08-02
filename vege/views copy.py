from django.shortcuts import render,redirect
from .models import *
from django.http import HttpResponse
from django .contrib.auth.models import User
from django.contrib import messages
from django.contrib.auth import authenticate
def receipes(request):
    if request.method=="POST":
        data=request.POST
        receipe_image=request.FILES.get('receipe_image')
        receipe_name=data.get('receipe_name')
        receipe_description=data.get('receipe_description')
        Receipe.objects.create(
             receipe_image=receipe_image,
             receipe_name=receipe_name,
             receipe_description=receipe_description,
        )
        return redirect('/receipes')
    queryset=Receipe.objects.all()
    if request.GET.get('search'):
       queryset=quesryset.filter(receipe_name__icontains=request.GET.get('search'))

    context={'receipes':queryset}
    print(queryset)
    return render(request,'receipes.html',context)


def update_receipe(request,id):
    queryset=Receipe.objects.get(id=id)
    if request.method=="POST":
        data=request.POST
        receipe_image=request.FILES.get('receipe_image')
        receipe_name=data.get('receipe_name')
        receipe_description=data.get('receipe_descriptioin')

        queryset.receipe_name=receipe_name
        queryset.receipe_description=receipe_description
        if(receipe_image):
            queryset.receipe_image=receipe_image
        
        queryset.save()
        return redirect('/receipes/')
    context={'receipe':queryset}
    return render(request,'update_receipes.html',context)
def delete_receipe(request,id):
    queryset=Receipe.objects.get(id=id)
    queryset.delete()
    return redirect('/receipes')
def logoutp(request):
    logout(request)
    return redirect('/login/')
def loginp(request):
    if request.method=='POST':
        
        username=request.POST.get('username')
        password=request.POST.get('password')
        if not User.objects.filter(username=username).exists():
            messages.error(request,'Invalid Username')
            return redirect('/loginp/')
        user=authentication(username=username,password=password)
        if user is None:
            messages.error(request,'Invakid Password')
            return redirect('/login/')
        else:
            login(user)
            return redirect('/receipes/')
    return render(request,'loginp.html')

def register(request):
    if request.method=='POST':
        first_name=request.POST.get('first_name')
        last_name=request.POST.get('last_name')
        username=request.POST.get('username')
        password=request.POST.get('password')

        user=User.objects.filter(username=username)
        if user.exists():
            messages.info(request,'username alaready taken')
            return redirect('/register/')


        user=User.objects.create(
            first_name=first_name,
            last_name=last_name,
            username=username
        )
        user.set_password(password)
        user.save()
        messages.info(request,'Account created')
        return redirect('/register/')
    return render(request,'register.html')

