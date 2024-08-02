from django.contrib import admin
from .models import *
@admin.register(Image_Prediction)
class ImagePredictionAdmin(admin.ModelAdmin):
    list_display = (
        'id', 'user', 'email', 'mobile', 'healthy', 'mild', 'moderate', 'proliferate', 'severe', 'image', 'timestamp'
    )
    # Optional: You can also include search fields and filters if needed
    search_fields = ('user__username', 'email', 'mobile')
    list_filter = ('timestamp', 'user')
