from django.urls import path
from .views import PushUpCountView

urlpatterns = [
    path('pushup-count/', PushUpCountView.as_view(), name='pushup-count'),

]
