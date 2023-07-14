from django.urls import path

from . import views

urlpatterns = [
    path('index/', views.index, name="index"),
    path('requestapi/', views.requestapi),
    path("upload/",views.upload_data_view,name="upload_data"),

]
