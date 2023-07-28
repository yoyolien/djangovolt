from django.urls import path

from . import views

urlpatterns = [
    path('index/', views.index, name="index"),
    path("upload/",views.upload_data_view,name="upload_data"),
    # path("requestml/",views.requestmlresult),

]
