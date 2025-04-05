from django.urls import path

from . import views

urlpatterns = [
	path('index/',views.index,name="index"),
	path("upload/",views.upload_data_view,name="upload_data"),
	# path("requestml/",views.requestmlresult),
	path("requestnttu",views.requestnttu),
	path("requesttai",views.requesttaipower),
	path("test",views.test),
	path('predictionresult/edit',views.edit_result,name='edit_result'),

	#
]
