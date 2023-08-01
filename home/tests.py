import json
from django.contrib.auth.models import User
import requests
from django.test import TestCase
import admin_volt.views as a
import home.views as h
from home.models import *
# Create your tests here.
user = User.objects.get(id=2)
class MyFunctionTestCasea(TestCase):
    ele = eledata.objects.filter(user_id=user.id).exclude(
        id__in=predictionresult.objects.filter(user_id=user.id).values_list('id', flat=True))
    print(ele[0].report_time.month)

