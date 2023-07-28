import json
from django.contrib.auth.models import User
import requests
from django.test import TestCase
import admin_volt.views as a
import home.views as h
from home.models import *
# Create your tests here.
user = User.objects.get(id=1)
class MyFunctionTestCasea(TestCase):
    ele = eledata.objects.filter(user_id=user.id)
    dayusage = [sum(map(float, e.daliyusage.split(","))) / 1000 for e in ele]
    print(dayusage)
