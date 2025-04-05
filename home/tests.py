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
    a=h.requestmlresult(user)
    print(a)

