from django.db import models
from django.contrib.auth.models import User
from django.db.models.signals import post_save
from django.core.signals import request_started
from django.dispatch import receiver
import os
# Create your models here.
class predictionresult(models.Model):
    user = models.OneToOneField(User,default=None, on_delete=models.CASCADE,primary_key=True)
    result=models.TextField(null=False)
class Slide(models.Model):
    id= models.TextField(primary_key=True)
    image = models.TextField()
    link = models.TextField()
    description = models.TextField()
    prev_id=models.TextField(null=True)
    next_id=models.TextField(null=True)
    def __str__(self):
        return self.id
class eledata(models.Model):
    id = models.TextField(primary_key=True)
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    report_time = models.DateField()
    daliyusage = models.TextField()