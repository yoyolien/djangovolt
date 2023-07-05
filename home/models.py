from django.db import models
from django.db.models.signals import post_save
from django.core.signals import request_started
from django.dispatch import receiver
import os
# Create your models here.
class predictionresult(models.Model):
    userid=models.TextField(null=False,primary_key=True)
    result=models.TextField(null=False)
class Slide(models.Model):
    id= models.TextField(primary_key=True)
    image = models.ImageField(upload_to='slides')
    title = models.CharField(max_length=100)
    description = models.TextField()
    def __str__(self):
        return self.id
