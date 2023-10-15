from django.contrib.postgres.fields import ArrayField
from django.db import models

# Create your models here.
class edata(models.Model):
    userid = models.TextField(null=False, primary_key=True)
    date = models.DateField(auto_now_add=True, auto_now=False,null=False)
    elet = models.TextField(null=True)
class a(models.Model):
    a=models.TextField()