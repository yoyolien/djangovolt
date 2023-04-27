from django.db import models

# Create your models here.
class predictionresult(models.Model):
    userid=models.TextField(null=False,primary_key=True)
    result=models.TextField(null=False)