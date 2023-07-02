from django.db import models

# Create your models here.
class predictionresult(models.Model):
    userid=models.TextField(null=False,primary_key=True)
    result=models.TextField(null=False)
class Slide(models.Model):
    image = models.ImageField(upload_to='slides/')  # 幻灯片图片
    title = models.CharField(max_length=100)  # 幻灯片标题
    description = models.TextField()  # 幻灯片描述
    order = models.PositiveIntegerField(default=0)  # 幻灯片顺序

    def __str__(self):
        return self.title