# Generated by Django 4.2.2 on 2023-07-10 14:41

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='predictionresult',
            fields=[
                ('userid', models.TextField(primary_key=True, serialize=False)),
                ('result', models.TextField()),
            ],
        ),
        migrations.CreateModel(
            name='Slide',
            fields=[
                ('id', models.TextField(primary_key=True, serialize=False)),
                ('image', models.ImageField(upload_to='slides')),
                ('link', models.TextField()),
                ('description', models.TextField()),
                ('prev_id', models.TextField(null=True)),
                ('next_id', models.TextField(null=True)),
            ],
        ),
    ]
