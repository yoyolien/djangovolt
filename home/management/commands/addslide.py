import os
from django.core.management.base import BaseCommand
from home.models import Slide

class Command(BaseCommand):
    help = 'Add slides to the database from static folder'

    def handle(self, *args, **options):
        Slide.objects.all().delete()
        slide_folder = 'static/slides' 
        slides=[]
        slides_id = list(Slide.objects.values_list('id', flat=True))
        if len(slides_id) !=len(os.listdir(slide_folder)):
            for index, filename in enumerate(os.listdir(slide_folder)):
                if filename not in slides_id:
                    if filename.startswith('slide'):
                        slide = Slide()
                        slide.image = os.path.join(slide_folder, filename)
                        slide.id = "img-"+str(index)
                        slide.save()
                        slides.append(slide)

        for index,slide in enumerate(slides):
            slide.prev_id = slides[index - 1].id
            slide.next_id = slides[(index + 1) % len(slides)].id
            slide.save()