import datetime
import json

from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User, Group
from django.http import JsonResponse
from rest_framework import viewsets
from rest_framework import permissions
from rest_framework.response import Response
from rest_framework.decorators import api_view

from home.models import predictionresult, eledata
from quickstart.serializers import UserSerializer, GroupSerializer
import pandas as pd
from mlmodule.LoadModel import Model_load


class UserViewSet(viewsets.ModelViewSet):
    """
    API endpoint that allows users to be viewed or edited.
    """
    queryset = User.objects.all().order_by('-date_joined')
    serializer_class = UserSerializer
    permission_classes = [permissions.IsAuthenticated]


class GroupViewSet(viewsets.ModelViewSet):
    """
    API endpoint that allows groups to be viewed or edited.
    """
    queryset = Group.objects.all()
    serializer_class = GroupSerializer
    permission_classes = [permissions.IsAuthenticated]


def newpredict(u):
    edata = eledata.objects.filter(user_id=u.id).exclude(
        id__in=predictionresult.objects.filter(user_id=u.id).values_list('id', flat=True)
    )
    for e in edata:
        elist = e.daliyusage.split(",")
        # elist = json.JSONDecoder().decode(e.daliyusage)
        a = [i for i in range(96)]
        a.insert(0, "Report_time")
        elist = list(map(float, elist))
        elist.insert(0, e.report_time)
        data = pd.DataFrame(elist, index=a)
        data = data.transpose()
        user00_model_load = Model_load(user_id="user00")
        predict = user00_model_load.predict(data)
        result = predictionresult(
            user=u,
            date=e.id[:10],
            result=predict['label'][0],
            id=e.id,
            checked=0 if predict['label'][0] == 1 else 7
        )
        result.save()
    return JsonResponse({"result": 123})
