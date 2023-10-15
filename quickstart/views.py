import datetime
import json
from django.contrib.auth.models import User, Group
from rest_framework import viewsets
from rest_framework import permissions
from rest_framework.response import Response
from rest_framework.decorators import api_view
from quickstart.serializers import UserSerializer, GroupSerializer
from quickstart.models import  edata
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

@api_view(['GET'])
def predict_view(request):

    # Get the value parameter from the query string
    value = float(request.query_params.get('value'))
    userid=request.query_params.get('userid')
    value = '%.2f'%value
    print(value,userid)

    try:
        # Get the edata object for today's date
        d = edata.objects.get(date=str(datetime.datetime.now().date()),userid=userid)

        # Add the value to the elet list and save the edata object
        elist = json.JSONDecoder().decode(d.elet)
        elist.append(value)
        d.elet=json.dumps(elist)
        l=len(elist)
        d.save()
        elesum=sum(map(float,elist))
        # 預測
        if len(elist)==96:
            a = [i for i in range(96)]
            a.insert(0, "Report_time")
            elist=list(map(float,elist))
            elist.insert(0,d.date)
            data=pd.DataFrame(elist,index=a)
            data=data.transpose()
            print(data)
            user00_model_load = Model_load(user_id="user00")
            predict = user00_model_load.predict(data)
            return Response({'result': predict,'sum':elesum,'len':96})

    except edata.DoesNotExist:
        # If there is no edata object for today's date, create a new one
        d = edata()
        d.userid=userid
        d.date=datetime.datetime.now().date()
        d.elet = json.dumps([value])
        d.save()
        elesum=value
        l=1

    return Response({'result':'success','sum':elesum,'len':l})
@api_view(['GET'])
def testdelet(request):
    d = edata.objects.get(date=str(datetime.datetime.now().date()),userid=0)
    elist = json.JSONDecoder().decode(d.elet)
    elist.pop(-1)
    datalen=len(elist)
    d.elet = json.dumps(elist)
    d.save()
    return Response({'len':datalen})