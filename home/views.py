from django.shortcuts import render, redirect
from django.http import HttpResponse
from home.models import predictionresult
from datetime import datetime
import requests
import json
# Create your views here.

def index(request):

    # Page from the theme 
    return render(request, 'pages/index.html')

def createuser(request):
    userid = request.GET.get('u')
    try:
        user=predictionresult.objects.get(userid=userid)
        return redirect('index')
    except:
        d=predictionresult()
        d.userid=userid
        pr={}
        d.result = json.dumps(pr)
        d.save()
        return redirect('index')
def requestapi(request):
    userid=request.GET.get('u')
    electdata=request.GET.get('e')
    mlapiurl = f"http://127.0.0.1:8000/predict/?value={electdata}&userid={userid}"
    print(userid,electdata)
    response = requests.get(mlapiurl)
    print(response.status_code)
    # print(response.content)
    result = json.loads(response.content)
    try:
        d = predictionresult.objects.get(userid=str(userid))
        pr = json.JSONDecoder().decode(d.result)
        if result['len']<96:
            pr[str(datetime.now().date())] = [result['sum']]
        elif result['len']==96:
            pr[str(datetime.now().date())].append(result['result']['label'][0])
            pr[str(datetime.now().date())][0] = result['sum']
        d.result = json.dumps(pr)
        d.save()
    except predictionresult.DoesNotExist:
        return redirect('index')
    d = predictionresult.objects.get(userid=str(userid))
    context={
        "eledata":d,
    }
    return render(request,'test.html',context=context)

def todashboard(request):
    a=predictionresult.objects.all()
    return render(request,'test.html',{'a':a})