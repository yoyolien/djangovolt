from django.contrib.auth.models import AnonymousUser
from django.shortcuts import render, redirect
from django.http import HttpResponse
from home.models import predictionresult,eledata,Slide
from datetime import datetime
import requests,json,csv,io
# Create your views here.

def index(request):

    # Page from the theme 
    return render(request, 'pages/index.html')
def upload_data_view(request):
    if request.method == 'POST' and request.FILES.get('csv_file'):
        # csvfile = request.FILES['csv_file']
        # reader = csv.reader(csvfile)
        # print(csvfile,reader)
        with io.TextIOWrapper(request.FILES["csv_file"], encoding="utf-8") as csvfile:
            reader = csv.reader(csvfile)
            user_id = request.user
            print(user_id)
            if str(user_id) =="AnonymousUser":
                return redirect("login")
            else:
                next(reader)  # 跳過標題列
                for row in reader:
                    print(row)
                    report_time = datetime.strptime(row[0], '%Y-%m-%d').date()
                      # 使用當前登入使用者的 ID
                    daliyusage = ','.join(row[1:])
                    data = eledata(user=user_id, report_time=report_time, daliyusage=daliyusage,id=str(report_time)+'-'+str(user_id))
                    data.save()
        csvfile.close()

        return redirect("dashboard")  # 轉到上傳成功的頁面

    return render(request, 'upload.html')  # 顯示上傳表單

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
            pr[str(datetime.now().date())] = [result['sum'],0]
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

def test(request):
    a=Slide.objects.all()
    print(a)
    return render(request,'test.html',{"fslide":a[0],'slides':a[1:]})
def testt():
    a=Slide.objects.all()
    print(a[0].image)
    return
