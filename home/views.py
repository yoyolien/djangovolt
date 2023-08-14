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
def chart(request):
    month = int(request.GET.get('m'))
    # year = int(request.GET.get('y'))
    prediction = predictionresult.objects.filter(user_id=request.user.id,date__month=month)
    print(prediction,request.user)
    result = list(map(lambda x: x.result[-5], prediction))
    ele = eledata.objects.filter(user_id=request.user.id,report_time__month=month)
    label = [i.report_time.day for i in ele]
    dayusage = [sum(map(float, i.daliyusage.split(","))) / 1000 for i in ele]

    # for i in ele:
    #     month = i.report_time.month
    #     usage = sum(map(float, i.daliyusage.split(","))) / 1000
    #     usage = float('%.2f' % usage)
    #     label[month - 1].append(i.report_time.day)
    #     dayusage[month - 1].append(usage)

    wu = float('%.2f' % sum(dayusage))
    print(label,dayusage)
    context={
        'ele': dayusage,
        'date': label,
        'presult': result,
        "todayusage":dayusage,
        "wholeusage":wu,
        'standard': wu/len(label),
    }

    return render(request, 'chart.html', context)
def upload_data_view(request):
    if request.method == 'POST' and request.FILES.get('csv_file'):
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
                    data = eledata(user=user_id, report_time=report_time, daliyusage=daliyusage,id=str(report_time)+str(user_id))
                    data.save()
        csvfile.close()

        return redirect("dashboard")  # 轉到上傳成功的頁面

    return render(request, 'upload.html')  # 顯示上傳表單


def requestmlresult(u):
    ele = eledata.objects.filter(user_id=u.id).exclude(id__in=predictionresult.objects.filter(user_id=u.id).values_list('id', flat=True))
    for e in ele:
        mlapiurl = f"http://127.0.0.1:8000/predict/?value=[{e.daliyusage}]&date={e.id[:10]}"
        response = requests.get(mlapiurl)
        if response.status_code == 200:
            print(1)
            result = predictionresult(
                user=u,
                date=e.id[:10],
                result=json.loads(response.content)["result"],
                id=e.id
            )
            result.save()
    return

def test(request):
    a=Slide.objects.all()
    print(a)
    return render(request,'test.html',{"fslide":a[0],'slides':a[1:]})

# def requestapi(request):
#     userid=request.GET.get('u')
#     electdata=request.GET.get('e')
#     mlapiurl = f"http://127.0.0.1:8000/predict/?value={electdata}&userid={userid}"
#     print(userid,electdata)
#     response = requests.get(mlapiurl)
#     print(response.status_code)
#     # print(response.content)
#     result = json.loads(response.content)
#     try:
#         d = predictionresult.objects.get(userid=str(userid))
#         pr = json.JSONDecoder().decode(d.result)
#         if result['len']<96:
#             pr[str(datetime.now().date())] = [result['sum'],0]
#         elif result['len']==96:
#             pr[str(datetime.now().date())].append(result['result']['label'][0])
#             pr[str(datetime.now().date())][0] = result['sum']
#         d.result = json.dumps(pr)
#         d.save()
#     except predictionresult.DoesNotExist:
#         return redirect('index')
#     d = predictionresult.objects.get(userid=str(userid))
#     context={
#         "eledata":d,
#     }
#     return render(request,'test.html',context=context)
