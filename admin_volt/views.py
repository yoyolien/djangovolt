import random,urllib.request

import requests
from bs4 import BeautifulSoup
from django.shortcuts import render, redirect
from admin_volt.forms import RegistrationForm, LoginForm, UserPasswordResetForm, UserPasswordChangeForm, UserSetPasswordForm
from django.contrib.auth.views import LoginView, PasswordResetView, PasswordChangeView, PasswordResetConfirmView
from django.contrib.auth import logout
from home.models import predictionresult,Slide
from django.contrib.auth.decorators import login_required
import json
import datetime
import os
def requesttaipower():
  url = "https://www.taipower.com.tw/tc/news.aspx?mid=17"
  save_path = "static/slides/"
  print(save_path)
  titles=[]
  links=[]
  imglinks=[]
  # 發送GET請求並取得響應
  response = requests.get(url)
  # 檢查是否成功取得響應
  if response.status_code == 200:
    Slide.objects.all().delete()
    soup = BeautifulSoup(response.text, "html.parser")
    # 找到包含新聞列表的 div 元素
    box_list_div = soup.find("div", class_="box_list")
    # 找到所有的 li 元素
    news_list = box_list_div.find_all("li")
    for i,news in enumerate(news_list):
          title = news.select_one("a").text.strip()
          # title=title.replace(" ", "").replace("\n", "").replace("\t", "")
          img = news.select_one("img")["src"]
          link = news.select_one("a")["href"]
          if "/upload/" in img:
            img_link = f"https://www.taipower.com.tw{img}"
          else:
            img_link = f"https://www.taipower.com.tw/tc/{img}"
          titles.append(title)
          links.append(f"https://www.taipower.com.tw/tc/{link}")
          imglinks.append(img_link)
    slides=[]
    for i,title in enumerate(titles):
      slide = Slide()
      slide.link=links[i]
      slide.image = imglinks[i]
      slide.description=title
      slide.id = "img-" + str(i+1)
      slide.save()
      slides.append(slide)
    for index, slide in enumerate(slides):
      slide.prev_id = slides[index - 1].id
      slide.next_id = slides[(index + 1) % len(slides)].id
      slide.save()
  else:
    print("無法取得響應")
# Index
def index(request):
  return render(request, 'pages/index.html')

# Dashboard
@login_required(login_url="/accounts/login/")
def dashboard(request):
  print(request.user)
  slides = list(Slide.objects.all())
  # 資料傳入dashboard.html
  # predictionandele = predictionresult.objects.get(userid=request.user.id)
  # result = json.JSONDecoder().decode(predictionandele.result)
  # label = [i for i in result]
  # dayusage = [result[i][0] for i in result]
  # dayusage = list(map(lambda x:x/1000,dayusage))
  # daypredictresult = [result[i][1] for i in result]
  # tree = 0.509*sum(dayusage)/550.5

  arr = [i for i in range(1, 6)]
  a = [i for i in range(1, 26)]
  b = []
  date = []
  ele = []

  for i in range(1,13):
    date.append([])
    ele.append([])
  for i in range(1, 31):
    date[3].append(str(datetime.date(2023, 4, i))[8:])
    ele[3].append(random.randint(10, 60))
    # b.append({'x: ' + str(datetime.date(2023, 4, i)), 'y: ' + str(random.randint(10, 60))})
  for i in range(1, 32):
    date[4].append(str(datetime.date(2023, 5, i))[8:])
    ele[4].append(random.randint(10, 60))
    # b.append({'x: ' + str(datetime.date(2023, 4, i)), 'y: ' + str(random.randint(10, 60))})
 

  tree = 7.5
  treec=[]
  for i in range(int(tree)+1):
    if i +1<tree:
      treec.append([100,0])
    else:
      treec.append([100*(tree-i),100-100*(tree-i)])
  print(treec,tree)
  print(slides)
  context = {
    'segment': 'dashboard',
    # 'ele': dayusage,
    # 'date': label,
    # 'presult': daypredictresult,
    # "todayusage":dayusage[-1],
    # "wholeusage":sum(dayusage),

    'arr': arr,
    'a': a,
    'date': date,
    'ele': ele,
    'standard': 35,
    'b': b,
    "treec":treec,
    "slides":slides[1:],
    "fslide":slides[0],
  }
  return render(request, 'pages/dashboard/dashboard.html', context)

# Pages
@login_required(login_url="/accounts/login/")
def transaction(request):
  context = {
    'segment': 'transactions'
  }
  return render(request, 'pages/transactions.html', context)

@login_required(login_url="/accounts/login/")
def settings(request):
  context = {
    'segment': 'settings'
  }
  return render(request, 'pages/settings.html', context)

# Tables
@login_required(login_url="/accounts/login/")
def bs_tables(request):
  context = {
    'parent': 'tables',
    'segment': 'bs_tables',
  }
  return render(request, 'pages/tables/bootstrap-tables.html', context)

# Components
@login_required(login_url="/accounts/login/")
def buttons(request):
  context = {
    'parent': 'components',
    'segment': 'buttons',
  }
  return render(request, 'pages/components/buttons.html', context)

@login_required(login_url="/accounts/login/")
def notifications(request):
  context = {
    'parent': 'components',
    'segment': 'notifications',
  }
  return render(request, 'pages/components/notifications.html', context)

@login_required(login_url="/accounts/login/")
def forms(request):
  context = {
    'parent': 'components',
    'segment': 'forms',
  }
  return render(request, 'pages/components/forms.html', context)

@login_required(login_url="/accounts/login/")
def modals(request):
  context = {
    'parent': 'components',
    'segment': 'modals',
  }
  return render(request, 'pages/components/modals.html', context)

@login_required(login_url="/accounts/login/")
def typography(request):
  context = {
    'parent': 'components',
    'segment': 'typography',
  }
  return render(request, 'pages/components/typography.html', context)


# Authentication
def register_view(request):
  if request.method == 'POST':
    form = RegistrationForm(request.POST)
    if form.is_valid():
      print("Account created successfully!")
      form.save()
      return redirect('/accounts/login/')
    else:
      print("Registration failed!")
  else:
    form = RegistrationForm()
  
  context = { 'form': form }
  return render(request, 'accounts/sign-up.html', context)

class UserLoginView(LoginView):
  form_class = LoginForm
  template_name = 'accounts/login-form-20/index.html'

class UserPasswordChangeView(PasswordChangeView):
  template_name = 'accounts/password-change.html'
  form_class = UserPasswordChangeForm

class UserPasswordResetView(PasswordResetView):
  template_name = 'accounts/forgot-password.html'
  form_class = UserPasswordResetForm

class UserPasswrodResetConfirmView(PasswordResetConfirmView):
  template_name = 'accounts/reset-password.html'
  form_class = UserSetPasswordForm

def logout_view(request):
  logout(request)
  return redirect('/accounts/login/')

def lock(request):
  return render(request, 'accounts/lock.html')

# Errors
def error_404(request):
  return render(request, 'pages/examples/404.html')

def error_500(request):
  return render(request, 'pages/examples/500.html')

# Extra
def upgrade_to_pro(request):
  return render(request, 'pages/upgrade-to-pro.html')