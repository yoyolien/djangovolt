from django.contrib.auth.models import AnonymousUser
from django.shortcuts import render,redirect
from django.http import HttpResponse,JsonResponse
from selenium.webdriver.common.by import By

from home.models import predictionresult,eledata,Slide
from datetime import datetime
import requests,json,csv,io
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.edge.service import Service
import time


# Create your views here.

def index(request):
	# Page from the theme
	return render(request,'pages/index.html')


def upload_data_view(request):
	if request.method == 'POST' and request.FILES.get('csv_file'):
		with io.TextIOWrapper(request.FILES["csv_file"],encoding="utf-8") as csvfile:
			reader = csv.reader(csvfile)
			user_id = request.user
			print(user_id)
			next(reader)  # 跳過標題列
			for row in reader:
				report_time = datetime.strptime(row[0],'%Y-%m-%d').date()
				# 使用當前登入使用者的 ID
				daliyusage = ','.join(row[1:])
				data = eledata(
					user=user_id,report_time=report_time,daliyusage=daliyusage,
					id=str(report_time) + str(user_id)
				)
				data.save()
	csvfile.close()
	return redirect("dashboard")  # 轉到上傳成功的頁面


def requestmlresult(u):
	ele = eledata.objects.filter(user_id=u.id).exclude(
		id__in=predictionresult.objects.filter(user_id=u.id).values_list('id',flat=True)
	)
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
	a = Slide.objects.all()
	print(a)
	return render(request,'test.html',{"fslide":a[0],'slides':a[1:]})


def requesttaipower():
	url = "https://www.taipower.com.tw/tc/news.aspx?mid=17"
	titles = []
	links = []
	imglinks = []
	# 發送GET請求並取得響應
	response = requests.get(url)
	# 檢查是否成功取得響應
	if response.status_code == 200:
		Slide.objects.all().delete()
		soup = BeautifulSoup(response.text,"html.parser")
		# 找到包含新聞列表的 div 元素
		box_list_div = soup.find("div",class_="box_list")
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
		slides = []
		for i,title in enumerate(titles):
			slide = Slide()
			slide.link = links[i]
			slide.image = imglinks[i]
			slide.description = title
			slide.id = "img-" + str(i + 1)
			slide.save()
			slides.append(slide)
		for index,slide in enumerate(slides):
			slide.prev_id = slides[index - 1].id
			slide.next_id = slides[(index + 1) % len(slides)].id
			slide.save()
	else:
		print("無法取得響應")


def requestnttu(reqest):
	service = Service(executable_path="msedgedriver.exe")
	op = webdriver.EdgeOptions()
	op.add_argument('--headless')
	driver = webdriver.Edge(service=service,options=op)
	url = "https://wdsa.nttu.edu.tw/p/403-1009-424-1.php?Lang=zh-tw"
	driver.get(url=url)
	a = driver.find_elements(By.CLASS_NAME,'mtitle')
	titles = []
	links = []
	for i in a:
		titles.append(i.text)
		links.append(i.find_element(By.TAG_NAME,'a').get_attribute('href'))
	a = {
		"title":titles,
		"link" :links
	}
	return JsonResponse(a)
