import time
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.edge.service import Service

service = Service(executable_path="msedgedriver.exe")
op = webdriver.EdgeOptions()
# op.add_argument("--headless")
# op.add_argument('--no-sandbox')
# op.add_argument('--disable-gpu')
# op.add_argument('--disable-dev-shm-usage')
driver = webdriver.Edge(service=service,options=op)
url = "https://green.nttu.edu.tw/?Lang=zh-tw"
driver.get(url=url)
time.sleep(1)
elem = BeautifulSoup(driver.page_source,"html.parser")
reports=[]
for i in range(1,16):
	reports.append(elem.find("div",id=f"SDG{i}"))
for i,rep in enumerate(reports):
	print(
		str(rep.prettify())
		.replace("/var/file/","https://green.nttu.edu.tw/var/file/")
		.replace("collapse","carousel-item")
		.replace("row gy-5","row")
		)
driver.close()
