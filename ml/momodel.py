


import pandas as pd
from selenium import webdriver
from bs4 import BeautifulSoup
import requests
import re
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from datetime import datetime
'''
path = 'C:/Python/chromedriver.exe'
# 농수산식품기업지원센터 가격정보 url
source_url = 'https://www.kamis.or.kr/customer/price/wholesale/period.do?action=monthly&yyyy=2021&period=5&countycode=&itemcategorycode=100&itemcode=152&kindcode=01&productrankcode=1&convert_kg_yn=N'

#드라이버 실행
try:
    driver = webdriver.Chrome(path)
    driver.get(source_url)
except:
    print("드라이버 오류")

# kg 단위 환산 누르기
try:
    kg_choice = driver.find_element_by_xpath("//*[@id='frm_search']/section/div[7]/a")
    kg_choice.click()
    time.sleep(2)
except:
    print("단위환산 오류")

# 검색 버튼 클릭
try:
    shear_1 = driver.find_element_by_xpath("//*[@id='btn_search']")
    shear_1.click()
    time.sleep(2)
except:
    print("검색버튼 오류")
'''

'''
# 기간 선택
montly_choice = driver.find_element_by_xpath("//*[@id='period']")
montly_choice.click()
time.sleep(2)

# 5년간 선택
years_choice = driver.find_element_by_xpath("//*[@id='period']/option[2]")
years_choice.click()
time.sleep(2)

# 서울 지역 선택
seoul = driver.find_element_by_xpath("//*[@id='ulcountycode']/li[2]/a")
seoul.click()
time.sleep(2)

# 품목, 품종
item = soup.find(name="h3", attrs={'class':'s_tit6 fl'}).get_text()
item = item.replace('\t', '')
item = item.replace('\n', '')
item
'''

'''
# 검색 결과 페이지 소스를 가져오기
try:
    html = driver.page_source
    soup = BeautifulSoup(html, "html.parser")
    time.sleep(2)
except:
    print("soup오류")

# 부류
category = soup.find(name="h3", attrs={'class':'s_tit5 fl'}).get_text()
category = category.replace('\t', '')
category = category.replace('\n', '')
category

# 등급
level = soup.find_all(name="h3", attrs={'class':'s_tit6 fl'})
level = str(level)
level = level.replace('[<h3 class="s_tit6 fl">', '').replace('</h3>,', '').replace('<h3 class="s_tit6 fl">', '').replace('</h3>]', '')
level = level.replace('\n', '').replace('\t', '').replace(':', '_').replace(' ', '').replace('\xa0\xa0', '')
level

# 단위
unit = soup.find(name="h3", attrs={'class':'s_tit6 color fl'}).get_text()
unit = unit.replace('\t', '')
unit = unit.replace('\n', '')
unit = unit.replace('\xa0\xa0', '')
unit

name = category + level + unit


# 테이블 목차 긁어오기
data = soup.find(name="table", attrs={'class':'wtable3'})
col_name = []
for c in data.find_all("tr"):
    datalist = []
    for d in c.find_all('th'):
        data_raw = d.get_text()
        datalist.append(data_raw)
    col_name.append(datalist)

# 빈 리스트 정리    
mokcha = list(filter(None, col_name))
mokcha

# 테이블 숫자 크롤링
infotable = soup.find(name="table", attrs={'class':'wtable3'})
info = []
for a in infotable.find_all("tr"):
    infolist = []
    for b in a.find_all("td"):
        info_raw = b.get_text()
        info_raw = info_raw.replace('-', '0')
        info_N = float(info_raw.replace(',', ''))
        infolist.append(info_N)
    info.append(infolist)

# 빈 리스트 정리
table_data = list(filter(None, info))
table_data

dataframe = pd.DataFrame(table_data, columns=mokcha)

# csv파일 저장
name_d = name.replace('/', '_')
name_d = name_d.replace(' ', '')
name_d = name_d.replace(',', '_')
name_d = name_d.replace(':', '_')
print(name_d)
print(dataframe)


dataframe.to_csv(name_d + '.csv', encoding='utf-8-sig')

try:
    driver.quit() #크롬브라우저 닫기.
except:
    print("드라이버 종료 오류")
'''

########################################## 데이터 받고 ds,y 로 저장 ##########################################
from fbprophet import Prophet
import pandas as pd
#potato_su_good = pd.read_csv(name_d + '.csv')
potato_su_good = pd.read_csv('C:/Users/whdrb/.spyder-py3/python/workspace/mon8_Project/식량작물_210801/식량작물_쌀_전체_등급_상품_단위_20kg.csv')

# 연평균 컬럼 따로 빼놓고 potato_su_good에서 삭제
potato_su_good_mean = potato_su_good['연평균']
del potato_su_good['연평균']

# 연도
import datetime
currentDateTime = datetime.datetime.now()
date = currentDateTime.date()
Year = date.strftime("%Y")
Year = int(Year)

# ds와 y로 변형
year = [Year-5, Year-4, Year-3, Year-2, Year-1, Year]
month = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
ds = []
for i in range(0,len(year)) :
    for j in range(0,len(month)) :
        ds.append(str(year[i]) + '-' + str(month[j]))
print(ds)

y = []
for i in range(0,len(year)) :
    for j in range(2,len(month)+2) :
        y.append(potato_su_good.iloc[i,j])
print(y)

df = pd.DataFrame({'ds' : ds, 'y' : y})
df = df[df['y'] != 0]     # 아직 값이 없는 날짜 삭제
df['ds'] = pd.to_datetime(df['ds']) #ds 컬럼을 시계열로
df = df.set_index('ds')

##############################################################################################################
############################################ ARIMA 모델 구축 ################################################
# 모델 구축
from sklearn.model_selection import train_test_split
train_data, test_data = train_test_split(df, test_size = 0.2, shuffle = False)

# 최적의 파라미터 찾기 시작
from statsmodels.tsa.arima_model import ARIMA
print('Examplt of parameter combinations for Seasonal ARIMA')
p = range(0, 3) #자기회귀 부분의 차수
d = range(1, 2) #1차 차분이 포함된 정도
q = range(0, 3) #이동 평균 부분의 차수
import itertools
pdq = list(itertools.product(p, d, q))

aic = []
try:
    for i in pdq:
        model = ARIMA(train_data.values, order=(i))
        model_fit2 = model.fit()
        print(f'ARIMA: {i} >> AIC : {round(model_fit2.aic, 2)}')
        aic.append(round(model_fit2.aic, 2))
except ValueError:
    print("파라미터가 맞지 않습니다.")

optimal = [(pdq[i], j) for i, j in enumerate(aic) if j == min(aic)]
optimal #AIC가 가장 적은 pdq값 산출.

############################################ ARIMA 모델 예측 ################################################

# 예측을 위해 1달 뒤, 5달 뒤의 년/월 출력하기
from datetime import datetime
last_month = int(df[-1:].index.strftime('%m')[0])
last_year = int(df[-1:].index.strftime('%Y')[0])

if len(str(last_month+1)) == 1 :
    start = str(str(last_year) + '-0' + str(last_month + 1))
elif last_month+1 > 12 :
    start = str(str(last_year+1) + '-0' + str((last_month + 1) - 12))
else :
    start = str(str(last_year) + '-' + str(last_month + 1))

if len(str(last_month+5)) == 1 :
    end = str(str(last_year) + '-0' + str(last_month + 5))
elif last_month+5 > 12 :
    end = str(str(last_year+1) + '-0' + str((last_month + 5) - 12))
else :
    end = str(str(last_year) + '-' + str(last_month + 5))
print(start)
print(end)

# ARIMA 분석을 위한 데이터 변형
import numpy as np
df_cut2 = df.iloc[0:len(df),]     
df_cut2_float = df_cut2[:].astype(np.float)     # ARIMA에 적합한 float으로 바꾸기
# df_cut2_float.plot()     # 그래프로 확인

# ARIMA 적용
from statsmodels.tsa.arima_model import ARIMA
model = ARIMA(df_cut2_float, order=optimal[0][0])
model_fit = model.fit(trend='nc',full_output=True, disp=1) 
print(model_fit.summary())


############################################### png 파일 저장 ################################################
# 미래 예측
preds = model_fit.predict(start, end, typ='levels') 
preds

# 실제가격과 예측가격 그래프 출력 + 이미지 저장
import matplotlib.pyplot as plt
plt.rc('font',family='Malgun Gothic')

model_fit.plot_predict()
title_font = {'fontsize': 16,'fontweight': 'bold'}
plt.title('실제가격과 예측가격',fontdict=title_font,loc='left',pad=20)
plt.legend(['예측값','실제값'])
plt.savefig('./ARIMA_1.png')
plt.close("all")

# 그래프 한 번에 출력
fig = plt.figure(figsize = (10,5))
ax1 = fig.add_subplot(1,2,1)# 전체 그래프 + 예측 그래프
ax1.plot(df_cut2_float)     # 현재 값
ax1.plot(preds,'r--')     # 2021년 12월까지의 예측 값
title_font = {'fontsize': 16,'fontweight': 'bold'}
ax1.set_title('가격현황 및 추이',fontdict=title_font,loc='left',pad=20)
ax2 = fig.add_subplot(1,2,2) # 예측 그래프 (크게 보기 위해)
ax2.plot(preds,'r--')
ax2.tick_params (axis = 'x', labelrotation =45) 
title_font = {'fontsize': 16,'fontweight': 'bold'}
ax2.set_title('가격 추이 [확대 버전]',fontdict=title_font,loc='left',pad=20)
fig.savefig('./ARIMA_2.png')
plt.close("all")


##############################################################################################################
################################################ Prophet #####################################################
# df 재설정 (ARIMA 적용하기 위해 변형했던 df 원래 형태로 재설정)
df = pd.DataFrame({'ds' : ds, 'y' : y})
df = df[df['y'] != 0]     # 아직 값이 없는 날짜 삭제

# r2 score & rmse를 구하기 위해 과거 데이터만 가지고 prophet
prophet = Prophet(seasonality_mode='multiplicative', #seasonality_mode='multiplicative' : 점차 증가하는 시계열 데이터에 대한 계절성.
                  yearly_seasonality=True,
#                  weekly_seasonality=True,
#                  daily_seasonality=True,                  
                  changepoint_prior_scale=0.99) #Trend 변화 민감도 설정.

prophet.fit(df)     # 학습하기

# 4개월 앞을 예측하기
future_data = prophet.make_future_dataframe(periods=5,freq='M')
# 예측하기
forecast_data = prophet.predict(future_data)

# 실제가격과 예측가격간의 차이 분석 => 성능
yy = df.y.values[:]     # 실제데이터.
yy_pred = forecast_data.yhat.values[:-5] #예측한 결과의 오늘날짜부분 까지만.

# r2 score RMSE값을 출력하기
from sklearn.metrics import mean_squared_error
from math import sqrt
yy = yy.reshape(-1, 1)
yy_pred = yy_pred.reshape(-1, 1)
from sklearn.linear_model import LinearRegression
model2 = LinearRegression()
model2.fit(yy,yy_pred)
r2 = model2.score(yy,yy_pred)
r2     # 0.919455827260635 : 91.9%의 설명력을 가짐
rmse = sqrt(mean_squared_error(yy,yy_pred))     # rmse는 낮을수록 좋음
rmse      # 3829.71376809466


# 실제가격과 예측가격간의 차이 분석 => 성능
days = len(df)
yy_hat = forecast_data.yhat.values[days:]

# 예측최소데이터
yy_pred_lower = forecast_data.yhat_lower.values[days:]
# 예측최대데이터
yy_pred_upper = forecast_data.yhat_upper.values[days:]

############################################### png 파일 저장 ################################################
# 예상가격 그래프 + 이미지로 저장
import matplotlib.pyplot as plt
plt.figure(2)
plt.rc('font',family='Malgun Gothic')
plt.plot(yy_pred_upper,color='blue', label='예상 가격 (최대)')     # 모델이 예상한 최대가격 그래프
plt.plot(yy_hat,color='gold', label='예상 가격')     # 모델이 예상한 가격 그래프
plt.plot(yy_pred_lower,color='red', label='예상 가격 (최소)')     # 모델이 예상한 최소가격 그래프
plt.legend()
plt.xlabel('개월')
plt.ylabel('가격')
title_font = {'fontsize': 16,'fontweight': 'bold'}
plt.title('향후 4개월의 가격 예측 그래프',fontdict=title_font,loc='left',pad=20)
plt.savefig('./Prophet_1.png')
plt.close("all")


# 예측/가격변화 그래프 + 이미지로 저장
import matplotlib
matplotlib.rcParams['axes.unicode_minus'] = False
fig1 = prophet.plot(forecast_data)
plt.xlabel('연도')
plt.ylabel('가격')
plt.savefig('./Prophet_2.png')
plt.close("all")

# from fbprophet.plot import add_changepoints_to_plot
# add_changepoints_to_plot(fig1.gca(), prophet, forecast_data)     # 빨간 점선이 ChangePoint를 뜻하며, 빨간 실선은 Trend
fig2 = prophet.plot_components(forecast_data)
plt.savefig('./Prophet_3.png')
plt.close("all")

############################################### csv 파일 저장 ################################################
# csv 파일로 저장
from statsmodels.tsa.stattools import adfuller
result = adfuller(model_fit.resid)
adf = result[0]     # ADF는 정상성 확인
percent = list(result[4].keys())
result2 = list(result[4].values())
p_value = result[1] 
pred_date = []
for i in range(len(preds)):
    mon = datetime.today().month + i
    pred_date.append(str('%2d월 예상금액 : ' % mon))

data = {'ADF' : [adf], 'Critical_Values_1%': [result2[0]], 'Critical_Values_5%': [result2[1]], 'Critical_Values_10%': [result2[2]], 'P-value':[p_value], 'RMSE':[rmse], 'R2_score':[r2]}
data2 = {'MONTH' : pred_date,'result of ARIMA' : preds.values, 'result of Prophet' : yy_hat}
data_csv = pd.DataFrame(data)
data2_csv = pd.DataFrame(data2)
data_csv.to_csv('result.csv', index=False)   # ADF가 Critical Value 1% 값보다 작으므로 유의한 결과
data2_csv.to_csv('result of ARIMA & Prophet.csv', index=False, encoding='utf-8-sig')
