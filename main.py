# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 10:54:41 2022

@author: user
"""

import numpy as np
import pandas as pd

#%% 데이터 불러오기
data = pd.read_excel("C:/Users/user/Documents/GitHub/Industry-University-Project/data/플레도AI블록_아이정보포함_학습데이터_220628.xlsx",header= 1)
user = pd.read_excel("C:/Users/user/Documents/GitHub/Industry-University-Project/data/아이정보_데이터_220628.xlsx",header= 1)

data["아이 고유 식별 값"].value_counts()
data["아이 성별"].value_counts()
data["컨텐츠 분류1"].value_counts()
data["향상 능력"].value_counts()
data["문제풀이 소요시간"].value_counts()
data["아이 나이"] = data["아이 생년월일"].apply(lambda x : 2023 - int(str(x)[:4])) # 생년월일로 나이 계산 후 저장
                     
#user
data["아이 고유 식별 값"]
user_index1 = data[data["아이 나이"]<=4]["아이 고유 식별 값"].drop_duplicates() #3        
user_index2 =data[data["아이 나이"]==5]["아이 고유 식별 값"].drop_duplicates()#32
user_index3 =data[data["아이 나이"]==6]["아이 고유 식별 값"].drop_duplicates() #29
user_index4 =data[data["아이 나이"]==7]["아이 고유 식별 값"].drop_duplicates()#17
user_index5 =data[(data["아이 나이"]>=8) & (data["아이 나이"]<=10)]["아이 고유 식별 값"].drop_duplicates() #14    

#문제풀이 소요시간 이상치 제거
data[data["문제풀이 소요시간"]<34]["문제풀이 소요시간"].plot() #이상치 제거 후 63366-> 63335
data = data[data["문제풀이 소요시간"]<34]

#예체능 분야 문제 고유 식별값으로 중복 제거
data_drop = data.drop_duplicates(["아이 고유 식별 값","문제 고유 식별 값"],keep ="last")

a = data_drop[data_drop["아이 나이"]<=4].groupby(["컨텐츠 분류1"]).size()
b = data[data["아이 나이"]==5]["향상 능력"].value_counts()
data[data["아이 나이"]==6]["향상 능력"].value_counts()
data[data["아이 나이"]==7]["향상 능력"].value_counts()
data[(data["아이 나이"]>=8) & (data["아이 나이"]<=10)]["향상 능력"].value_counts()

#나이별 컨텐츠분류

#%%ANOVA 분석
import scipy.stats as stats
import pandas as pd
import urllib
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
import matplotlib.pyplot as plt
import numpy as np
%matplotlib inline
#한글 폰트 깨짐 방지
from matplotlib import font_manager, rc
font_path = "C:/Windows/Fonts/NGULIM.TTF"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)



#일원배치 분산분석(연령대가 문제풀이 소요시간에 영향을 미치는지?)
user_1 = data[data["아이 나이"]<=4]["문제풀이 소요시간"]
user_2 = data[data["아이 나이"]==5]["문제풀이 소요시간"]
user_3 = data[data["아이 나이"]==6]["문제풀이 소요시간"]
user_4 = data[data["아이 나이"]==7]["문제풀이 소요시간"]
user_5 =data[(data["아이 나이"]>=8) & (data["아이 나이"]<=10)]["문제풀이 소요시간"]

F_statistic, pVal = stats.f_oneway(user_1,user_2,user_3,user_4,user_5)

print('Altman 910 데이터의 일원분산분석 결과 : F={0:.1f}, p={1:.10f}'.format(F_statistic, pVal))
if pVal < 0.05:
    print('P-value 값이 충분히 작음으로 인해 그룹의 평균값이 통계적으로 유의미하게 차이납니다.')
    
# matplotlib plotting
plot_data = [user_1, user_2,user_3,user_4,user_5]
ax = plt.boxplot(plot_data,labels =["1세~4세","5세","6세","7세","8세~10세"] )
plt.title("나이 그룹별 문제풀이 소요시간")
plt.xlabel("나이")
plt.ylabel("문제풀이 소요시간")
plt.show()
    
# 일원배치 분산분석(성별이 문제풀이 소요시간에 영향을 미치는가?)
male = data[data["아이 성별"]=="MALE"]["문제풀이 소요시간"]
female = data[data["아이 성별"]=="FEMALE"]["문제풀이 소요시간"]

F_statistic, pVal = stats.f_oneway(male,female)
print('Altman 910 데이터의 일원분산분석 결과 : F={0:.1f}, p={1:.10f}'.format(F_statistic, pVal))
if pVal < 0.05:
    print('P-value 값이 충분히 작음으로 인해 그룹의 평균값이 통계적으로 유의미하게 차이납니다.')
    
# matplotlib plotting
plot_data = [male,female]
ax = plt.boxplot(plot_data,labels =["남자","여자"] )
plt.title("성별에 따른 문제풀이 소요시간")
plt.xlabel("성별")
plt.ylabel("문제풀이 소요시간")
plt.show()
#%% 카이제곱 분석

#이원카이제곱검정 : 독립성 검정(연령대가 컨텐츠 선택에 영향을 미치는가?)
from scipy.stats import chi2_contingency
append_data = pd.Series([0],index=["미술"])
contents_1 = data_drop[data_drop["아이 나이"]<=4]["컨텐츠 분류1"].value_counts().append(append_data)
contents_2 = data_drop[data_drop["아이 나이"]==5]["컨텐츠 분류1"].value_counts()
contents_3 = data_drop[data_drop["아이 나이"]==6]["컨텐츠 분류1"].value_counts()
contents_4 = data_drop[data_drop["아이 나이"]==7]["컨텐츠 분류1"].value_counts()
contents_5 = data_drop[(data_drop["아이 나이"]>=8) & (data_drop["아이 나이"]<=10)]["컨텐츠 분류1"].value_counts()

chi2, pvalue, dof, expected = chi2_contingency([contents_2,contents_3,contents_4,contents_5])

msg = 'Test Statistic: {}\np-value: {}\nDegree of Freedom: {}'
print(msg.format(chi2, pvalue, dof))
print(expected)

# matplotlib plotting
plot_data = pd.DataFrame([contents_1, contents_2,contents_3,contents_4,contents_5],index =["1세~4세","5세","6세","7세","8세~10세"] )
plot_data.plot(kind ="bar")
plt.title("나이 그룹별 컨텐츠 선택")
plt.xlabel("나이")
plt.ylabel("컨텐츠 선택 개수")
plt.show

#이원카이제곱검정 : 독립성 검정(연령대가 향상 능력에 영향을 미치는가?)
from scipy.stats import chi2_contingency
append_data = pd.Series([0,0],index=["상상력","창의력"])
ability_1 = data_drop[data_drop["아이 나이"]<=4]["향상 능력"].value_counts().append(append_data)
ability_2 = data_drop[data_drop["아이 나이"]==5]["향상 능력"].value_counts()
ability_3 = data_drop[data_drop["아이 나이"]==6]["향상 능력"].value_counts()
ability_4 = data_drop[data_drop["아이 나이"]==7]["향상 능력"].value_counts()
ability_5 = data_drop[(data_drop["아이 나이"]>=8) & (data_drop["아이 나이"]<=10)]["향상 능력"].value_counts()

chi2, pvalue, dof, expected = chi2_contingency([ability_2,ability_3,ability_4,ability_5])

msg = 'Test Statistic: {}\np-value: {}\nDegree of Freedom: {}'
print(msg.format(chi2, pvalue, dof))
print(expected)

# matplotlib plotting
plot_data = pd.DataFrame([ability_1, ability_2, ability_3,ability_4,ability_5],index =["1세~4세","5세","6세","7세","8세~10세"] )
plot_data.plot(kind ="bar")
plt.title("나이 그룹별 향상능력 분류")
plt.xlabel("나이")
plt.ylabel("향상 능력별 개수")
plt.show

# 성별이 컨텐츠 선택에 영향을 미치는가?
male = data_drop[data_drop["아이 성별"]=="MALE"]["컨텐츠 분류1"].value_counts()
female = data_drop[data_drop["아이 성별"]=="FEMALE"]["컨텐츠 분류1"].value_counts()

chi2, pvalue, dof, expected = chi2_contingency([male,female])

msg = 'Test Statistic: {}\np-value: {}\nDegree of Freedom: {}'
print(msg.format(chi2, pvalue, dof))
print(expected)

# matplotlib plotting
plot_data = pd.DataFrame([male,female],index =["남자","여자"] )
plot_data.plot(kind ="bar")
plt.title("성별에 따른 컨텐츠 선택")
plt.xlabel("성별")
plt.ylabel("컨텐츠 선택 개수")
plt.show

# 성별이 향상 능력에 영향을 미치는가?
male = data_drop[data_drop["아이 성별"]=="MALE"]["향상 능력"].value_counts()
female = data_drop[data_drop["아이 성별"]=="FEMALE"]["향상 능력"].value_counts()

chi2, pvalue, dof, expected = chi2_contingency([male,female])

msg = 'Test Statistic: {}\np-value: {}\nDegree of Freedom: {}'
print(msg.format(chi2, pvalue, dof))
print(expected)

# matplotlib plotting
plot_data = pd.DataFrame([male,female],index =["남자","여자"] )
plot_data.plot(kind ="bar")
plt.title("성별에 따른 향상 능력 분류")
plt.xlabel("성별")
plt.ylabel("향상 능력별 개수")
plt.show

#%%시각화
import matplotlib.pyplot as plt
import seaborn as sns
# 도화지
plt.style.use("default")
fig, ax = plt.subplots()
fig.set_size_inches(12, 9)

# 평균
ax.plot(["Topic1","Topic2","Topic3","Topic4"], pd.DataFrame(A_rate_list).quantile(q=0.5),linestyle = "-",color = "black",linewidth = 4)
ax.plot(["Topic1","Topic2","Topic3","Topic4"], pd.DataFrame(B_rate_list).quantile(q=0.5), linestyle = "--",color = "black",linewidth = 4)

# x축, y축 폰트 사이즈
ax.tick_params(axis = 'x', labelsize = 10)
ax.tick_params(axis = 'y', labelsize = 10)

#legend
ax.set_xlabel("Name of topics",fontsize = 15)
ax.set_ylabel("Average topic ratio",fontsize = 15)
plt.legend(['After_COVID19', 'Before_COVID19'],fontsize =15)
plt.title("Rate of topics",fontsize = 20)

plt.show()