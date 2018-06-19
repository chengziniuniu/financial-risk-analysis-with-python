# -*- coding: utf-8 -*-
"""
@author: cx
"""
import datetime
import math
import random
import pandas as pd
import tushare as ts
import numpy as np
import scipy.stats as sts
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import fmin_slsqp
from scipy.special import gamma

#下载数据并保存
def DownloadData(code,start = '2012-12-31',end = datetime.datetime.now().strftime("%Y-%m-%d"),index = False):
    df = ts.get_k_data(code,start = start,end = end,index = index) 
    df.to_csv(code + '.csv')
    print('代码为 ' + code + ' ，' + start + ' 至 ' +end + ' 的历史价格数据已下载。')

#读取数据，提取收盘价、对数收益率
def ReadCsv(code):
    name = code + '.csv'
    df = pd.read_csv(name)
    df.index = df["date"]
    clo = df['close']
    close = np.array(df.close)
    rts = (np.log(clo) - np.log(clo).shift(1).dropna()).dropna()
    return df,close,rts

#计算均值、方差、偏度、峰度、最大值、最小值
def describe(a,maxmin = 0):  
    mean = np.mean(a)
    var = np.var(a)
    skew = sts.skew(a)
    kurt = sts.kurtosis(a)
    Max = np.max(a)
    Min = np.min(a)
    if maxmin == 0:
        des = pd.Series([mean, var, skew, kurt],index = ["mean","variance","skewness","kurtosis"])
    else :
        des = pd.Series([mean, var, Max, Min],index = ["mean","variance","max","min"])
    return des

#画出acf图，并排版（3张）
def drawacf(data1,title1,data2,title2,data3,title3):
    fig,(ax0,ax1,ax2) = plt.subplots(ncols=3,figsize=(30,5))
    fig = sm.graphics.tsa.plot_acf(data1,lags = 100,ax = ax0,title = title1+' acf', alpha = 0.05,zero= False)
    fig = sm.graphics.tsa.plot_acf(data2,lags = 100,ax = ax1,title = title2+' acf', alpha = 0.05,zero= False)
    fig = sm.graphics.tsa.plot_acf(data3,lags = 100,ax = ax2,title = title3+' acf', alpha = 0.05,zero= False)

#画出直方图，并排版（3张）
def drawhist(data1,title1,data2,title2,data3,title3):
    fig,(ax0,ax1,ax2) = plt.subplots(ncols=3,figsize=(30,10)) 
    ax0.hist(data1,40)
    ax0.set_title(title1,fontsize = 20)
    ax1.hist(data2,40)
    ax1.set_title(title2,fontsize = 20)
    ax2.hist(data3,40)
    ax2.set_title(title3,fontsize = 20)

#画出QQ图，并排版（3张）
def drawqq(data1,title1,data2,title2,data3,title3):
    fig,(ax0,ax1,ax2) = plt.subplots(ncols=3,figsize=(30,5))
    fig = sm.qqplot(data1,line='s',ax = ax0)
    ax0.set_title(title1,fontsize = 20)
    fig = sm.qqplot(data2,line='s',ax = ax1) 
    ax1.set_title(title2,fontsize = 20)
    fig = sm.qqplot(data3,line='s',ax = ax2)
    ax2.set_title(title3,fontsize = 20)

#根据公式，求历史波动率，Close-to-Close方法
def CtC(data, N):
    T = np.size(data)
    std_CtC = np.zeros((T-N,1))
    for i in range(0,T-N):
        std_CtC[i] = np.sqrt((1/(N-1))*np.sum((data[i:i+N]-(1/N)*np.sum(data[i:i+N]))**2))
    return std_CtC

#根据公式，求历史波动率，Parkinson方法
def PHV(data,N):
    un = np.log(data.high/data.open)
    dn = np.log(data.low/data.open)
    T = np.size(data.close)
    std_P = np.zeros((T-N,1))
    for i in range(0,T-N):
        std_P[i] = np.sqrt((1/(4*N*np.log(2)))*np.sum((un[i:i+N]-dn[i:i+N])**2))
    return std_P
    
#根据公式，求历史波动率，Garman-Klass方法
def GKHV(data,N):
    un = np.log(data.high/data.open)
    dn = np.log(data.low/data.open)
    cn = np.log(data.close/data.open)
    T = np.size(data.close)
    std_GK = np.zeros((T-N,1))
    for i in range(0,T-N):
        pat1 = (0.511/N)*np.sum((un[i:i+N]-dn[i:i+N])**2)
        pat2 = (0.019/N)*np.sum(cn[i:i+N]*(un[i:i+N]+dn[i:i+N])-2*un[i:i+N]*dn[i:i+N])
        pat3 = (0.383/N)*np.sum(cn[i:i+N]**2)
        std_GK[i] = np.sqrt(pat1-pat2-pat3)
    return std_GK

#根据公式，求历史波动率，Rogers方法
def RHV(data,N):
    un = np.log(data.high/data.open)
    dn = np.log(data.low/data.open)
    cn = np.log(data.close/data.open)
    T = np.size(data.close)
    std_R = np.zeros((T-N,1))
    for i in range(0,T-N):
        std_R[i] = np.sqrt((1/N)*np.sum(un[i:i+N]*(un[i:i+N]-cn[i:i+N])+dn[i:i+N]*(dn[i:i+N]-cn[i:i+N])))
    return std_R

#画出历史波动率
def drawhv(df,rts,name):
    plt.grid()
    plt.plot(CtC(rts,30),linestyle="-",label="CtC")
    plt.plot(PHV(df,30),linestyle="-",label="Parkinson")    
    plt.plot(GKHV(df,30),linestyle="-",label="Garman-Klass")
    plt.plot(RHV(df,30),linestyle="-",label="Rogers")  
    plt.title(name + ' historical volatility',fontsize = 20)
    plt.legend(loc='upper left')

#画出ES、VAR
def drawes(df,name):
    data_VaR = - GKHV(df,30)*norm.ppf(0.01)
    data_ES = GKHV(df,30)*norm.pdf(norm.ppf(0.01))/0.01
    plt.grid()
    plt.plot(data_VaR,linestyle="-", label ="VaR")
    plt.plot(data_ES,linestyle="-", label ="ES")
    plt.title(name + ' VaR vs ES, p = 0.01, type =Garman-Klass',fontsize = 20)
    plt.legend(loc = "upper left")

#极大似然求GARCH（1，1）波动率
def garch_loglikelihood(parameters, rets, out = None):
    alpha = parameters[0]
    beta = parameters[1]
    omega = (1-alpha-beta)*rets.var()
    T =len(rets)
    sigma2 = np.ones(T)*rets.var()
    Rt2 = rets**2
    # Data and Sigma2 are assumed as T by 1 vectors
    for i in range(T-1):
        sigma2[i+1]=alpha*rets[i]**2+beta*sigma2[i] + omega
        
    logliks = 0.5*(np.log(2*np.pi) + np.log(sigma2) + Rt2/sigma2)
    loglik = np.sum(logliks)
    
    if out is None:
        return loglik/100
    else:
        return loglik, logliks, np.copy(sigma2)

#garch 的边界条件
def garch_constraint(parameters, rets, out=None):
    alpha = parameters[0]
    beta = parameters[1]
    return np.array([1-alpha-beta])

#画出garch11波动率
def garch_hv(data,name,p = 1):
    rts = np.array(data)
    startingVals = np.array([0.1,0.85]) 
    bounds  =[(0.0,1.0),(0.0,1.0)]
    args = (np.asarray(rts),)
    estimates = fmin_slsqp(garch_loglikelihood, startingVals, f_ieqcons=garch_constraint,bounds=bounds, args =args)   
    analized= 1
    __, _, sigma2final = garch_loglikelihood(estimates,rts,out=True)
    garch_vol = np.sqrt(analized * sigma2final)
    ested_vol = pd.DataFrame(garch_vol,columns=['estimated vols'])
    if p == 1:
        print('Initial Values=',startingVals)
        print('Estimated Values=',estimates)
        ested_vol.plot(grid='on',color = '#CC0066',title = name + ' volatility with GARCH11',figsize =(16,3) )
        plt.legend(loc='upper left')
    else :
        return garch_vol

#求Ngarch的似然函数
def Ngarch_loglikelihood(parameters, rets, out = False):
    alpha = parameters[0] 
    beta = parameters[1]
    theta = parameters[2]
    omega = parameters[3]
    Rt2 = rets**2
    T = len(rets)
    sigma2 = np.ones(T)*rets.var()
    for i in range(T-1):
        sigma2[i+1]  = omega + alpha*(rets[i] - theta*np.sqrt(sigma2[i]))**2 + beta*sigma2[i]
    logliks = 0.5*(np.log(2*np.pi) + np.log(sigma2) + Rt2/sigma2)
    loglik = np.sum(logliks)
    if out == False:
        return loglik/10000
    if out == True:
        return loglik,np.copy(sigma2)    

#Ngarch的边界条件
def Ngarch_constraint(parameters, data, out=None):
    alpha = parameters[0]
    beta = parameters[1]
    theta = parameters[2]
    return np.array([1-(alpha*(1+theta**2))-beta])

#画出NGARCH波动率
def Ngarch_hv(data,name,p = 1):
    rts = np.array(data)
    startingVals = np.array([0.07,0.85,0.5,0.000005]) 
    bounds =[(0.0,1.0),(0.0,1.0),(-10.0,10.0),(0.0,1.0)] 
    args = (np.asarray(rts),)
    estimates = fmin_slsqp(Ngarch_loglikelihood, startingVals,bounds=bounds,f_ieqcons=Ngarch_constraint, args = args)
    analized = 1
    _, sigma2final = Ngarch_loglikelihood(estimates,rts, out=True)
    garch_vol= np.sqrt(analized*sigma2final)
    ested_vol = pd.DataFrame(garch_vol,columns=['estimated vols'])
    if p == 1:
        print('Initial Values=',startingVals)
        print('Estimated Values=',estimates)
        ested_vol.plot(grid='on',color = '#B22222',title=name + ' volatility with NGACH',figsize=(16,3))
        vol = analized*sigma2final
        return vol
    if p == 2:
        return garch_vol
    else:
        return garch_vol,estimates

#成分garch的似然函数
def component_garch_loglikelihood(parameters ,rets, out = False):
    alpha_sigma = parameters[0] 
    beta_sigma = parameters[1]
    alpha_vega = parameters[2]
    beta_vega = parameters[3]
    sigma_square = parameters[4]
    T = len(rets)
    Rt2 = rets**2
    sigma2 = np.ones(T)*rets.var()
    vega = np.ones(T)*rets.var()
    for i in range(T-1):
        vega[i+1] = sigma_square + alpha_vega*(Rt2[i] - sigma2[i]) + beta_vega*(vega[i] - sigma_square)
        sigma2[i+1] = vega[i+1] + alpha_sigma*(Rt2[i] - vega[i]) + beta_sigma*(sigma2[i] - vega[i])

    logliks = 0.5*(np.log(2*np.pi) + np.log(sigma2) + Rt2/sigma2)
    loglik = np.sum(logliks)
    if out == False:
        return loglik
    if out == True:
        return loglik,sigma2

#求解成分garch的波动率        
def component_garch_hv(data,name,p = 1):
    rts = np.array(data)
    startingVals = [0.04, 0.92, 0.05, 0.9, rts.var()]
    bounds =[(0,1),(0,1),(0,1),(0,1),(0,1)]
    args = (np.asarray(rts),)
    estimates = fmin_slsqp(component_garch_loglikelihood, startingVals , args =args)
    analized=252
    loglik, sigma2final = component_garch_loglikelihood(estimates, rts, out=True)
    garch_vol = np.sqrt(analized*sigma2final)
    ested_vol = pd.DataFrame(garch_vol,columns=['estimated vols'])
    if p ==1:
        print('Initial Values=',startingVals)
        print('Estimated Values=',estimates)
        ested_vol.plot(grid='on',color = '#B22222',title=name + ' volatility with component_GARCH',figsize = (16,3))
        vol = analized*sigma2final
        return vol
    else:
        return garch_vol


#参数法、历史数据法、蒙特卡罗法计算VaR
def VaR(rts,name):
    #参数法
    VaR1 = -norm.ppf(0.01)*np.sqrt(np.var(rts)) - np.mean(rts)   
    #历史法
    rts_sort = sorted(rts)
    a = math.ceil(0.01 * len(rts))
    VaR2 = - rts_sort[a]   
    #蒙特卡洛法
    vol,est = Ngarch_hv(rts,name,p = 0)
    sigma_forecast = est[3] + est[0]*vol[-1]**2*(rts[-1]/vol[-1]-est[2])**2 + est[1]*vol[-1]**2
    shock = np.random.randn(100000)
    R_forecast = shock*np.sqrt(sigma_forecast)
    R_forecast =sorted(R_forecast)
    b = math.ceil(0.01 * len(R_forecast))
    VaR3 =- R_forecast[b]
    
    VaR = pd.Series([VaR1, VaR2,VaR3],index = ["参数法VaR","历史法VaR","蒙特卡罗法VaR"])
    return VaR

#服从t分布的Ngarch的似然函数
def Ngarch_t_loglikelihood(parameter,rts,sigma):
    d = parameter[0]
    loglikes =  - np.log(gamma((d+1)/2)) + np.log(gamma(d/2)) + np.log(np.pi)/2 + 0.5*np.log(d-2) + 0.5*(1+d)*np.log(1+(rts/sigma)**2/(d-2))
    loglike =  loglikes.sum()
    return loglike

#求解RiskM
def RiskM(rts,name):
    initial_value = rts[:251].var()
    T = len(rts) - 251
    sigma2 = np.ones(T+1)*initial_value 
    for i in range(T):
        sigma2[i+1] = 0.94* sigma2[i] + 0.06*rts[i + 250]**2
    return np.sqrt(sigma2)

#求FHIST
def FHIST(rts,name):
    Standardized_Return = rts/Ngarch_hv(rts,name,p = 2)
    sigma = Ngarch_hv(rts,name,p = 2)
    T = len(rts) - 251
    FHIST = np.ones(T)
    for i in range(T):
        FHIST[i] = -sigma[ i + 250]*np.percentile(Standardized_Return[i+1:i+ 250],1)
    return FHIST

#求HIST
def HIST(rts,name):
    T = len(rts) - 251
    HIST = np.ones(T)
    for i in range(T):
        HIST[i] = - np.percentile(rts[i+1:i+ 250],1)
    return HIST

#非对称t分布
def Asymmetric_t_dist(d1,d2,out = None):
    C = gamma((d1+1)*0.5) / (gamma(d1/2)*np.sqrt(np.pi*(d1-2)))
    A = 4*d2*C*(d1-2)/(d1-1)
    B = np.sqrt(1+3*d2**2-A**2)
    m2 = 1+ 3*d2**2
    m3 = 16*C*d2*(1+d2**2)*(d1-2)**2 / ((d1-1)*(d1-3))
    m4 = 3*(d1-2)*(1+10*d2**2+5*d2**4)/(d1-4)
    skew = (m3-3*m2*A +2*A**3)/B**3
    excess_kur = ((m4-4*A*m3+6*A**2*m2-3*A**4)/B**4) - 3
    pdf=[
    B*C*(1+(B*x+A)**2/((1-d2)**2*(d1-2)))**(-0.5*(1+d1)) if x < -(A/B) else B*C*(1+(B*x+A)**2/((1+d2)**2*(d1-2)))**(-0.5*(1+d1)) for x in  np.linspace(-4,4,1000)
]
    if out == None:
        return pdf
    if out == "skew":
        return skew
    if out =="e_kur":
        return excess_kur

#DCC似然函数
def DCC_loglikelihood(parameter,z1,z2,out = None):
    lamda = parameter[0]
    df = pd.concat([z1,z2],axis = 1)
    df.columns = ["z1","z2"]
    df = df.dropna(axis=0 ,how ="any")
    T = len(df)
    q11 = np.ones(T) 
    q12 = np.ones(T)*(df.z1 * df.z2).sum()/T
    q22 = np.ones(T)
    for i in range(T-1):
        q11[i+1] = (1 - lamda)*(z1**2)[i] + lamda*q11[i]
        q22[i+1] = (1 - lamda)*(z2**2)[i] + lamda*q22[i]
        q12[i+1] = (1 - lamda)*z1[i]*z2[i] + lamda*q12[i]
    q11 = pd.Series(q11, index = df.index)
    q22 = pd.Series(q22, index = df.index)
    rol12 = q12/np.sqrt(q11*q22)
    DCC_loglikelihoods = (np.log(1 - rol12**2) + ((z1**2 + z2**2 - 2*rol12*z1*z2)/(1-rol12**2)))*0.5
    DCC_loglike =  DCC_loglikelihoods.sum()
    if out == None:
        return DCC_loglike
    else:
        return DCC_loglikelihood,rol12

#Garch_DCC 似然函数 用于求解 alpha beta 参数
def GARCH_DCC_loglikelihood(parameter,z1,z2,out = None):
    alpha = parameter[0]
    beta = parameter[1]
    df = pd.concat([z1,z2],axis = 1)
    df.columns = ["z1","z2"]
    df = df.dropna(axis=0 ,how ="any")
    T = len(df)
    q11 = np.ones(T) 
    q12 = np.ones(T)*(df.z1 * df.z2).sum()/T
    initialrol12 = (df.z1 * df.z2).sum()/T
    q22 = np.ones(T)
    for i in range(T-1):
        q11[i+1] = 1+ alpha*((z1**2)[i]-1) + beta *(q11[i]-1)
        q12[i+1] = initialrol12 + alpha*(z1[i]*z2[i]- initialrol12) + beta*(q12[i] - initialrol12)
        q22[i+1] = 1+ alpha*((z2**2)[i]-1) + beta *(q22[i]-1)
    q11 = pd.Series(q11, index = df.index)
    q22 = pd.Series(q22, index = df.index)
    rol12 = q12/np.sqrt(q11*q22)
    DCC_loglikelihoods = (np.log(1 - rol12**2) + ((z1**2 + z2**2 - 2*rol12*z1*z2)/(1-rol12**2)))*0.5
    DCC_loglike =  DCC_loglikelihoods.sum()
    if out == None:
        return DCC_loglike
    else:
        return DCC_loglikelihood,rol12

#Garch_DCC 的边界条件
def GARCH_DCC_constraint(parameter,z1,z2,out= None):
    alpha = parameter[0]
    beta = parameter[1]
    return np.array([1 - alpha - beta])

#使用蒙特卡洛模拟预测动态相关系数
def DCC_Monte(rts1,rts2,estimats,x):
    shock1 = np.random.randn(10000)
    shock2 = np.random.randn(10000)
    alpha = estimats[0]
    beta = estimats[1]
    df = pd.concat([rts1,rts2],axis =1)
    unconditional_corr = df.corr().iloc[0,1]
    q11 = shock1**2*alpha + beta + (1-alpha-beta)
    q22 = shock2**2*alpha + beta + (1-alpha-beta)
    q12 = (1-alpha-beta)*unconditional_corr + beta*x + alpha*shock1*shock2
    rol_0 = np.average(q12/np.sqrt(q11*q22))
    rol = []
    rol.append(rol_0)
    for i in range(20):
        np.random.seed(i)
        shock1 = np.random.randn(10000)
        shock2 = np.random.randn(10000)
        q11 = shock1**2*alpha + beta*q11 + (1-alpha-beta)
        q22 = shock2**2*alpha + beta*q22 + (1-alpha-beta)
        q12 = (1-alpha-beta)*unconditional_corr + beta*q12 + alpha*shock1*shock2
        rol12 = q12/np.sqrt(q11*q22)
        rol.append(np.average(rol12))
        i +=1
    return pd.Series(rol,name=str(x))

def shu(df,DV,dv):
    df['d'] = (np.log(df['S (Index)']/df['X (Strike)']) + (df['r'] - df['q'] + (DV**2)/2) * df['T (DTM)']) / (DV * np.sqrt(df['T (DTM)']))
    df['F(d)'] = norm.cdf(df['d'])
    df['f(d)'] = norm.pdf(df['d'])
    df['Position'] = -1
    df['Delta'] = df['Position'] * np.exp(-1 * df['q'] * df['T (DTM)']) * df['F(d)']
    df['Gamma'] = df['Position'] * df['f(d)'] * np.exp(-1 * df['q'] * df['T (DTM)']) / (df['S (Index)'] * DV * np.sqrt(df['T (DTM)']))   
    df['Delta-Based Portfolio Variance'] = df['Delta']**2 * df['S (Index)']**2 * dv**2
    df['10-day 1% $VaR'] = -1 * np.sqrt(df['Delta-Based Portfolio Variance']) * np.sqrt(10) * norm.ppf(0.01)
    
    
    
def Q1(code1,code2,code3):  
    DownloadData(code1)
    DownloadData(code2)
    DownloadData(code3,index = True)
       
def Q2(df1,data1,name1,df2,data2,name2,df3,data3,name3):
    des_closeprice = pd.concat([describe(data1,maxmin = 0),describe(data2,maxmin = 0),describe(data3,maxmin = 0)],axis = 1)
    des_closeprice.columns = [name1,name2,name3]
    print(des_closeprice)
    print('===========correlation matrix===============')
    df_price = pd.concat([df1['close'],df2['close'],df3['close']],axis = 1)
    df_price.columns = [name1,name2,name3] 
    print(df_price.corr())
    print('============================================')
    drawacf(data1,name1,data2,name2,data3,name3)
    drawhist(data1,name1,data2,name2,data3,name3)
    drawqq(data1,name1,data2,name2,data3,name3)
    
def Q3(rts1,name1,rts2,name2,rts3,name3):
    des_return = pd.concat([describe(rts1,maxmin = 0),describe(rts2,maxmin = 0),describe(rts3,maxmin = 0)],axis = 1)
    des_return.columns = [name1+'rts',name2+'rts',name3+'rts']
    print(des_return)
    print('============correlation matrix==============')
    df_rtn = pd.concat([rts1,rts2,rts3],axis = 1)
    df_rtn.columns = ['rts '+name1,'rts '+name2,'rts '+name3]
    print(df_rtn.corr())
    print('============================================')
    drawacf(rts1,'rts '+name1,rts2,'rts '+name2,rts3,'rts '+name3)
    drawhist(rts1,'rts '+name1,rts2,'rts '+name2,rts3,'rts '+name3)
    drawqq(rts1,'rts '+name1,rts2,'rts '+name2,rts3,'rts '+name3)
    
def Q4(df1,rts1,name1,df2,rts2,name2,df3,rts3,name3):
    plt.figure(figsize=(17,10))
    plt.subplot(311)
    drawhv(df1,rts1,name1)
    plt.subplot(312)
    drawhv(df2,rts2,name2)
    plt.subplot(313)
    drawhv(df3,rts3,name3)
    

def Q5(df1,name1,df2,name2,df3,name3):
    plt.figure(figsize=(17,10))
    plt.subplot(311)
    drawes(df1,name1)
    plt.subplot(312)
    drawes(df2,name2)
    plt.subplot(313)
    drawes(df3,name3)

def Q6_1(rts1,name1,rts2,name2,rts3,name3):   
    garch_hv(rts1,name1,p = 0)
    print('==========================')
    garch_hv(rts2,name2,p = 0)
    print('==========================')
    garch_hv(rts3,name3,p = 0)
    
def Q6_2(rts1,name1,rts2,name2,rts3,name3):
    vol1 = Ngarch_hv(rts1,name1,p = 1)
    print('==========================')
    vol2 = Ngarch_hv(rts2,name2,p = 1)
    print('==========================')
    vol3 = Ngarch_hv(rts3,name3,p = 1)
    drawacf(rts1**2,name1+' R^2',rts2**2,name2+' R^2',rts3**2,name3+' R^2')
    drawacf((rts1**2/vol1),name1+' R^2/sigma^2',(rts2**2/vol2),name2+' R^2/sigma^2',(rts3**2/vol3),name3+' R^2/sigma^2')

def Q6_3(rts1,name1,rts2,name2,rts3,name3):
    vol1 = component_garch_hv(rts1,name1,p = 1)
    print('==========================')
    vol2 = component_garch_hv(rts2,name2,p = 1)
    print('==========================')
    vol3 = component_garch_hv(rts3,name3,p = 1)
    
def Q7(df1,rts1,name1,df2,rts2,name2,df3,rts3,name3):
    for i,j,n in ([df1,rts1,name1],[df2,rts2,name2],[df3,rts3,name3]):
        des_hv = pd.concat([describe(CtC(j,30),maxmin = 1),describe(PHV(i,30),maxmin = 1),describe(GKHV(i,30),maxmin = 1),describe(RHV(i,30),maxmin = 1),describe(garch_hv(j,n,p = 0),maxmin = 1),describe(Ngarch_hv(j,n,p = 2),maxmin = 1)],axis = 1)
        des_hv.columns = ['CtC','Parkinson','Garman-Klass','Rogers','GARCH11','NGARCH']
        print(n + '波动率的统计分析')
        print(des_hv)    
        print('\n\n==========================')

def Q8(data1,name1,data2,name2,data3,name3):
    var = pd.concat([VaR(data1,name1),VaR(data2,name2),VaR(data3,name3)],axis = 1)
    var.columns = [name1,name2,name3]
    print(var)        

def Q9_1(rts1,name1,rts2,name2,rts3,name3):
    nomalized_return1 = rts1/rts1.var()
    nomalized_return2 = rts2/rts2.var()
    nomalized_return3 = rts3/rts3.var()
    drawqq(nomalized_return1,"garch_normalized_return_"+name1,nomalized_return2,"garch_normalized_return_"+name2,nomalized_return3,"garch_normalized_return_"+name3)

def Q9_2_3(rts1,name1,rts2,name2,rts3,name3):
    Ngarch_vol1 = Ngarch_hv(rts1,name1,p = 2)
    Ngarch_vol2 = Ngarch_hv(rts2,name2,p = 2)
    Ngarch_vol3 = Ngarch_hv(rts3,name3,p = 2)
    normalized_rturn1 = rts1/Ngarch_vol1
    normalized_rturn2 = rts2/Ngarch_vol2
    normalized_rturn3 = rts3/Ngarch_vol3
    drawqq(normalized_rturn1,"Ngarch_normalized_return_"+name1,normalized_rturn2,"Ngarch_normalized_return_"+name2,normalized_rturn3,"Ngarch_normalized_return_"+name3)

def Q9_4(rts1,name1,rts2,name2,rts3,name3):
    sigma1 = Ngarch_hv(rts1,name1,p = 2)
    args1 = (rts1, sigma1)
    sigma2 = Ngarch_hv(rts2,name2,p = 2)
    args2 = (rts2, sigma2)
    sigma3 = Ngarch_hv(rts3,name3,p = 2)
    args3 = (rts3, sigma3)
    startingVals = np.array([float(10.0)])
    bounds1 = [(5.0,1000.0)]
    bounds2 = [(5.1,1000.0)]
    bounds3 = [(5.6,1000.0)]
    estimats1 = fmin_slsqp(Ngarch_t_loglikelihood,startingVals,bounds= bounds1,args = args1)
    estimats2 = fmin_slsqp(Ngarch_t_loglikelihood,startingVals,bounds= bounds2,args = args2)
    estimats3 = fmin_slsqp(Ngarch_t_loglikelihood,startingVals,bounds= bounds3,args = args3)
    for x,z in ([estimats1,rts1/sigma1],[estimats2,rts2/sigma2],[estimats3,rts3/sigma3]):
        sm.qqplot(z, sts.t, distargs=(x,), line='45')
        plt.title("T Q-Q plot")
        plt.show()
    
def Q9_7(rts1,name1,rts2,name2,rts3,name3):
    for i, j, in ([rts1,name1],[rts2,name2],[rts3,name3]):
        Risk_M = pd.Series(RiskM(i,j)[1:])
        Ngarch_t = pd.Series(Ngarch_hv(i,j,p = 2)[251:])
        fhist = pd.Series(FHIST(i,j))
        hist = pd.Series(HIST(i,j))
        df = pd.concat([Risk_M,Ngarch_t,fhist,hist],axis = 1)
        df.columns = ['Risk_M','Narch_t','FHIST','HIST']
        df.plot(figsize= (12,6))

def Q9_8():
    pdf_rskew = Asymmetric_t_dist(8,0.4) 
    pdf_lskew = Asymmetric_t_dist(8,-0.4)
    x = np.linspace(-4,4,1000)
    plt.figure(figsize = (10,8))
    plt.plot(x,pdf_rskew,label = "d1 = 8,d2 = 0.4")
    plt.plot(x,pdf_lskew,label = "d1 = 8,d2 = -0.4")
    plt.legend()
    plt.grid()
    plt.title("The asymmetric t distribution")
    
def Q9_9_1():
    skew_5 = [ Asymmetric_t_dist(d1=5, d2 = x, out="skew") for x in np.linspace(-0.9,0.9,20)]
    skew_10 = [ Asymmetric_t_dist(d1=10, d2 = x, out="skew") for x in np.linspace(-0.9,0.9,20)]
    plt.figure(figsize = (10,8))
    plt.plot(np.linspace(-0.9,0.9,20),skew_5, label = "d1= 5")
    plt.plot(np.linspace(-0.9,0.9,20),skew_10, label = "d1= 10")
    plt.xlabel("d2")
    plt.ylabel("Skewness")
    plt.legend()
    plt.grid()
    plt.title("Skewness as a function of d2")

def Q9_9_2():
    ekur_05 = [ Asymmetric_t_dist(d1=x, d2 = 0.5, out="e_kur") for x in np.linspace(4.5,14,20)]
    ekur_0 = [ Asymmetric_t_dist(d1=x, d2 = 0, out="e_kur") for x in np.linspace(4.5,14,20)]
    plt.figure(figsize = (10,8))
    plt.plot(np.linspace(4.5,14,20),ekur_05, label = "d2= 0.5")
    plt.plot(np.linspace(4.5,14,20),ekur_0, label = "d2= 0")
    plt.xlabel("d1")
    plt.ylabel("Excess kurtosis")
    plt.legend()
    plt.grid()
    plt.title("Excess kurtosis as a function of d1")

def Q10(rts1,vol1,name1,rts2,vol2,name2):
    args = (rts1,rts2)
    bounds =[(0.0,0.9999)]
    startingVals = np.array([0.94])
    estimats = fmin_slsqp(DCC_loglikelihood,startingVals , args = args, bounds= bounds)
    _,rol_rts1_rts2 = DCC_loglikelihood(np.array([.94]),rts1,rts2, out = True)
    df = pd.concat([vol1,vol2,rol_rts1_rts2],axis=1)
    df = df.dropna(axis=0 ,how="any")
    df.columns = ["pfyh_vol","szzs_vol","rol"]
    df["beta"] = df.rol*df.pfyh_vol/df.szzs_vol
    
    title1 = "Dynamic Conditional Correlation of " + name1 + " and " + name2
    title2 = "Dynamic beta of "+ name1 +" and "+ name2
    rol_rts1_rts2.plot(title = title1,figsize = (12,8),label="corr")
    df.beta.plot(title = title2,figsize=(12,8))
    plt.legend()

def Q11():
    data={
        "0.6":DCC_Monte(standardized_rertun1,standardized_rertun3,estimats,x = 0.6),
        "0.7":DCC_Monte(standardized_rertun1,standardized_rertun3,estimats,x = 0.7),
        "0.4":DCC_Monte(standardized_rertun1,standardized_rertun3,estimats,x = 0.4),              
        "0.3":DCC_Monte(standardized_rertun1,standardized_rertun3,estimats,x =0.3)                          
                        }
    df = pd.DataFrame(data)
    df.plot(figsize = (10,6))
    plt.grid()
    plt.title("forecast correlation of pfyh & szzs after 10 days")
    plt.ylabel("corr")
    

def Q12(df,rts,name):
    np.random.seed(0)
    vol,para = Ngarch_hv(rts,name,p = 3)
    r = np.zeros((31,100000)) 
    v = np.zeros((31,100000)) 
    S = np.zeros((31,100000))
    r[0,:] = rts2[-1]
    v[0,:] = vol[-1]
    S[0,:] = df.close[-1]
    for i in range(30):
        shock = np.random.randn(100000)
        v[1+i,:] = np.sqrt(para[3] + para[0] * (r[0+i,:] - para[2]*v[0+i,:])**2  + para[1]*(v[0+i,:])**2)
        r[1+i,:] = shock * v[1+i,:]
        S[1+i,:] = np.exp(np.log(S[0+i,:]) + r[1+i,:])
    St = S[-1,:]
    K = df1.close[-1] + 1
    C = St-K
    C[C<0] = 0
    c  = C.mean()/np.power((1 + 0.03/360),30)
    print(name + ' 的欧式期权价格为：  ',c)

def Q13_1_2():
    df = pd.read_excel('Chapter2_Data.xls',parse_dates=[0],sheetname='Question 2.1 & 2.2 (Short)')
    df.index=df.pop('Date')
    new_df = pd.DataFrame(df.Close)
    rts = 100 * np.log(new_df/new_df.shift(1)).dropna()
    minus_rts = -1 * rts

    new_df['VaR Short'] = 0
    for i in range(len(minus_rts)-252):
        new_df['VaR Short'][252+i:] = np.percentile(minus_rts[2+i:252+i],1)
    new_df['WHS Short'] = 0
    for i in range(len(rts)-252):
        rts1 = sorted(rts.Close[-249-(len(rts)-252)+i : -(len(rts)-252)+i])
        new_df['WHS Short'][252+i:] = rts1.pop()
    new_df['VaR Long'] = 0
    for i in range(len(minus_rts)-252):
        new_df['VaR Long'][252+i:] = np.percentile(minus_rts[2+i:252+i],99)
    new_df['WHS Long'] = 0
    for i in range(len(rts)-252):
        rts1 = sorted(rts.Close[-249-(len(rts)-252)+i : -(len(rts)-252)+i])
        new_df['WHS Long'][252+i:] = -1 * rts1.pop(0)


    plt.figure(figsize=(15,5))
    plt.subplot(221)
    plt.grid()
    plt.plot(minus_rts[250:],color = '#4682B4',linewidth = 3,label = 'return')
    plt.plot(new_df['VaR Short'][252:],color = '#006400',linewidth = 3,label = 'VaR Short (HS)')
    plt.legend(loc='upper left')

    plt.subplot(222)
    plt.grid()
    plt.plot(minus_rts[250:],color = '#4682B4',linewidth = 3,label = 'return')
    plt.plot(new_df['WHS Short'][252:],color = '#006400',linewidth = 3,label = 'VaR Short (WHS)')
    plt.legend(loc='upper left')

    plt.subplot(223)
    plt.grid()
    plt.plot(minus_rts[250:],color = '#4682B4',linewidth = 3,label = 'return')
    plt.plot(new_df['VaR Long'][252:],color = '#006400',linewidth = 3,label = 'VaR Long (HS)')
    plt.legend(loc='upper left')

    plt.subplot(224)
    plt.grid()
    plt.plot(minus_rts[250:],color = '#4682B4',linewidth = 3,label = 'return')
    plt.plot(new_df['WHS Long'][252:],color = '#006400',linewidth = 3,label = 'VaR Long (WHS)')
    plt.legend(loc='upper left')

def Q13_3_4():    
    df = pd.read_excel('Chapter2_Data.xls',parse_dates=[0],sheetname='Question 2.3 & 2.4')
    df.index=df.pop('Date')
    new_df = pd.DataFrame(df['S&P500'])
    rts = np.log(new_df/new_df.shift(1)).dropna()
    lam = 0.94
    RiskM = [0] * (len(rts) + 2)
    RiskM10 = [0] * (len(rts) + 2)
    for i in range(len(rts)-1):
        RiskM[3+i] = (1 - lam) * (rts['S&P500'][0+i])**2 + lam * RiskM[2+i]
        RiskM10[3+i] = -1 * np.sqrt(10) * norm.ppf(0.01) * np.sqrt(RiskM[3+i])
    HS10 = [0] * (len(rts) + 2)
    for i in range(len(rts) - 250):
        HS10[252+i] = -1 * np.sqrt(10) * np.percentile(rts[0+i:250+i],1)
    new_df['RiskM-10'] = RiskM10
    new_df['HS-10'] = HS10
    
    new_df['PRM'] = 0
    new_df['PHS'] = 0
    new_df['PLRM'] = 0
    new_df['PLHS'] = 0
    new_df['CPLRM'] = 0
    new_df['CPLHS'] = 0
    data = new_df['2008-06-30':'2009-12-31']
    data['PRM'] = 100000 / data['RiskM-10']
    data['PHS'] = 100000 / data['HS-10']
    for i in range(len(data['PHS']) - 1):
        data['PLRM'][i+1] = (data['S&P500'][i+1] / data['S&P500'][i] - 1) * data['PRM'][i+1]
        data['PLHS'][i+1] = (data['S&P500'][i+1] / data['S&P500'][i] - 1) * data['PHS'][i+1]
        data['CPLRM'][i+1] = data['CPLRM'][0+i] + data['PLRM'][1+i]
        data['CPLHS'][i+1] = data['CPLHS'][0+i] + data['PLHS'][1+i]
    
    plt.figure(figsize=(15,5))
    plt.subplot(121)
    plt.grid()
    plt.plot(new_df['RiskM-10']['2008-07-01':'2009-12-31'],color = '#4682B4',linewidth = 3,label = 'RiskMetrics 10days')
    plt.plot(new_df['HS-10']['2008-07-01':'2009-12-31'],color = '#006400',linewidth = 3,label = 'HS 10days')
    plt.legend(loc='upper left')

    plt.subplot(122)
    plt.grid()
    plt.plot(data['CPLRM']['2008-07-01':'2009-12-31'],color = '#4682B4',linewidth = 3,label = 'P/L from Risk Metrics VaR')
    plt.plot(data['CPLHS']['2008-07-01':'2009-12-31'],color = '#006400',linewidth = 3,label = 'P/L from HS VaR')
    plt.legend(loc='upper left')

def Q14_1():    
    df = pd.read_excel('Chapter11_Data.xls',parse_dates=[0],sheetname='Question 11.1')
    shu(df,DV,dv)
    VaR = np.array(df['10-day 1% $VaR'])
    print('10-day 1% $VaR is :')
    print(VaR)

def Q14_2():
    df = pd.read_excel('Chapter11_Data.xls',parse_dates=[0],sheetname='Question 11.2')
    shu(df,DV,dv)
    Delta = np.sum(df['Delta'])
    Gamma = np.sum(df['Gamma'])
    Delta_Based = [0] * len(df['Simulated    10-day returns']) 
    Gamma_Based = [0] * len(df['Simulated    10-day returns'])
    for i in range(len(df['Simulated    10-day returns'])):
        df['Simulated    10-day returns'][0+i] = random.normalvariate(0,1) * dv* np.sqrt(HOR)    
        Delta_Based[0+i] = Delta * df['S (Index)'][1] * df['Simulated    10-day returns'][0+i]
        Gamma_Based[0+i] = Delta_Based[0+i] + 0.5 * Gamma * (df['S (Index)'][1])**2 * (df['Simulated    10-day returns'][0+i])**2
    mDV = 0.5* Gamma * (df['S (Index)'][1])**2 * (dv)**2 * HOR
    s2DV = Delta**2 * (df['S (Index)'][1])**2 * (dv)**2 * HOR + 0.5*Gamma**2 * ((df['S (Index)'][1])**2)**2 * (dv**2)**2 *(HOR**2)**2
    zDV = (4.5*(Delta)**2*Gamma*np.power(df['S (Index)'][1],4)*np.power(dv,4)*np.power(HOR,4) + 1.875*np.power(Gamma,3)*np.power(df['S (Index)'][1],6)*np.power(dv,6)*np.power(HOR,3) - 3*mDV*(Delta**2*(df['S (Index)'][1])**2*dv**2*HOR+0.75* Gamma**2* np.power(df['S (Index)'][1],4)*np.power(dv,4)*HOR**2)+2*np.power(mDV,3) )/ np.power(mDV,1.5)
    DVaR = -np.percentile(Delta_Based,0.01)
    GVaR = -np.percentile(Gamma_Based,0.01)
    Cornish_Fisher = -mDV-np.sqrt(s2DV)*(norm.ppf(0.01)+(norm.ppf(0.01)**2-1)*(mDV/6))
    print('Delta VaR  :  ' , DVaR)
    print('\nGamma VaR   :  ', GVaR)
    print('\nCornish_Fisher :  ',Cornish_Fisher)
    
def Q14_3(): 
    T=43
    X=1135	
    C=26.54	
    St=1137.14	
    r=0.000682/100
    q=0.005697/100	
    d=(np.log(St/X)+(r-q+0.5*(DV)**2)*T)/(DV*np.sqrt(T))
    delta = -1*np.exp(-T*q)*norm.cdf(d)
    gamma =-1*norm.pdf(d)*np.exp(-T*q)/(St*DV*np.sqrt(T))
    Rh = [0]*5000
    Sh = [0]*5000
    BSTt = [0]*5000
    BSD = [0]*5000
    BSds = [0]*5000
    BSc = [0]*5000
    db = [0]*5000
    gb = [0]*5000
    fv = [0]*5000
    for i in range(5000):
        Rh[i] = random.normalvariate(0,1)*dv*np.sqrt(HOR)
        Sh[i] = St * np.exp(Rh[i])
        BSTt[i] = T - 14
        BSD[i] = (np.log(Sh[i]/X)+(r-q+(DV**2)/2)*BSTt[i])/(DV*np.sqrt(BSTt[i]))
        BSds[i] = BSD[i]-DV*np.sqrt(BSTt[i])
        BSc[i] = Sh[i]*np.exp(-q*BSTt[i])*norm.cdf(BSD[i])-X*np.exp(-r*BSTt[i])*norm.cdf(BSds[i])
        db[i] = delta*St*Rh[i]
        gb[i] = db[i]+0.5*gamma*St**2*Rh[i]**2
        fv[i] = C-BSc[i]
    DVaR = -np.percentile(db,0.01)
    GVaR = -np.percentile(gb,0.01)
    fvVaR = -np.percentile(fv,0.01)
    print('Delta VaR  :  ' , DVaR)
    print('\nGamma VaR   :  ', GVaR)
    print('\nFull Valuation  :  ',fvVaR)
    drawhist(db,'Delta-Based',gb,'Gamma-Based',fv,'Full Valuation')
    
def Q14_4():
    df = pd.read_excel('Chapter11_Data.xls',parse_dates=[0],sheetname='Question 11.4')
    T = np.array(df['T (DTM)'][1:])
    S = np.array(df['S'][1:])
    r = np.array(df['r'][1:])
    dx95 = [0]*len(T)
    dx105 = [0]*len(T)
    fv = [0]*len(T)
    fv1 = [0]*len(T)
    fv2 = [0]*len(T)
    fv3 = [0]*len(T)
    db = [0]*len(T)
    gb = [0]*len(T)	
    c95 = -1.5
    sp1 = 95
    c105 = 2.5
    sp2 = 105
    p = -1
    for i in range(len(T)):
        dx95[i] = (np.log(S[i]/sp1)+(r[i]+0.5*DV**2)*T[i])/(DV*np.sqrt(T[i]))
        dx105[i] = (np.log(S[i]/sp2)+(r[i]+0.5*DV**2)*T[i])/(DV*np.sqrt(T[i]))
        fv1[i] = round(c95*(S[i]*norm.cdf(dx95[i])-sp1*np.exp(-r[i]*T[i])*norm.cdf(dx95[i]-DV*np.sqrt(T[i]))),2)
        fv2[i] = round(c105*(S[i]*norm.cdf(dx105[i])-sp2*np.exp(-r[i]*T[i])*norm.cdf(dx105[i]-DV*np.sqrt(T[i]))),2)
        fv3[i] = round(p*(-S[i]*norm.cdf(-dx95[i])+sp1*np.exp(-r[i]*T[i])*norm.cdf(-(dx95[i]-DV*np.sqrt(T[i])))),2)
        fv[i] = fv1[i]+fv2[i]+fv3[i]
    VPF = fv.pop(0)
    d = c95*norm.cdf(dx95[0])+c105*norm.cdf(dx105[0])+p*(norm.cdf(dx95[0])-1)
    g = c95*norm.pdf(dx95[0])/(S[0]*DV*np.sqrt(T[0]))+c105*norm.pdf(dx105[0])/(S[0]*DV*np.sqrt(T[0]))+p*norm.pdf(dx95[0])/(S[0]*DV*np.sqrt(T[0]))
    for i in range(len(T)):
        db[i] = VPF+d*(S[i]-S[0])
        gb[i] = VPF+d*(S[i]-S[0])+0.5*g*((S[i]-S[0])**2)
    del(db[0])
    del(gb[0])
    plt.figure(figsize=(15,5))
    plt.grid()
    plt.plot(fv,linewidth = 3,label = 'Full Valuation')
    plt.plot(db,linewidth = 3,label = 'Delta-based')
    plt.plot(gb,linewidth = 3,label = 'Gamma-based')
    plt.legend(loc='upper left')




code1 = "600000"
name1 = "pfyh"
code2 = "600099"
name2 = "lhgf"
code3 = "000001"
name3 = "szzs"
print("\nQuestion 1:\n")
Q1(code1,code2,code3)
DV = 0.015
dv = 0.0181
HOR = 10
df1,data1,rts1 = ReadCsv(code1)
df2,data2,rts2 = ReadCsv(code2)
df3,data3,rts3 = ReadCsv(code3)
pfyh_vol = pd.Series(Ngarch_hv(rts1,name1,p=2),index = rts1.index)
lhgf_vol = pd.Series(Ngarch_hv(rts2,name2,p=2),index = rts2.index)
szzs_vol = pd.Series(Ngarch_hv(rts3,name3,p=2),index = rts3.index)

df = pd.concat([pfyh_vol,lhgf_vol,szzs_vol],axis = 1)
df.columns = ["pfyh_vol","lhgf_vol","szzs_vol"]

standardized_rertun1 = rts1/pfyh_vol
standardized_rertun2 = rts2/lhgf_vol
standardized_rertun3 = rts3/szzs_vol

startingVals = np.array([0.05,0.9])
args = (standardized_rertun1,standardized_rertun3)
bounds = [(0.0,1.0),(0.0,1.0)]
estimats = fmin_slsqp(GARCH_DCC_loglikelihood,startingVals,args = args,bounds = bounds,f_ieqcons=GARCH_DCC_constraint)

print("\nQuestion 2:\n")
Q2(df1,data1,name1,df2,data2,name2,df3,data3,name3)
print("\nQuestion 3:\n")
Q3(rts1,name1,rts2,name2,rts3,name3)
print("\nQuestion 4:\n")
Q4(df1,rts1,name1,df2,rts2,name2,df3,rts3,name3)
plt.show()
print("\nQuestion 5:\n")
Q5(df1,name1,df2,name2,df3,name3)
plt.show()
print("\nQuestion 6_1:\n")
Q6_1(rts1,name1,rts2,name2,rts3,name3)
plt.show()
print("\nQuestion 6_2:\n")
Q6_2(rts1,name1,rts2,name2,rts3,name3)
plt.show()
print("\nQuestion 6_3:\n")
Q6_3(rts1,name1,rts2,name2,rts3,name3)
plt.show()
print("\nQuestion 7:\n")
Q7(df1,rts1,name1,df2,rts2,name2,df3,rts3,name3)
plt.show()
print("\nQuestion 8:\n")
Q8(rts1,name1,rts2,name2,rts3,name3)
plt.show()
print("\nQuestion 9_1:\n")
Q9_1(rts1,name1,rts2,name2,rts3,name3)
plt.show()
print("\nQuestion 9_2&3:\n")
Q9_2_3(rts1,name1,rts2,name2,rts3,name3)
plt.show()
print("\nQuestion 9_4:\n")
Q9_4(rts1,name1,rts2,name2,rts3,name3)
plt.show()
print("Q4 & Q5 没有\nQuestion 9_7:\n")
Q9_7(rts1,name1,rts2,name2,rts3,name3)
plt.show()
print("\nQuestion 9_8:\n")
Q9_8()
plt.show()
print("\nQuestion 9_9:\n")
Q9_9_1()
plt.show()
Q9_9_2()
plt.show()
print("\nQuestion 10:\n")
Q10(standardized_rertun1,pfyh_vol,name1,standardized_rertun3,szzs_vol,name3)
plt.show()
Q10(standardized_rertun2,lhgf_vol,name2,standardized_rertun3,szzs_vol,name3)
plt.show()
print("\nQuestion 11:\n")
Q11()
plt.show()
print("\nQuestion 12:\n")
Q12(df1,rts1,name1)
plt.show()
print("\nQuestion 13:\n")
Q13_1_2()
plt.show()
Q13_3_4()
plt.show()
print("\nQuestion 14:\n")
Q14_1()
plt.show()
Q14_2()
plt.show()
Q14_3()
plt.show()
Q14_4()
plt.show()
