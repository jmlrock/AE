#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 10:37:18 2019

@author: julesroche
"""

import os
import numpy as np
import pandas as pd
import json

import scipy
import statsmodels.api as sm


import sklearn.decomposition as sck_dec
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error


import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.dates as mdates

import tensorflow as tf
from tensorflow.python.tools import inspect_checkpoint as chkp

from sklearn.model_selection import train_test_split

from random import randint

from tensorflow import contrib

from pypfopt import *
from pypfopt import expected_returns
from pypfopt import risk_models
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import discrete_allocation

from empyrical import *
from empyrical import max_drawdown, alpha_beta

from pyfolio import *
import pyfolio as pf
#from pyfolio import utils
# silence warnings
import warnings
warnings.filterwarnings('ignore')

print('ok')

'Belgique'
u_bel=pd.read_csv('YC-BEL.csv',index_col='Date',parse_dates=True)
for i in range(1,9):
    del u_bel[str(i)+'-Year']
del u_bel['10-Year']
del u_bel['12-Year']
u_bel=u_bel.dropna()
u_bel.name='BEL'
#print(u_bel.name)

'Espagne'
u_esp=pd.read_csv('YC-ESP.csv',index_col='Date',parse_dates=True)

del u_esp['3-Month']
del u_esp['6-Month']
del u_esp['15-Year']
del u_esp['6to12-Month']
del u_esp['1to2-Year']
del u_esp['12-Month']
u_esp=u_esp.replace([np.inf, -np.inf], np.nan)
u_esp=u_esp.dropna()
u_esp.name='ESP'
#suppression valeur aberante'


'France'
u_fr=pd.read_csv('YC-FRA.csv',index_col='Date',parse_dates=True)
#u_fr=u_fr.iloc[:7200]
del u_fr['1-Year']
del u_fr['2-Year']
del u_fr['5-Year']
u_fr=u_fr.dropna()
u_fr.name='FR'

'Allemagne'
u_all=pd.read_csv('YC-DEU.csv',index_col='Date',parse_dates=True)
T_all=[6/12,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]
u_all=u_all.replace([np.inf, -np.inf], np.nan)
u_all=u_all.dropna()
u_all.name='All'



'Italie'
u_it=pd.read_csv('YC-ITA.csv',index_col='Date',parse_dates=True)
u_it=u_it.dropna()
u_it.name='ita'

'Grande Bretagne'
u_gb=pd.read_csv('YC-GBR.csv',index_col='Date',parse_dates=True)
u_gb=u_gb.dropna()
u_gb.name='GB'

'Japon'
u_jap=pd.read_csv('YC-JPN.csv',index_col='Date',parse_dates=True)

'Russie'
u_russ=pd.read_csv('YC-RUS.csv',index_col='Date',parse_dates=True) 
T_russ=[0.25,0.5,0.75,1,2,3,5,7,10,15,20,30]

'Suisse'
u_sui=pd.read_csv('YC-CHE.csv',index_col='Date',parse_dates=True)
#u_sui=u_sui.dropna()
u_sing=pd.read_csv('YC-SGP.csv',index_col='Date',parse_dates=True)

u_nz=pd.read_csv('YC-NZL.csv',index_col='Date',parse_dates=True)
u_nz=u_nz.dropna()
T_nz=[1/12,3/12,6/12,1,2,5,10]

u_costa=pd.read_csv('YC-CRI.csv',index_col='Date',parse_dates=True)





#i_m=u_alln['6-Month'].idxmin()
#
#y=u_alln.iloc[1]
y=u_all['6-Month']

def norm(data):
    data_n=(data-data.mean(axis=0))/data.std(axis=0)
    return(data_n)


def get_return(Y):
    R = Y.pct_change()[1:]
    R=R.dropna()
    return(R)
    
yr=get_return(y)[:1000]
allr=get_return(u_all)
#plt.plot(yr)

i=u_bel.iloc[1:91].index
j=u_esp.iloc[:90].index
'verifie si les index sont egaux'
#print(i.equals(j))



'histo des rendements'
#I=[-0.01]
#for i in range(1000):
#    I.append(I[-1]+0.00001)
#    
#a=plt.hist(yr, range =(-0.5,0.5),bins=I)
#x=[100,200,300,200,3000]
#plt.hist(x,bins=[x for x in range(1,4000,100)])   

'construction index'
#y_all=u_all['1-Year'][:500]
#y_esp=u_esp['12-Month'][:500]
#y_esp.index=y_all.index
#y_bel=u_bel['1-Year'][:500]
#y_bel.index=y_all.index
##y_s=u_sui['12-Month'][:500]
##y_s.index=y_all.index
#
#u_index=0.50*y_all +0.25*y_esp+0.25*y_bel
#plt.plot(u_index)

#E=pd.read_csv('eu.csv')
#E=E.iloc[:100]
#returns.index=pd.date_range('2010-01-01', periods=(self.date_window+self.rolling_window), freq="D")

u_fr=u_fr.iloc[:len(u_all)]
u_esp=u_esp.iloc[:len(u_all)]
u_bel=u_bel.iloc[:len(u_all)]
u_it=u_it.iloc[:len(u_all)]
u_gb=u_gb.iloc[:len(u_all)]



    
u_fr.index=u_all.index
u_bel.index=u_all.index
u_esp.index=u_all.index
u_it.index=u_all.index
u_gb.index=u_all.index

u_fr.name='fr'
u_bel.name='bel'
u_esp.name='esp'
u_it.name='it'
u_gb.name='gb'
u_all.name='all'

mit=u_it.mean(axis=1)
mit.name='moyen'
Ita=pd.concat([mit,u_it],axis=1)
#Ita.plot()




def index_gene():
    mf=u_fr.mean(axis=1)
    me=u_esp.mean(axis=1)
    ma=u_all.mean(axis=1)
    mi=u_it.mean(axis=1)
    mb=u_bel.mean(axis=1)
    mg=u_gb.mean(axis=1)
    
    y=ma*0.3+mf*0.2+mg*0.2+me*0.1+mi*0.1+mb*0.1
    
#    plt.plot(ma,color='red',label='allemagne')
#    plt.plot(me,label='UK')
#    plt.plot(mf,label='france')
#    plt.plot(y,'blue',label='index')
#    plt.xlabel('date')
#    plt.ylabel('taux')
#    plt.legend()
#    r=get_return(y)
    y.name='index'
    return(y)
#index_gene()

'euro index'
u_eu_1Y=pd.read_csv('EU_1Y.csv',index_col='date',parse_dates=True)
u_eu_10Y=pd.read_csv('EU_10Y.csv',index_col='date',parse_dates=True)
u_eu_3M=pd.read_csv('EU_3M.csv',index_col='date',parse_dates=True)

del u_eu_1Y['conf']
del u_eu_10Y['conf']
del u_eu_3M['conf']

u_eu_1Y=u_eu_1Y.sort_index(ascending=False)
u_eu_10Y=u_eu_10Y.sort_index(ascending=False)
u_eu_3M=u_eu_3M.sort_index(ascending=False)
##
#u_alea=index_gene().iloc[:len(u_eu_1Y)] 
#y.index=u_eu_1Y.index

u_eu=pd.concat([u_eu_3M,u_eu_1Y,u_eu_10Y],axis=1)
u_eu.columns=['3mois','1 ans','10 ans']
u_eu_index=u_eu.mean(axis=1)
u_eu_index.name='moyen'
Ueuro=pd.concat([u_eu,u_eu_index],axis=1)
#Ueuro.plot()
#Indice=pd.concat([u_eu_index,u_alea],axis=1)
#Indice.columns=['eurobond','fictif']
#Indice.plot()

#u_eu_index.plot()
#plt.plot(u_eu_1Y,color='blue')
#plt.plot(u_eu_10Y,color='red')
#plt.plot(u_eu_3M,color='green')
#plt.plot(y,color='black')

#plt.plot(u_eu_index,color='green')
#plt.plot(y,color='black')   

#Donne le nombre de bond impliqué dans le calcul de l'index= 55
#print(len(u_all.columns)+len(u_fr.columns)+len(u_esp.columns)+len(u_it.columns)+len(u_bel.columns)+len(u_gb.columns))

    
#call=[c for c in u_all.columns]
#u_all[call[8]].plot()

"Chargement des donnée et construction de la dataframe"
usy=pd.read_csv('USy.csv', index_col='Date', parse_dates=True)
"on restrint la base de donnée= debut: 2006-02-15+ on enlève la colonne de 2 mois de maturité"
usy_n=usy.copy()
usy_n=usy_n.drop('2 MO',axis=1)
usy_n=usy_n.iloc[:2500]

#yl=usy_n['30 YR']
#yc=usy_n['1 MO']
#plt.plot(yl, color='red')
#plt.plot(yc)



c=pd.read_csv('cac.csv',index_col='Date', parse_dates=True)
col=c['Open']
cr=col.diff()[1:]


'supprime les col et ligne avec NAn'
#df=usy.dropna()
#df=usy.interpolate()
#usy=usy.set_index('Date')

"Manipulation et tracés"

us30=usy[['30 YR']][0:1000]

#us30.plot()
'selectionne la premiere ligne'
def yield_aff():
    y1=usy.iloc[0]
    y2=usy.iloc[400]
    y3=usy.iloc[1200]
    y4=usy.iloc[2000]
    y5=usy.iloc[3000]
    time=[1/12,2/12,3/12,6/12,1,2,3,5,7,10,20,30]
    'evolution de la courbe de taux au cours du temps'
    plt.plot(time,y1, linestyle='-',marker='o',color='blue')
    plt.plot(time,y2, linestyle='-',marker='o',color='red')
    plt.plot(time,y3, linestyle='-',marker='o',color='orange')
    plt.plot(time,y4, linestyle='-',marker='o',color='green')
    plt.plot(time,y5, linestyle='-',marker='o')
    plt.legend(loc='lower right', frameon=True)
    plt.xlabel('Maturité')
    plt.ylabel('Rendement à maturité')

"PCA"


def taux_aff():
    ust=usy.iloc[:1000]
    ya=ust['1 MO']
    yb=ust['1 YR']
    yc=ust['20 YR']
    plt.plot(ya,color='blue')
    plt.plot(yb,color='red')
    plt.plot(yc,color='orange')
    plt.ylabel('Yield')
    plt.legend(loc='lower right', frameon=True)
    plt.xlabel('Date')
#taux_aff()

'PCA hedging'
U=usy_n.iloc[:1000]

time=[1/12,3/12,6/12,1,2,3,5,7,10,20,30]
M=pd.DataFrame(U.mean(axis=0))

#pca=PCA(n_components=3)
#pc=pca.fit_transform(u) 
#Pc=pd.DataFrame(pc,index=(u.index))
#Pc.columns=['pc1','pc2','pc3']

#Pc.plot()

'accès au vecteur propre: loadings'
pca=PCA(n_components=3)
pc=pca.fit(U)
PCs=pc.components_
PCs=np.transpose(PCs)
PCs=pd.DataFrame(PCs)
PCs.columns=['pc1','pc2','pc3']
p_1=pd.DataFrame(PCs['pc3'])
p_1.index=M.index

Fp=pd.concat([p_1,M],axis=1)
Fm=pd.concat([-p_1,M],axis=1)
Sp=Fp.sum(axis=1)
Sm=Fm.sum(axis=1)
#plt.plot(M,marker='o',label='moyenne')
#plt.plot(Sp,marker='o',label='moyenne+PC3')
#plt.plot(Sm,marker='o',label='moyenne-PC3')
#plt.ylabel('taux')
#plt.xlabel('maturité')
#plt.legend()


#Sp=M.add(PCs['pc1'],axis=0)

#plt.plot(PCs['pc1'],marker='o',label='pc1')
#plt.plot(PCs['pc2'],marker='o',label='pc2')
#plt.plot(PCs['pc3'],marker='o',label='pc3')
#plt.xlabel('maturités')
#plt.xlabel('taux')
#plt.legend()

    


def PCA1_aff():
    
    
    pca=PCA(n_components=3)
    pc=pca.fit_transform(usy_n)
    pcd=pd.DataFrame(data=pc,columns=['pc1','pc2','pc3'])
    
    u=pd.read_csv('USy.csv',parse_dates=True)
    u=u.iloc[:2500]
    m=u.mean(axis=1)
    m=pd.DataFrame(m,index=u.index)
    pcf=pd.concat([pcd,u['Date']],axis=1)
    pcf=pcf.set_index('Date')
    p3=pcf[['pc1','pc2','pc3']]
    m.index=p3.index
#    plt.plot(p3[0:260])
    #p3[0:260].plot()
#    p3=p3.sort_index()
    F=pd.concat([p3['pc1'],norm(m)],axis=1)
    F.columns=['pc1','moyenne']
    F.plot()
#    plt.plot(m)
    
#    plt.legend(['pc1','pc2','pc3'])
#    plt.xlabel('Date')
#    plt.ylabel('Yield')
    return(F)

  



#Pn=pd.DataFrame(P)

'score'
#P_score=pc.fit_transform(usy_n)
#plt.plot(Pc)




#plt.plot(T,P[:,0],marker='o')
#plt.plot(T,us_moy,marker='o')



'base de donnée final: les maturité concaténé avec les pc:'
#usfinal=pd.concat([usy_n,p3],axis=1)

'affichage des cp et des maturités: comparaison'
#usfinal[['1 MO','1 YR','10 YR','pc1','pc2','pc3']][0:260].plot()

'affichage des 3 composante principal au cours du temps'
#usfinal[['pc1','pc2','pc3']][0:260].plot()

'%pourcentage de variance'
pca=PCA(n_components=10)
pcn=pca.fit(usy_n)
var=pcn.explained_variance_ratio_

def var_aff():
    

    p=[1,2,3,4,5,6,7,8,9,10]
#    p=[1,2,3]
    plt.bar(p,var, align='center')
    plt.ylabel('% de variance expliquée')
    plt.xlabel('principal component')
    

    
'Visualisation 2D: on utilise pc1 et pc2'
def Dim_aff():
    print(type(usfinal['pc1']))
    plt.scatter(usfinal['pc1'],usfinal['pc2'],usfinal['10 YR'])
    plt.xlabel('pc 1')
    plt.ylabel('pc 2')

'Forcasting'


#X=np.matrix(usy_n)
#Xm=X-np.mean(X,axis=0) #'prend la moyenne de chaque colonne'


'Reconstruction'

#pca=PCA()
#pca.fit(usy_n)
#print(pca.n_components_) #'par defaut:11 pc'
#print(pca.explained_variance_) #donnne les valeur propres associé a chaque pc
#tot = sum(pca.explained_variance_)
#var_exp = [(i/tot)*100 for i in sorted(pca.explained_variance_, reverse=True)] 
#print(var_exp[0:11])#donne ratio de vairance expliqué
#print(pca.explained_variance_ratio_)# fait la meme chose

# somme cumulé
#cum_var_exp = np.cumsum(var_exp)
#print(cum_var_exp)

# PLOT OUT THE EXPLAINED VARIANCES SUPERIMPOSED 
#plt.figure(figsize=(10, 5))
#plt.step(range(1, 12), cum_var_exp, where='mid',label='cumulative explained variance')

#nombre de composent utiliser pour avoir % de variance expliqué
T=[1/12,3/12,6/12,1,2,3,5,7,10,20,30]

def PCA2_aff():
    
    pcaP=PCA(0.999)
    pcaP.fit(usy_n)
    print('nbre pc',pcaP.n_components_)
    comp=pcaP.transform(usy_n)
    
    approx=pcaP.inverse_transform(comp)
    
    'transforme en dataframe'
    approx=pd.DataFrame(approx,columns=['1 MO','3 MO','6 MO','1 YR','2 YR','3 YR','5 YR','7 YR','10 YR','20 YR','30 YR'])
    
    'comparason'
    i=randint(0,100)
    print('i=',i)
    y=usy_n.iloc[i]
    yapprox=approx.iloc[i]
    
    'calcul de erreur MSE'
    print('MSE',mean_squared_error(y,yapprox))
   
    
#    plt.plot(T,y, linestyle='-',marker='o',color='black',linewidth=3)
    plt.plot(T,y, linestyle='-',marker='o',color='blue')
    plt.plot(T,yapprox, linestyle='--',marker='o',color='red')
    
    plt.xlabel('Maturité')
    plt.ylabel('Yield')
    

def PCA_final_p(i,data,perc):
        
    pcaP=PCA(perc)
    pcaP.fit(data)
#    print('nbre pc',pcaP.n_components_)
    comp=pcaP.transform(data)
    
    approx=pcaP.inverse_transform(comp)
    
    'transforme en dataframe'
    approx=pd.DataFrame(approx,columns=data.columns)
    

    y=data.iloc[i]
    yapprox=approx.iloc[i]
    
    'calcul de erreur MSE'
    mse=mean_squared_error(y,yapprox)

    return([yapprox,mse,pcaP.n_components_])

def PCA_final_c(k,data,n):
        
    pcaP=PCA(n_components=n)
    pcaP.fit(data)
#    print('nbre pc',pcaP.n_components_)
    comp=pcaP.transform(data)
    
    #Base de donnée reconstruite
    approx=pcaP.inverse_transform(comp)

    'transforme en dataframe'
    #Cas particulier de l'Allemagne
#    approx=pd.DataFrame(approx,columns=['1 MO','3 MO','6 MO','1 YR','2 YR','3 YR','5 YR','7 YR','10 YR','20 YR','30 YR'])
    approx=pd.DataFrame(approx,columns=data.columns)
    
    E=[]
    for i in range(len(approx)):
        a=approx.iloc[i]
        b=data.iloc[i]
        e=mean_squared_error(a,b)
        E.append(e)
    E=pd.DataFrame(E,columns=['mse'])
    'erreur moyenne sur tt la base de donnée'
    mse_global=E.mean()
    F=pd.concat([approx,E],axis=1)
    

    y=data.iloc[k]
    yapprox=approx.iloc[k]
    
    'calcul de erreur MSE'
    mse=mean_squared_error(y,yapprox)

    return([yapprox,mse,pcaP.n_components_,F,mse_global])
    
#F=PCA_final_c(2,usy_n,3)[4]  
#a=F['mse'].idxmax()
    
p=[0.8,0.9,0.99,0.9999]

def PCA3_aff():
    n=len(p)
    y=usy_n.iloc[0]
    plt.plot(T,y, linestyle='-',marker='o',color='black',linewidth=4,label='courbe réelle')
    
    for k in range(n):
        pcak=PCA(k)
        
        pcak.fit(usy_n)
        
        comp=pcak.transform(usy_n)
        approx=pcak.inverse_transform(comp)
        approx=pd.DataFrame(approx,columns=['1 MO','3 MO','6 MO','1 YR','2 YR','3 YR','5 YR','7 YR','10 YR','20 YR','30 YR'])
        yapprox=approx.iloc[0]
        #T=[1/12,3/12,6/12,1,2,3,5,7,10,20,30]
        plt.plot(T,yapprox, linestyle='--',marker='o',label='% var expliqué: '+str(p[k]))
        plt.legend(loc='lower right', frameon=True)
        plt.xlabel('maturité')
        plt.ylabel('taux')


PCA3_aff()

"AUTOENCODER"

'JAPON'
del u_jap['40-Year']
del u_jap['30-Year']
del u_jap['25-Year']
del u_jap['20-Year']
del u_jap['15-Year']
del u_jap['10-Year']
u_japn=u_jap.dropna()

T_jap=[i for i in range(1,10)]





'US:preparation base de donnée '

X_train1=train_test_split(usy_n,test_size=0.1, random_state=42)
data_train1=X_train1[0]
data_train1_s=pd.DataFrame(data_train1,columns=['1 MO','3 MO','6 MO','1 YR','2 YR'])
data_train1_l=pd.DataFrame(data_train1,columns=['5 YR','7 YR','10 YR','20 YR','30 YR'])
data_test1=X_train1[1]

'US:elargisssement de la base de données'

usy_n2=usy.copy()
del usy_n2['1 MO']
del usy_n2['2 MO']
del usy_n2['30 YR']
usy_n2=usy_n2.iloc[:6300]
'netoyyage de la base de donnée'
usy_n2n=usy_n2.dropna()

T2=[3/12,6/12,1,2,3,5,7,10,20] 

'Split de la base de donnnée'
X_train2=train_test_split(usy_n2n,test_size=0.1, random_state=42)
data_train2=X_train2[0]
data_test2=X_train2[1]
'la fonction melange et split la base de donnée'
def split(data):
    data_s=train_test_split(data,test_size=0.1, random_state=42)
    return([data_s[0],data_s[1]])
    
H=split(u_all)


'sparse autoencoder'
def kl_divergence(rho, rho_hat):
    return rho * tf.log(rho) - rho * tf.log(rho_hat) + (1 - rho) * tf.log(1 - rho) - (1 - rho) * tf.log(1 - rho_hat)

'batch normalization'
def batchnorm(Ylogits, is_test, iteration, offset):
    exp_moving_avg = tf.train.ExponentialMovingAverage(0.999, iteration) # adding the iteration prevents from averaging across non-existing iterations
    bnepsilon = 1e-5
    
    mean, variance = tf.nn.moments(Ylogits, [0])
    update_moving_averages = exp_moving_avg.apply([mean, variance])
    m = tf.cond(is_test, lambda: exp_moving_avg.average(mean), lambda: mean)
    v = tf.cond(is_test, lambda: exp_moving_avg.average(variance), lambda: variance)
    Ybn = tf.nn.batch_normalization(Ylogits, m, v, offset, None, bnepsilon)
    return Ybn, update_moving_averages 


# train/test selector for batch normalisation
#tst = tf.placeholder(tf.bool)
## training iteration
#iter = tf.placeholder(tf.int32)
def get_random_block_from_data(data, batch_size):
    start_index = np.random.randint(0, len(data) - batch_size)
    return data[start_index:(start_index + batch_size)]
#
#
#
#tf_features=tf.placeholder(tf.float32,shape=[None,9],name='entre')
#w1=tf.Variable(tf.random_normal([9,7 ]),name='w1')
#w2=tf.Variable(tf.random_normal([7,9 ]),name='w2')
#b1=tf.Variable(tf.zeros(7),name='b1')
#
#b2=tf.Variable(tf.zeros(9),name='b2')
#z1=tf.matmul(tf_features,w1)+b1
#
#z1a=tf.nn.leaky_relu(z1,alpha=0.001,name=None)
#z2=tf.matmul(z1a,w2)+b2
#output=z2
#
#learning_rate=0.001
#
#erreur=tf.reduce_mean(tf.square(output-tf_features))
#optimizer = tf.train.AdamOptimizer(learning_rate)
#'opération dentrainement'
#train = optimizer.minimize(erreur)
#
#'sauvegarde'
#saver = tf.train.Saver()
#save_dir = 'checkpoints/'
#'creation repertoire'
#if not os.path.exists(save_dir):
#    os.makedirs(save_dir)
##    
#save_path = os.path.join(save_dir, 'best')
##
#batch_size = 100
#n_samples=5670
#
#sess=tf.Session()
#sess.run(tf.global_variables_initializer())
##batch=data_train2[:200]
#
#for e in range(1):
#    avg_e = 0
#    total_batch = int(n_samples / batch_size)
#    for i in range(total_batch):
#        batch_xs = get_random_block_from_data(data_train2, batch_size)
#        sess.run(train,feed_dict={tf_features:batch_xs})
#        e=sess.run(erreur,feed_dict={tf_features:batch_xs})
#        avg_e += e / n_samples * batch_size
##saver.save(sess,save_path )



#print('sortie=',sess.run(erreur,feed_dict={tf_features:data_test2}))

#with tf.Session() as session:
#    saver.restore(session, save_path)
#    print("Model restored.")
#  # Check the values of the variables
#    print("erruer : %s" % erreur.eval())



#save_path = os.path.join(save_dir, 'best_v')    
#chkp.print_tensors_in_checkpoint_file(save_path, tensor_name='', all_tensors=True)    
#chkp.print_tensors_in_checkpoint_file(save_path, tensor_name='v1', all_tensors=False)
    
    
def get_random_block_from_data(data, batch_size):
    start_index = np.random.randint(0, len(data) - batch_size)
    return data[start_index:(start_index + batch_size)]
        
def autoprime1g(data,d,dp,Tp,epoch,alpha,batch_size):
    
    tst = tf.placeholder(tf.bool)
    iter = tf.placeholder(tf.int32)
    'split la data'
    data_train=split(data)[0]
    data_test=split(data)[1]

    'pour nimporte quel taille de data set [None,11] ici None = 2500'
    tf_features=tf.placeholder(tf.float32,shape=[None,d])
    tf_target= tf_features
    #print(tf_features)
    'poids'
    w1=tf.Variable(tf.random_normal([d,dp ]))
#    we=w1=tf.Variable(tf.random_normal([dp,dp ]))
#    wd=w1=tf.Variable(tf.random_normal([dp,dp ]))
    w2=tf.Variable(tf.random_normal([dp,d ]))

    
    'biais'
    b1=tf.Variable(tf.zeros(dp))
#    be=tf.Variable(tf.zeros(dp))
#    bd=tf.Variable(tf.zeros(dp))
    b2=tf.Variable(tf.zeros(d))
    

    
    'preactivation du neurone:multiplication des poids par les entrée: multiplication matricielle'
    '+activation'
    z1=tf.matmul(tf_features,w1)+b1
    z1a=tf.nn.leaky_relu(z1,alpha=0.5,name=None)
    
#    ze=tf.matmul(z1a,we)+be
#    zea=tf.nn.leaky_relu(ze,alpha=0.5,name=None)
#    
#    zd=tf.matmul(zea,wd)+bd
#    zda=tf.nn.leaky_relu(zd,alpha=0.5,name=None)
    
    z2=tf.matmul(z1a,w2)+b2
    z2a=tf.nn.leaky_relu(z2,alpha=0.5,name=None)
#    
    

    

    print('o')
    output=z2a

    'regularisation: contre overfitting'
#    alpha=0.001
    regularizer = tf.nn.l2_loss(w1) + tf.nn.l2_loss(w2)
    'definition de lerreur MSE : moindre carré'
#    erreur=tf.reduce_mean(tf.square(output-tf_features))

    erreur=0.5*tf.reduce_mean(tf.square(output-tf_features))+ 0.5*alpha*regularizer

    'minimisation de lerreur'
    decay_step = tf.Variable(2, trainable=False)
    starter_learning_rate=0.1
#    decay_step=1
    global_step=tf.placeholder(tf.int32)
    decay_rate=0.15
#    learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, decay_step, decay_rate, staircase=True)
    learning_rate = tf.train.inverse_time_decay(starter_learning_rate, global_step, decay_step, decay_rate, staircase=True)
#    learning_rate=0.1
#    learning_rate=tf.Variable(0.9, trainable=False)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    'opération dentrainement'
    train = optimizer.minimize(erreur)

    
    'parametre de batch'
#    batch_size = 100
    n_samples=len(data_train)
    
    'definition session: contient le graphe defini précédemment + initialisation'
    sess=tf.Session()
    sess.run(tf.global_variables_initializer())


    'on renvoie lerreur'
#    print('erreur avant',sess.run(erreur,feed_dict={tf_features:usy_n/100}))
    E_train=[]
    E_test=[]
    ax=[]
    LR=[]
 
    r=0

    for e in range(epoch + 1):
        avg_err = 0
        total_batch = int(n_samples / batch_size)
        for i in range(total_batch):
            batch_xs = get_random_block_from_data(data_train, batch_size)
            sess.run(train,feed_dict={tf_features:batch_xs,tst:False,iter:e,global_step:e})
            eta=sess.run(learning_rate,feed_dict={tf_features:data_test,global_step:e})
#            print(eta)
            err=sess.run(erreur,feed_dict={tf_features:batch_xs})
            avg_err += err / n_samples * batch_size
            

#            print('erreur_train=',avg_err)
#            print('erreur_test=',sess.run(erreur,feed_dict={tf_features:data_test2,tst:True,iter:e}))            
        if (e%10==0):
            err_t=sess.run(erreur,feed_dict={tf_features:data_test})
            
            print(eta)
            LR.append(eta)
            if (sess.run(erreur,feed_dict={tf_features:data_test,tst:True,iter:e})<10):
                    
                ax.append(e)
                E_train.append(avg_err)
                E_test.append(sess.run(erreur,feed_dict={tf_features:data_test,tst:True,iter:e}))
                
            print(str(e) + ": erreur_train:" + str(avg_err) + " erreur_test: " + str(err_t) )
#            print('erreur_train=',avg_e)
#            print('erreur_test=',sess.run(erreur,feed_dict={tf_features:data_test2}))
            
    
#            E_train.append(avg_e)
#            E_test.append(sess.run(erreur,feed_dict={tf_features:data_test2,tst:True,iter:e}))



#        r=e
#        print(r)

        
    erreur_f=sess.run(erreur,feed_dict={tf_features:data_test,tst:True,iter:r})




    'base de donnée reconstruite'
    app=pd.DataFrame(sess.run(output,feed_dict={tf_features:data_test,tst:False,iter:r}))
    P=pd.DataFrame(sess.run(w1,feed_dict={tf_features:data_test,tst:False,iter:r}))
#    code=pd.DataFrame(sess.run(z1a,feed_dict={tf_features:data_test,tst:False,iter:r}))
#    code.columns=['code1','code2','code3']
#    app.index=data_test.index
#    print(data_test.index)
    appfixe=app.copy()
    E_ann=[]
    for i in range(len(appfixe)):
        a_ann=appfixe.iloc[i]
        b_ann=data_test.iloc[i]
        e_ann=mean_squared_error(a_ann,b_ann)
        E_ann.append(e_ann)
    E_ann=pd.DataFrame(E_ann,columns=['mse_ann'])
    appfixe_ann=pd.concat([appfixe,E_ann],axis=1)
        
        
        
    
    'fonction affichage'
    i=randint(0,len(data_test))
    print(i)
    i=0
    
    'pca'
    l=PCA_final_c(i,data_test,3)
    y_pca=l[0]
    mse=l[1]
    mse=round(mse,6)
    n_comp=l[2]
    F=l[3]
    mse_pca_global=l[4]

#    print('n_comp=',n_comp)
    
    'ann'
    yapp1=app.iloc[i]
    y1=data_test.iloc[i]
    name=y1.name

    mse_ann=mean_squared_error(y1,yapp1)
    mse_ann=round(mse_ann,6)


    'affichage dun courbe random'
#    plt.figure(1)
#    
#    plt.plot(Tp,yapp1,marker='o', linestyle='--',color='orange',label='alpha='+str(alpha))
##    plt.plot(Tp,y1,marker='o', color= 'blue',label='courbe réelle')
###    plt.plot(Tp,y_pca,marker='o',linestyle='--',color='green',label='PCA')
##    plt.title('Random_autoencodeur, erreur:'+str(mse_ann) +'\n'+ 'PCA, erreur:'+str(mse))
#    plt.xlabel('Maturités'+'\n'+str(name))
#    plt.ylabel('Rendement à maturité')
#    plt.legend()
    
    
#    print('erreur ann_global=',erreur_f)
#    print('erreur pca_global=',mse_pca_global)
#    
#    'pour indice random'
#    print('erreur ann_local=',mse_ann)
#    print('erreur pca_local=',mse)
#    

    
    'erreur de pCA la plus grande'
    i_max=F['mse'].idxmax()
    yapp_max=app.iloc[i_max]
    y_max=data_test.iloc[i_max]
    mse_ann_max=mean_squared_error(y_max,yapp_max)
    F_pca=F.copy()
    del F_pca['mse']
    y_pca_max=F_pca.iloc[i_max]
    name=y_max.name
#    plt.figure(2)
#    plt.subplot(212)
#    plt.plot(Tp,yapp_max,marker='o', linestyle='--',color='red',label='autoencodeur')
#    plt.plot(Tp,y_max,marker='o', color= 'blue',label='courbe réelle')
#    plt.plot(Tp,y_pca_max,marker='o',linestyle='--',color='green',label='PCA')
#    plt.title('MAX_PCA, autoencodeur, erreur:'+str(mse_ann_max) +'\n'+ 'PCA, erreur:'+str(F['mse'].max()))
#    plt.xlabel('Maturités'+'\n'+str(name))
#    plt.ylabel('Rendement à maturité')
#    plt.legend()
    
    '1_erreur de PCA la plus faible'
    i_min=F['mse'].idxmin()
    yapp_min=app.iloc[i_min]
    y_min=data_test.iloc[i_min]
    mse_ann_min=mean_squared_error(y_min,yapp_min)
    y_pca_min=F_pca.iloc[i_min]
    name=y_min.name
#    plt.figure(3)
#    plt.subplot(212)
#    plt.plot(Tp,yapp_min,marker='o', linestyle='--',color='red',label='autoencodeur')
#    plt.plot(Tp,y_min,marker='o', color= 'blue',label='courbe réelle')
#    plt.plot(Tp,y_pca_min,marker='o',linestyle='--',color='green',label='PCA')
#    plt.title('MIN_PCA, autoencodeur, erreur:'+str(mse_ann_min) +'\n'+ 'PCA, erreur:'+str(F['mse'].min()))
#    plt.xlabel('Maturités'+'\n'+str(name))
#    plt.ylabel('Rendement à maturité')
#    plt.legend()   
    
    '2_erreur ann la plus faible'
    i_min_ann=appfixe_ann['mse_ann'].idxmin()
    yapp_min2=app.iloc[i_min_ann]
    y_min2=data_test.iloc[i_min_ann]
    y_pca_min2=F_pca.iloc[i_min_ann]
    mse_ann_min2=mean_squared_error(y_min2,yapp_min2)
    mse_pca_min2=mean_squared_error(y_min2,y_pca_min2)
    name2=y_min2.name
    
#    plt.figure(4)
#    plt.plot(Tp,yapp_min2,marker='o', linestyle='--',color='red',label='autoencodeur')
#    plt.plot(Tp,y_min2,marker='o', color= 'blue',label='courbe réelle')
#    plt.plot(Tp,y_pca_min2,marker='o',linestyle='--',color='green',label='PCA')
#    plt.title('MIN_ANN, autoencodeur, erreur:'+str(mse_ann_min2) +'\n'+ 'PCA, erreur:'+str(mse_pca_min2))
#    plt.xlabel('Maturités'+'\n'+str(name2))
#    plt.ylabel('Rendement à maturité')
#    plt.legend()   
    
    
    
    'affichage de la courbe derreur'
#    plt.figure(5)
#    E_train=E_train[2:]
#    E_test=E_test[2:]
###    E_train=np.log(pd.DataFrame(E_train))
###    X=np.array([i for i in range(0,r,100)])
#    plt.plot(ax[2:],E_train, color='red',label='dimension '+str(dp))
##    plt.plot(ax[2:],E_test,color='blue',label='test')
###    plt.plot(ax[2:],E_test,color='red',marker='o',label='constant'+str(eta))
###    plt.plot(e_final)
#    plt.xlabel('nombre sessions dentrainement')
#    plt.ylabel('erreur')
#    plt.legend()
#    
#    plt.figure(6)
##    print(LR)
#    plt.plot(ax,LR,color='blue',label='inverse time'+str(decay_rate))
##    plt.plot(ax,LR,color='red',label='constant'+str(eta))
#    plt.xlabel('nombre sessions dentrainement')
#    plt.ylabel('learning rate')
#    plt.legend()
    
#    code.index=data_test.index
    
#    return(app,data_test,code)
    return(app,data_test,P)
    

    
#    
#Dd=autoprime1g(usy_n2n.iloc[100:],9,3,T2,3,0,100)
#D=autoprime1g(u_all,31,10,T_all,300,0,100)
#autoprime1g(u_japn,9,3,T_jap,200,0,100)
#autoprime1g(u_nz,7,3,T_nz,6)


#data_n=D[0]
#data_old=D[1] 
#data_n.index=data_old.index
#data_n.columns=data_old.columns
 


#E=[]
#for m in data_old.columns:
#    ynew=data_n[m]
#    yold=data_old[m]
#    mse=mean_squared_error(ynew,yold)
#    E.append(mse)
#S=pd.Series(E,index=data_old.columns)    
#mini=S.idxmin()
#maxi=S.idxmax()

#odronne la dataframe pae data (melange lors du split)
#data_old = data_old.sort_index()
#data_n = data_n.sort_index()

#plt.plot(data_old[maxi])
#plt.plot(data_n[maxi])



    
def PCA_selection(data,n):
    pcaP=PCA(n_components=n)
    pcaP.fit(data)
    comp=pcaP.transform(data)
    #Base de donnée reconstruite
    approx=pcaP.inverse_transform(comp)
    approx=pd.DataFrame(approx,columns=data.columns)
    var=pcaP.explained_variance_ratio_
    col=['pc:'+str(i) for i in range(1,n+1)]
    comp=pd.DataFrame(comp,index=data.index,columns=col)
    print('pct variance expliqué',var)
    return([approx,data,var,comp])
#    
#H1=PCA_selection(u_fr,3)
#H1[3].plot()

#rename all data base columns
D=[u_all,u_fr,u_it,u_bel,u_esp,u_gb]
for df in D:
    L=[c for c in df]
    df.columns=[str(df.name)+': '+c for c in L]
    
    
def join_data():
    G=pd.concat([u_all,u_fr,u_it,u_bel,u_esp,u_gb],axis=1)
    return(G)
    

    

    
#J=join_data()
#J=J.sort_index()
#Ra=get_return(J)
    

'selectionne les bonds pire/meilleurs de chaque pays'
def select_bond(data_pays,epoch,pca):
 

    k=len(data_pays.columns)
    
    #AUTOENCODEUR:
    if pca==0:
       #on entraine l'autoencodeur hors de la fenetre'
       D=autoprime1g(data_pays,k,3,T,epoch,0,100)
    else:
       D=PCA_selection(data_pays,3)
   
    data_n=D[0]
    data_old=D[1] 
    data_n.index=data_old.index
    data_n.columns=data_old.columns
       
    E=[]
    for m in data_old.columns:
        ynew=data_n[m]
        yold=data_old[m]
        mse=mean_squared_error(ynew,yold)
        E.append(mse)
    S=pd.Series(E,index=data_old.columns)    
    mini=S.idxmin()
    maxi=S.idxmax()
    
    #ordronne la dataframe par data (melange lors du split) (uniquement utile pr affichage)
    data_old = data_old.sort_index()
    data_n = data_n.sort_index()
    
#    plt.plot(data_old[maxi],label='courbe initiale')
#    plt.plot(data_n[maxi],label='courbe reconstuite')
#    plt.plot(data_old[mini],label='courbe initiale')
#    plt.plot(data_n[mini],label='courbe reconstuite')
#    plt.legend()
    sample=data_pays
    G=pd.DataFrame([sample[mini]])
#    print(S)


#    plt.plot(sample[maxi])
    F_pays=pd.DataFrame([sample[mini],sample[maxi]])
    #transpose la dataframe
    F_pays=F_pays.T
    F_pays.columns=[str(data_pays.name)+': '+str(mini),str(data_pays.name)+': '+str(maxi)]
    return(F_pays)
    

#m=select_bond(u_fr,20,0)


def new_select_bond(data_pays,epoch,pca):
    #nombre de maturité disponible pour le pays'

    k=len(data_pays.columns)
    
    #AUTOENCODEUR:
    if pca==0:
       #on entraine l'autoencodeur hors de la fenetre'
       D=autoprime1g(data_pays,4,3,T,epoch,0,100)
       code=D[2]
       pc=code['code1']
       data_old=D[1]
    else:
       D=PCA_selection(norm(data_pays),3)
       code=D[3]
       pc=code['pc1']
       data_old=D[1]
          
    E=[]
    
    for m in data_old.columns:
        
        yold=data_old[m]
        mse=mean_squared_error(pc,yold)
        E.append(mse)
    S=pd.Series(E,index=data_old.columns)    
    mini=S.idxmin()
    
    #ordronne la dataframe par data (melange lors du split) (uniquement utile pr affichage)
    data_old = data_old.sort_index()
    
#    pc=pc.sort_index()
    
#    plt.plot(data_old[maxi],label='bond')
#    plt.plot(pc,label='pc')
#    plt.ylabel()
#    plt.legend()
    
    sample=data_pays
    


#    plt.plot(sample[maxi])
#    F_pays=pd.DataFrame(sample[mini])
    F_pays=pd.concat([sample[mini],pc],axis=1)
    #transpose la dataframe
#    F_pays=F_pays.T

    F_pays.columns=[str(data_pays.name)+': '+str(mini),'pc']
    print(S)
    
    return(F_pays)
    
#Bb=new_select_bond(u_all,2,1)
#plt.plot(u_all)
#plt.plot(Bb)    
#u_itn=(u_it-u_it.mean(axis=0))/u_it.std(axis=0)
#u_itn.name='it'

#H=new_select_bond(u_it,2,1)
#H.plot()
#print('a')
#PF a partir de la base de donné général    
def selec_general(pca,epoch,pc,ng,nb):
    data_pays=join_data()
    k=len(data_pays.columns)
    #AUTOENCODEUR:
    if pca==0:
       #on entraine l'autoencodeur hors de la fenetre'
       D=autoprime1g(data_pays,k,5,T,epoch,0,100)
       print('AUTOENCODER4')
    else:
       D=PCA_selection(data_pays,pc)
       print('PCA')
       #afficher le pct var explained??
    data_n=D[0]
    data_old=D[1] 
    data_n.index=data_old.index
    data_n.columns=data_old.columns
    E=[]
    for m in data_old.columns:
        ynew=data_n[m]
        yold=data_old[m]
        mse=mean_squared_error(ynew,yold)
        E.append(mse)
    S=pd.Series(E,index=data_old.columns)
    S=S.sort_values(axis=0)
    print(S)
    index_g=[i for i in S.iloc[:ng].index]
    index_b=[i for i in S.iloc[55-nb:].index]
    F_pays=pd.concat([data_pays[index_b],data_pays[index_g]],axis=1)
#    Comp=pd.concat([data_old[index_g],data_n[index_g]],axis=1)
#    Comp.columns=['initiale'+str(index_g),'reconstruit']
#    Comp.plot()
    return(F_pays)
    
#K= selec_general(1,1,3,1,1)
    
def new_selec_general(pca,epoch,pc,ng,nb):
    data_pays=join_data()[:50]
    k=len(data_pays.columns)
    #AUTOENCODEUR:
    if pca==0:
       #on entraine l'autoencodeur hors de la fenetre'
       D=autoprime1g(data_pays,k,5,T,epoch)
       print('AUTOENCODER4')
    else:
       D=PCA_selection(norm(data_pays),pc)
       print('PCA')
       #afficher le pct var explained??
    data_norm=D[1]
    code=D[3]
    data_old=data_pays
    col=[c for c in code.columns]

    
    IND=[0]*3
    ER=[]
    for i in range(len(col)):
        E=[]
        for m in data_old.columns:
            ynew=code[col[i]]
            yold=data_norm[m]
            mse=mean_squared_error(ynew,yold)
            E.append(mse)
        S=pd.Series(E,index=data_old.columns)
        S=S.sort_values(axis=0)
        print(S)
        ind=S.iloc[:1].index
        ER.append(S.iloc[:2])
        print(ind)
        print(ER)
        IND[i]=ind
    F=data_pays[IND[0]]
    F=pd.concat([F,data_pays[IND[1]],data_pays[IND[2]]],axis=1)
    plt.plot(norm(F),label='sel')
#    plt.plot(F)
    plt.plot(code,label='code')
    plt.legend()
#    plt.legend()
#    print(S)
#    index_g=[i for i in S.iloc[:ng].index]
#    index_b=[i for i in S.iloc[55-nb:].index]
#    F_pays=pd.concat([data_pays[index_b],data_pays[index_g]],axis=1)
    return(F)
    
#H=new_selec_general(1,0,3,5,5) 
#J= selec_general(0,100,3,5,5)   
    
#selection par pays
def selection_alea(data_pays):
    L=[c for c in data_pays]
    i=randint(0,len(L)-1)
    j=randint(0,len(L)-1)
    F_pays=pd.concat([data_pays[L[i]],data_pays[L[j]]],axis=1)
    F_pays.columns=[str(data_pays.name)+': '+str(L[i]),str(data_pays.name)+': '+str(L[j])]
    return(F_pays)
    
def new_selection_alea(data_pays):
    L=[c for c in data_pays]
    i=randint(0,len(L)-1)
    F_pays=pd.DataFrame(data_pays[L[i]])
    F_pays.columns=[str(data_pays.name)+': '+str(L[i])]
    return(F_pays)
    

    
#selection aleaparmis tous les pays
def selection_alea_gene(n):
    data_pays=join_data()
    L=[c for c in data_pays]
    i=randint(0,len(L)-1)
    A=data_pays[L[i]]
    for k in range(n):
        i=randint(0,len(L)-1)
        A=pd.concat([A,data_pays[L[i]]],axis=1)
    return(A)
    
#D=create_data_gene()
#A=selection_alea_gene(6)    
    
'assemble le porteflolio utiliser pour le tracking'
def create_portefolio(epoch,pca,alea):
        
    L=[u_all,u_fr,u_esp,u_bel,u_it,u_gb]
    if alea:
       Lf=[selection_alea(pays) for pays in L]
    else:
       Lf=[select_bond(pays,epoch,pca) for pays in L]
    P=pd.concat([F for F in Lf],axis=1)
    return(P)
  
#P=create_portefolio(0,1,0)
'prediction'
def prediction_track(epoch,pca,epoch_auto,window,alea,alea_g,gene,n,ng):
    #Preaparation des données
    if alea:
        if alea_g:
            Pf=selection_alea_gene(n)
            print('alea generalisé')
        else:
            Pf=create_portefolio(2,1,1)
            print('alea pays par pays')

    #Une base d'entraineme
    else:
        if gene:
            nb=n-ng
            Pf=selec_general(pca,epoch_auto,10,ng,nb)
            print('data generalisé')
        else:
            if (pca):
                Pf=create_portefolio(2,1,0)
                print('pca pp')
            else:
                Pf=create_portefolio(epoch_auto,0,0)
                print('auto pp')
 
    P=Pf.iloc[window:]
    Y=index_gene()[window:]
    F=pd.concat([P,Y],axis=1)
    S=split(F)
    data_train=S[0]
    data_test=S[1]

    
    y_index_train=data_train['index']
    y_index_test=data_test['index']
    y_index_size=len(y_index_test)
    #necessaire pr feeder le tf_target, ne marche pas sinon
 
#    y_index_test = np.reshape(y_index_test, [y_index_size, 1])
    y_index_test=y_index_test.values.reshape(y_index_size,1) 
    print('o')
   
    data_PF_test=data_test.copy()
    del data_PF_test['index']
    
    #Une base de test out-of-sample de 100 jours
    Y_out=index_gene()[:window]
    P_out=Pf.iloc[:window]
    #ok trié ds l'ordre chrono
    P_out=P_out.sort_index()
    Y_out=Y_out.sort_index()
    Y_out_size=len(Y_out)
    Y_out_feed=Y_out.values.reshape(Y_out_size,1)
    
    print('donnée prep')
    #Defintion de l'arbre
    d=len(data_PF_test.columns)
    tf_features=tf.placeholder(tf.float32,shape=[None,d])
    tf_target= tf.placeholder(tf.float32,shape=[None,1])
    
    di=7
    dp=5
    
    w1=tf.Variable(tf.random_normal([d,dp ]))
    w2=tf.Variable(tf.random_normal([dp,1]))
    
#    w1=tf.Variable(tf.random_normal([d,di ]))
#    w2=tf.Variable(tf.random_normal([di,dp]))
#    w3=tf.Variable(tf.random_normal([dp,1]))
    
    
    b1=tf.Variable(tf.zeros(dp))
    b2=tf.Variable(tf.zeros(1))
#    b1=tf.Variable(tf.zeros(di))
#    b2=tf.Variable(tf.zeros(dp))
#    b3=tf.Variable(tf.zeros(1))
    
    
    z1=tf.matmul(tf_features,w1)+b1
    z1a=tf.nn.leaky_relu(z1,alpha=0.5,name=None)
    
    z2=tf.matmul(z1a,w2)+b2
    z2a=tf.nn.leaky_relu(z2,alpha=0.5,name=None)
    
#    z3=tf.matmul(z2a,w3)+b3
#    z3a=tf.nn.leaky_relu(z3,alpha=0.5,name=None)
    
    output=z2a
    
    'regularisation: contre overfitting'
    alpha=0.00001
    regularizer = tf.nn.l2_loss(w1) + tf.nn.l2_loss(w2)
    'definition de lerreur MSE : moindre carré'
    #erreur=tf.reduce_mean(tf.square(output-tf_features))
    erreur=0.5*tf.reduce_mean(tf.square(output-tf_target))+0.5*alpha*regularizer
    
    'minimisation de lerreur'
    decay_step = tf.Variable(2, trainable=False)
    starter_learning_rate=0.1
#    decay_step=1
    global_step=tf.placeholder(tf.int32)
    decay_rate=0.15
#    learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, decay_step, decay_rate, staircase=True)
    learning_rate = tf.train.inverse_time_decay(starter_learning_rate, global_step, decay_step, decay_rate, staircase=True)
    #    learning_rate=0.0001
    optimizer = tf.train.AdamOptimizer(learning_rate)
    'opération dentrainement'
    train = optimizer.minimize(erreur)
    
    'parametre de batch'
    batch_size = 100
    n_samples=len(data_train)
    
    'definition session: contient le graphe defini précédemment + initialisation'
    sess=tf.Session()
    sess.run(tf.global_variables_initializer())
            
    'on renvoie lerreur'

    E_train=[]
    E_test=[]
    ax=[]
     
    #    r=0
    print('debut')
    for e in range(epoch + 1):
        avg_err = 0
        total_batch = int(n_samples / batch_size)
        for i in range(total_batch):
            batch_xs = get_random_block_from_data(data_train, batch_size)
            batch_y=batch_xs['index']
            batch_size=len(batch_y)
            batch_y = batch_y.values.reshape(batch_size, 1)
            
            batch_PF=batch_xs.copy()
            del batch_PF['index']
        
            sess.run(train,feed_dict={tf_features:batch_PF,tf_target:batch_y,global_step:e})
            
            err=sess.run(erreur,feed_dict={tf_features:batch_PF,tf_target:batch_y})
            avg_err += err / n_samples * batch_size
            
        #        s=sess.run(output,feed_dict={tf_features:batch_PF,tf_target:batch_y})
            
            
        #        print('erreur_train=',avg_err)
        #        print('erreur_test=',sess.run(erreur,feed_dict={tf_features:data_test2,tst:True,iter:e}))            
        if (e%10==0):
            ax.append(e)
            E_train.append(avg_err)
            E_test.append(sess.run(erreur,feed_dict={tf_features:data_PF_test,tf_target:y_index_test}))
            err_t=sess.run(erreur,feed_dict={tf_features:data_PF_test,tf_target:y_index_test})
            print(str(e) + ": erreur_train:" + str(avg_err) + " erreur_test: " + str(err_t) )
        #        print('erreur_train=',avg_e)
        #        print('erreur_test=',sess.run(erreur,feed_dict={tf_features:data_test2}))
    
    
    #Finn de l'entrainement:phase de prédiction
    #tout 'out of sample sans reeentrainement
    sortie1=sess.run(output,feed_dict={tf_features:P_out,tf_target:Y_out_feed})
    sf1=pd.DataFrame(sortie1,columns=['tracking'])
    sf1.index=Y_out.index
#    S=pd.concat([sf1,Y_out],axis=1)
    print('erreur_out',mean_squared_error(sf1,Y_out))
#    print(S.mean(axis=0))
#    print(S.std(axis=0))
    plt.plot(sf1,label='pca')
    plt.legend()

    
    'rentrainement'
#    n=len(P_out)
#    sortie=[]
#    p=P_out.iloc[0]
#    p=pd.DataFrame(p).T
#    batch_f=p
##    print(Y_out_feed[0])
##    print(Y_out_feed)
#    for i in range(1,n):
#        p=P_out.iloc[i]
#        p=pd.DataFrame(p)
#        p=p.T
#        s=sess.run(output,feed_dict={tf_features:p,tf_target:[Y_out_feed[i]]})
#
##        print(s)
#        sortie.append(float(s))
#        batch_f=pd.concat([batch_f,p],axis=0)
##        print(batch_f)
##        #reajustement des poids en fonction de la date du lendemain
#        sess.run(train,feed_dict={tf_features:batch_f,tf_target:Y_out_feed[:i+1],global_step:e})
##        b=sess.run(tf_target,feed_dict={tf_target:Y_out_feed[:i+1]})
#
##        print('b',len(b))
#        
#    print('renentrainement fait')
#
#    sf=pd.DataFrame(sortie)
##    print(len(Y_out.iloc[1:]))
##    print(len(sf))
#    sf.index=(Y_out.iloc[1:]).index
#    S=pd.concat([sf,Y_out.iloc[1:],sf1.iloc[1:]],axis=1)
#    S.columns=['tracking','index','trackini']
##    print('erreur',mean_squared_error(S['trackini'],S['index']))
##    print('erreur_apres',mean_squared_error(S['tracking'],S['index']))
#    print('erreur:'+str(mean_squared_error(S['trackini'],S['index'])) +'erreur_apres'+ str(mean_squared_error(S['tracking'],S['index'])))
#    #calcul moyenne variance
#    M=S.mean(axis=0)
#    V=S.var(axis=0)
#    print("m_index:" + str(M['index']) + " m_tracking " + str(M['tracking'])+ " m_trackini " + str(M['trackini']) )
#    print("v_index:" + str(V['index']) + " v_tracking " + str(V['tracking'])+ " V_trackini " + str(V['trackini']) )
    #sortie jour pas jour et reentrainement chaque jours
#    plt.figure(1)
#    S.columns=['réentrainé','index','non réentrainé']
#    plt.plot(S['index'],label='index')
#    plt.plot(S['réentrainé'],label='réentrainé')
#    plt.legend()

    
    
#    plt.figure(2)
#    E_train=E_train[2:]
#    E_test=E_test[2:]
###    E_train=np.log(pd.DataFrame(E_train))
###    X=np.array([i for i in range(0,r,100)])
#    plt.plot(ax[2:],E_train,linestyle='--', color='red',label='train',)
##    plt.plot(ax[2:],E_test,linestyle='--',color='blue',label='test','--')
###    plt.plot(ax[2:],E_test,color='red',marker='o',label='constant'+str(eta))
###    plt.plot(e_final)
#    plt.xlabel('nombre sessions dentrainement')
#    plt.ylabel('erreur')
#    plt.legend()
    
    return(S)

#prediction_track(epoch,pca,epoch_auto,window,alea,aleag,gene,n,ng)    
#s=prediction_track(200,1,2,300,1,1,4) 
#prediction_track(epoch,pca,epoch_auto,window,alea,alea_g,gene,n,ng)
#s_alea=prediction_track(6,1,2,300,0,0,0,12,5)
    
#    (epoch,pca,epoch_auto,window,alea,alea_g,gene,n,ng)
#s_auto=prediction_track(150,0,250,20,0,0,0,12,6)
#s_alea=prediction_track(150,0,2,100,1,1,0,12,5)  
#s_pca=prediction_track(150,1,2,100,0,0,0,10,5) 

#F=pd.concat([s_alea['index'],s_alea['tracking'],s_pca['tracking'],s_auto['tracking']],axis=1)
#F.columns=['index','alea','pca','autoencodeur']
#F.plot()
#compare les différentes selections   
def compare(epoch,epoch_auto,window):
#    s_alea=prediction_track(epoch,1,2,300,1,0)
    s_alea=prediction_track(epoch,0,2,window,1,1,0,12,5)
    print('alea done')
    s_pca=prediction_track(epoch,1,2,window,0,0,1,10,5)
    print('pca done gene')
#    s_pca_p=prediction_track(epoch,1,2,300,0,0,0,10,5)
#    print('pca done gene')
    s_auto=prediction_track(epoch,0,epoch_auto,window,0,0,1,12,6)
    print('aut_done')
#    F=pd.concat([s_alea['index'],s_alea['tracking'],s_pca['tracking'],s_auto['tracking']],axis=1)
#    F.columns=['index','alea','pca','autoencodeur']
    F=pd.concat([s_alea['index'],s_alea['tracking'],s_alea['trackini'],s_pca['tracking'],s_pca['trackini'],s_auto['tracking'],s_auto['trackini']],axis=1)
    F.columns=['index','alea','aleafaible','pca','pcafaible','autoencodeur','autofaibe']
    
#    F=pd.concat([s_pca['index'],s_alea['tracking'],s_pca['tracking'],s_pca_p['tracking']],axis=1)
#    F.columns=['index','alea','pca','pcap']
    return(F)  
#
#F=compare(3,0,100)
#simulation de PF aleatoire
def simul_alea(n,window,epoch):
    s_alea=prediction_track(epoch,0,2,window,1,1,0,12,5)
    I=pd.DataFrame(s_alea['index'])
    F=pd.DataFrame(s_alea['index'])
    for i in range(n):
        s_alea=prediction_track(epoch,0,2,window,1,1,0,12,5)
        F=pd.concat([F,s_alea['tracking']],axis=1)
    del F['index']
    M=F.mean(axis=1)
    plt.plot(I,color='black')
    plt.plot(M,color='blue')
    plt.plot(F)
    return(M,F)
    
#S=simul_alea(6,300,200)    
#
#G=compare(2,1,100) 
#G.plot()

def get_block(data,p,k):
    block=data.iloc[k:k+p]
    if (k+p>len(data)):
        block=data.iloc[k:len(data)]
    return(block)


'prediction des returns'
def prediction0(data,epoch,window,day):
    #Preaparation des données
    data=data.sort_index()
    shift_day=day
    data_target=data.shift(-shift_day)
    data_target=data_target.dropna()
    data_input=data.iloc[:len(data)-shift_day]
    
    data_input.name='input'
    data_input.columns=[str(data_input.name)+': '+c for c in data_input.columns]
    
    c_input=[c for c in data_input.columns]
    c_target=[c for c in data_target.columns ]
    
    D=pd.concat([data_input,data_target],axis=1)
    D_train=D[:len(D)-window]
    D_test=D[len(D)-window:]
    data_cible=data_target[len(D)-window:]

    
    print('donnée prep')
    #Defintion de l'arbre
    d=len(data_input.columns)
    tf_features=tf.placeholder(tf.float32,shape=[None,d])
    tf_target= tf.placeholder(tf.float32,shape=[None,d])
    
    dp=d
    
    w1=tf.Variable(tf.random_normal([d,dp ]))
    w2=tf.Variable(tf.random_normal([dp,d]))
    
#    w1=tf.Variable(tf.random_normal([d,di ]))
#    w2=tf.Variable(tf.random_normal([di,dp]))
#    w3=tf.Variable(tf.random_normal([dp,1]))
    
    
    b1=tf.Variable(tf.zeros(dp))
    b2=tf.Variable(tf.zeros(d))
#    b1=tf.Variable(tf.zeros(di))
#    b2=tf.Variable(tf.zeros(dp))
#    b3=tf.Variable(tf.zeros(1))
    
    
    z1=tf.matmul(tf_features,w1)+b1
    z1a=tf.nn.leaky_relu(z1,alpha=0.5,name=None)
    
    z2=tf.matmul(z1a,w2)+b2
    z2a=tf.nn.leaky_relu(z2,alpha=0.5,name=None)
    
#    z3=tf.matmul(z2a,w3)+b3
#    z3a=tf.nn.leaky_relu(z3,alpha=0.5,name=None)
    
    output=z2a
    
    'regularisation: contre overfitting'
    alpha=0.00001
    regularizer = tf.nn.l2_loss(w1) + tf.nn.l2_loss(w2)
    'definition de lerreur MSE : moindre carré'
    #erreur=tf.reduce_mean(tf.square(output-tf_features))
    erreur=0.5*tf.reduce_mean(tf.square(output-tf_target))+0.5*alpha*regularizer
    
    'minimisation de lerreur'
    decay_step = tf.Variable(2, trainable=False)
    starter_learning_rate=0.1
#    decay_step=1
    global_step=tf.placeholder(tf.int32)
    decay_rate=0.72
#    learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, decay_step, decay_rate, staircase=True)
    learning_rate = tf.train.inverse_time_decay(starter_learning_rate, global_step, decay_step, decay_rate, staircase=True)
    #    learning_rate=0.0001
    optimizer = tf.train.AdamOptimizer(learning_rate)
    'opération dentrainement'
    train = optimizer.minimize(erreur)
    
    'parametre de batch'
    batch_size = 100
    n_samples=len(D)
    
    'definition session: contient le graphe defini précédemment + initialisation'
    sess=tf.Session()
    sess.run(tf.global_variables_initializer())
            
    'on renvoie lerreur'

    E_train=[]
    E_test=[]
    ax=[]
     
    #    r=0
    print('debut')
    for e in range(epoch + 1):
        avg_err = 0
        total_batch = int(n_samples / batch_size)
        for i in range(total_batch):
            batch = get_random_block_from_data(D_train, batch_size)
            batch_input=batch[c_input]
            batch_target=batch[c_target]        
            sess.run(train,feed_dict={tf_features:batch_input,tf_target:batch_target,global_step:e})
            
            err=sess.run(erreur,feed_dict={tf_features:batch_input,tf_target:batch_target})
            avg_err += err / n_samples * batch_size
           
        if (e%10==0):
            print(sess.run(learning_rate,feed_dict={tf_features:D_test[c_input],tf_target:D_test[c_target],global_step:e}))
            ax.append(e)
            E_train.append(avg_err)
            E_test.append(sess.run(erreur,feed_dict={tf_features:D_test[c_input],tf_target:D_test[c_target]}))
            err_t=sess.run(erreur,feed_dict={tf_features:D_test[c_input],tf_target:D_test[c_target]})
            print(str(e) + ": erreur_train:" + str(avg_err) + " erreur_test: " + str(err_t) )
            
    sortie=sess.run(output,feed_dict={tf_features:D_test[c_input]})
    sortie=pd.DataFrame(sortie,columns=D_test[c_input].columns,index=D_test.index)
#    
#    #affichage
#    plt.figure(1)
    A=pd.concat([sortie,D_test[c_target]],axis=1)
#    A.columns=['reel','reel','estimé','estimé']
#    plt.plot(A)
#    plt.legend()
#    plt.figure(1)
#    plt.plot(A)

    
    
#    C=[c for c in D_test[c_input].columns]
#    s=sortie[C[0]]
#    s_test=D_test[c_target][C[0]]
#    S=pd.concat([s,s_test],axis=1)
#    plt.plot(S)
    
    
    D_final=pd.concat([D_train[c_input],sortie],axis=0)
    D_final.columns=sortie.columns
#    print(D_final.shape)
    R_final=get_return(D_final)
    debut=len(D_train)-100
    fin=len(R_final)
    pas=1
    N=30
    M=[]
    I=[]
    W=pd.DataFrame(columns=sortie.columns)
    P=pd.DataFrame(columns=sortie.columns)
    for k in range(debut,fin,pas):

        if (k+N>len(R_final)):
            break
        else:
            
#            1 ere possibilité'
            block=get_block(R_final,N,k)
            S=block.cov()
            mu=block.iloc[-1]
            I.append(mu.name)
#            p=sortie.loc[mu.name]
#            P=pd.concat([P,p],axis=0)
#            
            ef = EfficientFrontier(mu, S)
            'long/short'
            #ef = EfficientFrontier(mu, S, weight_bounds=(-1, 1))
#            raw_weights = ef.max_sharpe()
#            raw_weights =ef.min_volatility()
            raw_weights =ef.efficient_return(0.05)
            cleaned_weights = ef.clean_weights()
            w=pd.DataFrame(cleaned_weights,index=[0])
            W=pd.concat([W,w],axis=0)
            K=ef.portfolio_performance(verbose=True)
            M.append(K)
            
    I=pd.DataFrame(I,columns=['date'])
    M=pd.DataFrame(M,columns=['return','vol','sr'])
    M.index=I['date']
    W.index=I['date']
    P=W.mul(sortie.loc[I['date']],1)
    
#    print('nouvelle opooooo')
#    
#    debut_prime=len(D_train)-100
#    fin_prime=len(R_final)
#    pas=3
#    N=50
#    M_prime=[]
#    I_prime=[]
#    W_prime=pd.DataFrame(columns=sortie.columns)
#    P_prime=pd.DataFrame(columns=sortie.columns)
#    for k in range(debut_prime,fin_prime,pas):
#
#        if (k+N>len(R_final)):
#            break
#        else:
#            
##            1 ere possibilité'
#            block_prime=get_block(R_final,N,k)
#            mu_prime = block_prime.mean(axis=0)
#            S_prime = block_prime.cov()
#            ef_prime= EfficientFrontier(mu_prime, S_prime)
#
#            I_prime.append(mu_prime.name)
#            'long/short'
#            #ef = EfficientFrontier(mu, S, weight_bounds=(-1, 1))
#            raw_weights_prime = ef_prime.max_sharpe()
#            cleaned_weights_prime = ef_prime.clean_weights()
#            w_prime=pd.DataFrame(cleaned_weights_prime,index=[0])
#            W_prime=pd.concat([W_prime,w_prime],axis=0)
#            K_prime=ef_prime.portfolio_performance(verbose=True)
#            M_prime.append(K_prime)
#            
#    I_prime=pd.DataFrame(I_prime,columns=['date'])
#    M_prime=pd.DataFrame(M_prime,columns=['return','vol','sr'])
#    M_prime.index=I['date']
#    W_prime.index=I['date']
#    P_prime=W_prime.mul(sortie.loc[I['date']],1)
    
#    troisieme option:
#    print('LASTTTT')
#    R_final_c=get_return(data_target)
#    debut_c=len(D_train)-100
#    fin_c=len(data_target)
#    pas=3
#    N=50
#    M_c=[]
#    I_c=[]
#    W_c=pd.DataFrame(columns=sortie.columns)
#    P_c=pd.DataFrame(columns=sortie.columns)
#    for k in range(debut_c,fin_c,pas):
#
#        if (k+N>len(R_final_c)):
#            print('oui')
#            break
#        else:
#            
##            1 ere possibilité'
#            
#            block_c=get_block(R_final_c,N,k)
#            S_c=block_c.cov()
#            mu_c=block_c.iloc[-1]
#            I.append(mu_c.name)
##            p=sortie.loc[mu.name]
##            P=pd.concat([P,p],axis=0)
##            
#            ef_c = EfficientFrontier(mu_c, S_c)
#            'long/short'
#            #ef = EfficientFrontier(mu, S, weight_bounds=(-1, 1))
#            raw_weights = ef_c.max_sharpe()
#            cleaned_weights = ef_c.clean_weights()
#            w_c=pd.DataFrame(cleaned_weights,index=[0])
#            
#            W_c=pd.concat([W_c,w_c],axis=0)
#            K_c=ef_c.portfolio_performance(verbose=True)
#            M_c.append(K_c)
            
#    I_c=pd.DataFrame(I,columns=['date'])
#    M_c=pd.DataFrame(M,columns=['return','vol','sr'])
#    M_c.index=I_c['date']
#    W_c.index=I_c['date']
#    P_c=W.mul(sortie.loc[I_c['date']],1)
    
            
    


#    plt.figure(2)
#    plt.plot(M['return'])
#    return(A,M,W,P,M_prime,W_prime,P_prime)
    return(A,M,W,P)
  


def generate_prix(returns):
    prices=100*(returns+1).cumprod()
    return(prices) 
    
def sharpe_ratio(Y,rf):
    R=Y
    r=(R.mean()-rf)/R.std()
    print('r',R.mean())
    print('std',R.std())
    return(r)
    
def maxDD(Y):
    max_Y=Y.cummax()
    i_creux=(max_Y-Y).idxmax()
    creux=Y[i_creux]
    i_sommet=Y[:i_creux].idxmax()
    sommet=Y[i_sommet]
    return([sommet-creux,i_creux,i_sommet,(sommet-creux)/sommet]) 
    
def affichage_maxDD(Vn,lab):
    i=maxDD(Vn)[1]
    j=maxDD(Vn)[2]
    xi=Vn.loc[i]
    xj=Vn.loc[j]

    plt.plot(Vn,label='long only')
    plt.plot([i,j],[xi,xj],color='Red')
    plt.title('Equity curve')
    plt.ylabel("P&L")
    plt.legend()

    
def drawdown_curve(Y):
    D=Y.cummax()-Y
#        plt.figure()
    plt.plot(D)
#        plt.title('Drawdown de equity curve')
    plt.ylabel('drawdown curve')
    plt.xlabel("Dates")
    return(D)
    





#data=selection_alea_gene(2)
#Pf=selec_general(1,0,3,5,5)
#S=prediction0(Pf,100,100,1)
#'Benchmark'
#
##get_return(u_i1).plot()
#
#    
##'affichage SR optimisation'
#u_i1=pd.read_csv('SPPX.DE.csv',index_col='Date',parse_dates=True)
#
#
#S_p=S[0]
#M=S[1]
#sig=M['vol']
###M_prime=S[4]
#
#u_i1=u_i1['Adj Close']
#rb1=get_return(u_i1).iloc[:len(M)]
#rb1.index=M.index
#rb1=rb1.dropna(axis=0)
#
#'position'
#df=S[2]
#C_df=[c for c in df.columns]
#
#
#
#'transaction'
#P=S[3].abs()
#Cost=(P.sum(axis=1))*100
#V=generate_prix(M['return'])
##V_prime=generate_prix(M_prime['return'])
#Vc=V-Cost
#Rc=get_return(Vc)

'rendement global'
#Rg=pd.concat([M['return'],Rc,rb1],axis=1)
#Rg.columns=['sans fees', 'avec fees','index' ]
#Rg=pd.concat([M['return'],Rc],axis=1)
#Rg.columns=['sans fees', 'avec fees']

#plt.plot(V,color='red')
#plt.plot(V_prime,color='blue')
###V.plot()
#plt.figure(1)
#affichage_maxDD(V,'sans fees')
#
#plt.figure(2)
#plt.plot(sig,label='sigma')
#plt.legend()
#
#plt.figure(3)
#plt.plot(M['return'],label='return')
#plt.legend()
##
#plt.figure(2)
#plt.plot(M['sr'])
#plt.ylabel('sharp ratio')
##
#plt.figure(3)
#pf.plotting.plot_rolling_returns(M['return'])
#
##
#plt.figure(5)
#pf.plot_drawdown_periods(M['return'])
#plt.figure(6)
#pf.plot_drawdown_underwater(M['return'])
#
#plt.figure(7)
#pf.plot_drawdown_periods(rb1)
#plt.figure(8)
#pf.plot_drawdown_underwater(rb1)
#
#
#
#plt.figure(9)
#pf.plotting.plot_annual_returns(M['return'])
#
#plt.figure(10)
#pf.plot_monthly_returns_dist(M['return'])
#plt.figure(11)
#pf.plot_monthly_returns_heatmap(M['return'])
#
#plt.figure(12)
#plt.plot(df[C_df[0]],marker='o')
#plt.ylabel('positions')

#

#

#C=[c for c in S_p.columns]
#k=2
#sortie=S_p[C[k]]
#verif=S_p[C[6+k]]
###SV=pd.concat([sortie,verif],axis=1)
##plt.figure(1)
#plt.plot(sortie,'--')
#plt.plot(verif)

#plt.figure(1)
#plt.plot(M['return'],marker='o')
#

#plt.figure(2)
#drawdown_curve(V)

#plt.figure(2)
#plt.plot(V)

#print('sr',sharpe_ratio(M['return'],0.02))
#plt.plot(M['sr'],marker='o')
#data=usy_n2
#data=join_data()
#'TRACE PF DE MARKOVITZ'
def rand_weights(n):
    ''' Produces n random weights that sum to 1 '''
    k = np.random.rand(n)
    return k / sum(k)  

#n_obs=100
#returns=get_return(data).sort_index(ascending=False).iloc[2000:4000]
#returns=returns.replace([np.inf, -np.inf], np.nan)
#returns=returns.dropna()
##returns=get_return(data).iloc[:n_obs]

def random_portfolio(returns):
#    p=[-0.1,0.1,-0.56,0.78]
#    p=np.asmatrix(p)
    p=np.asmatrix(returns.mean(axis=0))
    print(p)
    C=np.asmatrix(returns.cov())
    w = np.asmatrix(rand_weights(returns.shape[1]))
    mu = w * p.T
    sigma = np.sqrt(w * C * w.T)
    
    
    'frontiere efficient'
#    S=returns.cov()
#    mup=returns.mean(axis=0)
#    print(S)
#
#    ef = EfficientFrontier(mup, S)
##    raw_weights = ef.max_sharpe()
#    raw_weights = ef.max_sharpe()
#    cleaned_weights = ef.clean_weights()
#    K=ef.portfolio_performance(verbose=False)
#    r_fr=K[0]
#    vol_fr=K[1]
#    L_fr=(r_fr,vol_fr)
#    
    L_random=(mu,sigma)
    return(L_random)

    
#    return(L_random,L_fr)
    



'affichage des portefeuilles aleatoires'    
#n_portfolios = 10000
##means, stds = np.column_stack([random_portfolio(returns)[0] for i in range(n_portfolios)])
#means, stds = np.column_stack([random_portfolio(returns) for i in range(n_portfolios)])
#plt.plot(stds, means, 'o', markersize=5)
#plt.xlabel('std')
#plt.ylabel('mean')
#plt.title('Mean and standard deviation of returns of randomly generated portfolios')
#M=random_portfolio(returns)[1]
#print(M)
#plt.plot(M[1],M[0],'y-o')



#def random_portfolio(returns):
#    ''' 
#    Returns the mean and standard deviation of returns for a random portfolio
#    '''
#
#    p = np.asmatrix(np.mean(returns, axis=1))
#    w = np.asmatrix(rand_weights(returns.shape[0]))
#    C = np.asmatrix(np.cov(returns))
#    
#    mu = w * p.T
#    sigma = np.sqrt(w * C * w.T)
#    
#     This recursion reduces outliers to keep plots pretty
#    if sigma > 2:
#        return (random_portfolio(returns))
#    return (mu, sigma  )


#q=random_portfolio(r)


  
#Robj=pd.DataFrame(Robj)
#V=generate_prix(Robj)
#Sh=pd.DataFrame(Sh)
#V.plot()
#Robj.plot()
#Sh.plot()


#mu = expected_returns.mean_historical_return(D)
#Sc=Rd.cov()
#S = risk_models.sample_cov(D)
#'long only'
##ef = EfficientFrontier(mu, Sc)
#'long/short'
##ef = EfficientFrontier(mu, Sc, weight_bounds=(-1, 1))
#'regularisation'
##ef = EfficientFrontier(mu, Sc, gamma=1)
#'limite de poids'
##ef = EfficientFrontier(mu, Sc, weight_bounds=(0, 0.1))
#raw_weights = ef.max_sharpe()
#raw_weights2=ef.efficient_risk(0.1)
#cleaned_weights = ef.clean_weights()
#
##print(raw_weights2)
#print(cleaned_weights)
#ef.portfolio_performance(verbose=True)
#
#latest_prices = discrete_allocation.get_latest_prices(D)
#allocation, leftover = discrete_allocation.portfolio(
#    raw_weights, latest_prices, total_portfolio_value=10000
#)
#print(allocation)
#print("Funds remaining: ${:.2f}".format(leftover))

    
 
    
def prediction_return(r,p,epoch,window):
    #Preaparation des données
    Asset=selection_alea_gene(5) 
    Asset=Asset.sort_index()
    R=get_return(Asset)
    
    if r:
        #Une base d'entrainement
        data_train=R.iloc[:len(R)-window]
        #une base de test
        data_test=R.iloc[len(R)-window:]
        
    else:
        data_train=Asset.iloc[:len(R)-window]
        data_test=Asset.iloc[len(R)-window:]
    
    print('donnée prep')
    #Defintion de l'arbre
    d=len(data_train.columns)
    tf_features=tf.placeholder(tf.float32,shape=[None,d])
    tf_target= tf.placeholder(tf.float32,shape=[None,d])
    
    di=7
    dp=5
    
    w1=tf.Variable(tf.random_normal([d,dp ]))
    w2=tf.Variable(tf.random_normal([dp,d]))
    
#    w1=tf.Variable(tf.random_normal([d,di ]))
#    w2=tf.Variable(tf.random_normal([di,dp]))
#    w3=tf.Variable(tf.random_normal([dp,1]))
    
    
    b1=tf.Variable(tf.zeros(dp))
    b2=tf.Variable(tf.zeros(d))
#    b1=tf.Variable(tf.zeros(di))
#    b2=tf.Variable(tf.zeros(dp))
#    b3=tf.Variable(tf.zeros(1))
    
    
    z1=tf.matmul(tf_features,w1)+b1
    z1a=tf.nn.leaky_relu(z1,alpha=0.5,name=None)
    
    z2=tf.matmul(z1a,w2)+b2
    z2a=tf.nn.leaky_relu(z2,alpha=0.5,name=None)
    
#    z3=tf.matmul(z2a,w3)+b3
#    z3a=tf.nn.leaky_relu(z3,alpha=0.5,name=None)
    
    output=z2a
    
    'regularisation: contre overfitting'
    alpha=0.00001
    regularizer = tf.nn.l2_loss(w1) + tf.nn.l2_loss(w2)
    'definition de lerreur MSE : moindre carré'
    #erreur=tf.reduce_mean(tf.square(output-tf_features))
    erreur=0.5*tf.reduce_mean(tf.square(output-tf_target))+0.5*alpha*regularizer
    
    'minimisation de lerreur'
    global_step = tf.Variable(0, trainable=False)
    starter_learning_rate=0.1
    learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 100000, 0.96, staircase=True)
    #    learning_rate=0.0001
    optimizer = tf.train.AdamOptimizer(learning_rate)
    'opération dentrainement'
    train = optimizer.minimize(erreur)
    
    'parametre de batch'
#    batch_size = 100
#    n_samples=len(data_train)
    
    'definition session: contient le graphe defini précédemment + initialisation'
    sess=tf.Session()
    sess.run(tf.global_variables_initializer())
            
    'on renvoie lerreur'

    E_train=[]
    E_test=[]
    ax=[]
     
    print('debut')
    for e in range(epoch + 1):
        print(e)
        k=0
        while (k+p<len(data_train)):
            
            batch=get_block(data_train,p,k)
#            R_input=batch[0]
#            r_cible=pd.DataFrame(batch[1]).T
#            sess.run(train,feed_dict={tf_features:R_input,tf_target:r_cible})
#            er=sess.run(erreur,feed_dict={tf_features:R_input,tf_target:r_cible})

            k+=1
        print('e',e)

    sortie=sess.run(output,feed_dict={tf_features:data_test})
    sortie=pd.DataFrame(sortie,columns=data_test.columns,index=data_test.index)
    
    #affichage
#    C=[c for c in data_test.columns]
#    s_test=sortie[C[0]]
#    s=data_test[C[0]]
#    S=pd.concat([s,s_test],axis=1)
#    S.plot()

    return(sortie)
    
    
    


#S=prediction_return(0,5,50,50)
#
#Asset=join_data()
#Asset=Asset.sort_index()
#R=get_return(Asset)
#S_test=R.iloc[len(R)-100:]

#print(C[0])

#
#plt.plot(s_test)
#plt.plot(s)
    
#        total_batch = int(n_samples / batch_size)
#        for i in range(total_batch):
#            batch_xs = get_random_block_from_data(data_train, batch_size)
#            batch_y=batch_xs['index']
#            batch_size=len(batch_y)
#            batch_y = batch_y.values.reshape(batch_size, 1)
#            
#            batch_PF=batch_xs.copy()
#            del batch_PF['index']
#        
#            sess.run(train,feed_dict={tf_features:batch_PF,tf_target:batch_y})
#            
#            err=sess.run(erreur,feed_dict={tf_features:batch_PF,tf_target:batch_y})
#            avg_err += err / n_samples * batch_size
            
        #        s=sess.run(output,feed_dict={tf_features:batch_PF,tf_target:batch_y})
            
            
        #        print('erreur_train=',avg_err)
        #        print('erreur_test=',sess.run(erreur,feed_dict={tf_features:data_test2,tst:True,iter:e}))            
#        if (e%10==0):
#            ax.append(e)
#            E_train.append(avg_err)
#            E_test.append(sess.run(erreur,feed_dict={tf_features:data_PF_test,tf_target:y_index_test}))
#            err_t=sess.run(erreur,feed_dict={tf_features:data_PF_test,tf_target:y_index_test})
#            print(str(e) + ": erreur_train:" + str(avg_err) + " erreur_test: " + str(err_t) )
    
    
    
    
'Brouillon'
#Y=index_gene()
#y=Y.iloc[0]
#
#Y=pd.Series(Y)
#Y=Y.sort_index()
#R=get_return(Y)
#R=pd.DataFrame(R)
##R.hist(rwidth=0.5)
##plt.plot(Y)
#Rz=R.copy()
#Rz[Rz['index']<-0.02]=abs(Rz[Rz['index']<-0.02])
#Rz[Rz['index']<-0.02]=-0.02
#plt.plot(R)
#plt.plot(Rz)
#Yn=(1/(R+1)).cumprod()
#Y=pd.DataFrame(Y.iloc[1:])
#Yn=(R+1).cumprod()
#Yn.index=Y.index
#ratio=Y['index'].div(Yn['index'],axis=0)
#Rmul=ratio.mean(axis=0)
##plt.plot(Yn*R)
#Ym=(Rz+1).cumprod()
#Ym=Ym*Rmul
#plt.plot(Ym)    
    
    
        
    
'TEST POUR LA REGULARIZATION'    
    
def auto1_reg(data,d,dp,Tp,epoch):
    tst = tf.placeholder(tf.bool)
    iter = tf.placeholder(tf.int32)
    'split la data'
    data_train=split(data)[0]
    data_test=split(data)[1]

    'pour nimporte quel taille de data set [None,11] ici None = 2500'
    tf_features=tf.placeholder(tf.float32,shape=[None,d])
    tf_target= tf_features
    #print(tf_features)
    'poids'
    w1=tf.Variable(tf.random_normal([d,dp ]))
    w2=tf.Variable(tf.random_normal([dp,d ]))

    
    'biais'
    b1=tf.Variable(tf.zeros(dp))
    b2=tf.Variable(tf.zeros(d))

    
    'preactivation du neurone:multiplication des poids par les entrée: multiplication matricielle'
    '+activation'
    z1=tf.matmul(tf_features,w1)+b1
#    z1bn, update_ema1 = batchnorm(z1, tst, iter, b1)
#    z1a=tf.nn.leaky_relu(z1bn,alpha=0.5,name=None)
    z1a=tf.nn.leaky_relu(z1,alpha=0.5,name=None)
    
    z2=tf.matmul(z1a,w2)+b2
#    z2bn, update_ema1 = batchnorm(z2, tst, iter, b2)
#    z2a=tf.nn.leaky_relu(z2bn,alpha=0.5,name=None)
    z2a=tf.nn.leaky_relu(z2,alpha=0.5,name=None)
    
    

    

    
    output=z2a

    'regularisation: contre overfitting'

    alpha0=0
    alpha=0.00001
    alpha2=0.01
    alpha3=0.1
    alpha4=1
    
    regularizer = tf.nn.l2_loss(w1) + tf.nn.l2_loss(w2)
    'definition de lerreur MSE : moindre carré'
#    erreur=tf.reduce_mean(tf.square(output-tf_features))
    erreur0=0.5*tf.reduce_mean(tf.square(output-tf_features))+ 0.5*alpha0*regularizer
    erreur1=0.5*tf.reduce_mean(tf.square(output-tf_features))+ 0.5*alpha1*regularizer
    erreur2=0.5*tf.reduce_mean(tf.square(output-tf_features))+ 0.5*alpha2*regularizer
    erreur3=0.5*tf.reduce_mean(tf.square(output-tf_features))+ 0.5*alpha3*regularizer
    erreur4=0.5*tf.reduce_mean(tf.square(output-tf_features))+ 0.5*alpha4*regularizer
    'minimisation de lerreur'
    global_step = tf.Variable(0, trainable=False)
    starter_learning_rate=0.1
    learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 100000, 0.96, staircase=True)
#    learning_rate=0.0001
    optimizer = tf.train.AdamOptimizer(learning_rate)
    'opération dentrainement'
    train0 = optimizer.minimize(erreur0)
    train1 = optimizer.minimize(erreur0)
    train2 = optimizer.minimize(erreur0)
    train3 = optimizer.minimize(erreur0)
    train4 = optimizer.minimize(erreur0)   
    
    'parametre de batch'
    batch_size = 100
    n_samples=len(data_train)
    
    'definition session: contient le graphe defini précédemment + initialisation'
    sess=tf.Session()
    sess.run(tf.global_variables_initializer())


    'on renvoie lerreur'
#    print('erreur avant',sess.run(erreur,feed_dict={tf_features:usy_n/100}))
    E_train=[]
    E_test=[]
    ax=[]
 
    r=0

    for e in range(epoch + 1):
        avg_err = 0
        total_batch = int(n_samples / batch_size)
        for i in range(total_batch):
            batch_xs = get_random_block_from_data(data_train, batch_size)
            sess.run(train0,feed_dict={tf_features:batch_xs,tst:False,iter:e})
            sess.run(train1,feed_dict={tf_features:batch_xs,tst:False,iter:e})
            sess.run(train2,feed_dict={tf_features:batch_xs,tst:False,iter:e})
            sess.run(train3,feed_dict={tf_features:batch_xs,tst:False,iter:e})
            sess.run(train4,feed_dict={tf_features:batch_xs,tst:False,iter:e}) 
            
            err0=sess.run(erreur0,feed_dict={tf_features:batch_xs})
            avg_err0 += err0 / n_samples * batch_size
            
            err1=sess.run(erreur1,feed_dict={tf_features:batch_xs})
            avg_err1 += err1 / n_samples * batch_size 
            
            err2=sess.run(erreur2,feed_dict={tf_features:batch_xs})
            avg_err2 += err2 / n_samples * batch_size
            
            err3=sess.run(erreur3,feed_dict={tf_features:batch_xs})
            avg_err3 += err3 / n_samples * batch_size
            
            err4=sess.run(erreur4,feed_dict={tf_features:batch_xs})
            avg_err4 += err4 / n_samples * batch_size
            

#            print('erreur_train=',avg_err)
#            print('erreur_test=',sess.run(erreur,feed_dict={tf_features:data_test2,tst:True,iter:e}))            
        if (e%10==0):
            ax.append(e)
            E_train.append(avg_err)
            E_test.append(sess.run(erreur,feed_dict={tf_features:data_test,tst:True,iter:e}))
            err_t=sess.run(erreur,feed_dict={tf_features:data_test})
            print(str(e) + ": erreur_train:" + str(avg_err) + " erreur_test: " + str(err_t) )
#            print('erreur_train=',avg_e)
#            print('erreur_test=',sess.run(erreur,feed_dict={tf_features:data_test2}))
            
    
#            E_train.append(avg_e)
#            E_test.append(sess.run(erreur,feed_dict={tf_features:data_test2,tst:True,iter:e}))



#        r=e
#        print(r)

        
    erreur_f=sess.run(erreur,feed_dict={tf_features:data_test,tst:True,iter:r})




    'base de donnée reconstruite'
    app=pd.DataFrame(sess.run(output,feed_dict={tf_features:data_test,tst:False,iter:r}))
    appfixe=app.copy()
    E_ann=[]
    for i in range(len(appfixe)):
        a_ann=appfixe.iloc[i]
        b_ann=data_test.iloc[i]
        e_ann=mean_squared_error(a_ann,b_ann)
        E_ann.append(e_ann)
    E_ann=pd.DataFrame(E_ann,columns=['mse_ann'])
    appfixe_ann=pd.concat([appfixe,E_ann],axis=1)
        
        
        
    
    'fonction affichage'
    i=randint(0,len(data_test))
    print(i)
    
    'pca'
    l=PCA_final_c(i,data_test,3)
    y_pca=l[0]
    mse=l[1]
    mse=round(mse,6)
    n_comp=l[2]
    F=l[3]
    mse_pca_global=l[4]

    print('n_comp=',n_comp)
    
    'ann'
    yapp1=app.iloc[i]
    y1=data_test.iloc[i]
    name=y1.name

    mse_ann=mean_squared_error(y1,yapp1)
    mse_ann=round(mse_ann,6)


    'affichage dun courbe random'
    plt.figure(1)
    
    plt.plot(Tp,yapp1,marker='o', linestyle='--',color='red',label='autoencodeur')
    plt.plot(Tp,y1,marker='o', color= 'blue',label='courbe réelle')
    plt.plot(Tp,y_pca,marker='o',linestyle='--',color='green',label='PCA')
    plt.title('Random_autoencodeur, erreur:'+str(mse_ann) +'\n'+ 'PCA, erreur:'+str(mse))
    plt.xlabel('Maturités'+'\n'+str(name))
    plt.ylabel('Rendement à maturité')
    plt.legend()
    
    
    print('erreur ann_global=',erreur_f)
    print('erreur pca_global=',mse_pca_global)
    
    'pour indice random'
    print('erreur ann_local=',mse_ann)
    print('erreur pca_local=',mse)
    

    
    'erreur de pCA la plus grande'
    i_max=F['mse'].idxmax()
    yapp_max=app.iloc[i_max]
    y_max=data_test.iloc[i_max]
    mse_ann_max=mean_squared_error(y_max,yapp_max)
    F_pca=F.copy()
    del F_pca['mse']
    y_pca_max=F_pca.iloc[i_max]
    name=y_max.name
    plt.figure(2)
#    plt.subplot(212)
    plt.plot(Tp,yapp_max,marker='o', linestyle='--',color='red',label='autoencodeur')
    plt.plot(Tp,y_max,marker='o', color= 'blue',label='courbe réelle')
    plt.plot(Tp,y_pca_max,marker='o',linestyle='--',color='green',label='PCA')
    plt.title('MAX_PCA, autoencodeur, erreur:'+str(mse_ann_max) +'\n'+ 'PCA, erreur:'+str(F['mse'].max()))
    plt.xlabel('Maturités'+'\n'+str(name))
    plt.ylabel('Rendement à maturité')
    plt.legend()
    
    '1_erreur de PCA la plus faible'
    i_min=F['mse'].idxmin()
    yapp_min=app.iloc[i_min]
    y_min=data_test.iloc[i_min]
    mse_ann_min=mean_squared_error(y_min,yapp_min)
    y_pca_min=F_pca.iloc[i_min]
    name=y_min.name
    plt.figure(3)
#    plt.subplot(212)
    plt.plot(Tp,yapp_min,marker='o', linestyle='--',color='red',label='autoencodeur')
    plt.plot(Tp,y_min,marker='o', color= 'blue',label='courbe réelle')
    plt.plot(Tp,y_pca_min,marker='o',linestyle='--',color='green',label='PCA')
    plt.title('MIN_PCA, autoencodeur, erreur:'+str(mse_ann_min) +'\n'+ 'PCA, erreur:'+str(F['mse'].min()))
    plt.xlabel('Maturités'+'\n'+str(name))
    plt.ylabel('Rendement à maturité')
    plt.legend()   
    
    '2_erreur ann la plus faible'
    i_min_ann=appfixe_ann['mse_ann'].idxmin()
    yapp_min2=app.iloc[i_min_ann]
    y_min2=data_test.iloc[i_min_ann]
    y_pca_min2=F_pca.iloc[i_min_ann]
    mse_ann_min2=mean_squared_error(y_min2,yapp_min2)
    mse_pca_min2=mean_squared_error(y_min2,y_pca_min2)
    name2=y_min2.name
    
    plt.figure(4)
    plt.plot(Tp,yapp_min2,marker='o', linestyle='--',color='red',label='autoencodeur')
    plt.plot(Tp,y_min2,marker='o', color= 'blue',label='courbe réelle')
    plt.plot(Tp,y_pca_min2,marker='o',linestyle='--',color='green',label='PCA')
    plt.title('MIN_ANN, autoencodeur, erreur:'+str(mse_ann_min2) +'\n'+ 'PCA, erreur:'+str(mse_pca_min2))
    plt.xlabel('Maturités'+'\n'+str(name2))
    plt.ylabel('Rendement à maturité')
    plt.legend()   
    
    
    
    'affichage de la courbe derreur'
    plt.figure(5)
#    X=np.array([i for i in range(0,r,100)])
    plt.plot(ax,E_train, color='blue', marker='x')
    plt.plot(ax,E_test,color='red',marker='x')
#    plt.plot(e_final)
    plt.xlabel('nombre sessions dentrainement')
    plt.ylabel('erreur')
    
#auto1_reg(usy_n2n,9,5,T2,10)
  

'sans batch'
def auto1(): 
    tst = tf.placeholder(tf.bool)
    # training iteration
    iter = tf.placeholder(tf.int32)
    

    'pour nimporte quel taille de data set [None,11] ici None = 2500'
    tf_features=tf.placeholder(tf.float32,shape=[None,11])
    tf_target= tf_features
    #print(tf_features)
    'poids'
    w1=tf.Variable(tf.random_normal([11,7 ]))
    w2=tf.Variable(tf.random_normal([7,5 ]))
    w3=tf.Variable(tf.random_normal([5,3]))
    w4=tf.Variable(tf.random_normal([3,5 ]))
    w5=tf.Variable(tf.random_normal([5,7]))
    w6=tf.Variable(tf.random_normal([7,11 ]))
    
    
    'biais'
    b1=tf.Variable(tf.zeros(7))
    b2=tf.Variable(tf.zeros(5))
    b3=tf.Variable(tf.zeros(3))
    b4=tf.Variable(tf.zeros(5))
    b5=tf.Variable(tf.zeros(7))
    b6=tf.Variable(tf.zeros(11))
    
    'preactivation du neurone:multiplication des poids par les entrée: multiplication matricielle'
    '+activation'
    z1=tf.matmul(tf_features,w1)+b1
    z1bn, update_ema1 = batchnorm(z1, tst, iter, b1)
    z1a=tf.nn.leaky_relu(z1bn,alpha=0.001,name=None)
#    z1a=tf.nn.elu(z1,name=None)
    
    z2=tf.matmul(z1a,w2)+b2
    z2bn, update_ema2 = batchnorm(z2, tst, iter, b2)
    z2a=tf.nn.leaky_relu(z2bn,alpha=0.001,name=None)
#    z2a=tf.nn.elu(z2,name=None)
    
    z3=tf.matmul(z2a,w3)+b3
#    z3bn, update_ema3 = batchnorm(z3, tst, iter, b3)
    z3a=tf.nn.sigmoid(z3)
#    z3a=tf.nn.leaky_relu(z3,alpha=0.001,name=None)
#    z3a=tf.nn.elu(z3,name=None)
    
    z4=tf.matmul(z3a,w4)+b4
    z4bn, update_ema1 = batchnorm(z4, tst, iter, b4)
    z4a=tf.nn.leaky_relu(z4bn,alpha=0.001,name=None)
#    z4a=tf.nn.elu(z4,name=None)
    z5=tf.matmul(z4a,w5)+b5
    z5bn, update_ema5 = batchnorm(z5, tst, iter, b5)
    z5a=tf.nn.leaky_relu(z5bn,alpha=0.001,name=None)

    z6=tf.matmul(z5a,w6)+b6
    z6bn, update_ema1 = batchnorm(z6, tst, iter, b6)
    z6a=tf.nn.leaky_relu(z6bn,alpha=0.001,name=None) 
    
    output=z6a



    'regularisation: contre overfitting'
    alpha=0.0001
    regularizer = tf.nn.l2_loss(w1) + tf.nn.l2_loss(w2)+tf.nn.l2_loss(w3) + tf.nn.l2_loss(w4)
    
    
    'regulatisation: sparse hidden layers'
    beta=3
    rho=0.05
    H=z3a
    rho_hat=tf.reduce_mean(H,axis=0)
    kl=kl_divergence(rho, rho_hat)

    
    
    'definition de lerreur MSE : moindre carré avec les regularisation'
#    erreur=tf.reduce_mean(tf.square(output-tf_features))
#    erreur=0.5*tf.reduce_mean(tf.square(output-tf_features))+ 0.5*alpha*regularizer
#    erreur=tf.reduce_mean(tf.square(output-tf_features))+beta*tf.reduce_sum(kl)
    erreur=0.5*tf.reduce_mean(tf.square(output-tf_features))+ 0.5*alpha*regularizer + beta*tf.reduce_sum(kl)
    
    
    'minimisation de lerreur'
    global_step = tf.Variable(0, trainable=False)
    starter_learning_rate=0.1
    learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 100000, 0.96, staircase=True)
#    learning_rate=0.0001
    optimizer = tf.train.AdamOptimizer(learning_rate)
    'opération dentrainement'
    train = optimizer.minimize(erreur)
    
    'saveur'
    saver = tf.train.Saver()
    save_dir = 'checkpoints/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    save_path = os.path.join(save_dir, 'best_validation')
    
    
    'definition session: contient le graphe defini précédemment + initialisation'
    sess=tf.Session()
    'initialisation'
    sess.run(tf.global_variables_initializer())


    'on renvoie lerreur'
#    print('erreur avant',sess.run(erreur,feed_dict={tf_features:usy_n/100}))
    E_train=[] 
    E_test=[]
    r=0

    s=randint(2,30)
    data_train1.sample(n=2250,random_state=s)
    for e in range(5000):
        
        'melange de la base de donnée'
        sess.run(train,feed_dict={tf_features:data_train1,tst:False,iter:e})
        
        
#        print('H=',sess.run(H,feed_dict={tf_features:data_train1}))
#        print('rhoH=',sess.run(rho_hat,feed_dict={tf_features:data_train1}))
#        print('kl=',sess.run(kl,feed_dict={tf_features:data_train1}))
        print('erreur train=',sess.run(erreur,feed_dict={tf_features:data_train1,tst:False,iter:e}))
        print('erreur test=',sess.run(erreur,feed_dict={tf_features:data_test1,tst:True,iter:e}))
        if e%30==0 and e>100:
            E_train.append(sess.run(erreur,feed_dict={tf_features:data_train1,tst:False,iter:e}))
            E_test.append(sess.run(erreur,feed_dict={tf_features:data_test1,tst:True,iter:e}))
        
        r+=1
        print(r)




#    e_final=pd.DataFrame({'e_train':E_train,'e_test':E_test})

#    e_final=e_final.iloc[200:]
    erreur_f=sess.run(erreur,feed_dict={tf_features:data_test,tst:True,iter:r})
    
    app=pd.DataFrame(sess.run(output,feed_dict={tf_features:data_test,tst:True,iter:r}))
    appfixe=app.copy()
    
    'fonction affichage'
    i=randint(0,200)
    print(i)
    
    'pca'
    l=PCA_final_c(i,data_test,2)
    y_pca=l[0]
    mse=l[1]
    n_comp=l[2]
    print('n_comp=',n_comp)
    
    yapp1=app.iloc[i]
    y1=data_test1.iloc[i]
    
    plt.figure(1)

    plt.plot(T,yapp1,marker='o', linestyle='--',color='red',label='autoencodeur')
    plt.plot(T,y1,marker='o', color= 'blue',label='courbe réelle')
    plt.plot(T,y_pca,marker='o',linestyle='--',color='green',label='PCA')
    plt.legend()
    print('erreur ann=',erreur_f)
    print('erreur pca=',mse)
    
#    plt.legend(loc='lower right', frameon=True)
    
    
 

    plt.figure(2)
    plt.plot(E_train, color='blue', marker='x')
    plt.plot(E_test,color='red',marker='x')
#    plt.plot(e_final)
    plt.xlabel('nombre sessions dentrainement')
    plt.ylabel('erreur')



def auto1_test(): 
    

    'pour nimporte quel taille de data set [None,11] ici None = 2500'
    tf_features=tf.placeholder(tf.float32,shape=[None,11])
    tf_target= tf_features
    #print(tf_features)
    'poids'
    w1=tf.Variable(tf.random_normal([11,7 ]))
    w2=tf.Variable(tf.random_normal([7,3 ]))
    w3=tf.Variable(tf.random_normal([3,7]))
    w4=tf.Variable(tf.random_normal([7,11 ]))
    
    'biais'
    b1=tf.Variable(tf.zeros(7))
    b2=tf.Variable(tf.zeros(3))
    b3=tf.Variable(tf.zeros(7))
    b4=tf.Variable(tf.zeros(11))
    
    'preactivation du neurone:multiplication des poids par les entrée: multiplication matricielle'
    '+activation'
    z1=tf.matmul(tf_features,w1)+b1
    z1a=tf.nn.leaky_relu(z1,alpha=0.001,name=None)
    'dropout'
#    pkeep=0.75
#    z1a=tf.nn.dropout(z1a,pkeep)
    

    
    z2=tf.matmul(z1a,w2)+b2
    z2a=tf.nn.leaky_relu(z2,alpha=0.001,name=None)
    'dropout'
#    pkeep=0.75
#    z2a=tf.nn.dropout(z2a,pkeep)

    
    z3=tf.matmul(z2a,w3)+b3
    z3a=tf.nn.leaky_relu(z3,alpha=0.001,name=None)
    'dropout'
#    pkeep=0.75
#    z3a=tf.nn.dropout(z3a,pkeep)

    
    z4=tf.matmul(z3a,w4)+b4
    z4a=tf.nn.leaky_relu(z4,alpha=0.001,name=None)
    'dropout'
#    pkeep=0.75
#    z4a=tf.nn.dropout(z4a,pkeep)

    
    output=z4a


    
    'regularisation: contre overfitting'
    alpha=0.00001
    regularizer = tf.nn.l2_loss(w1) + tf.nn.l2_loss(w2)+tf.nn.l2_loss(w3) + tf.nn.l2_loss(w4)
    
    
    'regulatisation: sparse hidden layers'
    beta=3
    rho=0.01
    H=z2a
    rho_hat=tf.reduce_mean(H,axis=0)
    kl=kl_divergence(rho, rho_hat)
    
    
    'definition de lerreur MSE : moindre carré'
#    erreur=tf.reduce_mean(tf.square(output-tf_features))
    erreur=0.5*tf.reduce_mean(tf.square(output-tf_features))+ 0.5*alpha*regularizer
#    erreur=tf.reduce_mean(tf.square(output-tf_features))+beta*tf.reduce_sum(kl)
#    erreur=0.5*tf.reduce_mean(tf.square(output-tf_features))+ 0.5*alpha*regularizer + beta*tf.reduce_sum(kl)
    
    'minimisation de lerreur'
    global_step = tf.Variable(0, trainable=False)
    starter_learning_rate=0.1
    learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 100000, 0.96, staircase=True)
#    learning_rate=0.0001
    optimizer = tf.train.AdamOptimizer(learning_rate)
    'opération dentrainement'
    train = optimizer.minimize(erreur)
    
       
    'definition session: contient le graphe defini précédemment + initialisation'
    sess=tf.Session()
    sess.run(tf.global_variables_initializer())


    'on renvoie lerreur'
#    print('erreur avant',sess.run(erreur,feed_dict={tf_features:usy_n/100}))
    E_train=[] 
    E_test=[]
    r=0

    s=randint(2,30)
    data_train1.sample(n=2250,random_state=s)
    
    
    
    for e in range(5):
        
        data_train1.sample(n=2250,random_state=s)
        
        for b in range(0,len(data_train1),50):
#        ia=random.randint(0,2200)
#        print('ia=',ia)
#        batch=data_train1[ia:ia +50]
            batch=data_train1[b:b +50]
        
            'melange de la base de donnée'
            sess.run(train,feed_dict={tf_features:batch})
            
        if (e%1==0):
            
            print('erreur train=',sess.run(erreur,feed_dict={tf_features:data_train1}))
            print('erreur test=',sess.run(erreur,feed_dict={tf_features:data_test1}))
            E_train.append(sess.run(erreur,feed_dict={tf_features:data_train1}))
            E_test.append(sess.run(erreur,feed_dict={tf_features:data_test1}))
            r+=1
            print(r)
            
            
        



  
    app=pd.DataFrame(sess.run(output,feed_dict={tf_features:data_test1}))
    appfixe=app.copy()
    
    'fonction affichage'
    i=randint(0,1)
    print(i)
    
    yapp1=app.iloc[i]
    y1=data_test1.iloc[i]
    
    plt.figure(1)
#    plt.subplot(211)
    plt.plot(T,yapp1,marker='o', linestyle='--',color='red')
    plt.plot(T,y1,marker='o', color= 'blue')
    
    plt.figure(2)
    plt.plot(E_train)
    plt.xlabel('nombre sessions dentrainement')
    plt.ylabel('erreur')
    



'1 seule couche'
def auto1prime(): 
    

    'pour nimporte quel taille de data set [None,11] ici None = 2500'
    tf_features=tf.placeholder(tf.float32,shape=[None,11])
    tf_target= tf_features
    #print(tf_features)
    'poids'
    w1=tf.Variable(tf.random_normal([11,5 ]))
    w2=tf.Variable(tf.random_normal([5,11 ]))

    
    'biais'
    b1=tf.Variable(tf.zeros(5))
    b2=tf.Variable(tf.zeros(11))

    
    'preactivation du neurone:multiplication des poids par les entrée: multiplication matricielle'
    '+activation'
    z1=tf.matmul(tf_features,w1)+b1
#    z1bn, update_ema1 = batchnorm(z1, tst, iter, b1)
#    z1a=tf.nn.leaky_relu(z1bn,alpha=0.5,name=None)
    z1a=tf.nn.leaky_relu(z1,alpha=0.5,name=None)
    
    z2=tf.matmul(z1a,w2)+b2
#    z2bn, update_ema1 = batchnorm(z2, tst, iter, b2)
#    z2a=tf.nn.leaky_relu(z2bn,alpha=0.5,name=None)
    z2a=tf.nn.leaky_relu(z2,alpha=0.5,name=None)
    
    

    

    
    output=z2a

    'regularisation: contre overfitting'
    alpha1=0.00001
    alpha2=0.1
    regularizer = tf.nn.l2_loss(w1) + tf.nn.l2_loss(w2)


    'definition de lerreur MSE : moindre carré'
#    erreur=tf.reduce_mean(tf.square(output-tf_features))
    erreur=0.5*tf.reduce_mean(tf.square(output-tf_features))+ 0.5*alpha1*regularizer
    'minimisation de lerreur'
    global_step = tf.Variable(0, trainable=False)
    starter_learning_rate=0.1
    learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 100000, 0.96, staircase=True)
#    learning_rate=0.0001
    optimizer = tf.train.AdamOptimizer(learning_rate)
    'opération dentrainement'
    train = optimizer.minimize(erreur)
    
    'definition de laccuracy: pourcentage de bonne réponse'
    correct_prediction=tf.equal(tf.round(output),tf_target)
    accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    
    
    'definition session: contient le graphe defini précédemment + initialisation'
    sess=tf.Session()
    sess.run(tf.global_variables_initializer())


    'on renvoie lerreur'
#    print('erreur avant',sess.run(erreur,feed_dict={tf_features:usy_n/100}))
    E_train=[]
    E_test=[]
 
    r=0

    s=randint(2,30)
    data_train1.sample(n=2250,random_state=s)
    for e in range(3000):
        'melange de la base de donnée'
        sess.run(train,feed_dict={tf_features:data_train1,tst:False,iter:e})
#        sess.run(train,feed_dict={tf_features:data_train1})
        print('erreur train',sess.run(erreur,feed_dict={tf_features:data_train1,tst:False,iter:e}))
        print('erreur test',sess.run(erreur,feed_dict={tf_features:data_test1,tst:False,iter:e}))
        if e%100==0 and e>100:
           E_train.append(sess.run(erreur,feed_dict={tf_features:data_train1,tst:False,iter:e}))
           E_test.append(sess.run(erreur,feed_dict={tf_features:data_test1,tst:True,iter:e}))



        r+=1
        print(r)
        
    erreur_f=sess.run(erreur,feed_dict={tf_features:data_test1,tst:True,iter:r})



  
    app=pd.DataFrame(sess.run(output,feed_dict={tf_features:data_test1,tst:False,iter:r}))
    appfixe=app.copy()
    
    'fonction affichage'
    i=randint(0,200)
    print(i)
    
    'pca'
    l=PCA_final_c(i,data_test1,3)
    y_pca=l[0]
    mse=l[1]
    n_comp=l[2]
    print('n_comp=',n_comp)
    
    yapp1=app.iloc[i]
    y1=data_test1.iloc[i]
    
    plt.figure(1)

    plt.plot(T,yapp1,marker='o', linestyle='--',color='red',label='autoencodeur')
    plt.plot(T,y1,marker='o', color= 'blue',label='courbe réelle')
    plt.plot(T,y_pca,marker='o',linestyle='--',color='green',label='PCA')
    plt.legend()
    print('erreur ann=',erreur_f)
    print('erreur pca=',mse)
    
    plt.figure(2)
#    X=np.array([i for i in range(0,r,100)])
    plt.plot(E_train, color='blue', marker='x')
    plt.plot(E_test,color='red',marker='x')
#    plt.plot(e_final)
    plt.xlabel('nombre sessions dentrainement')
    plt.ylabel('erreur')
    
    

    
'avec batch'
def auto2(): 

    'pour nimporte quel taille de data set [None,11] ici None = 2500'
    tf_features=tf.placeholder(tf.float32,shape=[None,11])
    tf_target= tf_features
    #print(tf_features)
    'poids'
    w1=tf.Variable(tf.random_normal([11,7 ]))
    w2=tf.Variable(tf.random_normal([7,5 ]))
    w3=tf.Variable(tf.random_normal([5,7]))
    w4=tf.Variable(tf.random_normal([7,11 ]))
    
    'biais'
    b1=tf.Variable(tf.zeros(7))
    b2=tf.Variable(tf.zeros(5))
    b3=tf.Variable(tf.zeros(7))
    b4=tf.Variable(tf.zeros(11))
    
    'preactivation du neurone:multiplication des poids par les entrée: multiplication matricielle'
    '+activation'
    z1=tf.matmul(tf_features,w1)+b1
    z1a=tf.nn.tanh(z1)
    
    z2=tf.matmul(z1a,w2)+b2
    z2a=tf.nn.tanh(z2)
    
    z3=tf.matmul(z2a,w3)+b3
    z3a=tf.nn.tanh(z3)
    
    z4=tf.matmul(z3a,w4)+b4
    z4a=tf.nn.sigmoid(z4)
    
    output=z4a



    'definition de lerreur MSE : moindre carré'
    erreur=tf.reduce_mean(tf.square(output-tf_features))
    
    'minimisation de lerreur'
    #global_step = tf.Variable(0, trainable=False)
    #starter_learning_rate=0.1
    #learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 100000, 0.96, staircase=True)
    learning_rate=0.0001
    optimizer = tf.train.AdamOptimizer(learning_rate)
    'opération dentrainement'
    train = optimizer.minimize(erreur)
    
    'definition de laccuracy: pourcentage de bonne réponse'
    correct_prediction=tf.equal(tf.round(output),tf_target)
    accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    
    'definition session: contient le graphe defini précédemment + initialisation'
    sess=tf.Session()
    sess.run(tf.global_variables_initializer())

    'on renvoie lerreur'
    
    E=[]
    A=[]
    r=0
    b_s=50
    s=randint(2,30)
    data_train1.sample(n=2250,random_state=s)
    for e in range(5000):
        'melange de la base de donnée'

        
        
        for b in range(0,len(usy_n),b_s):
            batch=data_train1[b:b+b_s]
            sess.run(train,feed_dict={tf_features:batch})
            print('erreur apres',sess.run(erreur,feed_dict={tf_features:batch}))
            #print('accuracy',sess.run(accuracy,feed_dict={tf_features:usy_n}))
            E.append(sess.run(erreur,feed_dict={tf_features:batch}))
            A.append(sess.run(accuracy,feed_dict={tf_features:batch}))
            r+=1
            print(r)



  
    app=pd.DataFrame(sess.run(output,feed_dict={tf_features:data_test1}))
    appfixe=app.copy()
    y=data_test1.iloc[0]
    yapp=app.iloc[0]
    plt.plot(T,yapp,marker='o', linestyle='--',color='red')
    plt.plot(T,y,marker='o', color= 'blue')
    
'affichage: courbe erreur'
def erreur_aff():
    plt.plot(E)
    
    plt.xlabel('nombre sessions dentrainement')
    plt.ylabel('erreur')
    plt.legend('r=0,0001')

def acc_aff():
    plt.plot(A)        


          





#usy_n2.reset_index(drop=True, inplace=True)
#usy_n2.reindex(np.random.permutation(usy_n2.index))

#u_sh=usy_n2.copy()
#u_sh.apply(np.random.shuffle(u_sh.values),axis=1)
ur=usy_n2.sample(n=6300,random_state=90)
#print(df.shape[0])













y2=usy_n2.iloc[0]
    

u=pd.read_csv('USy.csv',parse_dates=True)
u=u.iloc[:6300]
    
    
#

#'split la base de train en deux'
#data_train2_split=train_test_split(data_train2,test_size=0.5, random_state=42)
#d1=data_train2_split[0]
#d2=d1=data_train2_split[1]
#
#data_test2=X_train2[1] 




'melanger la base'
#np.random.shuffle(index)
#i=pd.DataFrame(data=index,columns=['indice'])
#i=pd.concat([i,u['Date']],axis=1)
#i=i.set_index('Date')
#new=pd.concat([train,i['indice']],axis=1)
# pcf=pcf.set_index('Date')


'1 couche, elargissement base de donnée'

def auto3():
    tf_features=tf.placeholder(tf.float32,shape=[None,9])
    tf_targets=tf_features
    
    #
    w1=tf.Variable(tf.random_normal([9,5]))
    w2=tf.Variable(tf.random_normal([5,9]))
    
    b1=tf.Variable(tf.zeros(5))
    b2=tf.Variable(tf.zeros(9))
    
    z1=tf.matmul(tf_features,w1)+b1
    z1a=tf.nn.tanh(z1)
    
    z2=tf.matmul(z1a,w2)+b2
    z2a=tf.nn.relu(z2)
    
    output=z2a
    'loss function: MSE'
    erreur=tf.reduce_mean(tf.square(output-tf_features))
    
    global_step = tf.Variable(0, trainable=False)
    starter_learning_rate=0.1
    learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 100000, 0.96, staircase=True)
 
    optimizer = tf.train.AdamOptimizer(learning_rate)
    'opération dentrainement'
    train = optimizer.minimize(erreur)
    
    
    
   
    
    r=0
    
    sess=tf.Session()
    sess.run(tf.global_variables_initializer())
    
    for e in range(5000):
        d1.sample(n=2835,random_state=2)
        d2.sample(n=2835,random_state=2)
        a = random.randint(0,1)
        if a==0:
            sess.run(train,feed_dict={tf_features:d1})
        if a==1:
            sess.run(train,feed_dict={tf_features:d2})
            
            

        
        
        
        print('erreur=',sess.run(erreur,feed_dict={tf_features:d2}))
        r+=1
        print(r)
        
        
    app2=pd.DataFrame(sess.run(output,feed_dict={tf_features:data_test2}))
    appfixe2=app2.copy()
    y=data_test2.iloc[0]
    yapp2=app2.iloc[0]
    plt.plot(T2,yapp2,marker='o', linestyle='--',color='red')
    plt.plot(T2,y2,marker='o', color= 'blue')

'2 couche, elargissement base de donnée'


def auto4():
    tf_features=tf.placeholder(tf.float32,shape=[None,9])
    tf_targets=tf_features
    
    'poids'
    w1=tf.Variable(tf.random_normal([9,7 ]))
    w2=tf.Variable(tf.random_normal([7,5 ]))
    w3=tf.Variable(tf.random_normal([5,7]))
    w4=tf.Variable(tf.random_normal([7,9 ]))
    
    'biais'
    b1=tf.Variable(tf.zeros(7))
    b2=tf.Variable(tf.zeros(5))
    b3=tf.Variable(tf.zeros(7))
    b4=tf.Variable(tf.zeros(9))
    
    'preactivation du neurone:multiplication des poids par les entrée: multiplication matricielle'
    '+activation'
    z1=tf.matmul(tf_features,w1)+b1
    z1a=tf.nn.tanh(z1)
    
    z2=tf.matmul(z1a,w2)+b2
    z2a=tf.nn.tanh(z2)
    
    z3=tf.matmul(z2a,w3)+b3
    z3a=tf.nn.tanh(z3)
    
    z4=tf.matmul(z3a,w4)+b4
    z4a=tf.nn.relu(z4)
    
    output=z4a
    


    'loss function: MSE'
    erreur=tf.reduce_mean(tf.square(output-tf_features))
    learning_rate=0.0001
    optimizer = tf.train.AdamOptimizer(learning_rate)
    'opération dentrainement'
    train = optimizer.minimize(erreur)
    
    
    
   
    
    r=0
    
    sess=tf.Session()
    sess.run(tf.global_variables_initializer())
    
    for e in range(5000):

        sess.run(train,feed_dict={tf_features:d1})
        sess.run(train,feed_dict={tf_features:d2})
        
        print(sess.run(erreur,feed_dict={tf_features:data_test2}))
        r+=1
        print(r)
        
        
    app2=pd.DataFrame(sess.run(output,feed_dict={tf_features:data_test2}))
    appfixe2=app2.copy()
    y=data_test2.iloc[0]
    yapp2=app2.iloc[0]
    plt.plot(T2,yapp2,marker='o', linestyle='--',color='red')
    plt.plot(T2,y2,marker='o', color= 'blue')            
    

  




    










print('ok')











