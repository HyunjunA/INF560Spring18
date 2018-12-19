'''
Created on Mar 20, 2018
   
@author: Hyun Jun Choi
   
@attention: Make Model saving part 
resampling is processed about each training data set.
'''

import copy
import time
import csv
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import random

import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn import datasets

from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import Imputer


from sklearn.decomposition import PCA

from collections import Counter
  
def find_type(a):
   try:
       var_type = type(int(a))
   except ValueError:
       try:
           var_type = type(float(a))
       except ValueError:
           var_type = type(a)
   return var_type 
    
# boston = load_boston()
    

start = time.time()

my_csv = pd.read_csv('complete_data_3 months.csv')
    
    
csvfile= open('complete_data_3 months.csv','r')
total=csv.reader(csvfile,delimiter=',')
wantedIndex=-1
colNames=[]
i=0
for row in total:
    i+=1
    if i==1:
        colNames=row
        break
        
    

   
# You can change this data file name.   
data=pd.read_csv('complete_data_3 months.csv',names=None)
originData=data.ix[:]
# data2=pd.read_csv('complete_data_3 months.csv',names=None)
# imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
data=pd.DataFrame(data.replace(np.nan,np.nan))
data=pd.DataFrame(data.replace('Unknown',np.nan))
data=pd.DataFrame(data.replace(np.nan,-1000))



values=data.values


df = pd.read_excel('STAR_FILE_DOC.xlsx',skiprows=1)
variableKinds=df[df.columns[0]]

listVariableKinds=list(variableKinds)

variableForm=df[df.columns[2]]

listVariableForm=list(variableForm)
# consideredForm=['CALCULATED','DDR','DDR/LDR','DDR/LDR-CALCULATED','LDR','RH','TCR','TCR (PRE 10/25/99 VERSION)','TCR-CALCULATED','TRF','TRF/TRR','TRR/TRF-CALCULATED','TRR>TCR','TRR-CALCULATED','WAITING LIST DATA','WL DATA','']
consideredForm=['CALCULATED','DDR','DDR/LDR','DDR/LDR-CALCULATED','LDR','RH']
variableType=df[df.columns[4]]

setDataTemp=[]





deleteFeatureSet=[]



#The number of Features in liver transplant dataset 
exceptDelCol=copy.copy(colNames)


starConsideredRowNum=[]
for index in range(len(listVariableForm)):
    
    if (listVariableForm[index] in consideredForm):
        starConsideredRowNum.append(index)

beforeSurgeryFeatures=[]

for indS in starConsideredRowNum:
    beforeSurgeryFeatures.append(listVariableKinds[indS])

removeRate=0.5

postSurgeryFeatures=set(listVariableKinds).difference(set(beforeSurgeryFeatures))

#Model target variable candidates
ourRealPostSurgeryFeature=set(colNames).intersection(postSurgeryFeatures)
lisOourRealPostSurgeryFeature=list(ourRealPostSurgeryFeature)
#Model Input data
ourRealBeforeSurgeryFeature=set(colNames).intersection(set(beforeSurgeryFeatures))
listOurRealBeforeSurgeryFeature=list(ourRealBeforeSurgeryFeature)

removeInd=listOurRealBeforeSurgeryFeature.index('TX_YEAR')
listOurRealBeforeSurgeryFeature.pop(removeInd)
removeInd=listOurRealBeforeSurgeryFeature.index('PTIME')
listOurRealBeforeSurgeryFeature.pop(removeInd)

lisOourRealPostSurgeryFeature.append('PTIME')

# removeInd=listOurRealBeforeSurgeryFeature.index('GTIME')
# listOurRealBeforeSurgeryFeature.pop(removeInd)

lisOourRealPostSurgeryFeature.append('GTIME')

# lisOourRealPostSurgeryFeature.append('G90')



colNames.pop(0)
remo=colNames.index('G90')
colNames.pop(remo)
#Impute data
for index in range(len(colNames)):
    location=listVariableKinds.index(colNames[index])  
#     if index==158:
#         print('1')
    
    if (variableType[location]=='CHAR'):
        

        if index==53:
            print(1)
        setDataTemp=list(set(data[colNames[index]]))
        
        chS=0
        for indS in setDataTemp:
            if indS==-1000:
                chS=1
                break
                
        if chS==1:
        
            keyVa=Counter(data[colNames[index]])
            keyVa=list(keyVa.most_common())
            
            miSeven=0
            
            
            for indeKIn in range(len(keyVa)):  
                miSeven+=keyVa[indeKIn][1]
                if keyVa[indeKIn][0]==-1000:
                    sevenIndex=indeKIn
                
                    
            if (keyVa[sevenIndex][1]/miSeven)>removeRate:
                deleteFeatureSet.append( colNames[index])
                
                tempIndex=exceptDelCol.index(colNames[index])
#                 exceptDelCol[tempIndex]='REMOVED'
                del data[colNames[index]]
                continue
            
            
            
            
            
            for indeK in range(len(keyVa)):
                
                
                    
                if keyVa[indeK][0]!=-1000:
#                     print(keyVa)
                    
                    # Imputee nan==-1000 frequent values
                    data[colNames[index]]=pd.Series(data[colNames[index]].replace(-1000,keyVa[indeK][0]))
                    data[colNames[index]]=pd.Series(data[colNames[index]].replace(np.nan,keyVa[indeK][0]))
                    
#                     if index==158:
#                         print(colNames[index])
                    
#                     print(data[colNames[index]])
                    
                    temp=array(data[colNames[index]])
                    
                    label_encoder = LabelEncoder()
                    integer_encoded = label_encoder.fit_transform(temp)
                    data[colNames[index]]=pd.Series(integer_encoded)
                    break
                
    #             data[colNames[index]].fillna(data[colNames[index]].mean(), inplace=True)
    #             print('nan')
     
        if chS==0:
            temp=array(data[colNames[index]])
            temp=array(list(map(str,temp)))
#             print(temp)      
            label_encoder = LabelEncoder()
            integer_encoded = label_encoder.fit_transform(temp)
            data[colNames[index]]=pd.Series(integer_encoded)
            
           
        
        
        
    
    if variableType[location]=='NUM':
#         data[colNames[index]].pop(0)
        
#         data[colNames[index]].replace(np.nan,np.nan)
#         data[colNames[index]].replace('Unknown',-1000)
#         data[colNames[index]].replace(np.nan,-1000)
        
        
        setDataTemp=list(set(data[colNames[index]]))
        T3=map(int,setDataTemp)
        
        chS=0
        for indS in setDataTemp:
            if indS==-1000:
                chS=1
                break
            
#             data[colNames[index]].fillna(data[colNames[index]].mean(), inplace=True)

        if chS==1:
            data[colNames[index]]=pd.Series(list(map(float,data[colNames[index]])))
            
            
            keyVa=Counter(data[colNames[index]])
            keyVa=list(keyVa.most_common())
            
            
            miSeven=0
            for indeKIn in range(len(keyVa)):  
                miSeven+=keyVa[indeKIn][1]
                if keyVa[indeKIn][0]==-1000:
                    sevenIndex=indeKIn
                
                    
            if (keyVa[sevenIndex][1]/miSeven)>removeRate:
#             if (keyVa[sevenIndex][1]/miSeven)>0.001:
                deleteFeatureSet.append( colNames[index])
                
                tempIndex=exceptDelCol.index(colNames[index])
#                 exceptDelCol[tempIndex]='REMOVED'
                del data[colNames[index]]
                continue
            
            
            
            
            for indeK in range(len(keyVa)):
                
 
                
                if keyVa[indeK][0]!=-1000:
    #                     print(keyVa)
                    
                    # Imputee nan==-1000 frequent values
                    data[colNames[index]]=pd.Series(data[colNames[index]].replace(-1000,keyVa[indeK][0]))
#                     data[colNames[index]]=pd.Series(data[colNames[index]].replace(np.nan,keyVa[indeK][0]))
                    data[colNames[index]]=pd.Series(data[colNames[index]].replace(np.nan,data[colNames[index]].mean()))
#                     data[colNames[index]]=pd.Series(data[colNames[index]].replace(np.nan,keyVa[indeK][0]))
#                     if index==158:
#                         print(data[colNames[index]])
# #                     print(data[colNames[index]])
                    break
            
#             print('nan')
        


# 



listOurRealBeforeSurgeryFeature.append('G90')
# re live
# data=data[listOurRealBeforeSurgeryFeature]
data.index = pd.RangeIndex(len(data.index))


#Data removed 
# del data['PX_STAT']
# del data['PSTATUS']
# del data['PTIME']
# del data['LOS']
del data['CTR_CODE']
data=data.loc[:, ~data.columns.str.contains('^Unnamed')]
# del data['DONOR_ID']
# del data['TRR_ID_CODE']
# del data['WL_ID_CODE']
del data['LISTING_CTR_CODE']
del data['OPO_CTR_CODE']
del data['PTIME']
del data['LOS']
del data['ACUTE_REJ_EPI']
del data['GRF_STAT']
# del data['ETHNICITY']
del data['TX_YEAR']
# del data['GTIME']
# del data['GSTATUS']
# del data['PSTATUS']



finalTest=data.ix[:].values



# plot feature importance manually
from numpy import loadtxt
from xgboost import XGBClassifier
from xgboost import plot_importance
from matplotlib import pyplot

from keras.utils import plot_model

from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# load data
# dataset = loadtxt('complete_data_3 months.csv', delimiter=",")
# split data into X and y
# X = dataset[:,0:8]

y = data['G90'].values

# y = data['PTIME'].values

gstatus=data['G90']

# gstatus=data['PTIME']

del data['G90']


X = data.values
model = XGBClassifier()
model.fit(X, y)
oriMfi=list(model.feature_importances_)
mfiSort=list(model.feature_importances_)
mfiSort.sort(reverse=True)




# importances = pd.DataFrame({'feature':data.columns,'importance':np.round(model.feature_importances_,3)})
importances = pd.DataFrame({'feature':data.columns,'importance':model.feature_importances_})
# importances = importances.sort_values('importance',ascending=False).set_index('feature')
importances = importances.sort_values('importance',ascending=True).set_index('feature')
# print (importances)
# sorted(importances)
# importances.plot.bar()
# importances.plot(kind='barh',fontsize=4,figsize=(150,150))
importances.plot(kind='barh',fontsize=3)
#fig = plt.gcf()
#fig.savefig('C:/Users/User/workspace/Hello_Test/src/result/test3.pdf')
































#number of major factors which have high F-Importance
timeNum=len(importances)

majorList=[]
featureImporList=list(model.feature_importances_)
for elem in mfiSort:
    if elem!=0:
        tempLocation=featureImporList.index(elem)
        featureImporList[tempLocation]=np.nan
        majorList.append(tempLocation)
        
        timeNum-=1
        if timeNum==0:
            break
        






#Print what are important features
for ind in range(0,len(majorList)):
    print(data.columns[majorList[ind]])



#Select 10% of data as test dataset. 90% data is used as training and validate set. 
x_train, x_val, y_train, y_val = train_test_split(X, y,test_size = .1,random_state=12)





# inDimen=len(majorList)
inDimen=60
# x_train_res=x_train_res[:,majorList[0:inDimen]]
x_train=x_train[:,majorList[0:inDimen]]
x_val=x_val[:,majorList[0:inDimen]]






from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
import numpy

from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc
from sklearn import metrics
from scipy import interp

from mpl_toolkits.mplot3d import Axes3D
from sklearn import svm
from keras.layers import Dense
from keras.layers import Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras.constraints import maxnorm
from keras.optimizers import SGD
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
# from keras.regularizers import l2, activity_l2

# import scikitplot as skplt
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)


kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)

Xfpr=[] 

tprs = []
aucs = []
mean_fpr = np.linspace(0,1,100)
i = 1
historySet=[]
cvscores = []


#model definition
model = Sequential()
model.add(Dense(12,input_dim=inDimen, kernel_initializer='uniform', activation='relu',kernel_constraint=maxnorm(3)))
model.add(Dropout(0.2))
model.add(Dense(9, kernel_initializer='uniform', activation='relu',kernel_constraint=maxnorm(3)))
model.add(Dropout(0.2))
model.add(Dense(8, kernel_initializer='uniform', activation='relu',kernel_constraint=maxnorm(3)))
model.add(Dropout(0.2))
model.add(Dense(8, kernel_initializer='uniform', activation='relu',kernel_constraint=maxnorm(3)))
model.add(Dropout(0.2))
model.add(Dense(7, kernel_initializer='uniform', activation='relu',kernel_constraint=maxnorm(3)))
model.add(Dropout(0.2))
model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

historySet=[]
cvscores = []


i=0
#model training validating testing 
for train, test in kfold.split(x_train, y_train):
    

    sm = SMOTE(random_state=12, ratio = 1.0)

    #Oversampling 
    x_train_res_train, y_train_res_train = sm.fit_sample(x_train[train], y_train[train])

    # model fiting with train set and model validatation with validation data set
    history=model.fit(x_train_res_train, y_train_res_train,validation_data=(x_train[test], y_train[test]), epochs=2000, batch_size=300,verbose=0)

    historySet.append(history)

    
    scores = model.evaluate(x_train[test], y_train[test], verbose=0)
    

#predict test dataset.
predicted = model.predict(x_val)
predicted = (predicted > 0.5)
#making confuction matrix
cm = confusion_matrix(y_val, predicted)

fpr, tpr, thresholds = metrics.roc_curve(y_val, predicted)
tprs.append(interp(mean_fpr, fpr, tpr))
roc_auc = auc(fpr, tpr)
print('ROC_AUC')
print(roc_auc)
print('TPR')
print(tpr[1])
print('FPR')
print(fpr[1])
print('PPV')
print(cm[1][1]/(cm[1][1]+cm[0][1]))

print('F1')
print(2*cm[1][1]/(2*cm[1][1]+cm[0][1]+cm[1][0]))
print('ACC')
print((cm[0][0]+cm[1][1])/sum(sum(cm)))

plt.plot(fpr, tpr, lw=2, alpha=0.3, label=' AUC = %0.2f' % (roc_auc))
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC')
plt.show()











