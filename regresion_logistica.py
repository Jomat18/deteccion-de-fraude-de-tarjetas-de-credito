
#Libreria a utilizar
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn import metrics


#cargando la dataset
df=pd.read_csv("creditcard.csv")

target=df['Class']
target

df.drop('Class',axis=1,inplace=True)
df.shape

#mostrando el tamaño de las salidas y entradas
X=np.array(df)
y=np.array(target)
X.shape
y.shape

len(y[y==1])
len(y[y==0])

#Entrenamiento y testing (75:25)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=1)
print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)

sm=SMOTE(random_state=2)
X_sm,y_sm=sm.fit_sample(X_train,y_train)
print(X_sm.shape,y_sm.shape)
print(len(y_sm[y_sm==1]),len(y_sm[y_sm==0]))

logreg=LogisticRegression()
logreg.fit(X_sm,y_sm)
y_logreg=logreg.predict(X_test)
y_logreg_prob=logreg.predict_proba(X_test)[:,1]


print("Matrix de Confusion:\n",metrics.confusion_matrix(y_test,y_logreg))
print("Exactitud:\n",metrics.accuracy_score(y_test,y_logreg))
print("Precision:\n",metrics.precision_score(y_test,y_logreg))
print("Recall:\n",metrics.recall_score(y_test,y_logreg))
print("AUC:\n",metrics.roc_auc_score(y_test,y_logreg_prob))
auc=metrics.roc_auc_score(y_test,y_logreg_prob)

#ROC
fpr,tpr,thresholds=metrics.roc_curve(y_test,y_logreg_prob)
plt.plot(fpr,tpr,'b', label='AUC = %0.2f'% auc)
plt.plot([0,1],[0,1],'r-.')
plt.xlim([-0.2,1.2])
plt.ylim([-0.2,1.2])
plt.title('Regresion Logistica')
plt.legend(loc='lower right')
plt.ylabel('Rango Verdaderos Positivos')
plt.xlabel('Rango Falsos Positivos')
plt.show()

