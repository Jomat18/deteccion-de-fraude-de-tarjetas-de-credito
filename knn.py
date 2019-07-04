
#Libreria a utilizar
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D
import time
import math
import scipy

#cargando la dataset
dataset = pd.read_csv("creditcard.csv", header = 0)
print(dataset.head())
print(dataset.describe())

x = dataset.iloc[: , 1:30].values
y = dataset.iloc[:, 30].values

#mostrando el tamaño de las salidas y entradas
print("Entrada : ", x.shape)
print("Salida : ", y.shape)

#conjunto para el entrenamiento
data, data_test = train_test_split(dataset, test_size=0.25)

#Separando los fraudes de los no fraudes
non_fraud = data[data['Class'] == 0].sample(1000)
fraud = data[data['Class'] == 1]

df = non_fraud.append(fraud).sample(frac=1).reset_index(drop=True)
X = df.drop(['Class'], axis = 1).values
Y = df["Class"].values


print("Porcentaje total de transaciones que son fraudulentas")
print(dataset["Class"].mean()*100)

print("Perdidas debido al fraude:")
print("Cantidad total perdida por fraude")
print(dataset.Amount[dataset.Class == 1].sum())
print("Importe medio por transacción fraudulenta")
print(dataset.Amount[dataset.Class == 1].mean())
print("Compare con las transacciones normales:")
print("Importe total de transacciones normales")
print(dataset.Amount[dataset.Class == 0].sum())
print("Monto medio por transacciones normales")
print(dataset.Amount[dataset.Class == 0].mean())


#Visulizando los datos
f, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12,4))

bins = 40

ax1.hist(dataset.Amount[dataset.Class == 1], bins = bins, normed = True, alpha = 0.75, color = 'red')
ax1.set_title('Fraude')

ax2.hist(dataset.Amount[dataset.Class == 0], bins = bins, normed = True, alpha = 0.5, color = 'blue')
ax2.set_title('No Fraude')

plt.xlabel('Cantidad')
plt.ylabel('% de Transaciones')
plt.yscale('log')
plt.show()


bins = 75
plt.hist(dataset.Time[dataset.Class == 1], bins = bins, normed = True, alpha = 0.75, label = 'Fraude', color = 'red')
plt.hist(dataset.Time[dataset.Class == 0], bins = bins, normed = True, alpha = 0.5, label = 'No Fraude', color = 'blue')
plt.legend(loc='upper right')
plt.xlabel('Tiempo (segundos)')
plt.ylabel('% of ')
plt.title('Transaciones por Tiempo')
plt.show()


tic=time.time()
full_data=pd.read_csv("creditcard.csv")

full_features=full_data.drop(["Time","Class"],axis=1)
full_labels=pd.DataFrame(full_data[["Class"]])

full_features_array=full_features.values
full_labels_array=full_labels.values

#Normalizando
train_features,test_features,train_labels,test_labels=train_test_split(full_features_array,full_labels_array,train_size=0.90)
train_features=normalize(train_features)
test_features=normalize(test_features)

#k_means_clustering, confusion_matrix
kmeans=KMeans(n_clusters=2,random_state=0,algorithm="elkan",max_iter=10000,n_jobs=-1)
kmeans.fit(train_features)
kmeans_predicted_train_labels=kmeans.predict(train_features)


#confusion matrix
# tn fp
# fn tp
print("tn --> Verdadero negativos")
print("fp --> Falso positivos")
print("fn --> Falso negativos")
print("tp --> Verdadero positivos")

tn,fp,fn,tp=confusion_matrix(train_labels,kmeans_predicted_train_labels).ravel()

reassignflag=False

if tn+tp<fn+fp:
	reassignflag=True
kmeans_predicted_test_labels=kmeans.predict(test_features)
if reassignflag:
	kmeans_predicted_test_labels=1-kmeans_predicted_test_labels


#confusion matrix para kmeans
tn,fp,fn,tp=confusion_matrix(test_labels,kmeans_predicted_test_labels).ravel()


#resultados
kmeans_accuracy_score=accuracy_score(test_labels,kmeans_predicted_test_labels)
kmeans_precison_score=precision_score(test_labels,kmeans_predicted_test_labels)
kmeans_recall_score=recall_score(test_labels,kmeans_predicted_test_labels)
kmeans_f1_score=f1_score(test_labels,kmeans_predicted_test_labels)


print("")
print("K-Means")
print("Confusion Matrix")
print("tn =",tn,"fp =",fp)
print("fn =",fn,"tp =",tp)
print("Puntuaciones")
print("Exactitud -->",kmeans_accuracy_score)
print("Precision -->",kmeans_precison_score)
print("Recall -->",kmeans_recall_score)
print("F1 -->",kmeans_f1_score)


#k_nearest_neighbours
knn=KNeighborsClassifier(n_neighbors=5,algorithm="kd_tree",n_jobs=-1)
knn.fit(train_features,train_labels.ravel())
knn_predicted_test_labels=knn.predict(test_features)


#confusion matrix para knn
tn,fp,fn,tp=confusion_matrix(test_labels,knn_predicted_test_labels).ravel()


#resultados knn
knn_accuracy_score=accuracy_score(test_labels,knn_predicted_test_labels)
knn_precison_score=precision_score(test_labels,knn_predicted_test_labels)
knn_recall_score=recall_score(test_labels,knn_predicted_test_labels)
knn_f1_score=f1_score(test_labels,knn_predicted_test_labels)


print("")
print("K-Nearest Neighbours")
print("Confusion Matrix")
print("tn =",tn,"fp =",fp)
print("fn =",fn,"tp =",tp)
print("Puntuaciones")
print("Exactitud -->",knn_accuracy_score)
print("Precision -->",knn_precison_score)
print("Recall -->",knn_recall_score)
print("F1 -->",knn_f1_score)

#ROC Knn
fpr, tpr, threshold = roc_curve(test_labels, knn_predicted_test_labels)
roc_auc = auc(fpr, tpr)

plt.title('Característica de funcionamiento del receptor')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('Rango Verdaderos Positivos')
plt.xlabel('Rango Falsos Positivos')
plt.title('ROC curva de kNN')
plt.show()


toc=time.time()
elapsedtime=toc-tic
print("")
print("Tiempo tomado : "+str(elapsedtime)+"segundos")


