import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score,confusion_matrix,f1_score

#sklearn datasetlerinden iris dataseti alıyor ve değişkenleri tanımlıyoruz.

iris = datasets.load_iris()
x = pd.DataFrame(iris.data)
x.columns = ['AltYaprakUzunlugu', 'AltYaprakGenisligi', 'UstYaprakUzunlugu', 'UstYaprakGenisligi']
y = pd.DataFrame(iris.target)
y.columns = ['Tur']

#Alt yaprak ve üst yaprak için grafik çizdiriyoruz.

plt.figure(figsize=(14, 7))
colormap = np.array(['red', 'lime', 'black'])
plt.subplot(1, 2, 1)
plt.scatter(x.AltYaprakUzunlugu, x.AltYaprakGenisligi, c=colormap[y.Tur], s=40)
plt.title('Alt Yaprak')
plt.subplot(1, 2, 2)
plt.scatter(x.UstYaprakUzunlugu, x.UstYaprakGenisligi, c=colormap[y.Tur], s=40)
plt.title('Ust Yaprak')
plt.show()

#KMeans için optimum k belirlemek adına dirsek yöntemini(Elbow Method) kullanıyoruz.

wcss = []   #Kümeler içi kareler toplamını hesaplamak için boş bir liste oluşturduk.

for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss)
plt.title('Dirsek Yöntemi')
plt.xlabel('Kume Sayisi')
plt.ylabel('Kümeler İçi Kareler Toplamı')
plt.show()

#Dirsek yöntemi grafiğinde ani kırılmanın yaşandığı noktadaki küme sayısı optimal k'yı verir. Burada k=3.

model = KMeans(n_clusters=3,random_state=0)
model.fit(x)

#Çiçek türlerinin gerçek kümelenmesi ile K Means ile tamin edilen kümelemenin karşılaştırmalı grafiğini çizdiriyoruz.

plt.figure(figsize=(14, 7))
colormap = np.array(['red', 'lime', 'black'])
plt.subplot(1, 2, 1)
plt.scatter(x.UstYaprakUzunlugu, x.UstYaprakGenisligi, c=colormap[y.Tur], s=40)
plt.title('Gerçek Kümeleme')
plt.subplot(1, 2, 2)
plt.scatter(x.UstYaprakUzunlugu, x.UstYaprakGenisligi, c=colormap[model.labels_], s=40)
plt.title('K Means Kümeleme')
plt.show()

#K Means ile tahmin yapıyoruz.

tahmin = np.choose(model.labels_, [1, 0, 2]).astype(np.int64)
print(model.labels_)
print(tahmin)

plt.figure(figsize=(14, 7))
colormap = np.array(['red', 'lime', 'black'])

plt.subplot(1, 2, 1)
plt.scatter(x.UstYaprakUzunlugu, x.UstYaprakGenisligi, c=colormap[y.Tur], s=40)
plt.title('Gerçek Sınıflandırma')

plt.subplot(1, 2, 2)
plt.scatter(x.UstYaprakUzunlugu, x.UstYaprakGenisligi, c=colormap[tahmin], s=40)
plt.title('K Means Sınıflandırma')
plt.show()

#KMeans ile elde ettiğimiz sonuçları yazdırıyoruz.

print("Basari: ",accuracy_score(y,tahmin))
print("f1 makro skor:",f1_score(y,tahmin,average='macro'))
print("Hata Matrisi: ",confusion_matrix(y,tahmin))
