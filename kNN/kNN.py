import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

#pandas ile bir urlden veri setini çekiyoruz.

dataset = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/bezdekIris.data',header=None,
                   names=["AltYaprakUzunlugu", "AltYaprakGenisligi", "UstYaprakUzunlugu", "UstYaprakGenisligi", "Tur"])

#Veriyi tanımak için verinin ilk 5 satırını yazdırıyoruz.

print(dataset.iloc[:5])

#Tur değişkeninin kategorik olduğunu gördük. Bunu tanımlamamız gerekiyor.

dataset['Tur'].unique()
setosa=dataset[dataset['Tur']=='Iris-setosa']
versicolor =dataset[dataset['Tur']=='Iris-versicolor']
virginica =dataset[dataset['Tur']=='Iris-virginica']

#Veri setini tanımak adına özetleyici istatitiklerini görüntülüyoruz.

print(dataset.describe())

#Datasetini görsel olarak görmek için grafik oluşturuyoruz.

plt.figure()
fig,ax=plt.subplots(1,2,figsize=(21, 10))

#Veriyi tanımak için alt yaprakların uzunluğu ve genişliğini karşılaştıran grafik oluşturuyoruz.

setosa.plot(x="AltYaprakUzunlugu", y="AltYaprakGenisligi", kind="scatter",ax=ax[0],label='setosa',color='r')
versicolor.plot(x="AltYaprakUzunlugu",y="AltYaprakGenisligi",kind="scatter",ax=ax[0],label='versicolor',color='b')
virginica.plot(x="AltYaprakUzunlugu", y="AltYaprakGenisligi", kind="scatter", ax=ax[0], label='virginica', color='g')

#Veriyi tanımak için üst yaprakların uzunluğu ve genişliğini karşılaştıran grafik oluşturuyoruz.

setosa.plot(x="UstYaprakUzunlugu", y="UstYaprakGenisligi", kind="scatter",ax=ax[1],label='setosa',color='r')
versicolor.plot(x="UstYaprakUzunlugu",y="UstYaprakGenisligi",kind="scatter",ax=ax[1],label='versicolor',color='b')
virginica.plot(x="UstYaprakUzunlugu", y="UstYaprakGenisligi", kind="scatter", ax=ax[1], label='virginica', color='g')

ax[0].set(title='Alt Yaprak Karsilastirmasi ', ylabel='Alt-Yaprak-Genisligi')
ax[1].set(title='Ust Yaprak Karsilastirmasi ',  ylabel='Ust-Yaprak-Genisligi')
ax[0].legend()
ax[1].legend()
plt.show()

#Bağımlı değişkenimiz(y) ve bağımsız değişkenlerimizi(x) belirliyoruz.

X = np.array(dataset.ix[:, 0:4])
y = np.array(dataset['Tur'])

#Veri setini eğitim-test olarak ayırıyoruz.

X_egitim, X_test, y_egitim, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Veri setini ayırırken oluşabilecek yanlılık, sapma vb hataları önlemek için cross validation kullanıyoruz.
# Her k değeri için knn modelinin başarısını ölçerek skor değişkenine atıyor ve sonunda skor'ların ortalamasını alarak
# basari_skoru değişkenine ekliyor.

Liste = list(range(1,50))
basari_skoru = []
for k in Liste:
    knn = KNeighborsClassifier(n_neighbors=k)
    skor = cross_val_score(knn, X_egitim, y_egitim, cv=10, scoring='accuracy')
    basari_skoru.append(skor.mean())

# MSE'si en düşük olan k'yı optimal k olarak seçiyoruz.

MSE = [1 - x for x in basari_skoru]  #MSE = Mean Squarred Error-Hata Kareler Ortalaması
optimal_k = Liste[MSE.index(min(MSE))]
print("Komsu sayisi k icin en uygun deger %d dir." % optimal_k)

#k ile MSE'den oluşan bir grafik çizdiriyoruz.

plt.plot(Liste, MSE)
plt.xlabel('Komsu Sayisi')
plt.ylabel('Siniflandirma Hatasi')
plt.show()

#Optimal k'yı kullanarak modeli oluşturuyoruz.

knn = KNeighborsClassifier(n_neighbors=optimal_k)

#Oluşturduğumuz modeli eğitiyoruz.

knn.fit(X_egitim, y_egitim)

#Eğittiğimiz modelle tahmin yapıyoruz.

tahmin = knn.predict(X_test)

#kNN ile yaptığımız tahminin başarısını tanımlıyoruz.

basari_yuzdesi = accuracy_score(y_test,tahmin)
print("kNN basarisi:",basari_yuzdesi)