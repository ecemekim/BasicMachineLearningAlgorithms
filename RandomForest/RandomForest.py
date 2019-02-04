import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

iris = datasets.load_iris()
dataset = pd.DataFrame(iris.data, columns=iris.feature_names)

#Sklearn sınıflandırma için hedef değişkenin numerik olmasını ister.
# Bu sebeple dataset dataframe'ine species(tür) değişkenini ekliyoruz.

dataset['species'] = np.array([iris.target_names[i] for i in iris.target])

#Kategorik değişkenin species olduğunu belirterek datayı visualize ediyoruz.

sns.pairplot(dataset, hue='species')
plt.show()

#Veriyi eğitim-test olarak ayırıyoruz.

x_egitim, X_test, y_egitim, y_test = train_test_split(dataset[iris.feature_names], iris.target, test_size=0.33, stratify=iris.target, random_state=1)

#Random Forest modelini oluşturup eğitiyoruz.

rf = RandomForestClassifier(n_estimators=100, oob_score=True, random_state=1)
rf.fit(x_egitim, y_egitim)

#Oluşturduğumuz modelle tahmin yapıyoruz.
#Model başarısını oluşturuyoruz ve out-of-bag skorunu tahmin ediyoruz.

tahmin = rf.predict(X_test)
basari = accuracy_score(y_test, tahmin)
print('Out-of-bag skor tahmini: ',rf.oob_score_)
print('Ortalama basari: ',basari)

#Hata matrisi ile, sınıfların tahminini gerçek sınıflarla karşılaştırıyoruz ve visualize ediyoruz.

hm = pd.DataFrame(confusion_matrix(y_test, tahmin), columns=iris.target_names, index=iris.target_names)
sns.heatmap(hm, annot=True)
plt.show()