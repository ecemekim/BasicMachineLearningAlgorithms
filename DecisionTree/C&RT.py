import os
import pydotplus
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.externals.six import StringIO
from IPython.display import Image
from sklearn.tree import export_graphviz

os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

#pandas ile bir urlden veri setini çekiyoruz.

dataset = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/bezdekIris.data',header=None,
                   names=["AltYaprakUzunlugu", "AltYaprakGenisligi", "UstYaprakUzunlugu", "UstYaprakGenisligi", "Tur"])

#Veriyi tanımak için verinin ilk 5 satırını yazdırıyoruz.

print(dataset.iloc[0:5:])

#Tur değişkeninin string olduğunu belirterek tüm değişkenler için ortak bir grafik çizdiriyoruz.

sns.pairplot(dataset, hue='Tur')
plt.show()

#Bağımlı değişkenimiz(y) ve bağımsız değişkenlerimizi(x) belirliyoruz.

x = dataset.iloc[:,0:4].values
y = dataset.iloc[:,4].values

#Veri setini eğitim-test olarak ayırıyoruz.

(x_egitim, x_test, y_egitim, y_test) = train_test_split(x, y, test_size=0.25, random_state=1)

#Modelimizi oluşturup, eğitiyoruz.

karar_agaci = DecisionTreeClassifier()
karar_agaci.fit(x_egitim, y_egitim)

#Karar ağacı başarısını yazdırıyoruz.

print("C&RT ile Siniflandirma Skoru: ",karar_agaci.score(x_test, y_test))

#Karar ağacını visualize ediyoruz.

dot_data = StringIO()
export_graphviz(karar_agaci, out_file=dot_data,
                filled=True, rounded=True,
                special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
Image(graph.create_png())
graph.write_jpg("cart-iris-gini.jpg")