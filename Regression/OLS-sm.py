import pandas as pd
from sklearn.model_selection import train_test_split
import statsmodels.api as sm

#pandas ile bir urlden veri setini çekiyoruz.

dataset = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/bezdekIris.data',header=None,
                   names=["AltYaprakUzunlugu", "AltYaprakGenisligi", "UstYaprakUzunlugu", "UstYaprakGenisligi", "Tur"])

#Bağımlı değişkenimiz(y) ve bağımsız değişkenlerimizi(x) belirliyoruz.

x = dataset.iloc[:,0:4].values
y_ = dataset.iloc[:,4].values

#Bağımlı değişkenin her bir kategorisini 0'dan başlayarak kodluyoruz.

y = pd.factorize(dataset['Tur'])[0]

#Verisetini eğitim-test olarak ayırıyoruz.

(x_egitim, x_test, y_egitim, y_test) = train_test_split(x, y, test_size=0.25, random_state=1)

#EKK modelini oluşturup eğitiyoruz.

model = sm.OLS(y_egitim, x_egitim).fit()

#Eğitilen modelle tahminde bulunuyoruz.

tahmin = model.predict(x_test)

#En küçük kareler regresyonun sonuçlarını yazdırıyoruz.

print(model.summary())