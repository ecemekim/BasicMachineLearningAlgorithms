import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

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

#Modeli oluşturuyoruz.

reg = LinearRegression()

#Modeli eğitiyoruz.

model = reg.fit(x_egitim, y_egitim)

#Eğittiğimiz modelle tahminde bulunuyoruz.

tahmin = reg.predict(x_test)

#Modelin tahmin hatasını MSE metriği ile oluşturuyoruz.

MSE = mean_squared_error(y_test,tahmin)
print("MSE:",MSE)

#Bağımlı değişkenin, bağımsız değişkenlerce açıklanma oranını (R^2) belirliyoruz.

r2 = r2_score(y_test,tahmin)
print("r kare skor: ",r2)

#reg.score(x,y) Bu satır da R^2 metriğini verecektir.