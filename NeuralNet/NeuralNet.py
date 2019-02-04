from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam

iris_data = load_iris()
x = iris_data.data
y_ = iris_data.target.reshape(-1, 1)
encoder = OneHotEncoder(sparse=False)
y = encoder.fit_transform(y_)

x_egitim, x_test, y_egitim, y_test = train_test_split(x, y, test_size=0.20)

model = Sequential()
model.add(Dense(10, input_dim=4, activation='relu', name='fc1'))
model.add(Dense(10, activation='relu', name='fc2'))
model.add(Dropout(0.1))  #overfittingi önlemek için
model.add(Dense(3, activation='softmax', name='output'))

optimizer = Adam(lr=0.001)
model.compile(optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

print('Yapay Sinir Agi Model Ozeti: ')
print(model.summary())

model.fit(x_egitim, y_egitim, verbose=2, batch_size=5, epochs=200)
sonuc = model.evaluate(x_test, y_test)

print('Test hatasi: {:4f}'.format(sonuc[0]))
print('Test basarisi: {:4f}'.format(sonuc[1]))