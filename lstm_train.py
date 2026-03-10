import numpy as np
from tensorflow.keras.models import Sequential # base model, katmanları bunun üzerine inşa edelim 
from tensorflow.keras.layers import LSTM, Dense # LSTM ve dense tam bağlantılı katmanlar
from tensorflow.keras.callbacks import EarlyStopping # erken durdurma 
from tensorflow.keras.losses import MeanSquaredError # MSE kayıp fonksiyonu 

import matplotlib.pyplot as plt

# veriyi yükle
X_train = np.load("X_train.npy")
X_test = np.load("X_test.npy")
y_train = np.load("y_train.npy")
y_test = np.load("y_test.npy")

# modeli tanımla
model = Sequential() #bunun içerisine model katmanları sıralı bir şekilde bağlanır

# LSTM katmanı
model.add(LSTM(
    256, # 64 tane LSTM hücresi
    activation = "tanh",
    input_shape = {X_train.shape[1], X_train.shape[2]} # (24,1) -> (zaman adımı, öznitelik sayısı)

))

# dense layer ekle
model.add(Dense(1)) #tek çıkışlı bir tam bağlantılı katman sadece 1 saatlik enerji tahmini yapar

# model compile (derleme)
model.compile(
    optimizer = "adam", # yaygın olarak kullanılır, hızlıdır, adaptif öğrenme söz konusudur
    loss = MeanSquaredError() # ortalama kare hata, regresyon problemleri için en çok kullanılan kayıp fonksiyonu 
)

# erken durdurma callback
early_stop = EarlyStopping(
    monitor = "val_loss", # doğrulama kaybı izlenir
    patience = 5, # art arda 5 epoch boyunca doğrulama iyileşmez ise eğitim durur
    restore_best_weights = True # böylece eğitim sırasında ki en iyi ağırlıklar (min val loss (en düşük doğrulama kaybı)) geri yüklenir
)

# eğitimi başlat
history = model.fit(
    X_train, y_train,
    validation_data = (X_test,y_test),
    epochs = 15,
    batch_size = 32, # her bir eğitim adımında 32 örnek işlenir
    callbacks = [early_stop], # eğitim sırasındaki erken durdurma
    verbose = 1 # eğitim sırasında detaylı çıktılar yazdırılır
)

# kayıp grafiği çizdirme
plt.plot(history.history["loss"], label = "Eğitim Kaybı")
plt.plot(history.history["val_loss"], label = "Doğrulama Kaybı")
plt.title("Model Kayıp Grafiği")
plt.xlabel("Epochs")
plt.ylabel("Loss (MSE)")
plt.legend()
plt.show()

# model kaydet
model.save("lstm_model.h5") # eğitim lstm modeli hdf5 formatında diske kaydedilir

















