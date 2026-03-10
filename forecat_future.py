import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import joblib
import pandas as pd

# model ve scaler yükle
model = load_model("lstm_model.h5")
scaler = joblib.load("scaler.save")

# zaman serisini yükle
df_hourly = pd.read_csv("df_hourly.csv", index_col = 0, parse_dates = True)

# son 48 saati alalım, 24 saatlik geçmişi kullanarak 24 saatlik geleceği tahmin edelim
last_48 = df_hourly.iloc[-48:].copy() # son 48 saatlik veriyi seç
last_24_real = last_48.iloc[:24].values # ilk 24 saat, modelin girdisi olacak
real_next_24 = last_48.iloc[24:].values # son 24 saat, model tahminleri ile karşılaştırmak için gerçek değerler

# normalize edilmiş veriyi al 
X_test = np.load("X_test.npy")
forecast_input = X_test[-1].copy() # test setinin son penceresi ileriye dönük tahminlerde başlangıç olacak

# modelimiz ile 24 saatlik tahmin yapacağız
future_predictions = []# tahmin edilen kw değerleri burada tutulacak

for _ in range(24):
    input_3d = forecast_input.reshape(1, forecast_input.shape[0], 1) # model 3 boyutlu giriş bekler (örnek sayısı, zaman adımı öznitelik sayısı)

    next_scaled = model.predict(input_3d, verbose = 0)[0] #modelin çıktısı ölçeklenmiş yani 0 ile 1 arasında olur

    next_value = scaler.inverse_transform(next_scaled.reshape(1,-1))[0][0] # orijinal değere dön

    future_predictions.append(next_value) # tahmin değerini listeye ekle

    # yeni girdi penceresini güncelle, ilk değer al, tahmin edilen değeri sona ekle
    forecast_input = np.vstack((forecast_input[1:], next_scaled.reshape(1,1)))


# karşılaştırmalı grafik
plt.figure()
plt.plot(real_next_24.flatten(), label = "Gelecek (Gelecek 24 saat)", linewidth = 2)
plt.plot(future_predictions, label = "Tahmin (Gelecek 24 saat)", linewidth = 2, linestyle = "--")

plt.title("Gelecek 24 saat: Gerçek vs Tahmin")
plt.xlabel("Saat")
plt.ylabel("kwh")
plt.legend()
plt.grid(True)
plt.show()



















