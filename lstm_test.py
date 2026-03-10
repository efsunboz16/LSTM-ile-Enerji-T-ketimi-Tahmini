
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model 
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

# verileri ve modeli yükle
X_test = np.load("X_test.npy")
y_test = np.load("y_test.npy")

model = load_model("lstm_model.h5")
scaler = joblib.load("scaler.save") # normalizasyon sırasında kullanılan scaler nesnesini tekrar yükler

# tahmin yap
# model test verisi ile tahmin yapar
y_pred_scaled = model.predict(X_test)

# tahmin ve gerçek değerleri ölçeklendir
y_pred = scaler.inverse_transform(y_pred_scaled)
y_true = scaler.inverse_transform(y_test)

# hata metrikleri 
mae = mean_absolute_error(y_true,y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))

print(f"MAE: {mae}")
print(f"RMSE: {rmse}")

# ilk 200 tahmin grafiğini çizdir
plt.figure()
plt.plot(y_true[:200], label = "Gerçek", linewidth = 2)
plt.plot(y_pred[:200], label = "Tahmin", linestyle = "--")
plt.title("Gerçek ve Tahmin")
plt.xlabel("Saat")
plt.ylabel("Güç Tüketimi")
plt.legend()
plt.show()









