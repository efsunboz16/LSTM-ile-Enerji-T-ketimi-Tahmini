import pandas as pd
import numpy as np
import matplotlib.pyplot as plt # görselleştirme

# 1. Adım: Veriyi parse_dates kullanmadan, düz bir şekilde okuyoruz
df = pd.read_csv(
    "household_power_consumption.txt",
    sep=";", 
    na_values="?", 
    low_memory=False 
)

# 2. Adım: Date ve Time sütunlarındaki metinleri aralarında bir boşluk bırakarak birleştiriyoruz
# ve formatı (Gün/Ay/Yıl Saat:Dakika:Saniye) belirterek datetime objesine çeviriyoruz
df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %H:%M:%S')

# 3. Adım: Artık işimiz biten eski 'Date' ve 'Time' sütunlarını datasetten silebiliriz (İsteğe bağlı ama önerilir)
df = df.drop(columns=['Date', 'Time'])

# datetime sütununu indeks yap
df.set_index("datetime", inplace = True)

# Global active power sütununu seç

df["Global_active_power"] = pd.to_numeric(
    df["Global_active_power"], # sayıya çevirmek istediğimiz sütun / şuanda string haldeler
    errors = "coerce" # sayıya çevrilemeyenleri boşluklar harfler falan bunları NaN yap
)

# eksik verilerden kurtul
# sadece Global_active_power içerisindeki nana değerleri sil, eğitim esnasında hata olmasını engeller
df = df.dropna(subset = ["Global_active_power"])

# saatlik ortlamaya göre yeniden örnekleme
df_hourly = df["Global_active_power"].resample("h").mean()

# zaman serisini görselleştirme
plt.figure()
plt.plot(df_hourly, label = "Saatlik Enerji Tüketimi")
plt.title("Enerji Tüketimi")
plt.xlabel("Zaman")
plt.ylabel("kw")
plt.legend()
plt.show()

# saatlik yeniden örneklenmiş veriyi kaydet
df_hourly.to_csv("df_hourly.csv")