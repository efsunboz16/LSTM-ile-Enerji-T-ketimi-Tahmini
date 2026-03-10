import numpy as np
from sklearn.preprocessing import MinMaxScaler # veri ölçeklemek yani normalize etmek
import pandas as pd
import joblib #model ve işlem nesnelerini kaydetmek ve yüklemek

df_hourly = pd.read_csv(
    "df_hourly.csv",
    index_col = 0, # ilk sütun yani datetimei index olarak kullan
    parse_dates = True #indeks sütunundaki tarih-saat verisini datetime formatına çevirir
)

# NaN değerleri temizle
df_hourly.dropna(inplace = True)

# pandas dataframe formatından numpy array formatına çevrilir
values = df_hourly.values.reshape(-1,1)

# normalizasyon (0-1 arasına sıkıştırma)
scaler = MinMaxScaler() # 0-1 arasına ölçeklemek için gerekli sınıf
scaled = scaler.fit_transform(values) # önce veriye göre min max değerlerini hesaplıyor sonrasında dönüştürüyor.
# neden normalizasyon: lstm modellerde, modelin daha hızlı ve stabil öğrenmesi için önemli

# scaler kaydetme
joblib.dump(scaler, "scaler.save") # test veya gerçek zamanlı tahminde aynı ölçekleyici kullan

# sliding window
def create_sliding_window(data, window_size = 24):
    """
        data: normalleştirilmiş zaman serisi verisi
        window_size: geçmiş kaç adim kullanilacak son 24 saat
    
    
    """

    X, y = [],[]
    for i in range(len(data) - window_size):
        X.append(data[i: i + window_size]) # belirlenen pencere kadar geçmiş veri
        y.append(data[i + window_size ])
                 
    return np.array(X), np.array(y)

# giriş ve çıkış verilerini oluştur
window_size = 24
X,y = create_sliding_window(scaled, window_size)

# train test split - eğitim ve test ayrımı
# bu ayrım sayesinde model geçmiş veriler ile eğitilir ve gelecek veriler ile test edilir
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:] # ilk %80 eğitim kalan %20 test
y_train, y_test = y[:split], y[split:] 

print(f"X_train shape: {X_train.shape}") # X_train shape: (27315, 24, 1) -> (örnek sayısı, zaman adımı, özellik sayısı)
print(f"y_train shape: {y_train.shape}") 

np.save("X_train", X_train)
np.save("X_test", X_test)
np.save("y_train", y_train)
np.save("y_test", y_test)