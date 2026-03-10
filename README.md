# ⚡ LSTM ile Elektrik Tüketimi Tahmini (Time Series Forecasting)

Bu proje, geçmiş elektrik tüketim verilerini kullanarak gelecekteki (24 saatlik) enerji tüketimini tahmin etmeyi amaçlayan Derin Öğrenme (Deep Learning) tabanlı bir zaman serisi analizidir. Model mimarisi olarak sıralı veri işleme kapasitesi yüksek olan **LSTM (Long Short-Term Memory)** ağları kullanılmıştır.

Proje, modelin kendi ürettiği tahminleri bir sonraki adımın girdisi olarak kullandığı (autoregressive) bir yapı ile çalışır ve ileriye dönük ardışık tahminler üretir.

## 📊 Veri Seti Hakkında (Önemli Not)

GitHub'ın 100 MB'lık dosya boyutu sınırı nedeniyle, projenin eğitiminde kullanılan orijinal `household_power_consumption.txt` (yaklaşık 124 MB) dosyası bu repoya dahil **edilmemiştir**. 

Projeyi kendi bilgisayarınızda çalıştırmak için:
1. Veri setini [UCI Machine Learning Repository - Individual household electric power consumption](https://archive.ics.uci.edu/ml/datasets/individual+household+electric+power+consumption) adresinden indirin.
2. Zipten çıkardığınız `household_power_consumption.txt` dosyasını projenin ana dizinine yerleştirin.

*(Not: İşlenmiş büyük boyutlu `.npy` matrisleri ve eğitilmiş `.h5` model dosyaları da boyuttan tasarruf etmek amacıyla `.gitignore` ile repodan hariç tutulmuştur. Kodları çalıştırdığınızda bu dosyalar bilgisayarınızda otomatik olarak üretilecektir.)*

## 📁 Proje Yapısı ve İş Akışı

Proje, modüler bir yapıda tasarlanmış olup adım adım çalıştırılacak şekilde 5 ana Python betiğinden oluşmaktadır:

* `clean_dataset.py`: Ham `.txt` verisini okur, eksik verileri temizler ve ön işleme (preprocessing) adımlarını uygular.
* `data_preparation.py`: Temizlenmiş veriyi saatlik formata (`df_hourly.csv`) dönüştürür, *MinMaxScaler* ile normalize eder ve modeli besleyecek olan 3 boyutlu zaman serisi matrislerini (`X_train.npy`, `y_test.npy` vb.) oluşturur.
* `lstm_train.py`: TensorFlow/Keras kullanarak LSTM sinir ağını inşa eder, modeli eğitir ve ağırlıkları `lstm_model.h5` olarak kaydeder.
* `lstm_test.py`: Eğitilmiş modeli test verisi üzerinde değerlendirir ve başarı metriklerini ölçer.
* `forecat_future.py`: Son 24 saatlik gerçek veriyi alarak modelin gelecek 24 saati tahmin etmesini sağlar. Tahminleri ve gerçek verileri karşılaştırmalı bir Matplotlib grafiği ile görselleştirir.

## 🚀 Kurulum ve Çalıştırma

Projeyi yerel makinenizde (local) çalıştırmak için aşağıdaki adımları izleyebilirsiniz:

**1. Repoyu Klonlayın:**
```bash
git clone [https://github.com/efsunboz16/LSTM-ile-Enerji-T-ketimi-Tahmini.git](https://github.com/efsunboz16/LSTM-ile-Enerji-T-ketimi-Tahmini.git)
cd LSTM-ile-Enerji-T-ketimi-Tahmini

Sanal ortam oluşturun 
python -m venv venv
# Windows için:
.\venv\Scripts\activate
# MacOS/Linux için:
source venv/bin/activate

Gerekli kütüphaneleri indirin
pip install -r requirements.txt

Çalıştırma sırası
python clean_dataset.py
python data_preparation.py
python lstm_train.py
python lstm_test.py
python forecat_future.py

🛠️ Kullanılan Teknolojiler
Dil: Python 3.x

Makine Öğrenmesi & Derin Öğrenme: TensorFlow, Keras, Scikit-Learn

Veri Manipülasyonu: Pandas, NumPy

Görselleştirme: Matplotlib

👤 Geliştirici
Efsun

GitHub: @efsunboz16
