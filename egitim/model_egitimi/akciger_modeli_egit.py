import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

# --- AYARLAR ---
# Veri setinin olduğu klasör (İçinde Asthma, COPD, Healthy vb. klasörler olmalı)
veri_klasoru = r"C:\Users\merts\OneDrive\Masaüstü\gemini deneme\ISLENMIS_VERI_SETI"

# Resim Boyutu (Standart 128x128)
img_yukseklik = 128
img_genislik = 128
batch_boyutu = 32 # Her adımda kaç resim öğrensin?

print("🚀 BÜYÜK EĞİTİM BAŞLIYOR...")
print(f"TensorFlow Versiyonu: {tf.__version__}")

# 1. Veri Setini Yükle ve Ayır
print("\n--- Veriler Yükleniyor (Bu biraz sürebilir) ---")

# Eğitim Seti (%80)
train_ds = tf.keras.utils.image_dataset_from_directory(
  veri_klasoru,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_yukseklik, img_genislik),
  batch_size=batch_boyutu)

# Test (Doğrulama) Seti (%20)
val_ds = tf.keras.utils.image_dataset_from_directory(
  veri_klasoru,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_yukseklik, img_genislik),
  batch_size=batch_boyutu)

# Sınıf İsimlerini Al ve Yazdır
sinif_isimleri = train_ds.class_names
print(f"\n✅ Tespit Edilen Sınıflar: {sinif_isimleri}")
print(f"Toplam Sınıf Sayısı: {len(sinif_isimleri)}")

# Performans Ayarı (Veriyi RAM'e alıp hızlandırır)
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# 2. CNN Modelini İnşa Et (Beyin Mimarisi)
print("\n--- CNN Modeli Oluşturuluyor ---")

model = Sequential([
  # Giriş: Resim verisini 0-1 arasına sıkıştır (Normalize)
  layers.Rescaling(1./255, input_shape=(img_yukseklik, img_genislik, 3)),
  
  # Katman 1: Özellik Çıkarıcı (Detaycı)
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  
  # Katman 2: Özellik Çıkarıcı (Daha Geniş)
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  
  # Katman 3: Özellik Çıkarıcı (Soyut)
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  
  # Katman 4: Derinleşme
  layers.Conv2D(128, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  
  # Düzleştirme ve Karar Verme
  layers.Flatten(),
  layers.Dense(256, activation='relu'),
  
  # Dropout: Ezberlemeyi önlemek için %50 unutma
  layers.Dropout(0.5),
  
  # Çıkış: Sınıf sayısı kadar karar hücresi
  layers.Dense(len(sinif_isimleri), activation='softmax')
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])

model.summary()

# 3. Eğitimi Başlat
print("\n--- EĞİTİM BAŞLIYOR (Arkanıza yaslanın) ---")
# epochs=15: Veriyi 15 kere dönecek. Veri çok olduğu için 15 yeterli olacaktır.
epochs = 15 
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)

# 4. Grafikleri Çiz (Başarıyı Gör)
print("\n--- Eğitim Bitti! Grafikler Çiziliyor... ---")

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Eğitim Başarısı')
plt.plot(epochs_range, val_acc, label='Test Başarısı')
plt.legend(loc='lower right')
plt.title('Doğruluk (Accuracy)')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Eğitim Hatası')
plt.plot(epochs_range, val_loss, label='Test Hatası')
plt.legend(loc='upper right')
plt.title('Hata (Loss)')
plt.show()

# 5. Modeli Kaydet
kayit_yolu = r"C:\Users\merts\OneDrive\Masaüstü\gemini deneme\akilli_steteskop_final_model.h5"
model.save(kayit_yolu)
print(f"\n✅ TEBRİKLER! Final modeli kaydedildi: {kayit_yolu}")