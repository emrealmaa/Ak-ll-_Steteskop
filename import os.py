import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras import layers, models
import shutil

# --- AYARLAR ---
# 1. Senin oluşturduğun ana klasör (İçinde GURULTU... ve Kalp_Sesi olan yer)
ANA_KLASOR = r"C:\Users\merts\OneDrive\Masaüstü\gemini deneme\Kalp_Bekci_Seti"

# 2. Modelin çıkacağı yer
KAYIT_YOLU = r"C:\Users\merts\OneDrive\Masaüstü\bekci_kalp_modeli.h5"
# ---------------

print("🛡️ FİNAL BEKÇİ EĞİTİMİ BAŞLIYOR... 🚧")

# --- ADIM 0: KLASÖR İSİMLERİNİ STANDARTLAŞTIR (HATA ÖNLEME) ---
# Amacımız: 0=Heart_Sound, 1=Noise sıralamasını garantilemek (H < N)
print("\n📂 Klasör isimleri kontrol ediliyor...")

# Eski isimler (Senin verdiklerin)
eski_gurultu = os.path.join(ANA_KLASOR, "GURULTU_GORSELLERI_FINAL")
eski_kalp = os.path.join(ANA_KLASOR, "Kalp_Sesi")

# Yeni standart isimler
yeni_noise = os.path.join(ANA_KLASOR, "Noise")
yeni_heart = os.path.join(ANA_KLASOR, "Heart_Sound")

# Değiştir
if os.path.exists(eski_gurultu) and not os.path.exists(yeni_noise):
    print("   -> 'GURULTU_GORSELLERI_FINAL' ismi 'Noise' olarak değiştiriliyor...")
    os.rename(eski_gurultu, yeni_noise)

if os.path.exists(eski_kalp) and not os.path.exists(yeni_heart):
    print("   -> 'Kalp_Sesi' ismi 'Heart_Sound' olarak değiştiriliyor...")
    os.rename(eski_kalp, yeni_heart)

print("✅ Klasör yapısı standartlaştırıldı.")


# --- ADIM 1: VERİ YÜKLEME ---
img_yukseklik = 128
img_genislik = 128
batch_boyutu = 64 # Veri çok olduğu için batch'i artırdık

print("\n--- Veriler Yükleniyor... ---")

train_ds = tf.keras.utils.image_dataset_from_directory(
  ANA_KLASOR,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_yukseklik, img_genislik),
  batch_size=batch_boyutu)

val_ds = tf.keras.utils.image_dataset_from_directory(
  ANA_KLASOR,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_yukseklik, img_genislik),
  batch_size=batch_boyutu)

class_names = train_ds.class_names
print(f"\n✅ SINIF SIRALAMASI: {class_names}") 
# BEKLENEN: ['Heart_Sound', 'Noise'] (H harfi N'den önce gelir -> 0:Heart, 1:Noise)
# Bu sıralama Sunucu koduyla %100 uyumludur.

# Performans
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# --- ADIM 2: BEKÇİ MODELİ (GÜÇLENDİRİLMİŞ CNN) ---
# Bekçi artık daha zeki olmalı, basit gürültüleri ezberlememeli.
model = models.Sequential([
  layers.Rescaling(1./255, input_shape=(img_yukseklik, img_genislik, 3)),
  
  # Katman 1
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Dropout(0.1), # Hafif unutkanlık (Ezberi bozar)
  
  # Katman 2
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Dropout(0.1),
  
  # Katman 3
  layers.Conv2D(128, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Dropout(0.2),
  
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dropout(0.3),
  
  layers.Dense(len(class_names), activation='softmax')
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])

model.summary()

# --- ADIM 3: EĞİTİM ---
print("\n--- EĞİTİM BAŞLIYOR (Bu sefer %100 hemen olmayacak!) ---")
# Veri çok ve zor olduğu için 8 Epoch idealdir.
epochs = 8 

history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)

# --- SONUÇ GRAFİĞİ ---
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Eğitim')
plt.plot(epochs_range, val_acc, label='Test')
plt.legend(loc='lower right')
plt.title('Doğruluk')
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Eğitim')
plt.plot(epochs_range, val_loss, label='Test')
plt.legend(loc='upper right')
plt.title('Kayıp')
plt.show()

# --- KAYDET ---
model.save(KAYIT_YOLU)
print(f"\n✅ YENİ NESİL BEKÇİ HAZIR: {KAYIT_YOLU}")