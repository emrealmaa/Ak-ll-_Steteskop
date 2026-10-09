import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

# --- AYARLAR (Yolları Kontrol Et) ---
# 1. Eğitim Setinin Olduğu Klasör (İçinde 'Noise' ve 'Lung_Sound' olmalı)
veri_klasoru = r"C:\Users\merts\OneDrive\Masaüstü\gemini deneme\BEKCI_VERI_SETI"

# 2. Modelin Kaydedileceği Yer
kayit_yolu = r"C:\Users\merts\OneDrive\Masaüstü\gemini deneme\bekci_modeli.h5"
# ------------------------------------

# Resim Ayarları (Ana modelle aynı olmalı)
img_yukseklik = 128
img_genislik = 128
batch_boyutu = 32

print("🛡️ BEKÇİ MODELİ EĞİTİMİ BAŞLIYOR...")
print(f"TensorFlow Versiyonu: {tf.__version__}")

# 1. Verileri Yükle (%80 Eğitim, %20 Test)
print("\n--- Veriler Yükleniyor... ---")

train_ds = tf.keras.utils.image_dataset_from_directory(
  veri_klasoru,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_yukseklik, img_genislik),
  batch_size=batch_boyutu)

val_ds = tf.keras.utils.image_dataset_from_directory(
  veri_klasoru,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_yukseklik, img_genislik),
  batch_size=batch_boyutu)

class_names = train_ds.class_names
print(f"\n✅ Sınıflar: {class_names}") 
# Çıktı şuna benzer olmalı: ['Lung_Sound', 'Noise']

# Performans Ayarı (Hızlandırma)
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# 2. Modeli Kur (Hızlı ve Hafif CNN)
print("\n--- Bekçi Modeli İnşa Ediliyor... ---")

model = Sequential([
  layers.Rescaling(1./255, input_shape=(img_yukseklik, img_genislik, 3)),
  
  # Katman 1
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  
  # Katman 2
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  
  # Katman 3
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  
  layers.Flatten(),
  layers.Dense(64, activation='relu'),
  layers.Dropout(0.3), # Ezberlemeyi önle
  
  # Çıkış Katmanı (2 Sınıf: Lung vs Noise)
  layers.Dense(len(class_names), activation='softmax')
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])

model.summary()

# 3. Eğitimi Başlat
print("\n--- EĞİTİM BAŞLIYOR ---")
# 8 Epoch yeterli olacaktır (Çünkü farklar çok net)
epochs = 8 
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)

# 4. Sonuçları Çiz
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Eğitim Başarısı')
plt.plot(epochs_range, val_acc, label='Test Başarısı')
plt.legend(loc='lower right')
plt.title('Doğruluk (Accuracy)')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Eğitim Kaybı')
plt.plot(epochs_range, val_loss, label='Test Kaybı')
plt.legend(loc='upper right')
plt.title('Hata (Loss)')
plt.show()

# 5. Kaydet
model.save(kayit_yolu)
print(f"\n✅ MÜKEMMEL! Bekçi Modeli Kaydedildi: {kayit_yolu}")