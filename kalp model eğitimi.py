import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

# --- AYARLAR ---
# 1. 90.000 Resmin Olduğu Klasör
veri_klasoru = r"C:\Users\merts\OneDrive\Masaüstü\gemini deneme\KALP_MFCC_SPEKTROGRAM"

# 2. Modelin Kaydedileceği Yer
kayit_yolu = r"C:\Users\merts\OneDrive\Masaüstü\kalp_modeli.h5"
# ---------------

# Resim Boyutları
img_yukseklik = 128
img_genislik = 128
batch_boyutu = 32 

print("🫀 KALP MODELİ EĞİTİMİ BAŞLIYOR... 🚀")
print(f"TensorFlow Versiyonu: {tf.__version__}")

# GPU Kontrolü
if tf.config.list_physical_devices('GPU'):
    print("🔥 GPU Algılandı! Eğitim ateş gibi hızlı olacak.")
else:
    print("⚠️ GPU Bulunamadı. İşlemci (CPU) kullanılacak, biraz sabır lazım.")

# --- 1. VERİLERİ YÜKLE (%80 Eğitim, %20 Test) ---
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
# Beklenen: ['Hasta', 'Normal'] (Alfabetik sıraya göre H önce gelir)

# Performans Ayarı (Hızlandırıcı)
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# --- 2. MODELİ KUR (Derin CNN Mimarisi) ---
# Veri çok olduğu için modeli biraz derinleştirdik (Daha zeki olsun diye)
print("\n--- Kalp Modeli İnşa Ediliyor... ---")

model = Sequential([
  layers.Rescaling(1./255, input_shape=(img_yukseklik, img_genislik, 3)),
  
  # Blok 1
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Dropout(0.2), 
  
  # Blok 2
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Dropout(0.2),

  # Blok 3
  layers.Conv2D(128, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Dropout(0.3),
  
  # Blok 4 (Ekstra Derinlik)
  layers.Conv2D(256, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Dropout(0.4),
  
  layers.Flatten(),
  layers.Dense(256, activation='relu'),
  layers.Dropout(0.5), # Ezberlemeyi önlemek için güçlü dropout
  
  # Çıkış Katmanı (2 Sınıf: Hasta vs Normal)
  layers.Dense(len(class_names), activation='softmax')
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])

model.summary()

# --- 3. EĞİTİMİ BAŞLAT ---
print("\n--- EĞİTİM BAŞLIYOR ---")
# Veri çok olduğu için 6-8 Epoch bile yeterli olabilir ama
# En iyi sonucu almak için 10 yapalım.
epochs = 10

history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)

# --- 4. SONUÇLARI GÖSTER ---
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

# --- 5. KAYDET ---
model.save(kayit_yolu)
print(f"\n✅ TEBRİKLER! Kalp Modeli Başarıyla Kaydedildi: {kayit_yolu}")
print("Artık 'KALP_MFCC_SPEKTROGRAM' klasörünü silebilirsin! 🗑️")