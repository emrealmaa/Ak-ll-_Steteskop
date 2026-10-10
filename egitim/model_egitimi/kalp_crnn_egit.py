import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras import layers, models

# --- AYARLAR ---
# 1. 64.000 Resmin Olduğu Klasör
veri_klasoru = r"C:\Users\merts\OneDrive\Masaüstü\gemini deneme\KALP_EGITIM_GORSELLERI"

# 2. Modelin Kaydedileceği Yer
kayit_yolu = r"C:\Users\merts\OneDrive\Masaüstü\gemini deneme\kalp_modeli.h5"
# ---------------

# Resim Boyutları (Dönüştürme kodundakiyle aynı olmalı)
img_yukseklik = 128
img_genislik = 128
batch_boyutu = 64 # Güçlü bir eğitim için batch'i biraz artırdık

print("🫀 CRNN (CNN + LSTM) KALP MODELİ EĞİTİMİ BAŞLIYOR... 🚀")
print(f"TensorFlow Versiyonu: {tf.__version__}")

# GPU Kontrolü
if tf.config.list_physical_devices('GPU'):
    print("🔥 GPU Algılandı! Eğitim ateş gibi hızlı olacak.")
else:
    print("⚠️ GPU Bulunamadı. İşlemci (CPU) kullanılacak, sabır lazım.")

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
# Beklenen: ['Hasta', 'Normal']

# Performans Ayarı (Önbellekleme)
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# --- 2. CRNN MİMARİSİ KURULUYOR ---
print("\n--- CRNN Modeli İnşa Ediliyor... ---")

# Giriş Katmanı
inputs = layers.Input(shape=(img_yukseklik, img_genislik, 3))
x = layers.Rescaling(1./255)(inputs) # Normalizasyon (0-1 arasına çek)

# --- BÖLÜM 1: CNN (Gözler - Özellik Çıkarma) ---
# Katman 1
x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = layers.MaxPooling2D((2, 2))(x)
x = layers.Dropout(0.2)(x) # Aşırı öğrenmeyi engelle

# Katman 2
x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = layers.MaxPooling2D((2, 2))(x)
x = layers.Dropout(0.2)(x)

# Katman 3
x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = layers.MaxPooling2D((2, 2))(x)
x = layers.Dropout(0.2)(x)

# --- BÖLÜM 2: KÖPRÜ (Reshape) ---
# CNN çıktısını (Görüntü) LSTM girişine (Zaman Serisi) çeviriyoruz.
# Şu anki boyut: (Batch, 16, 16, 128) -> Hedef: (Batch, 16, 2048)
# Zaman eksenini (16) koruyup, diğerlerini özellik vektörü olarak birleştiriyoruz.
target_shape = (16, 16 * 128) 
x = layers.Reshape(target_shape=target_shape)(x)

# Ara Katman (Dense) - Boyutu biraz düşürelim ki LSTM boğulmasın
x = layers.Dense(64, activation='relu')(x) 

# --- BÖLÜM 3: RNN / LSTM (Hafıza - Ritmi Duyar) ---
# Bidirectional LSTM: Sesi hem baştan sona, hem sondan başa dinler.
# Bu, ritim bozukluklarını yakalamak için çok etkilidir.
x = layers.Bidirectional(layers.LSTM(64, return_sequences=False))(x)
x = layers.Dropout(0.3)(x)

# --- BÖLÜM 4: KARAR ---
# Çıkış Katmanı (2 Sınıf: Hasta vs Normal)
outputs = layers.Dense(len(class_names), activation='softmax')(x)

model = models.Model(inputs=inputs, outputs=outputs)

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])

model.summary()

# --- 3. EĞİTİMİ BAŞLAT ---
print("\n--- EĞİTİM BAŞLIYOR (CRNN) ---")
# Veri çok olduğu için 8 Epoch yeterli olacaktır.
# Her Epoch'ta 64.000 resim göreceği için öğrenme hızlı olur.
epochs = 8

history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)

# --- 4. GRAFİK ÇİZ ---
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
print(f"\n✅ TEBRİKLER! CRNN Kalp Modeli Kaydedildi: {kayit_yolu}")
print("Bu model, hem görsel desenleri hem de ritmik bozuklukları analiz edebilir!")