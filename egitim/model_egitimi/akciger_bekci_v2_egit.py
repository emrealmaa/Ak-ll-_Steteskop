import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras import layers, models

# --- AYARLAR ---
# Bekçi verilerinin olduğu klasör
veri_klasoru = r"C:\Users\merts\OneDrive\Masaüstü\gemini deneme\BEKCI_VERI_SETI"

# Modelin kaydedileceği yer
kayit_yolu = r"C:\Users\merts\OneDrive\Masaüstü\bekci_modeli.h5"
# ---------------

img_yukseklik = 128
img_genislik = 128
batch_boyutu = 32

print("🛡️ BEKÇİ (GATEKEEPER) MODELİ EĞİTİLİYOR... 🚀")

# --- VERİ YÜKLEME ---
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
# Beklenen: ['Hearth_Sound', 'Noise'] (Senin klasör isimlerin neyse o çıkar)

# Performans
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# --- BASİT CNN MİMARİSİ (Bekçi İçin Yeterli) ---
# Bekçi'nin işi "Var/Yok" demek olduğu için çok derin bir ağa gerek yok.
# Hızlı ve hafif olması daha iyidir.

model = models.Sequential([
  layers.Rescaling(1./255, input_shape=(img_yukseklik, img_genislik, 3)),
  
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(len(class_names), activation='softmax')
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])

model.summary()

# --- EĞİTİM ---
print("\n--- BEKÇİ EĞİTİME BAŞLIYOR ---")
# Bekçi görevi basit olduğu için 5 Epoch yeter de artar.
epochs = 5 

history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)

# --- KAYDET ---
model.save(kayit_yolu)
print(f"\n✅ BEKÇİ MODELİ HAZIR: {kayit_yolu}")