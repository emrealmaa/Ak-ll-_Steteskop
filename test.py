import os
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# --- AYARLAR ---
# 1. Eğittiğin Final Modelin Yeri (Dosya adının doğru olduğundan emin ol)
model_yolu = r"C:\Users\merts\OneDrive\Masaüstü\gemini deneme\akilli_steteskop_final_model.h5"

# 2. Test Etmek İstediğin Ses Dosyası (Gerçek bir dosya yolu yapıştır)
test_edilecek_ses = r"C:\Users\merts\OneDrive\Masaüstü\test.wav"
# ----------------

# GÜNCELLENMİŞ SINIF LİSTESİ (Alfabetik Sıra - 5 Sınıf)
SINIFLAR = ['Asthma', 'Bronchial_Disease', 'COPD', 'Healthy', 'Pneumonia']

print("🩺 Yapay Zeka Doktoru (v2.0) Başlatılıyor...")

# 1. Modeli Yükle
try:
    model = tf.keras.models.load_model(model_yolu)
    print("✅ Model başarıyla yüklendi.")
except Exception as e:
    print(f"❌ Model yüklenemedi: {e}")
    exit()

def sesi_tahmin_et(ses_yolu):
    print(f"\n🔍 Analiz Edilen Dosya: {os.path.basename(ses_yolu)}")
    
    if not os.path.exists(ses_yolu):
        print("❌ HATA: Belirttiğin ses dosyası bulunamadı! Yolu kontrol et.")
        return

    # --- ADIM 1: SESİ GÖRÜNTÜYE ÇEVİR ---
    try:
        # Sesi Yükle
        y, sr = librosa.load(ses_yolu, sr=None)
        
        # 5 saniyeye ayarla (Kesme veya Padding)
        hedef_uzunluk = 5 * sr
        if len(y) > hedef_uzunluk:
            y = y[:hedef_uzunluk]
        else:
            y = librosa.util.fix_length(y, size=hedef_uzunluk)
            
        # Hibrit Resim Oluştur (Eğitimdeki ile birebir aynı)
        plt.figure(figsize=(10, 10))
        
        # Spektrogram
        plt.subplot(2, 1, 1)
        D = librosa.stft(y)
        S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
        librosa.display.specshow(S_db, sr=sr, y_axis='log')
        plt.axis('off')
        
        # MFCC
        plt.subplot(2, 1, 2)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        librosa.display.specshow(mfccs, sr=sr)
        plt.axis('off')
        
        plt.tight_layout()
        
        # Resmi geçici olarak kaydet
        gecici_resim = "temp_test_image.png"
        plt.savefig(gecici_resim, bbox_inches='tight', pad_inches=0)
        plt.close()
        
        # --- ADIM 2: TAHMİN ET ---
        # Resmi yükle (128x128)
        img = image.load_img(gecici_resim, target_size=(128, 128))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) # Batch boyutu ekle
        
        # Modele sor
        tahminler = model.predict(img_array)
        skorlar = tf.nn.softmax(tahminler[0])
        
        # --- SONUCU YAZDIR ---
        en_yuksek_skor_index = np.argmax(skorlar)
        tahmin_edilen_hastalik = SINIFLAR[en_yuksek_skor_index]
        guven_orani = 100 * np.max(skorlar)
        
        print("-" * 40)
        print(f"🩺 TANI SONUCU: {tahmin_edilen_hastalik.upper()}")
        print(f"📊 Güven Oranı: %{guven_orani:.2f}")
        print("-" * 40)
        
        print("Detaylı Olasılıklar:")
        for i, olasilik in enumerate(skorlar):
            print(f"   - {SINIFLAR[i]:<20}: %{100*olasilik:.2f}")
            
        # Temizlik
        if os.path.exists(gecici_resim):
            os.remove(gecici_resim)
            
    except Exception as e:
        print(f"Hata oluştu: {e}")

# Fonksiyonu Çalıştır
sesi_tahmin_et(test_edilecek_ses)