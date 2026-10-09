import os
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# --- AYARLAR ---
model_yolu = r"C:\Users\merts\OneDrive\Masaüstü\gemini deneme\akilli_steteskop_final_model.h5"
test_edilecek_ses = r"C:\Users\merts\OneDrive\Masaüstü\test.wav"
# ----------------

SINIFLAR = ['Asthma', 'Bronchial_Disease', 'COPD', 'Healthy', 'Pneumonia']

print("🩺 AKILLI STETESKOP: Kümülatif Tanı Modülü Başlatılıyor...\n")

try:
    model = tf.keras.models.load_model(model_yolu)
except Exception as e:
    print(f"❌ Model Yüklenemedi: {e}")
    exit()

def kümülatif_test(ses_yolu):
    if not os.path.exists(ses_yolu):
        print("❌ Dosya bulunamadı.")
        return

    try:
        # 1. Orijinal Hızda Yükle (Doğru bilen yöntem)
        y, sr = librosa.load(ses_yolu, sr=None)
        
        toplam_sure = librosa.get_duration(y=y, sr=sr)
        parca_suresi = 5
        
        if toplam_sure < parca_suresi: parca_sayisi = 1
        else: parca_sayisi = int(toplam_sure // parca_suresi)
        
        print(f"ℹ️  Analiz: {os.path.basename(ses_yolu)} ({toplam_sure:.1f} sn)")
        print("-" * 50)
        
        # Her sınıfın puan kumbarası
        # [Asthma_Puan, Bronchial_Puan, COPD_Puan...]
        toplam_skorlar = np.zeros(len(SINIFLAR))
        
        for i in range(parca_sayisi):
            # Kesme
            if toplam_sure < parca_suresi:
                y_parca = librosa.util.fix_length(y, size=int(parca_suresi*sr))
            else:
                y_parca = y[int(i*parca_suresi*sr) : int((i+1)*parca_suresi*sr)]
            
            # Resim (Eğitimdeki ile aynı)
            plt.figure(figsize=(10, 10))
            plt.subplot(2, 1, 1)
            D = librosa.stft(y_parca)
            librosa.display.specshow(librosa.amplitude_to_db(np.abs(D), ref=np.max), sr=sr, y_axis='log')
            plt.axis('off')
            plt.subplot(2, 1, 2)
            librosa.display.specshow(librosa.feature.mfcc(y=y_parca, sr=sr, n_mfcc=13), sr=sr)
            plt.axis('off')
            plt.tight_layout()
            plt.savefig("temp_score.png", bbox_inches='tight', pad_inches=0)
            plt.close()
            
            # Tahmin
            img = image.load_img("temp_score.png", target_size=(128, 128))
            img_array = np.expand_dims(image.img_to_array(img), axis=0)
            
            tahmin = model.predict(img_array, verbose=0)
            skorlar = tf.nn.softmax(tahmin[0]).numpy()
            
            # Puanları Kumbaraya Ekle! (Ortalama almıyoruz, topluyoruz)
            toplam_skorlar += skorlar
            
            en_yuksek_anlik = np.argmax(skorlar)
            print(f"   ⏱️  Parça {i+1}: {SINIFLAR[en_yuksek_anlik]} (+{skorlar[en_yuksek_anlik]:.2f} puan)")

        # --- FİNAL KARAR ---
        # En çok puan toplayan kazanır
        kazanan_index = np.argmax(toplam_skorlar)
        kazanan_sinif = SINIFLAR[kazanan_index]
        toplam_puan = toplam_skorlar[kazanan_index]
        
        # Toplam puanı, parça sayısına bölerek "Genel Güven" bulalım
        # Ama bu sefer diğer sınıflarla aradaki FARKA bakalım.
        
        print("=" * 50)
        print(f"🩺 TEŞHİS: {kazanan_sinif.upper()}")
        print(f"🏆 Toplam Skor: {toplam_puan:.2f} / {parca_sayisi}.00")
        print("-" * 50)
        print("Puan Tablosu:")
        for i, sinif in enumerate(SINIFLAR):
            # Bar grafiği gibi gösterelim
            bar = "█" * int(toplam_skorlar[i] * 5)
            print(f"   - {sinif:<20}: {toplam_skorlar[i]:.2f} {bar}")
            
        if os.path.exists("temp_score.png"): os.remove("temp_score.png")

    except Exception as e:
        print(f"Hata: {e}")

# Çalıştır
kümülatif_test(test_edilecek_ses)