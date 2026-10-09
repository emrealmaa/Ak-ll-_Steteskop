import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import random

# --- AYARLAR ---
# 1. Kaynak: 2000 adet ESC-50 Wav dosyasının olduğu yer
kaynak_klasor = r"C:\Users\merts\OneDrive\Masaüstü\gemini deneme\audio"

# 2. Hedef: Bekçi'nin eğitim seti
hedef_ana_klasor = r"C:\Users\merts\OneDrive\Masaüstü\gemini deneme\ISLENMIS_VERI_SETI\ESC-50"
# ---------------

# Noise klasörünü oluştur
hedef_noise_klasor = os.path.join(hedef_ana_klasor, "Noise")
if not os.path.exists(hedef_noise_klasor):
    os.makedirs(hedef_noise_klasor)

print("🔊 Gürültü Çoğaltma Operasyonu Başlatılıyor (x15)... 🚧")
print("Hedef: ~30.000 Gürültü Verisi")

# Efekt Fonksiyonları
def sese_efekt_uygula(data, sr):
    secim = random.choice([1, 2, 3, 4])
    
    if secim == 1: # Hız (Time Stretch)
        rate = np.random.uniform(low=0.8, high=1.2)
        data = librosa.effects.time_stretch(data, rate=rate)
        
    elif secim == 2: # Ton (Pitch Shift)
        n_steps = np.random.uniform(low=-4, high=4)
        data = librosa.effects.pitch_shift(data, sr=sr, n_steps=n_steps)
        
    elif secim == 3: # Ters Çevir (Reverse - Gürültü için uygundur)
        data = np.flip(data)
        
    else: # Gürültüye Gürültü Ekle (Daha kaotik olsun)
        noise_amp = 0.005 * np.random.uniform() * np.amax(data)
        data = data + noise_amp * np.random.normal(size=data.shape[0])
        
    return data

islenen_dosya = 0
toplam_yeni_resim = 0

# Klasördeki dosyaları gez
dosya_listesi = [f for f in os.listdir(kaynak_klasor) if f.endswith('.wav')]
print(f"Toplam Kaynak Dosya: {len(dosya_listesi)}")

for dosya_adi in dosya_listesi:
    tam_yol = os.path.join(kaynak_klasor, dosya_adi)
    
    try:
        # Sesi Yükle
        y_ana, sr = librosa.load(tam_yol, sr=None)
        
        # x15 Kat Çoğaltma Döngüsü
        TEKRAR_SAYISI = 15
        
        for i in range(TEKRAR_SAYISI):
            # Efekt Uygula
            y_yeni = sese_efekt_uygula(y_ana, sr)
            
            # 5 Saniyeye Ayarla
            hedef_uzunluk = 5 * sr
            if len(y_yeni) > hedef_uzunluk:
                y_yeni = y_yeni[:hedef_uzunluk]
            else:
                y_yeni = librosa.util.fix_length(y_yeni, size=hedef_uzunluk)
            
            # Dosya Adı: AUG_NOISE_...
            resim_adi = f"AUG_NOISE_{i}_{dosya_adi.replace('.wav', '')}.png"
            kayit_yolu = os.path.join(hedef_noise_klasor, resim_adi)
            
            # Eğer yoksa oluştur
            if not os.path.exists(kayit_yolu):
                plt.figure(figsize=(10, 10))
                
                # Spektrogram
                plt.subplot(2, 1, 1)
                D = librosa.stft(y_yeni)
                S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
                librosa.display.specshow(S_db, sr=sr, y_axis='log')
                plt.axis('off')
                
                # MFCC
                plt.subplot(2, 1, 2)
                mfccs = librosa.feature.mfcc(y=y_yeni, sr=sr, n_mfcc=13)
                librosa.display.specshow(mfccs, sr=sr)
                plt.axis('off')
                
                plt.tight_layout()
                plt.savefig(kayit_yolu, bbox_inches='tight', pad_inches=0)
                plt.close()
                
                toplam_yeni_resim += 1
        
        islenen_dosya += 1
        if islenen_dosya % 50 == 0:
            print(f"   ✅ {islenen_dosya} dosya çoğaltıldı...")
            
    except Exception as e:
        print(f"Hata: {dosya_adi} - {e}")

print("-" * 30)
print(f"🎉 GÜRÜLTÜ ORDUSU HAZIR!")
print(f"✅ Toplam Yeni Veri: {toplam_yeni_resim}")