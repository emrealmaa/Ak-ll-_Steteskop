import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import random

# --- AYARLAR ---
# 1. Kaynak (WAV dosyalarının olduğu yer)
kaynak_klasor = r"C:\Users\merts\OneDrive\Masaüstü\gemini deneme\archive"

# 2. Hedef (Resimlerin olduğu yer)
hedef_klasor = r"C:\Users\merts\OneDrive\Masaüstü\gemini deneme\ISLENMIS_VERI_SETI"
# ---------------

# --- SADECE EKSİK KALANLAR İÇİN ÇARPANLAR ---
# Hedef: Hepsini ~3000 bandına çekmek.
COGALTIM_KATSAYILARI = {
    'Asthma': 0,  #23        # ~289 wav * 10 = ~2890 yeni
    'Pneumonia': 0,  #19      # ~322 wav * 8  = ~2576 yeni
    'Healthy': 0,         # ~168 wav * 16 = ~2688 yeni
    'Bronchial': 0, #45
    'COPD': 0   #4
}

print("Telafi Operasyonu Başlatılıyor (Asthma, Pneumonia, Healthy)... 🚀")

def sese_efekt_uygula(data, sr):
    secim = random.choice([1, 2, 3, 4]) 
    if secim == 1: # Gürültü
        noise_amp = 0.005 * np.random.uniform() * np.amax(data)
        data = data + noise_amp * np.random.normal(size=data.shape[0])
    elif secim == 2: # Hız
        rate = np.random.uniform(low=0.9, high=1.2)
        data = librosa.effects.time_stretch(data, rate=rate)
    elif secim == 3: # Ton
        n_steps = np.random.uniform(low=-2, high=2)
        data = librosa.effects.pitch_shift(data, sr=sr, n_steps=n_steps)
    else: # Kombinasyon
        data = librosa.effects.time_stretch(data, rate=0.95)
        data = librosa.effects.pitch_shift(data, sr=sr, n_steps=1)
    return data

islenen_dosya = 0

# Ana Klasörü Gez
for hastalik_adi in os.listdir(kaynak_klasor):
    hastalik_yolu = os.path.join(kaynak_klasor, hastalik_adi)
    
    # Sadece hedeflediğimiz hastalıkları işle
    if os.path.isdir(hastalik_yolu) and hastalik_adi in COGALTIM_KATSAYILARI:
        
        tekrar_sayisi = COGALTIM_KATSAYILARI[hastalik_adi]
        
        if tekrar_sayisi == 0:
            continue # Zaten tamam olanları atla
            
        print(f"🔄 {hastalik_adi} işleniyor (x{tekrar_sayisi} kat çoğaltılacak)...")
        
        hedef_hastalik_yolu = os.path.join(hedef_klasor, hastalik_adi)
        if not os.path.exists(hedef_hastalik_yolu): os.makedirs(hedef_hastalik_yolu)
        
        for dosya_adi in os.listdir(hastalik_yolu):
            if dosya_adi.endswith(".wav"):
                tam_yol = os.path.join(hastalik_yolu, dosya_adi)
                
                try:
                    y_ana, sr = librosa.load(tam_yol, sr=None)
                    
                    for i in range(tekrar_sayisi):
                        y_yeni = sese_efekt_uygula(y_ana, sr)
                        
                        hedef_uzunluk = 5 * sr
                        if len(y_yeni) > hedef_uzunluk:
                            y_yeni = y_yeni[:hedef_uzunluk]
                        else:
                            y_yeni = librosa.util.fix_length(y_yeni, size=hedef_uzunluk)
                        
                        # İsimlendirme: AUG_TELAFI_
                        yeni_isim = f"AUG_TELAFI_{i}_{dosya_adi.replace('.wav', '')}.png"
                        kayit_yolu = os.path.join(hedef_hastalik_yolu, yeni_isim)
                        
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
                            
                    islenen_dosya += 1
                    if islenen_dosya % 50 == 0:
                        print(f"   ⚡ {islenen_dosya} dosya tamamlandı...")
                        
                except Exception as e:
                    print(f"Hata: {dosya_adi} - {e}")

print("-" * 30)
print(f"🎉 TELAFİ İŞLEMİ BİTTİ! Eksik sınıflar tamamlandı.")