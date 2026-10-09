import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal
import math
import random

# --- AYARLAR ---
# 1. Kaynak: Ses dosyalarının olduğu yer (.wav dosyaları)
KAYNAK_KLASOR = r"C:\Users\merts\OneDrive\Masaüstü\gemini deneme\KALP_VERI_SETI_FINAL"

# 2. Hedef: Resimlerin çıkacağı yer (Mevcut resimlerin üzerine ekleyecek)
HEDEF_KLASOR = r"C:\Users\merts\OneDrive\Masaüstü\gemini deneme\KALP_MFCC_SPEKTROGRAM"
# ---------------

print("🫀 Kalp Sesleri İçin 'Data Augmentation' (Veri Çoğaltma) Başlıyor... 🚀")
print("Hedef: Her sınıf için ~30.000 Görüntü")

# --- DİJİTAL FİLTRE ---
def temizle_filtrele(y, sr):
    nyquist = sr / 2
    ust_sinir = 1000 if nyquist > 1000 else nyquist - 10
    if ust_sinir <= 20: return y
    try:
        b, a = signal.butter(4, [20, ust_sinir], btype='bandpass', fs=sr)
        return signal.filtfilt(b, a, y)
    except: return y 

# --- EFEKTLER ---
def efekt_uygula(y, sr, tur):
    if tur == 1: # Hız Değişimi (Time Stretch)
        rate = np.random.uniform(0.8, 1.2) # %20 yavaş veya hızlı
        return librosa.effects.time_stretch(y, rate=rate)
    
    elif tur == 2: # Ton Değişimi (Pitch Shift)
        steps = np.random.uniform(-2, 2) # Kalınlaştır veya incelt
        return librosa.effects.pitch_shift(y, sr=sr, n_steps=steps)
    
    elif tur == 3: # Hafif Gürültü (Noise Injection)
        noise_amp = 0.005 * np.random.uniform() * np.amax(y)
        return y + noise_amp * np.random.normal(size=y.shape[0])
        
    return y

# --- İŞLEME MOTORU ---
def cogalt_ve_olustur(sinif_adi):
    kaynak_yol = os.path.join(KAYNAK_KLASOR, sinif_adi)
    hedef_yol = os.path.join(HEDEF_KLASOR, sinif_adi)
    
    if not os.path.exists(kaynak_yol): return

    print(f"\n📂 {sinif_adi} klasörü çoğaltılıyor (x3 Augmentation)...")
    
    dosyalar = [f for f in os.listdir(kaynak_yol) if f.endswith(".wav")]
    toplam_uretilen = 0
    
    for dosya in dosyalar:
        tam_yol = os.path.join(kaynak_yol, dosya)
        
        try:
            y_ham, sr = librosa.load(tam_yol, sr=None)
            y_temiz = temizle_filtrele(y_ham, sr)
            
            # 3 Farklı Efekt Türü İçin Döngü
            for efekt_turu in [1, 2, 3]: 
                
                # Efekti uygula
                y_efektli = efekt_uygula(y_temiz, sr, efekt_turu)
                
                # Dilimleme (Slicing)
                parca_uzunlugu = 5 * sr
                toplam_parca = math.ceil(len(y_efektli) / parca_uzunlugu)
                if toplam_parca == 0: toplam_parca = 1
                
                for i in range(toplam_parca):
                    baslangic = i * parca_uzunlugu
                    bitis = baslangic + parca_uzunlugu
                    
                    if bitis > len(y_efektli): y_parca = y_efektli[baslangic:]
                    else: y_parca = y_efektli[baslangic:bitis]
                    
                    if len(y_parca) < parca_uzunlugu:
                        y_parca = librosa.util.fix_length(y_parca, size=parca_uzunlugu)
                    
                    # Dosya Adı: AUG1_DosyaAdi_Part0.png
                    resim_adi = f"AUG{efekt_turu}_{dosya.replace('.wav', '')}_P{i}.png"
                    kayit_yolu = os.path.join(hedef_yol, resim_adi)
                    
                    if not os.path.exists(kayit_yolu):
                        plt.figure(figsize=(10, 10))
                        plt.subplot(2, 1, 1)
                        librosa.display.specshow(librosa.amplitude_to_db(np.abs(librosa.stft(y_parca)), ref=np.max), sr=sr, y_axis='log')
                        plt.axis('off')
                        plt.subplot(2, 1, 2)
                        librosa.display.specshow(librosa.feature.mfcc(y=y_parca, sr=sr, n_mfcc=13), sr=sr)
                        plt.axis('off')
                        plt.tight_layout()
                        plt.savefig(kayit_yolu, bbox_inches='tight', pad_inches=0)
                        plt.close('all')
                        
                        toplam_uretilen += 1
            
            if toplam_uretilen % 200 == 0:
                print(f"   ⚡ {sinif_adi}: {toplam_uretilen} yeni veri üretildi...")

        except Exception as e:
            print(f"Hata: {dosya} - {e}")
            
    print(f"✅ {sinif_adi} Bitti. Toplam {toplam_uretilen} YENİ veri eklendi.")

# --- BAŞLAT ---
cogalt_ve_olustur("Normal")
cogalt_ve_olustur("Hasta")

print("-" * 30)
print("🎉 ÇOĞALTMA İŞLEMİ TAMAMLANDI!")
print("Artık elinde devasa bir Kalp Veri Seti var. Eğitime hazır!")