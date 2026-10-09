import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal
import random

# --- AYARLAR ---
KAYNAK_KLASOR = r"C:\Users\merts\OneDrive\Masaüstü\gemini deneme\KALP_VERI_SETI_FINAL"
HEDEF_KLASOR = r"C:\Users\merts\OneDrive\Masaüstü\gemini deneme\KALP_EGITIM_GORSELLERI"
# ---------------

print("🫀 KALP VERİSİ HAZIRLANIYOR... (MOD: PRO / OVERLAP + BALANCE) 🚧")

hedef_normal = os.path.join(HEDEF_KLASOR, "Normal")
hedef_hasta = os.path.join(HEDEF_KLASOR, "Hasta")

if not os.path.exists(hedef_normal): os.makedirs(hedef_normal)
if not os.path.exists(hedef_hasta): os.makedirs(hedef_hasta)

# --- FİLTRE ---
def temizle_filtrele(y, sr):
    nyquist = sr / 2
    ust_sinir = 1000 if nyquist > 1000 else nyquist - 10
    if ust_sinir <= 20: return y
    try:
        b, a = signal.butter(4, [20, ust_sinir], btype='bandpass', fs=sr)
        return signal.filtfilt(b, a, y)
    except: return y

# --- EFEKT (HASTA VERİSİ İÇİN) ---
def efekt_uygula(y, sr):
    secim = random.choice([1, 2])
    if secim == 1: # Hız
        rate = np.random.uniform(0.9, 1.1)
        return librosa.effects.time_stretch(y, rate=rate)
    else: # Ton
        steps = np.random.uniform(-1, 1)
        return librosa.effects.pitch_shift(y, sr=sr, n_steps=steps)
    return y

# --- GÖRSEL KAYDET ---
def resim_kaydet(y, sr, yol):
    plt.figure(figsize=(5, 5))
    # Sadece Spektrogram (CRNN için en temiz veri budur)
    D = librosa.stft(y)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    librosa.display.specshow(S_db, sr=sr, y_axis='log', cmap='magma')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(yol, bbox_inches='tight', pad_inches=0)
    plt.close('all')

# --- İŞLEME MOTORU ---
def islem_yap(sinif_adi, hedef_yol, augment_yap=False):
    kaynak_yol = os.path.join(KAYNAK_KLASOR, sinif_adi)
    print(f"\n📂 {sinif_adi} işleniyor... [Augmentation: {'VAR' if augment_yap else 'YOK'}]")
    
    dosyalar = [f for f in os.listdir(kaynak_yol) if f.endswith(".wav")]
    sayac = 0
    
    for dosya in dosyalar:
        tam_yol = os.path.join(kaynak_yol, dosya)
        try:
            y_ham, sr = librosa.load(tam_yol, sr=None)
            y = temizle_filtrele(y_ham, sr)
            
            # --- ÖRTÜŞMELİ DİLİMLEME (OVERLAPPING) ---
            # 5 saniyelik pencere, 2.5 saniyelik kayma (Step)
            parca_len = 5 * sr
            adim = int(2.5 * sr) 
            
            # Eğer dosya 5 sn'den kısaysa en az 1 kez dönsün
            if len(y) < parca_len:
                y = librosa.util.fix_length(y, size=parca_len)
            
            # Kayar Pencere Döngüsü
            for start in range(0, len(y) - parca_len + 1, adim):
                y_parca = y[start : start + parca_len]
                
                # 1. Orijinal Parçayı Kaydet
                resim_adi = f"{dosya[:-4]}_T{start}.png"
                resim_kaydet(y_parca, sr, os.path.join(hedef_yol, resim_adi))
                sayac += 1
                
                # 2. Eğer HASTA ise ve Augment Açıksa -> Çoğalt
                if augment_yap:
                    y_aug = efekt_uygula(y_parca, sr)
                    # Uzunluk bozulduysa düzelt
                    if len(y_aug) != parca_len:
                        y_aug = librosa.util.fix_length(y_aug, size=parca_len)
                        
                    resim_adi_aug = f"{dosya[:-4]}_T{start}_AUG.png"
                    resim_kaydet(y_aug, sr, os.path.join(hedef_yol, resim_adi_aug))
                    sayac += 1

            if sayac % 500 == 0: print(f"   📸 {sayac} resim üretildi...")

        except Exception as e:
            print(f"Hata: {dosya} -> {e}")
            
    print(f"✅ {sinif_adi} Tamamlandı. Toplam: {sayac} resim.")

# --- BAŞLAT ---
# Normal: Sadece Overlap yap (Zaten çok var)
islem_yap("Normal", hedef_normal, augment_yap=False)

# Hasta: Overlap + Augmentation yap (Sayısı azdı, şimdi yetişecek)
islem_yap("Hasta", hedef_hasta, augment_yap=True)

print("-" * 30)
print("🎉 VERİ SETİ MÜKEMMEL HALE GETİRİLDİ!")