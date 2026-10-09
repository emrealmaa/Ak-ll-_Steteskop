import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# --- AYARLAR ---
# 1. ESC-50 Ses Dosyalarının Olduğu Yer
kaynak_klasor = r"C:\Users\merts\OneDrive\Masaüstü\gemini deneme\audio"

# 2. Resimlerin Kaydedileceği Yer
hedef_klasor = r"C:\Users\merts\OneDrive\Masaüstü\gemini deneme\ISLENMIS_VERI_SETI\ESC-50"
# ---------------

print("🔊 Gürültü Veri Seti İşleniyor (ESC-50)... 🚧")

if not os.path.exists(hedef_klasor):
    os.makedirs(hedef_klasor)

islenen = 0
hatali = 0

# Klasördeki ses dosyalarını gez
for dosya_adi in os.listdir(kaynak_klasor):
    if dosya_adi.endswith(".wav"):
        tam_yol = os.path.join(kaynak_klasor, dosya_adi)
        
        try:
            # Sesi Yükle
            y, sr = librosa.load(tam_yol, sr=None)
            
            # ESC-50 zaten 5 saniyelik dosyalardan oluşur.
            # Yine de garanti olsun diye 5 saniyeye sabitleyelim (Kesme/Padding)
            hedef_uzunluk = 5 * sr
            if len(y) > hedef_uzunluk:
                y = y[:hedef_uzunluk]
            else:
                y = librosa.util.fix_length(y, size=hedef_uzunluk)
            
            # Resim İsmi
            resim_adi = f"{dosya_adi.replace('.wav', '')}.png"
            kayit_yolu = os.path.join(hedef_klasor, resim_adi)
            
            # Eğer yoksa oluştur
            if not os.path.exists(kayit_yolu):
                plt.figure(figsize=(10, 10))
                
                # 1. Spektrogram
                plt.subplot(2, 1, 1)
                D = librosa.stft(y)
                S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
                librosa.display.specshow(S_db, sr=sr, y_axis='log')
                plt.axis('off')
                
                # 2. MFCC
                plt.subplot(2, 1, 2)
                mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
                librosa.display.specshow(mfccs, sr=sr)
                plt.axis('off')
                
                plt.tight_layout()
                plt.savefig(kayit_yolu, bbox_inches='tight', pad_inches=0)
                plt.close()
                
                islenen += 1
                if islenen % 100 == 0:
                    print(f"   ✅ {islenen} adet gürültü işlendi...")
                    
        except Exception as e:
            print(f"Hata: {dosya_adi} - {e}")
            hatali += 1

print("-" * 30)
print(f"🎉 İŞLEM TAMAMLANDI!")
print(f"✅ Oluşturulan Gürültü Resmi: {islenen}")
print(f"❌ Hatalı Dosya: {hatali}")
print(f"📂 Kayıt Yeri: {hedef_klasor}")