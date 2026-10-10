import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# --- AYARLAR (Senin İstediğin Yollar) ---
# 1. Kaynak Klasör (Ses dosyalarının olduğu yer)
kaynak_klasor = r"C:\Users\merts\OneDrive\Masaüstü\archive2\pneumonia"

# 2. Hedef Klasör (Resimlerin kaydedileceği yer)
hedef_klasor = r"C:\Users\merts\OneDrive\Masaüstü\ISLENMIS_VERI_SETI\Pneumonia"
# ----------------------------------------

print("Astım Verileri için İşlem Başlatılıyor (_A2 Etiketli)... 🚀")

# Hedef klasör yoksa oluştur
if not os.path.exists(hedef_klasor):
    os.makedirs(hedef_klasor)
    print(f"📁 Hedef klasör oluşturuldu: {hedef_klasor}")

islenen_dosya_sayisi = 0
toplam_resim_sayisi = 0

# Kaynak klasördeki dosyaları tara
if not os.path.exists(kaynak_klasor):
    print(f"❌ HATA: Kaynak klasör bulunamadı! Lütfen yolu kontrol et: {kaynak_klasor}")
    exit()

for dosya_adi in os.listdir(kaynak_klasor):
    if dosya_adi.endswith(".wav"):
        
        tam_yol = os.path.join(kaynak_klasor, dosya_adi)
        
        try:
            # Sesi Yükle
            y, sr = librosa.load(tam_yol, sr=None)
            
            # Parçalara Bölme Hesabı (5 Saniyelik)
            toplam_sure = librosa.get_duration(y=y, sr=sr)
            parca_suresi = 5
            
            # Dosya kısaysa en az 1 parça, uzunsa böl
            if toplam_sure < parca_suresi:
                parca_sayisi = 1
            else:
                parca_sayisi = int(toplam_sure // parca_suresi)
                
            for i in range(parca_sayisi):
                baslangic = i * parca_suresi
                bitis = (i + 1) * parca_suresi
                
                # --- Kesme veya Doldurma (Padding) ---
                if toplam_sure < parca_suresi:
                    # Kısa dosyayı 5 saniyeye tamamla
                    y_parca = y
                    hedef_uzunluk = int(parca_suresi * sr)
                    y_parca = librosa.util.fix_length(y_parca, size=hedef_uzunluk)
                else:
                    # Uzun dosyayı kes
                    y_parca = y[int(baslangic*sr) : int(bitis*sr)]
                
                # --- İsimlendirme (_A2 Etiketi) ---
                # Örn: dosyaadi_A2_parca0.png
                yeni_isim = f"{dosya_adi.replace('.wav', '')}_A2_parca{i}.png"
                kayit_yolu = os.path.join(hedef_klasor, yeni_isim)
                
                # Eğer resim zaten varsa tekrar yapma (Hız kazandırır)
                if not os.path.exists(kayit_yolu):
                    plt.figure(figsize=(10, 10))
                    
                    # 1. Spektrogram (Üst)
                    plt.subplot(2, 1, 1)
                    D = librosa.stft(y_parca)
                    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
                    librosa.display.specshow(S_db, sr=sr, y_axis='log')
                    plt.axis('off')
                    
                    # 2. MFCC (Alt)
                    plt.subplot(2, 1, 2)
                    mfccs = librosa.feature.mfcc(y=y_parca, sr=sr, n_mfcc=13)
                    librosa.display.specshow(mfccs, sr=sr)
                    plt.axis('off')
                    
                    plt.tight_layout()
                    plt.savefig(kayit_yolu, bbox_inches='tight', pad_inches=0)
                    plt.close() # Hafızayı temizle
                    
                    toplam_resim_sayisi += 1
            
            islenen_dosya_sayisi += 1
            if islenen_dosya_sayisi % 10 == 0:
                print(f"   ✅ {islenen_dosya_sayisi} dosya işlendi...")

        except Exception as e:
            print(f"   ⚠️ Hata oluştu: {dosya_adi} - {e}")

print("-" * 30)
print(f"🎉 İŞLEM TAMAMLANDI!")
print(f"📂 Kaynak: {kaynak_klasor}")
print(f"📂 Hedef: {hedef_klasor}")
print(f"✅ İşlenen Ses Dosyası: {islenen_dosya_sayisi}")
print(f"🖼️ Oluşturulan Yeni Resim: {toplam_resim_sayisi}")