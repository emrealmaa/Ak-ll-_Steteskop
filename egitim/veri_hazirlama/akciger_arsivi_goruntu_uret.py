import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# --- AYARLAR (Senin Yeni Birleştirdiğin Klasör) ---
# 1. Kaynak: Tüm seslerin (Arşiv 1 + 2) toplandığı yer
kaynak_klasor = r"C:\Users\merts\OneDrive\Masaüstü\gemini deneme\archive"

# 2. Hedef: Resimlerin kaydedileceği yer
hedef_klasor = r"C:\Users\merts\OneDrive\Masaüstü\gemini deneme\ISLENMIS_VERI_SETI"
# -------------------------------------------------

print("Büyük Dönüştürme Fabrikası Başlatılıyor... 🏭")
print(f"Kaynak: {kaynak_klasor}")
print(f"Hedef: {hedef_klasor}")

if not os.path.exists(hedef_klasor):
    os.makedirs(hedef_klasor)

toplam_islenen_dosya = 0
toplam_resim_sayisi = 0

# Kaynak klasördeki alt klasörleri (Hastalıkları) gez
for hastalik_adi in os.listdir(kaynak_klasor):
    hastalik_yolu = os.path.join(kaynak_klasor, hastalik_adi)
    
    # Sadece klasör olanları işle (Dosya varsa atla)
    if os.path.isdir(hastalik_yolu):
        
        # Hedefte de aynı hastalık klasörünü aç
        hedef_hastalik_yolu = os.path.join(hedef_klasor, hastalik_adi)
        if not os.path.exists(hedef_hastalik_yolu):
            os.makedirs(hedef_hastalik_yolu)
            
        print(f"\n📂 İşleniyor: {hastalik_adi} klasörü...")
        
        dosya_sayaci = 0
        
        for dosya_adi in os.listdir(hastalik_yolu):
            if dosya_adi.endswith(".wav"):
                
                tam_yol = os.path.join(hastalik_yolu, dosya_adi)
                
                try:
                    # 1. Sesi Yükle
                    y, sr = librosa.load(tam_yol, sr=None)
                    
                    # 2. Süreyi Hesapla ve Parça Sayısını Bul
                    toplam_sure = librosa.get_duration(y=y, sr=sr)
                    parca_suresi = 5 # 5 saniyelik dilimler
                    
                    # Eğer dosya 5 saniyeden kısaysa en az 1 parça çıkar
                    if toplam_sure < parca_suresi:
                        parca_sayisi = 1
                    else:
                        parca_sayisi = int(toplam_sure // parca_suresi)
                    
                    for i in range(parca_sayisi):
                        baslangic = i * parca_suresi
                        bitis = (i + 1) * parca_suresi
                        
                        # 3. Kesme veya Doldurma (Padding)
                        if toplam_sure < parca_suresi:
                            y_parca = y
                            hedef_uzunluk = int(parca_suresi * sr)
                            y_parca = librosa.util.fix_length(y_parca, size=hedef_uzunluk)
                        else:
                            y_parca = y[int(baslangic*sr) : int(bitis*sr)]
                        
                        # 4. Dosya İsmi Oluştur (Orijinal isim + Parça No)
                        # Örn: 101_1b1..._parca0.png VEYA P1Asthma..._parca0.png
                        yeni_isim = f"{dosya_adi.replace('.wav', '')}_parca{i}.png"
                        kayit_yolu = os.path.join(hedef_hastalik_yolu, yeni_isim)
                        
                        # Eğer resim zaten varsa atla (Zaman kazanmak için)
                        if not os.path.exists(kayit_yolu):
                            
                            # --- HİBRİT RESİM OLUŞTURMA ---
                            plt.figure(figsize=(10, 10))
                            
                            # Spektrogram
                            plt.subplot(2, 1, 1)
                            D = librosa.stft(y_parca)
                            S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
                            librosa.display.specshow(S_db, sr=sr, y_axis='log')
                            plt.axis('off')
                            
                            # MFCC
                            plt.subplot(2, 1, 2)
                            mfccs = librosa.feature.mfcc(y=y_parca, sr=sr, n_mfcc=13)
                            librosa.display.specshow(mfccs, sr=sr)
                            plt.axis('off')
                            
                            plt.tight_layout()
                            plt.savefig(kayit_yolu, bbox_inches='tight', pad_inches=0)
                            plt.close()
                            
                            toplam_resim_sayisi += 1
                            
                    dosya_sayaci += 1
                    toplam_islenen_dosya += 1
                    
                    if dosya_sayaci % 50 == 0:
                        print(f"   ✅ {dosya_sayaci} dosya tamamlandı...")

                except Exception as e:
                    print(f"   ⚠️ Hata: {dosya_adi} - {e}")

print("-" * 30)
print(f"🎉 TÜM İŞLEMLER BİTTİ!")
print(f"✅ Toplam İşlenen Ses Dosyası: {toplam_islenen_dosya}")
print(f"🖼️ Oluşturulan Toplam Resim: {toplam_resim_sayisi}")
print(f"📂 Resimlerin Yeri: {hedef_klasor}")