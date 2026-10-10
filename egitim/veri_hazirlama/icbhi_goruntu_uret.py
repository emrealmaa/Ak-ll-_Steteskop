import os
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# --- BURALARI KENDİNE GÖRE DÜZENLE ---
ses_klasoru = r"C:\Users\merts\OneDrive\Masaüstü\audio_and_txt_files"
csv_yolu = r"C:\Users\merts\OneDrive\Masaüstü\patient_diagnosis.csv"
hedef_klasor = r"C:\Users\merts\OneDrive\Masaüstü\ISLENMIS_VERI_SETI"
# -------------------------------------

print("Veri işleme fabrikası başlatılıyor (_A1 Etiketli)... 🏭")

# 1. CSV Dosyasını Oku
try:
    df = pd.read_csv(csv_yolu, header=None, names=['HastaNo', 'Hastalik'])
    hasta_sozlugu = dict(zip(df['HastaNo'].astype(str), df['Hastalik']))
    print(f"✅ CSV okundu. Toplam {len(hasta_sozlugu)} hasta kaydı bulundu.")
except Exception as e:
    print(f"❌ CSV okuma hatası: {e}")
    exit()

if not os.path.exists(hedef_klasor):
    os.makedirs(hedef_klasor)

islenen_dosya_sayisi = 0

print("Dosyalar işleniyor...")

for dosya_adi in os.listdir(ses_klasoru):
    if dosya_adi.endswith(".wav"):
        
        # Dosya adından hasta no bul
        try:
            hasta_no = dosya_adi.split('_')[0]
        except: continue
        
        # CSV'de var mı?
        if hasta_no in hasta_sozlugu:
            hastalik = hasta_sozlugu[hasta_no]
            
            hastalik_klasoru = os.path.join(hedef_klasor, hastalik)
            if not os.path.exists(hastalik_klasoru):
                os.makedirs(hastalik_klasoru)
            
            tam_yol = os.path.join(ses_klasoru, dosya_adi)
            
            try:
                y, sr = librosa.load(tam_yol, sr=None)
                
                toplam_sure = librosa.get_duration(y=y, sr=sr)
                parca_suresi = 5
                
                if toplam_sure < parca_suresi:
                    parca_sayisi = 1
                else:
                    parca_sayisi = int(toplam_sure // parca_suresi)
                
                for i in range(parca_sayisi):
                    baslangic = i * parca_suresi
                    bitis = (i + 1) * parca_suresi
                    
                    # --- GÜNCELLENEN KISIM: _A1 Etiketi Eklendi ---
                    # Örn: 101_1b1..._A1_parca0.png
                    resim_adi = f"{dosya_adi.replace('.wav', '')}_A1_parca{i}.png"
                    kayit_yolu = os.path.join(hastalik_klasoru, resim_adi)
                    
                    if not os.path.exists(kayit_yolu):
                        
                        # Kesme / Padding
                        if toplam_sure < parca_suresi:
                            y_parca = y
                            hedef_uzunluk = int(parca_suresi * sr)
                            y_parca = librosa.util.fix_length(y_parca, size=hedef_uzunluk)
                        else:
                            y_parca = y[int(baslangic*sr) : int(bitis*sr)]
                        
                        # Hibrit Resim
                        plt.figure(figsize=(10, 10))
                        
                        plt.subplot(2, 1, 1)
                        D = librosa.stft(y_parca)
                        S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
                        librosa.display.specshow(S_db, sr=sr, y_axis='log')
                        plt.axis('off') 
                        
                        plt.subplot(2, 1, 2)
                        mfccs = librosa.feature.mfcc(y=y_parca, sr=sr, n_mfcc=13)
                        librosa.display.specshow(mfccs, sr=sr)
                        plt.axis('off') 
                        
                        plt.tight_layout()
                        plt.savefig(kayit_yolu, bbox_inches='tight', pad_inches=0)
                        plt.close()
                
                print(f"✅ İşlendi: {dosya_adi} -> {hastalik}")
                islenen_dosya_sayisi += 1
            
            except Exception as e:
                print(f"⚠️ Hata: {dosya_adi} - {e}")

print("-" * 30)
print(f"🎉 İŞLEM BİTTİ! Toplam {islenen_dosya_sayisi} ses dosyası _A1 etiketiyle işlendi.")