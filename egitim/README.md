# Eğitim betikleri

Bu klasördeki betikler, `modeller/` altındaki modelleri üretmek için kullanılan **orijinal**
kodlardır. Dosya adları düzenlendi, içerikleri değiştirilmedi. Hepsinde `C:\Users\merts\...`
gibi sabit Windows yolları var; bu yüzden olduğu gibi çalışmazlar. Yeniden eğitim yapılacaksa
dosyanın başındaki `AYARLAR` bölümündeki yollar düzenlenmelidir.

Çıkarım (inference) `steteskop/` paketiyle yapılır. Buradaki görüntü üretim kodu
`steteskop/ozellik.py` ile **aynı kalmalıdır**.

## Hangi model, hangi betikle üretildi?

| Model (`modeller/`) | Veri hazırlama (`veri_hazirlama/`) | Eğitim (`model_egitimi/`) |
|---|---|---|
| `akilli_steteskop_final_model.h5`: akciğer, 5 sınıf CNN | `icbhi_goruntu_uret.py`, `pnomoni_arsivi_goruntu_uret.py`, `akciger_arsivi_goruntu_uret.py`, `akciger_veri_cogaltma.py` | `akciger_modeli_egit.py` |
| `bekci_modeli.h5`: akciğer sesi / gürültü | akciğer görüntüleri + `esc50_goruntu_uret.py`, `esc50_veri_cogaltma.py` | `akciger_bekci_egit.py` |
| `kalp_modeli.h5`: kalp, CNN + BiLSTM (CRNN) | `kalp_goruntu_uret.py` | `kalp_crnn_egit.py` |

Eşleştirme, `.h5` dosyalarındaki katman yapısı incelenerek yapıldı.

## Çıktısı repoda olmayan betikler

| Betik | Açıklama |
|---|---|
| `model_egitimi/kalp_bekci_egit.py` | Kalp sesi bekçisi (`bekci_kalp_modeli.h5`). Model dosyası repoda yok; eğitilip `config.yaml`'a eklenirse kalp kayıtlarında gürültü ayıklanabilir. |
| `model_egitimi/akciger_bekci_v2_egit.py` | Akciğer bekçisinin alternatif sürümü (Dense 128, dropout yok). |
| `model_egitimi/kalp_cnn_egit.py` + `veri_hazirlama/kalp_hibrit_veri_cogaltma.py` | Kalp için düz CNN (hibrit görüntü). Kullanımdaki model CRNN sürümü. |

## Veri kaynakları
- **ICBHI 2017 Respiratory Sound Database:** akciğer sesleri ve `patient_diagnosis.csv`
- **Pnömoni / astım arşivi** (Kaggle, `archive2`)
- **ESC-50:** gürültü sınıfı (bekçi)
- **Kalp sesleri:** `KALP_VERI_SETI_FINAL` (Normal / Hasta klasörleri)

## Yeniden eğitim için öneriler
- Doğrulama setini **hasta bazında** ayırın. Aynı hastanın parçaları ve çoğaltılmış kopyaları
  yalnızca bir sette yer almalı; şu anki ~%97 doğrulama başarısı bu sızıntıdan kaynaklanıyor.
- Eğitim çıktısında yazdırılan sınıf listesini `config.yaml` dosyasına aynen yazın.
- Görüntü üretimi için `steteskop.ozellik` fonksiyonlarını kullanın. Böylece eğitim ve çıkarım
  kodu ayrışmaz.
