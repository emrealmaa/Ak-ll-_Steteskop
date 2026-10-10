---
name: steteskop-analiz
description: Steteskop ses kayıtlarını (akciğer/solunum veya kalp sesi; .wav, .mp3, .flac, .ogg, .m4a) Akıllı Steteskop modelleriyle analiz eder ve sonucu kullanıcıya sade Türkçe ile yorumlar. Kullanıcı bir ses dosyası ya da klasör yolu verip analiz, teşhis, değerlendirme, "ne çıktı", "sağlıklı mı" gibi bir şey istediğinde kullan.
---

# Steteskop kaydı analizi

Bu skill'in amacı: kullanıcının verdiği ses kaydını **repodaki araçla** analiz etmek
ve çıkan JSON raporunu, tıp bilgisi olmayan birinin anlayacağı şekilde açıklamak.

## Kesin kurallar

- **Ön işlemeyi kendin yazma.** Sesi spektrograma çevirme, filtreleme, bölütleme ve
  model çalıştırma işlerini yalnızca `steteskop analiz` komutu yapar. Kendi librosa/keras
  kodunu yazarsan model eğitimde görmediği girdiler alır ve sonuç anlamsız olur.
- **Modelleri, `config.yaml`'ı ya da `steteskop/ozellik.py`'yi analiz sırasında değiştirme.**
- **Teşhis koyma.** Bu deneysel bir modeldir (akciğer modelinin gerçek doğruluğu yaklaşık %60).
  Her yanıtta bunun tıbbi teşhis olmadığını ve hekime başvurulması gerektiğini belirt.
- Komut hata verirse hatayı olduğu gibi aktar ve çözüm öner. Sonuç uydurma.

## Adımlar

### 1. Kaydın türünü belirle: `akciger` mi, `kalp` mi?
Kaydın türünü otomatik ayırt eden bir model yok. Sırasıyla şunlara bak:
- Kullanıcının ifadesi ("akciğer", "nefes", "solunum", "öksürük" → `akciger`; "kalp", "üfürüm", "ritim" → `kalp`)
- Dosya veya klasör adı (`lung`, `akciger`, `breath` / `heart`, `kalp`, `pcg`)
- Hâlâ belli değilse **kullanıcıya sor**. Tahmin ederek çalıştırma.

### 2. Ortam hazır mı?
İlk kullanımda ya da `uv run` hata verirse:
```bash
uv sync
uv run steteskop kontrol
```
`"durum": "tamam"` görmeden analize geçme.

### 3. Analizi çalıştır
Repo kökünde çalıştır. Yolları tırnak içinde yaz (Türkçe karakter ve boşluk olabilir):
```bash
uv run steteskop analiz "<dosya-veya-klasör>" --tip <akciger|kalp> \
  --goruntu-dizini raporlar/goruntu --cikti "raporlar/<kayıt-adı>.json"
```
- Klasör verilirse içindeki tüm ses dosyaları analiz edilir.
- Çıkış kodu `1` ise bazı dosyalar işlenememiştir; ayrıntı `hatalar` alanındadır, işlenen dosyaları yine yorumla.
- İstersen `raporlar/goruntu/` altındaki spektrogram PNG'lerine Read ile bakabilirsin.
  Bunlar modelin gördüğü görüntülerdir. Kullanıcıya göstermek faydalıysa yollarını ver.
- `--bekcisiz` seçeneğini yalnızca kullanıcı kaydın gerçek bir steteskop kaydı olduğunu
  söylüyorsa ve bekçi bütün parçaları elemişse kullan. Bu durumda sonucun daha az güvenilir
  olduğunu belirt.

### 4. JSON'u oku
Her `raporlar[]` öğesi için:

| Alan | Anlamı |
|---|---|
| `durum` | `tamam` ya da `gecersiz_kayit` (bekçi tüm parçaları gürültü saydı → **teşhis yorumu yapma**) |
| `sonuc.sinif` / `sinif_tr` | Kayıt düzeyindeki tahmin |
| `sonuc.ortalama_olasilik` | Geçerli parçalardaki ortalama olasılık |
| `sonuc.guven_duzeyi` | `yuksek` (≥0.75), `orta`, `belirsiz` (<0.50 ya da ilk iki sınıf arasındaki fark <0.15) |
| `sonuc.ikinci_sinif`, `fark` | En yakın alternatif sınıf ve aradaki fark |
| `sonuc.tutarlilik`, `oylar` | Parçaların kaçta kaçı aynı sonucu verdi |
| `sonuc.saglikli_mi` | Kazanan sınıf `Healthy` / `Normal` mi |
| `kalite.uyarilar` | Kısa kayıt, kırpılma, elenen parçalar, bekçi olmaması: **mutlaka aktar** |
| `parcalar[]` | 5 sn'lik parça sonuçları (zaman içindeki değişimi anlatmak için) |
| `model.bilinen_dogruluk` | Modelin bilinen doğruluğu (`null` = ölçülmedi) |

Yorumlarken:
- `guven_duzeyi` `belirsiz` ise net bir sınıf söyleme; "model kararsız kaldı" de ve ilk iki olasılığı ver.
- `tutarlilik` düşükse (<0.6) parçalar arasında farklı sonuçlar çıktığını söyle.
- Akciğer modelinde URTI (üst solunum yolu enfeksiyonu), bronşiolit gibi sınıflar **yok**. Bu tür bir
  durum en yakın sınıfa (çoğunlukla KOAH ya da Bronşiyal) atanabilir; bunu belirt.
- Kalp kayıtlarında bekçi yok: gürültülü bir kayıt da sonuç üretir. Kayıt kalitesini kullanıcıya sor.
- Model, eğitim veri setlerindeki stetoskop ve örnekleme hızlarıyla eğitildi (4 kHz, 44.1 kHz, 2 kHz).
  Çok farklı bir cihazdan gelen kayıtta güvenilirlik düşebilir.

### 5. Kullanıcıya yanıt
Kısa ve sade yaz. Her dosya için şu yapıyı kullan:

```
**<dosya adı>** (<tür>, <süre> sn, <geçerli>/<toplam> parça analiz edildi)

Sonuç: <sinif_tr> — güven: <yüksek/orta/belirsiz> (%<ortalama_olasilik*100>)
<1-2 cümle: sonuç ne demek, en yakın alternatif ne, parçalar tutarlı mı>

Kayıt kalitesi: <uyarılar ya da "sorun görülmedi">

Ne yapmalı: <yeniden kayıt önerisi ve/veya hekime başvurma önerisi>
```

Birden çok dosya varsa önce kısa bir özet tablo ver (dosya | sonuç | güven), sonra ayrıntıları yaz.
Sonuna her zaman şu notu ekle:
> Bu sonuç deneysel bir yapay zeka modelinin tahminidir ve tıbbi teşhis yerine geçmez. Nefes darlığı, göğüs ağrısı, yüksek ateş gibi belirtiler varsa vakit kaybetmeden bir hekime başvurun.

### Sınıfların sade dilde anlamı
- **Healthy / Sağlıklı**: Belirgin anormal solunum sesi bulunmadı.
- **Asthma / Astım**: Hava yollarının daralmasıyla ilişkili; tipik olarak hışıltılı (wheeze) sesler.
- **COPD / KOAH**: Kronik obstrüktif akciğer hastalığı; çoğunlukla hışıltı ve çıtırtı (crackle) sesleri.
- **Bronchial_Disease / Bronşiyal hastalık**: Bronşit, bronşiektazi gibi bronş kaynaklı durumlar.
- **Pneumonia / Zatürre**: Akciğer enfeksiyonu; tipik olarak ince çıtırtılar.
- **Normal (kalp)**: Kalp seslerinde belirgin anormallik bulunmadı.
- **Hasta (kalp)**: Üfürüm, ritim düzensizliği gibi anormal bir örüntü olabilir; kardiyolojik değerlendirme önerilir.

### Yeniden kayıt önerileri (`gecersiz_kayit`, kısa kayıt ya da kırpılma uyarısında)
- Sessiz bir ortamda, en az 20 saniye kaydedin.
- Steteskobu cilde doğrudan ve sabit tutun; kıyafet üzerinden kaydetmeyin.
- Kayıt seviyesini düşürün (kırpılma uyarısı varsa).
- Akciğer için derin ve ağızdan nefes alınırken kaydedin.
