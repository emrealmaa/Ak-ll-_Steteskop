# Akıllı Steteskop: Claude Code için proje notları

Steteskop kayıtlarını (akciğer / kalp) Keras modelleriyle sınıflandıran bir araç.
Kullanıcı bir ses dosyasını analiz ettirmek istediğinde **`steteskop-analiz` skill'ini kullan**
(`.claude/skills/steteskop-analiz/SKILL.md`).

## Komutlar
```bash
uv sync                                              # kurulum
uv run steteskop kontrol                             # ortam + model doğrulama
uv run steteskop analiz <yol> --tip akciger|kalp     # JSON rapor (stdout)
uv run pytest                                        # testler (-m "not model" = hızlı)
```

## Yapı
- `steteskop/`: çıkarım paketi (`ses.py` → `ozellik.py` → `modeller.py` → `analiz.py` → `cli.py`)
- `config.yaml`: model yolları, sınıf sıraları, hangi modelin hangi ön işlemeyi kullandığı
- `modeller/`: eğitilmiş `.h5` dosyaları (Keras 3.12; `keras>=3.12` gerekir)
- `egitim/`: orijinal veri hazırlama ve eğitim betikleri (sabit Windows yolları içerir, doğrudan çalışmaz; eşleşmeler `egitim/README.md`'de)
- `ses/`: kullanıcı kayıtları, `raporlar/`: analiz çıktıları (ikisi de gitignore'da)
- `docs/GELISTIRME.md`: durum, doğrulama sonuçları, yol haritası. Önemli bir iş bitince güncelle.

## Değişmezler (bozma)
- `steteskop/ozellik.py` ve `steteskop/ses.py` içindeki çizim, filtre ve bölütleme kodu
  `egitim/` altındaki betiklerle **birebir** aynıdır. Figür boyutu, renk haritası, `bbox_inches`,
  bilinear yeniden boyutlandırma gibi küçük bir değişiklik bile tahminleri sessizce bozar.
  Değiştirmek gerekiyorsa modeli de aynı kodla yeniden eğitmek gerekir.
- Akciğer modeli + bekçi: `hibrit` görüntü (spektrogram + MFCC), ardışık 5 sn parçalar, filtre yok.
- Kalp modeli (CRNN): 20–1000 Hz bant geçiren filtre, yalnızca `magma` spektrogram, 5 sn pencere / 2.5 sn kaydırma.
- Sınıf sırası `.h5` içinde saklanmaz; `config.yaml` tek kaynaktır. Gerçek ICBHI ve PhysioNet
  kayıtlarıyla doğrulandı (bekçi: `Lung_Sound`=0, `Noise`=1; kalp: `Hasta`=0, `Normal`=1).
- Modellerin çıkışı zaten softmax; ikinci kez softmax uygulanmaz.

## Bilinen sınırlamalar
- Akciğer modelinin eğitiminde hasta bazlı bölme yapılmadı (veri sızıntısı); gerçek doğruluk ~%60.
- Kalp sesi bekçisi yok; tür (akciğer/kalp) kullanıcıdan alınır.
