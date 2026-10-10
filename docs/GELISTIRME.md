# Geliştirme günlüğü ve yol haritası

## Durum özeti (2026-10-10)

Proje, bir AI agent'ının (Claude Code) kullanabileceği bir analiz aracına dönüştürüldü.
Kullanıcı bir ses kaydını `ses/` klasörüne koyup Claude Code'a yolunu söyler; Claude Code
`steteskop analiz` komutunu çalıştırır ve JSON raporu sade Türkçe ile yorumlar.

## Tamamlananlar

### v0.1.0: Agent entegrasyonu
- [x] `steteskop/` çıkarım paketi: yükleme → kalite kontrolü → filtre → bölütleme → görüntü → bekçi → teşhis → birleştirme
- [x] Eğitimle birebir aynı ön işleme (`ozellik.py`, `ses.py`); kalp modelinin farklı ön işlemesi ayrıldı
- [x] `config.yaml` model kayıt defteri (yollar, sınıf sırası, ön işleme türü, Türkçe etiketler)
- [x] `steteskop analiz` / `steteskop kontrol` CLI, JSON çıktı, dosya veya klasör girişi
- [x] Eski test betiklerindeki çift softmax hatası giderildi
- [x] Claude Code skill'i (`.claude/skills/steteskop-analiz/`) ve `CLAUDE.md`
- [x] `uv` ile tekrarlanabilir kurulum (Python 3.12, TensorFlow 2.21, Keras ≥3.12)
- [x] Birim ve entegrasyon testleri (`uv run pytest`)
- [x] Repo düzeni: `modeller/`, `egitim/`, `donanim/`, `docs/`, `ses/`; açıklayıcı dosya adları
- [x] Lisans: `Apache-2.0 OR MIT` (iki lisans, kullanıcı seçer); `NOTICE` dosyasında tıbbi uyarı ve eğitim veri seti atıfları
- [x] Kullanılmayan betikler (eski testler, alternatif bekçi, düz CNN kalp modeli) ve yanıltıcı eğitim grafiği kaldırıldı; hepsi git geçmişinde duruyor

### Doğrulama sonuçları
Bu kayıtlar büyük olasılıkla eğitim setinde de bulunduğu için aşağıdaki sonuçlar doğruluk
ölçümü değildir; yalnızca entegrasyonun doğru yapıldığını gösterir.

| Test | Sonuç |
|---|---|
| 12 ICBHI akciğer kaydı (9 KOAH, 1 zatürre, 1 URTI, 1 bronşiolit) | Bekçi hepsini geçirdi (≈1.0). KOAH 9/9, zatürre doğru. URTI ve bronşiolit → KOAH (bu sınıflar modelde yok) |
| Sentetik beyaz gürültü | Bekçi reddetti → `gecersiz_kayit` |
| 10 PhysioNet 2016 kalp kaydı (5 anormal, 5 normal) | 9/10 doğru |
| Uçtan uca Claude Code oturumu (`claude -p`) | Skill bulundu, araç çalıştı, sade Türkçe rapor ve tıbbi uyarı üretildi |

Doğrulanan varsayımlar: bekçi sınıf sırası `[Lung_Sound, Noise]`, kalp sınıf sırası `[Hasta, Normal]`.

## Yol haritası

### Model kalitesi (öncelikli)
- [ ] **Hasta bazlı veri bölme ile yeniden eğitim.** Şu anki doğrulama başarısı (~%97) veri sızıntısı içeriyor; gerçek doğruluk ~%60
  - **Not (model dosyaları):** Yeni modeli commit etmeden **önce** `git lfs install && git lfs track "*.h5"` çalıştırıp
    `.gitattributes` dosyasını commit edin. Böylece yeni model sürümleri git geçmişini büyütmez
    (her yeniden eğitim aksi halde ~26 MB'ı kalıcı olarak ekler; GitHub 100 MB üstü dosyaları reddeder).
    Mevcut geçmişe dokunulmaz, force push gerekmez. Repoyu klonlayanların `git-lfs` kurması gerekir;
    GitHub'ın ücretsiz LFS kotası ~1 GB depolama ve aylık ~1 GB indirme ile sınırlı.
    Alternatif: modelleri GitHub Releases / Hugging Face'te tutup bir `steteskop indir` komutuyla indirmek.
- [ ] Bağımsız bir test seti ile gerçek doğruluk, duyarlılık ve özgüllük ölçümü; `config.yaml` → `bilinen_dogruluk`
- [ ] Kalp sesi bekçisinin eğitilmesi (`egitim/model_egitimi/kalp_bekci_egit.py`) ve `config.yaml`'a eklenmesi
- [ ] Akciğer / kalp tür ayrımı yapan bir model (kullanıcının `--tip` belirtme zorunluluğu kalkar)
- [ ] Görüntü (PNG) yerine doğrudan mel-spektrogram ve MFCC dizileriyle eğitim: daha hızlı, bilgi kaybı yok
- [ ] Örnekleme hızı standardizasyonu (eğitim ve çıkarımda tek bir sr)
- [ ] Eğitim betiklerinin `steteskop.ozellik` fonksiyonlarını kullanacak şekilde yeniden yazılması
- [ ] Kalp eğitim verisinin (`KALP_VERI_SETI_FINAL`) kaynağının ve lisansının belgelenmesi (`NOTICE`)
- [ ] Bekçinin ESC-50 (CC BY-NC) yerine ticari kullanıma açık bir gürültü verisiyle yeniden eğitilmesi (modellerin ticari kullanımı gerekirse)

### Entegrasyon
- [ ] MCP sunucusu (`analyze_auscultation` aracı); Claude Desktop ve diğer agent'lar için
- [ ] ESP32 + Bluetooth ile kayıt alıp `ses/` klasörüne yazan alıcı
- [ ] Model dosyaları için Git LFS: şu an gerekli değil (~43 MB); yeniden eğitimle birlikte açılacak (yukarıdaki nota bakın)

### Donanım
- [ ] ESP32 tabanlı steteskop prototipi (mikrofon, ön yükselteç, Bluetooth ses aktarımı)

## Tasarım kararları
- **Ön işleme deterministik araçta, yorum agent'ta:** Agent ön işleme kodu yazarsa eğitimden
  sapabilir ve model sessizce yanlış sonuç verir. Bu yüzden agent sadece aracı çağırır.
- **Çıktı JSON:** Agent'ın güvenilir biçimde okuyabilmesi için. Loglar stderr'e, sonuç stdout'a gider.
- **Bekçi gürültü parçalarını eler ama rapora yazar:** Elenen parçalar sonuca katılmaz;
  hiç geçerli parça kalmazsa sonuç verilmez (`gecersiz_kayit`).
- **Güven düzeyi eşikleri** (`analiz.py`): yüksek ≥ 0.75; belirsiz < 0.50 ya da ilk iki sınıf farkı < 0.15.
