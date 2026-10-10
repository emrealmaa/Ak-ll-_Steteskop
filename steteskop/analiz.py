"""Uçtan uca analiz: ses dosyası -> yapılandırılmış (JSON'a hazır) rapor."""

from pathlib import Path

import numpy as np

from . import modeller, ozellik, ses
from .ayarlar import Ayarlar, ModelProfili

# Güven düzeyi eşikleri (ortalama olasılık ve 1.-2. sınıf farkı üzerinden)
YUKSEK_GUVEN = 0.75
BELIRSIZ_ALT = 0.50
BELIRSIZ_FARK = 0.15


def _y(x: float) -> float:
    return round(float(x), 4)


def guven_duzeyi(en_yuksek: float, fark: float) -> str:
    if en_yuksek < BELIRSIZ_ALT or fark < BELIRSIZ_FARK:
        return "belirsiz"
    if en_yuksek < YUKSEK_GUVEN:
        return "orta"
    return "yuksek"


def birlestir(olasiliklar: np.ndarray, siniflar: list[str]) -> dict:
    """Geçerli parçaların olasılıklarını kayıt düzeyinde tek sonuca indirger."""
    ortalama = olasiliklar.mean(axis=0)
    sira = np.argsort(ortalama)[::-1]
    birinci, ikinci = int(sira[0]), int(sira[1])
    parca_tahminleri = olasiliklar.argmax(axis=1)
    oylar = {s: int((parca_tahminleri == i).sum()) for i, s in enumerate(siniflar)}
    fark = ortalama[birinci] - ortalama[ikinci]
    return {
        "sinif": siniflar[birinci],
        "ortalama_olasilik": _y(ortalama[birinci]),
        "ikinci_sinif": siniflar[ikinci],
        "ikinci_olasilik": _y(ortalama[ikinci]),
        "fark": _y(fark),
        "guven_duzeyi": guven_duzeyi(ortalama[birinci], fark),
        "tutarlilik": _y(oylar[siniflar[birinci]] / len(parca_tahminleri)),
        "oylar": oylar,
        "_ortalama": ortalama,
    }


def analiz_et(
    yol: str | Path,
    tip: str,
    ayarlar: Ayarlar,
    goruntu_dizini: str | Path | None = None,
    bekci_kullan: bool = True,
) -> dict:
    yol = Path(yol).resolve()
    if tip not in ayarlar.modeller:
        raise ValueError(f"Bilinmeyen tip '{tip}'. Seçenekler: {', '.join(ayarlar.modeller)}")
    profil: ModelProfili = ayarlar.modeller[tip]
    parca_suresi = ayarlar.parca_suresi_sn

    y, sr = ses.yukle(yol)
    uyarilar = ses.kalite_kontrol(y, sr, parca_suresi)

    if profil.filtre:
        y = ses.FILTRELER[profil.filtre](y, sr)
    parcalar, atlanan_sn = ses.BOLUTLEYICILER[profil.bolutleme](y, sr, parca_suresi)
    if atlanan_sn >= 1.0:
        uyarilar.append(f"Kaydın son {atlanan_sn:.1f} sn'si tam bir parça oluşturmadığı için analize alınmadı.")

    # Aynı görüntü türünü kullanan modeller (akciğer + bekçi) aynı PNG'leri paylaşır.
    png_onbellek: dict[str, list[bytes]] = {}

    def pngler(tur: str) -> list[bytes]:
        if tur not in png_onbellek:
            uretici = ozellik.GORUNTU_URETICILERI[tur]
            png_onbellek[tur] = [uretici(p.y, sr) for p in parcalar]
        return png_onbellek[tur]

    def girdiler(tur: str) -> list[np.ndarray]:
        return [ozellik.png_to_tensor(p) for p in pngler(tur)]

    # 1) Bekçi: parça gerçekten bir akciğer sesi mi?
    bekci = profil.bekci if bekci_kullan else None
    bekci_sonuclari = [None] * len(parcalar)
    gecerli = np.ones(len(parcalar), dtype=bool)
    if bekci:
        b_olas = modeller.tahmin(bekci.dosya, girdiler(bekci.goruntu), len(bekci.siniflar))
        gi = bekci.siniflar.index(bekci.gecerli_sinif)
        for i, satir in enumerate(b_olas):
            gecerli[i] = satir[gi] >= bekci.esik
            bekci_sonuclari[i] = {
                "sinif": bekci.siniflar[int(satir.argmax())],
                "gecerli_olasiligi": _y(satir[gi]),
            }
    elif profil.bekci and not bekci_kullan:
        uyarilar.append("Bekçi (gürültü) kontrolü kullanıcı isteğiyle atlandı.")
    else:
        uyarilar.append(f"'{tip}' için bekçi modeli yok; gürültülü parçalar ayıklanamadı.")

    # 2) Teşhis modeli (tüm parçalar; birleştirmeye yalnız geçerliler girer)
    olas = modeller.tahmin(profil.dosya, girdiler(profil.goruntu), len(profil.siniflar))

    goruntu_yollari = []
    if goruntu_dizini:
        hedef = Path(goruntu_dizini)
        hedef.mkdir(parents=True, exist_ok=True)
        for p, png in zip(parcalar, pngler(profil.goruntu)):
            dosya = hedef / f"{yol.stem}_{tip}_parca{p.no}.png"
            dosya.write_bytes(png)
            goruntu_yollari.append(str(dosya.resolve()))

    etiket = lambda s: profil.etiketler.get(s, s)  # noqa: E731
    parca_raporlari = []
    for i, p in enumerate(parcalar):
        k = int(olas[i].argmax())
        parca_raporlari.append({
            "no": p.no,
            "bas_sn": round(p.bas_sn, 2),
            "bit_sn": round(p.bit_sn, 2),
            "gecerli": bool(gecerli[i]),
            "bekci": bekci_sonuclari[i],
            "tahmin": profil.siniflar[k],
            "tahmin_tr": etiket(profil.siniflar[k]),
            "guven": _y(olas[i, k]),
            "olasiliklar": {s: _y(v) for s, v in zip(profil.siniflar, olas[i])},
        })

    gecerli_sayisi = int(gecerli.sum())
    if gecerli_sayisi == 0:
        durum, sonuc, olasiliklar = "gecersiz_kayit", None, None
        uyarilar.append("Hiçbir parça akciğer sesi olarak doğrulanamadı; kayıt gürültü gibi görünüyor. Sonuç verilmedi.")
    else:
        durum = "tamam"
        b = birlestir(olas[gecerli], profil.siniflar)
        ortalama = b.pop("_ortalama")
        b["sinif_tr"] = etiket(b["sinif"])
        b["saglikli_mi"] = b["sinif"] == profil.saglikli_sinif
        sonuc = b
        olasiliklar = {s: _y(v) for s, v in zip(profil.siniflar, ortalama)}
        if gecerli_sayisi < len(parcalar):
            uyarilar.append(f"{len(parcalar) - gecerli_sayisi} parça gürültü olarak elendi ve sonuca katılmadı.")

    return {
        "dosya": str(yol),
        "tip": tip,
        "durum": durum,
        "ses": {"sure_sn": round(len(y) / sr, 2), "ornekleme_hizi": sr},
        "kalite": {
            "toplam_parca": len(parcalar),
            "gecerli_parca": gecerli_sayisi,
            "gurultu_parca": len(parcalar) - gecerli_sayisi,
            "bekci_kullanildi": bekci is not None,
            "uyarilar": uyarilar,
        },
        "sonuc": sonuc,
        "olasiliklar": olasiliklar,
        "parcalar": parca_raporlari,
        "goruntuler": goruntu_yollari,
        "model": {
            "ad": profil.ad,
            "siniflar": profil.siniflar,
            "etiketler": profil.etiketler,
            "bilinen_dogruluk": profil.bilinen_dogruluk,
        },
    }
