"""Komut satırı arayüzü. Çıktı her zaman stdout'a JSON'dur; agent'lar bunu okur.

    steteskop analiz <dosya|dizin> [...] --tip akciger|kalp [--goruntu-dizini DIZIN]
    steteskop kontrol

Çıkış kodları: 0 = tüm dosyalar işlendi, 1 = en az bir dosyada hata, 2 = kullanım/ayar hatası.
"""

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from . import __version__  # noqa: E402
from . import ayarlar as ayar_mod  # noqa: E402
from .ses import DESTEKLENEN_UZANTILAR, SesHatasi  # noqa: E402

TIBBI_UYARI = (
    "Bu çıktı deneysel bir makine öğrenmesi modelinin tahminidir, tıbbi teşhis değildir. "
    "Sağlıkla ilgili kararlar için bir hekime başvurun."
)


def _yaz(veri: dict, cikti: str | None, girintisiz: bool) -> None:
    metin = json.dumps(veri, ensure_ascii=False, indent=None if girintisiz else 2)
    if cikti:
        Path(cikti).parent.mkdir(parents=True, exist_ok=True)
        Path(cikti).write_text(metin + "\n", encoding="utf-8")
    print(metin)


def _dosyalari_topla(yollar: list[str]) -> tuple[list[Path], list[dict]]:
    dosyalar, hatalar = [], []
    for ham in yollar:
        yol = Path(ham).expanduser()
        if yol.is_dir():
            bulunan = sorted(p for p in yol.iterdir() if p.is_file() and p.suffix.lower() in DESTEKLENEN_UZANTILAR)
            if not bulunan:
                hatalar.append({"dosya": str(yol.resolve()), "hata": "Dizinde desteklenen ses dosyası yok."})
            dosyalar.extend(bulunan)
        elif yol.exists():
            dosyalar.append(yol)
        else:
            hatalar.append({"dosya": str(yol), "hata": "Dosya veya dizin bulunamadı."})
    return dosyalar, hatalar


def komut_analiz(args) -> int:
    from .analiz import analiz_et
    from .modeller import ModelHatasi

    ayarlar = ayar_mod.yukle(args.ayar)
    dosyalar, hatalar = _dosyalari_topla(args.yollar)
    raporlar = []
    for dosya in dosyalar:
        try:
            raporlar.append(analiz_et(dosya, args.tip, ayarlar, args.goruntu_dizini, bekci_kullan=not args.bekcisiz))
        except (SesHatasi, ModelHatasi) as e:
            hatalar.append({"dosya": str(dosya.resolve()), "hata": str(e)})
        except Exception as e:  # beklenmeyen hatada diğer dosyalara devam et
            hatalar.append({"dosya": str(dosya.resolve()), "hata": f"Beklenmeyen hata: {e.__class__.__name__}: {e}"})

    _yaz(
        {
            "arac": "akilli-steteskop",
            "surum": __version__,
            "tip": args.tip,
            "tibbi_uyari": TIBBI_UYARI,
            "raporlar": raporlar,
            "hatalar": hatalar,
        },
        args.cikti,
        args.girintisiz,
    )
    return 1 if hatalar else 0


def komut_kontrol(args) -> int:
    """Ortamı ve modelleri doğrular; agent ilk kullanımda bunu çalıştırabilir."""
    import platform

    from .modeller import ModelHatasi, yukle

    ayarlar = ayar_mod.yukle(args.ayar)
    sonuc = {"arac": "akilli-steteskop", "surum": __version__, "python": platform.python_version(), "modeller": {}}
    try:
        import keras
        import librosa
        import tensorflow as tf

        sonuc["kutuphaneler"] = {"tensorflow": tf.__version__, "keras": keras.__version__, "librosa": librosa.__version__}
    except Exception as e:
        sonuc["kutuphaneler"] = {"hata": str(e)}

    hepsi_tamam = "hata" not in sonuc["kutuphaneler"]
    profiller = [(t, p.dosya, p.siniflar) for t, p in ayarlar.modeller.items()]
    profiller += [(f"bekci:{p.bekci.anahtar}", p.bekci.dosya, p.bekci.siniflar) for p in ayarlar.modeller.values() if p.bekci]
    for ad, dosya, siniflar in profiller:
        try:
            model = yukle(dosya)
            cikis = int(model.output_shape[-1])
            durum = "tamam" if cikis == len(siniflar) else f"sinif_sayisi_uyusmuyor ({cikis} != {len(siniflar)})"
            sonuc["modeller"][ad] = {"dosya": str(dosya), "durum": durum, "girdi": list(model.input_shape[1:]), "siniflar": siniflar}
        except ModelHatasi as e:
            durum = str(e)
            sonuc["modeller"][ad] = {"dosya": str(dosya), "durum": durum}
        hepsi_tamam &= durum == "tamam"

    sonuc["durum"] = "tamam" if hepsi_tamam else "sorunlu"
    _yaz(sonuc, None, args.girintisiz)
    return 0 if hepsi_tamam else 1


def main(argv: list[str] | None = None) -> int:
    ana = argparse.ArgumentParser(prog="steteskop", description="Akıllı Steteskop ses analiz aracı (JSON çıktı).")
    ana.add_argument("--surum", action="version", version=f"%(prog)s {__version__}")
    ana.add_argument("--ayar", help="config.yaml yolu (varsayılan: repo kökündeki config.yaml)")
    alt = ana.add_subparsers(dest="komut", required=True)

    a = alt.add_parser("analiz", help="Ses dosyası(ları)nı veya bir dizini analiz et")
    a.add_argument("yollar", nargs="+", help="Ses dosyası ya da ses dosyaları içeren dizin")
    a.add_argument("--tip", required=True, choices=["akciger", "kalp"], help="Kaydın türü")
    a.add_argument("--goruntu-dizini", help="Modele verilen spektrogram görüntülerini bu dizine kaydet")
    a.add_argument("--cikti", help="JSON raporu ayrıca bu dosyaya yaz")
    a.add_argument("--bekcisiz", action="store_true", help="Gürültü bekçisini atla (yalnız bekçi gerçek kaydı yanlışlıkla eliyorsa)")
    a.add_argument("--girintisiz", action="store_true", help="JSON'u tek satır yaz")
    a.set_defaults(fonk=komut_analiz)

    k = alt.add_parser("kontrol", help="Ortamı ve modelleri doğrula")
    k.add_argument("--girintisiz", action="store_true")
    k.set_defaults(fonk=komut_kontrol)

    args = ana.parse_args(argv)
    try:
        return args.fonk(args)
    except ayar_mod.AyarHatasi as e:
        _yaz({"arac": "akilli-steteskop", "hata": str(e)}, None, False)
        return 2


if __name__ == "__main__":
    sys.exit(main())
