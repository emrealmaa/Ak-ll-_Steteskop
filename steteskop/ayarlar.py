"""config.yaml dosyasını okur ve doğrular."""

import os
from dataclasses import dataclass
from pathlib import Path

import yaml

KOK_DIZIN = Path(__file__).resolve().parent.parent
VARSAYILAN_AYAR_DOSYASI = KOK_DIZIN / "config.yaml"


class AyarHatasi(Exception):
    pass


@dataclass(frozen=True)
class BekciProfili:
    anahtar: str
    ad: str
    dosya: Path
    goruntu: str
    siniflar: list
    gecerli_sinif: str
    esik: float


@dataclass(frozen=True)
class ModelProfili:
    tip: str
    ad: str
    dosya: Path
    goruntu: str
    filtre: str | None
    bolutleme: str
    siniflar: list
    etiketler: dict
    saglikli_sinif: str
    bilinen_dogruluk: float | None
    bekci: BekciProfili | None


@dataclass(frozen=True)
class Ayarlar:
    parca_suresi_sn: float
    modeller: dict  # tip -> ModelProfili


def yukle(yol: str | os.PathLike | None = None) -> Ayarlar:
    yol = Path(yol or os.environ.get("STETESKOP_AYAR") or VARSAYILAN_AYAR_DOSYASI)
    if not yol.exists():
        raise AyarHatasi(f"Ayar dosyası bulunamadı: {yol}")
    ham = yaml.safe_load(yol.read_text(encoding="utf-8"))
    kok = yol.parent

    bekciler = {}
    for anahtar, b in (ham.get("bekciler") or {}).items():
        bekciler[anahtar] = BekciProfili(
            anahtar=anahtar,
            ad=b["ad"],
            dosya=kok / b["dosya"],
            goruntu=b["goruntu"],
            siniflar=list(b["siniflar"]),
            gecerli_sinif=b["gecerli_sinif"],
            esik=float(b["esik"]),
        )
        if b["gecerli_sinif"] not in b["siniflar"]:
            raise AyarHatasi(f"{anahtar}: gecerli_sinif sınıf listesinde yok")

    modeller = {}
    for tip, m in ham["modeller"].items():
        bekci_adi = m.get("bekci")
        if bekci_adi and bekci_adi not in bekciler:
            raise AyarHatasi(f"{tip}: tanımsız bekçi '{bekci_adi}'")
        if m["saglikli_sinif"] not in m["siniflar"]:
            raise AyarHatasi(f"{tip}: saglikli_sinif sınıf listesinde yok")
        modeller[tip] = ModelProfili(
            tip=tip,
            ad=m["ad"],
            dosya=kok / m["dosya"],
            goruntu=m["goruntu"],
            filtre=m.get("filtre"),
            bolutleme=m["bolutleme"],
            siniflar=list(m["siniflar"]),
            etiketler=dict(m.get("etiketler") or {}),
            saglikli_sinif=m["saglikli_sinif"],
            bilinen_dogruluk=m.get("bilinen_dogruluk"),
            bekci=bekciler.get(bekci_adi) if bekci_adi else None,
        )

    return Ayarlar(parca_suresi_sn=float(ham["parca_suresi_sn"]), modeller=modeller)
