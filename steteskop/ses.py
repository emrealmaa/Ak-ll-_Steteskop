"""Ses dosyası yükleme, kalite kontrolü, filtreleme ve bölütleme.

Bölütleme ve filtre fonksiyonları eğitim betiklerindeki mantığın birebir
kopyasıdır; değiştirilirse modeller eğitimde görmedikleri girdiler alır.
"""

from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import scipy.signal as signal

DESTEKLENEN_UZANTILAR = {".wav", ".flac", ".ogg", ".mp3", ".m4a", ".aac", ".aif", ".aiff"}

SESSIZLIK_ESIGI = 1e-4  # tepe genliği bunun altındaysa kayıt boş sayılır
KIRPILMA_ESIGI = 0.999
KIRPILMA_ORANI = 0.001


class SesHatasi(Exception):
    pass


@dataclass
class Parca:
    no: int
    bas_sn: float
    bit_sn: float
    y: np.ndarray


def yukle(yol: str | Path) -> tuple[np.ndarray, int]:
    """Sesi mono ve orijinal örnekleme hızında yükler (eğitimde de sr=None kullanıldı)."""
    yol = Path(yol)
    if not yol.exists():
        raise SesHatasi(f"Dosya bulunamadı: {yol}")
    if yol.suffix.lower() not in DESTEKLENEN_UZANTILAR:
        raise SesHatasi(
            f"Desteklenmeyen dosya türü '{yol.suffix}'. Desteklenenler: "
            + ", ".join(sorted(DESTEKLENEN_UZANTILAR))
        )
    try:
        y, sr = librosa.load(yol, sr=None, mono=True)
    except Exception as e:
        ipucu = " (m4a/aac dosyaları için ffmpeg kurulu olmalı: brew install ffmpeg)" if yol.suffix.lower() in {".m4a", ".aac"} else ""
        raise SesHatasi(f"Ses dosyası okunamadı: {e.__class__.__name__}: {e}{ipucu}") from e
    if y.size == 0:
        raise SesHatasi("Ses dosyası boş (0 örnek).")
    return y, int(sr)


def kalite_kontrol(y: np.ndarray, sr: int, parca_suresi: float) -> list[str]:
    """Kullanılamaz kayıtta SesHatasi fırlatır, sorunlu kayıtta uyarı listesi döner."""
    tepe = float(np.max(np.abs(y)))
    if tepe < SESSIZLIK_ESIGI:
        raise SesHatasi("Kayıt sessiz: sinyal yok denecek kadar zayıf. Mikrofon bağlantısını kontrol edip yeniden kaydedin.")

    uyarilar = []
    sure = len(y) / sr
    if sure < parca_suresi:
        uyarilar.append(
            f"Kayıt {sure:.1f} sn; model {parca_suresi:.0f} sn'lik parçalarla eğitildi. "
            "Eksik kısım sessizlikle dolduruldu, sonuç daha az güvenilir."
        )
    kirpik = float(np.mean(np.abs(y) >= KIRPILMA_ESIGI))
    if kirpik > KIRPILMA_ORANI:
        uyarilar.append(
            f"Örneklerin %{100 * kirpik:.1f}'i kırpılmış (aşırı yüksek kayıt seviyesi); "
            "spektrum bozulmuş olabilir."
        )
    return uyarilar


def bant_gecir_20_1000(y: np.ndarray, sr: int) -> np.ndarray:
    """Kalp sesi filtresi (egitim/veri_hazirlama/kalp_goruntu_uret.py: temizle_filtrele)."""
    nyquist = sr / 2
    ust_sinir = 1000 if nyquist > 1000 else nyquist - 10
    if ust_sinir <= 20:
        return y
    try:
        b, a = signal.butter(4, [20, ust_sinir], btype="bandpass", fs=sr)
        return signal.filtfilt(b, a, y)
    except Exception:
        return y


FILTRELER = {"bant_gecir_20_1000": bant_gecir_20_1000}


def bolutle_ardisik(y: np.ndarray, sr: int, parca_suresi: float) -> tuple[list[Parca], float]:
    """Akciğer bölütlemesi (egitim/veri_hazirlama/akciger_arsivi_goruntu_uret.py).

    Dönüş: (parçalar, analize alınmayan son kısmın süresi sn)
    """
    toplam_sure = len(y) / sr
    if toplam_sure < parca_suresi:
        y_parca = librosa.util.fix_length(y, size=int(parca_suresi * sr))
        return [Parca(1, 0.0, round(toplam_sure, 3), y_parca)], 0.0

    adet = int(toplam_sure // parca_suresi)
    parcalar = []
    for i in range(adet):
        bas, bit = i * parca_suresi, (i + 1) * parca_suresi
        parcalar.append(Parca(i + 1, bas, bit, y[int(bas * sr): int(bit * sr)]))
    return parcalar, round(toplam_sure - adet * parca_suresi, 3)


def bolutle_ortusmeli(y: np.ndarray, sr: int, parca_suresi: float) -> tuple[list[Parca], float]:
    """Kalp bölütlemesi (egitim/veri_hazirlama/kalp_goruntu_uret.py): pencere = parca_suresi, kaydırma = yarısı."""
    toplam_sure = len(y) / sr
    parca_len = int(parca_suresi * sr)
    adim = int(parca_suresi / 2 * sr)
    if len(y) < parca_len:
        y = librosa.util.fix_length(y, size=parca_len)

    parcalar = []
    for i, bas in enumerate(range(0, len(y) - parca_len + 1, adim)):
        parcalar.append(Parca(i + 1, bas / sr, min((bas + parca_len) / sr, toplam_sure), y[bas: bas + parca_len]))
    kapsanan = (parcalar[-1].bit_sn) if parcalar else 0.0
    return parcalar, round(max(toplam_sure - kapsanan, 0.0), 3)


BOLUTLEYICILER = {"ardisik": bolutle_ardisik, "ortusmeli": bolutle_ortusmeli}
