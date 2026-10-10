"""Keras modellerinin yüklenmesi ve tahmin."""

import os
from functools import lru_cache
from pathlib import Path

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")


class ModelHatasi(Exception):
    pass


@lru_cache(maxsize=None)
def yukle(dosya: Path):
    if not Path(dosya).exists():
        raise ModelHatasi(f"Model dosyası bulunamadı: {dosya}")
    import keras

    try:
        return keras.models.load_model(dosya, compile=False)
    except Exception as e:
        raise ModelHatasi(f"Model yüklenemedi ({dosya.name}): {e.__class__.__name__}: {e}") from e


def tahmin(dosya: Path, girdiler: list[np.ndarray], sinif_sayisi: int) -> np.ndarray:
    """(parça_sayısı, sınıf_sayısı) olasılık matrisi döner.

    Modellerin son katmanı zaten softmax; eski test betiklerindeki gibi ikinci
    kez softmax uygulanmaz.
    """
    model = yukle(dosya)
    olasiliklar = np.asarray(model.predict(np.stack(girdiler), verbose=0), dtype=np.float64)
    if olasiliklar.shape[1] != sinif_sayisi:
        raise ModelHatasi(
            f"{dosya.name} {olasiliklar.shape[1]} sınıf üretiyor, config.yaml'da {sinif_sayisi} sınıf tanımlı."
        )
    return olasiliklar
