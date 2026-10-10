"""Ses parçasından model girdisi (128x128x3 görüntü) üretimi.

Modeller ham sesi değil, matplotlib ile çizilip PNG olarak kaydedilmiş
spektrogram görüntülerini gördü. Buradaki çizim kodu eğitim betikleriyle
BİREBİR aynı tutulmalıdır (figür boyutu, renk haritası, eksen kapatma,
tight bbox). En küçük fark bile tahminleri sessizce bozar.
"""

import io

import matplotlib

matplotlib.use("Agg")

import librosa  # noqa: E402
import librosa.display  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

GORUNTU_BOYUTU = (128, 128)


def _png_olarak_kaydet() -> bytes:
    plt.tight_layout()
    tampon = io.BytesIO()
    plt.savefig(tampon, format="png", bbox_inches="tight", pad_inches=0)
    plt.close("all")
    return tampon.getvalue()


def hibrit_png(y: np.ndarray, sr: int) -> bytes:
    """Akciğer ve bekçi modelleri: üstte log-frekans STFT spektrogramı, altta 13 MFCC."""
    plt.figure(figsize=(10, 10))

    plt.subplot(2, 1, 1)
    S_db = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    librosa.display.specshow(S_db, sr=sr, y_axis="log")
    plt.axis("off")

    plt.subplot(2, 1, 2)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    librosa.display.specshow(mfccs, sr=sr)
    plt.axis("off")

    return _png_olarak_kaydet()


def kalp_spektrogram_png(y: np.ndarray, sr: int) -> bytes:
    """Kalp (CRNN) modeli: yalnızca log-frekans STFT spektrogramı, magma renk haritası."""
    plt.figure(figsize=(5, 5))
    S_db = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    librosa.display.specshow(S_db, sr=sr, y_axis="log", cmap="magma")
    plt.axis("off")
    return _png_olarak_kaydet()


GORUNTU_URETICILERI = {
    "hibrit": hibrit_png,
    "kalp_spektrogram": kalp_spektrogram_png,
}


def png_to_tensor(png: bytes) -> np.ndarray:
    """PNG'yi tf.keras.utils.image_dataset_from_directory ile aynı şekilde çözer:
    3 kanal (alfa atılır), bilinear yeniden boyutlandırma, float32, 0-255 aralığı.
    Normalizasyonu modelin kendi Rescaling katmanı yapar.
    """
    import tensorflow as tf

    img = tf.io.decode_image(png, channels=3, expand_animations=False)
    img = tf.image.resize(img, GORUNTU_BOYUTU, method="bilinear")
    return img.numpy().astype(np.float32)
