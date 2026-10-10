import json

import numpy as np
import pytest
import soundfile as sf

from steteskop import ayarlar
from steteskop.analiz import birlestir, guven_duzeyi
from steteskop.cli import main

SINIFLAR = ["A", "B", "C"]


def test_birlestir_ortalama_ve_oylar():
    olas = np.array([[0.8, 0.1, 0.1], [0.6, 0.3, 0.1], [0.2, 0.7, 0.1]])
    b = birlestir(olas, SINIFLAR)
    assert b["sinif"] == "A"
    assert b["ortalama_olasilik"] == pytest.approx(0.5333, abs=1e-4)
    assert b["ikinci_sinif"] == "B"
    assert b["oylar"] == {"A": 2, "B": 1, "C": 0}
    assert b["tutarlilik"] == pytest.approx(0.6667, abs=1e-4)


@pytest.mark.parametrize(
    "en_yuksek, fark, beklenen",
    [(0.9, 0.8, "yuksek"), (0.6, 0.3, "orta"), (0.45, 0.2, "belirsiz"), (0.8, 0.1, "belirsiz")],
)
def test_guven_duzeyi(en_yuksek, fark, beklenen):
    assert guven_duzeyi(en_yuksek, fark) == beklenen


def test_config_gecerli():
    a = ayarlar.yukle()
    assert set(a.modeller) == {"akciger", "kalp"}
    assert a.modeller["akciger"].bekci is not None
    for p in a.modeller.values():
        assert p.dosya.exists(), p.dosya


@pytest.mark.model
def test_cli_uctan_uca(tmp_path, capsys):
    sr = 4000
    rng = np.random.default_rng(0)
    sf.write(tmp_path / "gurultu.wav", rng.normal(size=sr * 11) * 0.3, sr)
    sf.write(tmp_path / "sessiz.wav", np.zeros(sr * 6), sr)

    kod = main(["analiz", str(tmp_path), "--tip", "akciger", "--goruntu-dizini", str(tmp_path / "img")])
    cikti = json.loads(capsys.readouterr().out)

    assert kod == 1  # sessiz.wav hata verir
    assert [h["dosya"].endswith("sessiz.wav") for h in cikti["hatalar"]] == [True]
    rapor = cikti["raporlar"][0]
    assert rapor["kalite"]["toplam_parca"] == 2
    assert len(rapor["parcalar"]) == 2
    for p in rapor["parcalar"]:
        assert sum(p["olasiliklar"].values()) == pytest.approx(1.0, abs=1e-3)
    assert len(rapor["goruntuler"]) == 2
    # Beyaz gürültü bir akciğer sesi değildir; bekçi reddetmeli.
    assert rapor["durum"] == "gecersiz_kayit" and rapor["sonuc"] is None


@pytest.mark.model
def test_kalp_bekcisiz_sonuc_uretir(tmp_path, capsys):
    sr = 2000
    t = np.arange(sr * 8) / sr
    sf.write(tmp_path / "kalp.wav", np.sin(2 * np.pi * 50 * t) * (np.sin(2 * np.pi * 1.2 * t) > 0.9), sr)

    assert main(["analiz", str(tmp_path / "kalp.wav"), "--tip", "kalp"]) == 0
    rapor = json.loads(capsys.readouterr().out)["raporlar"][0]
    assert rapor["durum"] == "tamam"
    assert rapor["sonuc"]["sinif"] in {"Hasta", "Normal"}
    assert rapor["kalite"]["bekci_kullanildi"] is False
    assert [p["bas_sn"] for p in rapor["parcalar"]] == [0.0, 2.5]
