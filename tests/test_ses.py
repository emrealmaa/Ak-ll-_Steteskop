import numpy as np
import pytest

from steteskop import ses

SR = 4000


def test_ardisik_bolutleme_tam_parcalar_ve_artan_sure():
    y = np.random.default_rng(0).normal(size=int(12.3 * SR))
    parcalar, atlanan = ses.bolutle_ardisik(y, SR, 5)
    assert [p.no for p in parcalar] == [1, 2]
    assert all(len(p.y) == 5 * SR for p in parcalar)
    assert atlanan == pytest.approx(2.3, abs=1e-3)


def test_ardisik_bolutleme_kisa_kayit_doldurulur():
    y = np.ones(2 * SR)
    parcalar, atlanan = ses.bolutle_ardisik(y, SR, 5)
    assert len(parcalar) == 1 and len(parcalar[0].y) == 5 * SR
    assert parcalar[0].bit_sn == pytest.approx(2.0)
    assert atlanan == 0


def test_ortusmeli_bolutleme_yarim_kaydirma():
    y = np.ones(15 * SR)
    parcalar, atlanan = ses.bolutle_ortusmeli(y, SR, 5)
    assert [p.bas_sn for p in parcalar] == [0.0, 2.5, 5.0, 7.5, 10.0]
    assert all(len(p.y) == 5 * SR for p in parcalar)
    assert atlanan == 0


def test_sessiz_kayit_reddedilir():
    with pytest.raises(ses.SesHatasi, match="sessiz"):
        ses.kalite_kontrol(np.zeros(SR * 6), SR, 5)


def test_kisa_ve_kirpilmis_kayit_uyarisi():
    y = np.ones(SR * 2)
    uyarilar = ses.kalite_kontrol(y, SR, 5)
    assert any("sessizlikle dolduruldu" in u for u in uyarilar)
    assert any("kırpılmış" in u for u in uyarilar)


def test_desteklenmeyen_uzanti(tmp_path):
    dosya = tmp_path / "not.txt"
    dosya.write_text("x")
    with pytest.raises(ses.SesHatasi, match="Desteklenmeyen"):
        ses.yukle(dosya)
