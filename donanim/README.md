# Donanım

## boost_konvertor/
ATmega328P (Arduino) ile basit bir boost converter kontrolü. Çıkış gerilimi A0'daki gerilim
bölücüyle (×2) okunur, D3 PWM görev oranı 5 V hedefe göre adım adım ayarlanır (üst sınır %85).
Steteskop prototipinin güç beslemesi için düşünülmüştür.

## Planlanan
ESP32 tabanlı, Bluetooth üzerinden ses aktaran steteskop prototipi (bkz. `docs/GELISTIRME.md`).
