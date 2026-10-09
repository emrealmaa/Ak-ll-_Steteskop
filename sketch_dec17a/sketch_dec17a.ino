/* * ATmega328P Boost Converter PID/PWM Kontrolü
 * Pin Mapping:
 * Arduino D3  = ATmega328P Pin 5 (PD3) -> PWM Çıkışı
 * Arduino A0  = ATmega328P Pin 23 (PC0) -> Voltaj Okuma
 */

const int pwmPin = 3;      // MOSFET Gate
const int feedbackPin = A0; // Gerilim bölücü orta uç

float targetVoltage = 5.0; // Hedef Voltaj
float currentVoltage = 0.0;
int pwmValue = 0;

void setup() {
  // PWM pini çıkış, Feedback pini giriş
  pinMode(pwmPin, OUTPUT);
  pinMode(feedbackPin, INPUT);
  
  // PWM Frekansını yükseltmek (İsteğe Bağlı - Proteus'ta kapatılabilir)
  // Gerçek devrede bobin ötmesin ve verim artsın diye bu satır açılır:
  // TCCR2B = (TCCR2B & 0b11111000) | 0x01; // 31.25 kHz
}

void loop() {
  // 1. Çıkış Voltajını Oku
  // Formül: (AnalogDeğer / 1023) * 5V * 2 (Gerilim bölücü katsayısı)
  int sensorValue = analogRead(feedbackPin);
  currentVoltage = (sensorValue * (5.0 / 1023.0)) * 2.0;

  // 2. Basit Kontrol Döngüsü
  if (currentVoltage < targetVoltage) {
    pwmValue++; // Voltaj düşük, yüklen
  } 
  else if (currentVoltage > targetVoltage) {
    pwmValue--; // Voltaj yüksek, kıs
  }

  // 3. Sınırlandırma (Safety)
  if (pwmValue > 220) pwmValue = 220; // %85 Duty Cycle sınırı (Bobin doymasın diye)
  if (pwmValue < 0) pwmValue = 0;

  // 4. PWM Sinyalinin Uygulanması
  analogWrite(pwmPin, pwmValue);

  // Simülasyon kararlılığı için minik gecikme
  delay(5); 
}