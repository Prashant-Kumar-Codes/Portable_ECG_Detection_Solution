/*
 * esp32_ad8232_250hz.ino
 * ----------------------
 * ESP32 firmware for the AD8232 ECG sensor module.
 * Samples the analog output at exactly 250 Hz using a hardware timer ISR
 * and streams samples over USB UART at 115200 baud.
 *
 * Wiring
 * ------
 *   AD8232 OUTPUT  ->  GPIO 34  (ADC1_CH6, input-only)
 *   AD8232 LO+     ->  GPIO 32
 *   AD8232 LO-     ->  GPIO 33
 *   AD8232 3.3V    ->  3.3V
 *   AD8232 GND     ->  GND
 *
 * Serial Protocol (115200 baud, newline-delimited)
 * ------------------------------------------------
 *   Format: {"v":<ADC_value>,"lo":<0|1>}\n
 *   v  : 12-bit ADC reading (0–4095)
 *   lo : 1 if either LO+ or LO- is HIGH (lead-off), 0 otherwise
 *
 * The Python SerialReader in Modules/serial_reader.py parses this JSON format.
 */

#define ECG_PIN       34   // ADC1_CH6 — AD8232 OUTPUT
#define LO_PLUS       32   // AD8232 LO+ (lead-off detection)
#define LO_MINUS      33   // AD8232 LO- (lead-off detection)
#define LED_PIN        2   // Built-in status LED (ESP32 GPIO 2)

// Precise 250 Hz non-blocking timer: 1,000,000 us / 250 = 4,000 us
const unsigned long SAMPLE_INTERVAL_US = 4000;
unsigned long lastSampleUs = 0;
unsigned long lastHeartbeatMs = 0;
bool ledState = false;

void setup() {
  Serial.begin(115200);

  pinMode(LO_PLUS,  INPUT);
  pinMode(LO_MINUS, INPUT);
  pinMode(LED_PIN,  OUTPUT);

  analogReadResolution(12);        // 12-bit ADC (0-4095)
  analogSetAttenuation(ADC_11db);  // Full 0-3.3V range

  // Wait briefly for serial connection and output ready beacon
  delay(200);
  Serial.println("{\"msg\":\"ECG_READY\",\"fs\":250}");
  lastSampleUs = micros();
}

void loop() {
  unsigned long currentUs = micros();

  // Trigger at exact 4000 us (250 Hz) intervals
  if (currentUs - lastSampleUs >= SAMPLE_INTERVAL_US) {
    lastSampleUs += SAMPLE_INTERVAL_US;

    // Guard against timer drift if loop was briefly delayed
    if (currentUs - lastSampleUs > SAMPLE_INTERVAL_US) {
      lastSampleUs = currentUs;
    }

    // Safe ADC read outside ISR context (prevents ESP32 Guru Meditation panics)
    int  v  = analogRead(ECG_PIN);
    bool lo = (digitalRead(LO_PLUS) == HIGH) || (digitalRead(LO_MINUS) == HIGH);

    // Stream compact JSON line: {"v":2048,"lo":0}
    Serial.print("{\"v\":");
    Serial.print(v);
    Serial.print(",\"lo\":");
    Serial.print(lo ? 1 : 0);
    Serial.println("}");
  }

  // Visual heartbeat: toggle LED every 500 ms to confirm firmware is running
  unsigned long currentMs = millis();
  if (currentMs - lastHeartbeatMs >= 500) {
    lastHeartbeatMs = currentMs;
    ledState = !ledState;
    digitalWrite(LED_PIN, ledState ? HIGH : LOW);
  }
}
