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

#define ECG_PIN    34   // ADC1_CH6 — AD8232 OUTPUT
#define LO_PLUS    32   // AD8232 LO+ (lead-off detection)
#define LO_MINUS   33   // AD8232 LO- (lead-off detection)

// Timer handle
hw_timer_t* timer = NULL;
portMUX_TYPE timerMux = portMUX_INITIALIZER_UNLOCKED;

// Volatile flags set in ISR
volatile bool sampleReady = false;
volatile int  adcValue    = 0;
volatile bool leadOff     = false;

// ---- Timer ISR — called at 250 Hz ----
void IRAM_ATTR onTimer() {
  portENTER_CRITICAL_ISR(&timerMux);
  adcValue    = analogRead(ECG_PIN);
  leadOff     = (digitalRead(LO_PLUS) == HIGH) || (digitalRead(LO_MINUS) == HIGH);
  sampleReady = true;
  portEXIT_CRITICAL_ISR(&timerMux);
}

void setup() {
  Serial.begin(115200);

  pinMode(LO_PLUS,  INPUT);
  pinMode(LO_MINUS, INPUT);
  analogReadResolution(12);   // 12-bit ADC (0-4095)
  analogSetAttenuation(ADC_11db);  // full 0-3.3V range

  // Hardware timer 0, prescaler 80 -> 1 MHz tick
  // Period: 1,000,000 / 250 = 4000 ticks = 4 ms -> 250 Hz
  timer = timerBegin(0, 80, true);
  timerAttachInterrupt(timer, &onTimer, true);
  timerAlarmWrite(timer, 4000, true);
  timerAlarmEnable(timer);
}

void loop() {
  if (sampleReady) {
    int   v  = 0;
    bool  lo = false;

    portENTER_CRITICAL(&timerMux);
    v  = adcValue;
    lo = leadOff;
    sampleReady = false;
    portEXIT_CRITICAL(&timerMux);

    // Stream compact JSON line: {"v":2048,"lo":0}
    Serial.print("{\"v\":");
    Serial.print(v);
    Serial.print(",\"lo\":");
    Serial.print(lo ? 1 : 0);
    Serial.println("}");
  }
}
