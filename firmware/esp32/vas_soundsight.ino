// ============================================================================
//  vas_soundsight.ino — SoundSight listening node.
//
//  ESP32-S3 + INMP441 I2S microphone. Captures 16 kHz mono, runs the same
//  log-mel front-end and the same CNN as the browser twin, and publishes
//  detections over MQTT in the format the twin's hub already expects:
//
//      topic    home/<room>/event
//      payload  {"c":"fire","p":0.98,"node":"kitchen","rssi":-54}
//
//  If the broker is unreachable it falls back to ESP-NOW straight to the beacon,
//  which is the failure mode the twin models: no phone push, but light, shaker
//  and band still alert.
//
//  Detection policy is deliberately identical to vas3d/js/audio/listener.js —
//  same gate, same smoothing, same refractory window. A node that behaves
//  differently from the twin makes the twin a lie.
// ============================================================================
#include <WiFi.h>
#include <PubSubClient.h>
#include <esp_now.h>
#include <driver/i2s.h>

extern "C" {
#include "vas_dsp.h"
#include "vas_nn.h"
}

// ------------------------------------------------------------------ config --
#define NODE_ROOM     "kitchen"          // one of: corridor | kitchen | living
#define NODE_NAME     "Kitchen Module"

static const char *WIFI_SSID = "your-ssid";
static const char *WIFI_PASS = "your-password";
static const char *MQTT_HOST = "192.168.1.10";
static const uint16_t MQTT_PORT = 1883;

// Beacon's MAC, for the ESP-NOW fallback path. Print it from the beacon sketch.
static uint8_t BEACON_MAC[6] = { 0x24, 0x6F, 0x28, 0x00, 0x00, 0x00 };

// INMP441 wiring (any GPIOs work on the S3; these avoid the strapping pins).
#define I2S_SCK  4     // BCLK
#define I2S_WS   5     // LRCL  — tie L/R to GND for the left channel
#define I2S_SD   6     // DOUT

#define I2S_PORT I2S_NUM_0

// Detection policy — keep in step with vas_ml/config.py.
static const float CONF_GATE   = VAS_CONF_THRESHOLD;
static const int   SMOOTH_WIN  = 5;
static const int   SMOOTH_HITS = 3;
static const float REFRACTORY_S = 8.0f;
static const float HOP_S = 0.25f;
// Below this the room is quiet enough that inference is a waste of power.
static const float SILENCE_RMS = 0.002f;

// Which classes this node is allowed to raise. Mirrors the `classes` array for
// this module in vas3d/js/config.js — a door node reporting baby-cry would
// contradict the twin's localisation story.
static const char *ENABLED[] = { "fire_alarm", "glass_break" };
static const int N_ENABLED = sizeof(ENABLED) / sizeof(ENABLED[0]);

// ------------------------------------------------------------------ state --
static float ring[VAS_N_SAMPLES];
static int ring_write = 0;
static bool ring_full = false;

static float window_buf[VAS_N_SAMPLES];
static float feat[VAS_N_BINS * VAS_N_FRAMES];
static float logits[VAS_N_CLASSES];
static float probs[VAS_N_CLASSES];

static float hist[SMOOTH_WIN][VAS_N_CLASSES];
static int hist_n = 0, hist_i = 0;
static float last_fired[VAS_N_CLASSES];
static float clock_s = 0.0f;

static WiFiClient wifi_client;
static PubSubClient mqtt(wifi_client);
static bool espnow_ready = false;

typedef struct __attribute__((packed)) {
  char cls[16];
  float conf;
  char room[16];
} vas_espnow_msg;

// -------------------------------------------------------------------- i2s --
static void i2s_setup() {
  i2s_config_t cfg = {};
  cfg.mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX);
  cfg.sample_rate = VAS_SAMPLE_RATE;
  cfg.bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT;   // INMP441 is 24-bit in a 32-bit slot
  cfg.channel_format = I2S_CHANNEL_FMT_ONLY_LEFT;
  cfg.communication_format = I2S_COMM_FORMAT_STAND_I2S;
  cfg.intr_alloc_flags = ESP_INTR_FLAG_LEVEL1;
  cfg.dma_buf_count = 8;
  cfg.dma_buf_len = 256;
  cfg.use_apll = true;                               // cleaner 16 kHz than the PLL divider
  cfg.tx_desc_auto_clear = false;

  i2s_pin_config_t pins = {};
  pins.bck_io_num = I2S_SCK;
  pins.ws_io_num = I2S_WS;
  pins.data_out_num = I2S_PIN_NO_CHANGE;
  pins.data_in_num = I2S_SD;

  i2s_driver_install(I2S_PORT, &cfg, 0, NULL);
  i2s_set_pin(I2S_PORT, &pins);
  i2s_zero_dma_buffer(I2S_PORT);
}

/** Pull whatever I2S has ready into the ring buffer. Returns samples read. */
static int i2s_pump() {
  static int32_t raw[256];
  size_t got = 0;
  if (i2s_read(I2S_PORT, raw, sizeof(raw), &got, 0) != ESP_OK || got == 0) return 0;

  const int n = got / sizeof(int32_t);
  for (int i = 0; i < n; i++) {
    // 24-bit sample left-aligned in 32 bits; >>8 gives the 24-bit value, then
    // scale to [-1, 1] to match what the model trained on.
    float s = (float)(raw[i] >> 8) / 8388608.0f;
    ring[ring_write] = s;
    ring_write = (ring_write + 1) % VAS_N_SAMPLES;
    if (ring_write == 0) ring_full = true;
  }
  return n;
}

static void ring_snapshot(float *out) {
  const int head = VAS_N_SAMPLES - ring_write;
  memcpy(out, ring + ring_write, sizeof(float) * head);
  memcpy(out + head, ring, sizeof(float) * ring_write);
}

// --------------------------------------------------------------- transport --
static void espnow_setup() {
  if (esp_now_init() != ESP_OK) return;
  esp_now_peer_info_t peer = {};
  memcpy(peer.peer_addr, BEACON_MAC, 6);
  peer.channel = 0;
  peer.encrypt = false;
  espnow_ready = (esp_now_add_peer(&peer) == ESP_OK);
}

static void publish(const char *cls, float conf) {
  char topic[48];
  snprintf(topic, sizeof(topic), "home/%s/event", NODE_ROOM);

  if (mqtt.connected()) {
    char payload[128];
    snprintf(payload, sizeof(payload),
             "{\"c\":\"%s\",\"p\":%.2f,\"node\":\"%s\",\"rssi\":%d}",
             cls, conf, NODE_ROOM, WiFi.RSSI());
    mqtt.publish(topic, payload, /*retained=*/false);
    Serial.printf("MQTT %s %s\n", topic, payload);
    return;
  }

  if (espnow_ready) {
    vas_espnow_msg m = {};
    strncpy(m.cls, cls, sizeof(m.cls) - 1);
    strncpy(m.room, NODE_ROOM, sizeof(m.room) - 1);
    m.conf = conf;
    esp_now_send(BEACON_MAC, (uint8_t *)&m, sizeof(m));
    Serial.printf("ESP-NOW -> beacon: %s %.2f (broker unreachable)\n", cls, conf);
    return;
  }
  Serial.printf("!! %s %.2f — no transport available\n", cls, conf);
}

static void mqtt_reconnect() {
  static uint32_t next_try = 0;
  if (mqtt.connected() || millis() < next_try) return;
  next_try = millis() + 5000;            // never block the audio loop retrying
  char id[32];
  snprintf(id, sizeof(id), "vas-%s", NODE_ROOM);
  mqtt.connect(id);
}

// ---------------------------------------------------------------- decision --
static bool class_enabled(const char *cls) {
  for (int i = 0; i < N_ENABLED; i++) if (strcmp(ENABLED[i], cls) == 0) return true;
  return false;
}

static void decide() {
  if (hist_n < SMOOTH_WIN) return;

  float mean[VAS_N_CLASSES] = { 0 };
  int hits[VAS_N_CLASSES] = { 0 };
  for (int h = 0; h < SMOOTH_WIN; h++) {
    int top = 0;
    for (int i = 0; i < VAS_N_CLASSES; i++) {
      mean[i] += hist[h][i] / SMOOTH_WIN;
      if (hist[h][i] > hist[h][top]) top = i;
    }
    if (hist[h][top] >= CONF_GATE) hits[top]++;
  }

  int best = 0;
  for (int i = 1; i < VAS_N_CLASSES; i++) if (mean[i] > mean[best]) best = i;
  const char *cls = VAS_CLASSES[best];

  if (!class_enabled(cls)) return;
  if (mean[best] < CONF_GATE || hits[best] < SMOOTH_HITS) return;
  if (clock_s - last_fired[best] < REFRACTORY_S) return;

  last_fired[best] = clock_s;
  publish(cls, mean[best]);
}

// -------------------------------------------------------------------- main --
void setup() {
  Serial.begin(115200);
  delay(200);
  Serial.printf("\nSoundSight node — %s (%s)\n", NODE_NAME, NODE_ROOM);

  vas_dsp_init();

  const size_t need = vas_nn_required_arena();
  Serial.printf("nn arena: need %u floats, have %u\n",
                (unsigned)need, (unsigned)VAS_NN_MAX_ACT);
  if (need > VAS_NN_MAX_ACT) {
    Serial.println("!! model too large for the compiled arena — raise VAS_NN_MAX_ACT and rebuild");
    while (true) delay(1000);            // refuse to run rather than corrupt memory
  }

  i2s_setup();

  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  for (int i = 0; i < 40 && WiFi.status() != WL_CONNECTED; i++) delay(250);
  Serial.println(WiFi.status() == WL_CONNECTED
                 ? "wifi up" : "wifi down — ESP-NOW only");

  mqtt.setServer(MQTT_HOST, MQTT_PORT);
  espnow_setup();

  for (int i = 0; i < VAS_N_CLASSES; i++) last_fired[i] = -1000.0f;
}

void loop() {
  static uint32_t last_infer = 0;

  i2s_pump();
  mqtt.loop();
  mqtt_reconnect();

  if (!ring_full) return;
  if (millis() - last_infer < (uint32_t)(HOP_S * 1000)) return;
  last_infer = millis();
  clock_s += HOP_S;

  ring_snapshot(window_buf);

  // Skip inference in a quiet room. Saves roughly two thirds of the duty cycle
  // in practice, which matters on the 18650.
  if (vas_rms(window_buf, VAS_N_SAMPLES) < SILENCE_RMS) {
    hist_n = 0;
    return;
  }

  const uint32_t t0 = micros();
  vas_logmel(window_buf, feat);
  vas_nn_forward(feat, logits);
  vas_nn_softmax(logits, probs);
  const uint32_t dt = micros() - t0;

  memcpy(hist[hist_i], probs, sizeof(probs));
  hist_i = (hist_i + 1) % SMOOTH_WIN;
  if (hist_n < SMOOTH_WIN) hist_n++;

  static uint32_t last_report = 0;
  if (millis() - last_report > 2000) {
    last_report = millis();
    const int top = vas_nn_argmax(probs);
    Serial.printf("%-12s %.2f   (%lu us/inference)\n", VAS_CLASSES[top], probs[top], (unsigned long)dt);
  }

  decide();
}
