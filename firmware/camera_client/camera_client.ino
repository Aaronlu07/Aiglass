#include "esp_camera.h"
#include <WiFi.h>
#include <WiFiUdp.h>
#include <HTTPClient.h>
#include <ArduinoWebsockets.h>
#include <WebServer.h>
#include <DNSServer.h>
#include <Preferences.h>
#define ENABLE_MQTT 0
#define ENABLE_BLE 0
#define ENABLE_AUDIO 1
#define ENABLE_MIC 1
#if ENABLE_MQTT
#include <PubSubClient.h>
#endif
#if ENABLE_BLE
#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>
#endif
#if ENABLE_AUDIO
#include "driver/i2s.h"
#endif

const char* ssid = "lxy";
const char* password = "12345678";
const char* websocket_server_host = "192.168.137.1";
const uint16_t websocket_server_port = 8000;
const char* websocket_path = "/ws/camera";
const char* http_infer_path = "/api/device/infer";
const uint16_t discovery_port = 8899;
const char* mqtt_server_host = "192.168.137.1";
const uint16_t mqtt_server_port = 1883;
const char* mqtt_pub_topic = "aiglass/telemetry";
const char* mqtt_cmd_topic = "aiglass/cmd";
const char* mqtt_result_prefix = "aiglass/device";
// 初始配置较高的分辨率，以保证画质和模型识别率
// 可选：FRAMESIZE_VGA (640x480), FRAMESIZE_SVGA (800x600), FRAMESIZE_XGA (1024x768)
framesize_t edge_frame_size = FRAMESIZE_XGA;
bool edge_grayscale = false;
uint8_t edge_jpeg_quality = 12; // 调低此值可提高画质（范围10-63，越低越好）
uint16_t edge_min_send_interval_ms = 40;
uint8_t edge_send_every_n = 1;
int edge_len_delta_threshold = 800;
const int button_mode_pin = 41;
const bool enable_button = false;
#if ENABLE_AUDIO
// 针对 PAM8403 等模拟功放的引脚定义 (适配 Seeed XIAO ESP32S3 引脚编号)
const int i2s_bclk = 5; // 对于模拟输出不重要
const int i2s_lrck = 6; // 不重要
const int i2s_dout = 4; // 关键：这是输出模拟 PWM 音频信号的引脚，对应 XIAO 板子上的 D4 引脚，接到 PAM8403 的 INL
// XIAO ESP32S3 Sense 自带的数字麦克风引脚
const int i2s_din  = 41; // PDM DATA
const int i2s_clk  = 42; // PDM CLK
#endif

#if ENABLE_MIC
const i2s_port_t i2s_mic_port = I2S_NUM_1;
const uint32_t mic_sample_rate = 16000;
const size_t mic_samples_per_read = 256;
bool mic_enabled = true;
bool mic_ready = false;
unsigned long last_mic_debug_ms = 0;
unsigned long last_mic_upload_ms = 0;
const int mic_voice_threshold = 900;
const unsigned long mic_upload_cooldown_ms = 6000;
const uint32_t mic_capture_ms = 2200;
bool mic_busy = false;
#endif

using namespace websockets;
WebsocketsClient client;
WiFiUDP discovery_udp;
unsigned long last_ws_attempt_ms = 0;
unsigned long last_discovery_ms = 0;
unsigned long last_http_post_ms = 0;
bool ws_connected = false;
String active_ws_host = websocket_server_host;
uint16_t active_ws_port = websocket_server_port;
String active_ws_path = websocket_path;
String active_ws_url = "";
String active_http_url = "";
String device_id = "";
String mqtt_result_topic = "";
bool use_http_upload = false;
#if ENABLE_MQTT
WiFiClient mqtt_wifi;
PubSubClient mqtt_client(mqtt_wifi);
unsigned long last_mqtt_attempt_ms = 0;
unsigned long last_mqtt_publish_ms = 0;
#endif
String last_cmd = "";

void handle_command(String cmd);

Preferences wifi_prefs;
WebServer prov_server(80);
DNSServer prov_dns;
bool prov_active = false;
String saved_wifi_ssid = "";
String saved_wifi_pass = "";
bool have_saved_wifi = false;
unsigned long wifi_lost_since_ms = 0;
unsigned long last_wifi_reconnect_ms = 0;
const unsigned long wifi_reconnect_interval_ms = 2000;
const unsigned long wifi_reconnect_timeout_ms = 1500;
const unsigned long wifi_enter_prov_after_ms = 15000;

bool connect_wifi_sta(const String& s, const String& p, uint32_t timeout_ms) {
  WiFi.mode(WIFI_STA);
  WiFi.setSleep(false);
  WiFi.setAutoReconnect(true);
  WiFi.persistent(false);
  WiFi.disconnect(true, false);
  delay(150);
  WiFi.begin(s.c_str(), p.c_str());
  uint32_t start = millis();
  while (WiFi.status() != WL_CONNECTED && (millis() - start) < timeout_ms) {
    delay(150);
  }
  return WiFi.status() == WL_CONNECTED;
}

bool connect_wifi_sta_keep_ap(const String& s, const String& p, uint32_t timeout_ms) {
  // 如果当前是配网模式，我们不能简单的 WIFI_AP_STA
  // 因为如果连接失败，底层有时候会把 AP 关掉或者搞乱
  WiFi.mode(WIFI_AP_STA);
  WiFi.setSleep(false);
  WiFi.setAutoReconnect(true);
  WiFi.persistent(false);
  WiFi.begin(s.c_str(), p.c_str());
  uint32_t start = millis();
  while (WiFi.status() != WL_CONNECTED && (millis() - start) < timeout_ms) {
    delay(150);
  }
  bool ok = WiFi.status() == WL_CONNECTED;
  if (!ok) {
    WiFi.disconnect(false, false);
    // 连接失败后，为了确保 AP 热点继续存活可见，重新激活 AP 模式
    WiFi.mode(WIFI_AP);
    WiFi.softAP(make_ap_ssid().c_str());
  }
  return ok;
}

bool load_saved_wifi(String& out_ssid, String& out_pass) {
  if (!wifi_prefs.begin("aiglass", true)) {
    return false;
  }
  out_ssid = wifi_prefs.getString("ssid", "");
  out_pass = wifi_prefs.getString("pass", "");
  wifi_prefs.end();
  return out_ssid.length() > 0;
}

void save_wifi(const String& s, const String& p) {
  if (!wifi_prefs.begin("aiglass", false)) {
    return;
  }
  wifi_prefs.putString("ssid", s);
  wifi_prefs.putString("pass", p);
  wifi_prefs.end();
}

void clear_wifi() {
  if (!wifi_prefs.begin("aiglass", false)) {
    return;
  }
  wifi_prefs.remove("ssid");
  wifi_prefs.remove("pass");
  wifi_prefs.end();
}

String make_ap_ssid() {
  String id = String((uint32_t)ESP.getEfuseMac(), HEX);
  id.toUpperCase();
  return String("aiglass-setup-") + id;
}

String html_escape(const String& s) {
  String o = s;
  o.replace("&", "&amp;");
  o.replace("<", "&lt;");
  o.replace(">", "&gt;");
  o.replace("\"", "&quot;");
  o.replace("'", "&#39;");
  return o;
}

void start_provision_ap() {
  prov_active = true;
  ws_connected = false;
  WiFi.mode(WIFI_AP_STA);
  WiFi.softAP(make_ap_ssid().c_str());
  IPAddress ap_ip = WiFi.softAPIP();
  prov_dns.start(53, "*", ap_ip);

  auto send_portal = []() {
    prov_server.sendHeader("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0");
    prov_server.sendHeader("Pragma", "no-cache");
    prov_server.sendHeader("Expires", "0");
    prov_server.sendHeader("Content-Type", "text/html; charset=utf-8");
    String page = "<!doctype html><html><head><meta charset=\"utf-8\"/>"
                  "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\"/>"
                  "<meta http-equiv=\"refresh\" content=\"0; url=/\"/>"
                  "<title>aiglass setup</title>"
                  "<style>"
                  "body{font-family:-apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Helvetica,Arial; background:#0b1220; color:#e6edf3; margin:0; padding:24px;}"
                  ".card{max-width:520px; margin:0 auto; background:#0f1a2c; border:1px solid #20304d; border-radius:14px; padding:18px 16px;}"
                  ".title{font-size:18px; font-weight:700; margin:0 0 8px 0;}"
                  ".muted{color:#9fb0c7; font-size:13px; line-height:1.5;}"
                  ".btn{display:inline-block; background:#2f81f7; color:#fff; padding:10px 12px; border-radius:10px; text-decoration:none; font-weight:600;}"
                  "</style></head><body>"
                  "<div class=\"card\">"
                  "<div class=\"title\">AI Glass 配网</div>"
                  "<div class=\"muted\">正在打开配置页面… 若未自动跳转，请点击下方按钮。</div>"
                  "<div style=\"height:12px\"></div>"
                  "<a class=\"btn\" href=\"/\">打开 WiFi 配置</a>"
                  "</div>"
                  "</body></html>";
    prov_server.send(200, "text/html; charset=utf-8", page);
  };

  prov_server.on("/", HTTP_GET, []() {
    // 强制先断开 STA 连接，以确保扫描稳定，并清空之前的扫描缓存
    WiFi.disconnect();
    delay(100);
    int n = WiFi.scanNetworks(false, true, false, 300);
    String options = "";
    String added_ssids = "";
    if (n > 0) {
      for (int i = 0; i < n; i++) {
        String s = WiFi.SSID(i);
        if (s.length() > 0 && added_ssids.indexOf("|" + s + "|") == -1) {
          options += "<option value=\"" + html_escape(s) + "\">" + html_escape(s) + "</option>";
          added_ssids += "|" + s + "|";
        }
      }
    }
    // 释放扫描占用的内存
    WiFi.scanDelete();
    String page = "<!doctype html><html><head><meta charset=\"utf-8\"/>"
                  "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\"/>"
                  "<title>aiglass setup</title>"
                  "<style>"
                  "body{font-family:-apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Helvetica,Arial; background:#0b1220; color:#e6edf3; margin:0; padding:24px;}"
                  ".card{max-width:520px; margin:0 auto; background:#0f1a2c; border:1px solid #20304d; border-radius:14px; padding:18px 16px;}"
                  ".title{font-size:18px; font-weight:700; margin:0 0 10px 0;}"
                  ".row{margin:10px 0;}"
                  "label{display:block; font-size:13px; color:#9fb0c7; margin-bottom:6px;}"
                  "select,input{width:100%; font-size:15px; padding:10px 10px; border-radius:10px; border:1px solid #223456; background:#0b1220; color:#e6edf3; box-sizing:border-box;}"
                  ".btn{width:100%; background:#2f81f7; color:#fff; padding:11px 12px; border:0; border-radius:10px; font-weight:700; font-size:15px;}"
                  ".btn2{width:100%; background:#24324b; color:#e6edf3; padding:10px 12px; border:1px solid #2a3a59; border-radius:10px; font-weight:600; font-size:14px;}"
                  ".muted{color:#9fb0c7; font-size:12px; line-height:1.5;}"
                  ".kv{display:flex; gap:8px; flex-wrap:wrap; margin-top:10px;}"
                  ".chip{background:#0b1220; border:1px solid #223456; padding:6px 8px; border-radius:999px; font-size:12px; color:#cdd9e5;}"
                  "</style></head><body>"
                  "<div class=\"card\">"
                  "<div class=\"title\">AI Glass 配网</div>"
                  "<div class=\"muted\">选择要连接的 WiFi，并输入密码。保存后设备会重启并连接。</div>"
                  "<form method=\"POST\" action=\"/save\">"
                  "<div class=\"row\"><label>选择附近 WiFi</label>"
                  "<select id=\"wifi-select\" onchange=\"document.getElementById('ssid-input').value = this.value;\">"
                  "<option value=\"\">-- 点击选择 --</option>" + options + "</select></div>"
                  "<div class=\"row\"><label>WiFi (SSID)</label>"
                  "<input id=\"ssid-input\" name=\"ssid\" type=\"text\" placeholder=\"点击上方选择或手动输入\" autocomplete=\"off\" required/></div>"
                  "<div class=\"row\"><label>密码</label><input name=\"pass\" type=\"password\" placeholder=\"若无密码可留空\"/></div>"
                  "<div class=\"row\"><button class=\"btn\" type=\"submit\">保存并连接</button></div>"
                  "</form>"
                  "<form method=\"POST\" action=\"/forget\">"
                  "<div class=\"row\"><button class=\"btn2\" type=\"submit\">清除已保存 WiFi</button></div>"
                  "</form>"
                  "<div class=\"kv\">"
                  "<div class=\"chip\">AP: " + html_escape(make_ap_ssid()) + "</div>"
                  "<div class=\"chip\">地址: 192.168.4.1</div>"
                  "</div>"
                  "</div>"
                  "</body></html>";
    prov_server.send(200, "text/html; charset=utf-8", page);
  });

  prov_server.on("/save", HTTP_POST, []() {
    String s = prov_server.arg("ssid");
    String p = prov_server.arg("pass");
    s.trim();
    if (s.length() == 0) {
      prov_server.send(400, "text/plain; charset=utf-8", "ssid为空");
      return;
    }
    prov_server.send(200, "text/plain; charset=utf-8", "正在连接…（若密码错误会保持在配网页）");
    delay(200);
    bool ok = connect_wifi_sta_keep_ap(s, p, 12000);
    if (ok) {
      save_wifi(s, p);
      delay(150);
      ESP.restart();
      return;
    }
  });

  prov_server.on("/forget", HTTP_POST, []() {
    clear_wifi();
    prov_server.send(200, "text/plain; charset=utf-8", "已清除，设备将重启…");
    delay(200);
    ESP.restart();
  });

  prov_server.on("/generate_204", HTTP_GET, send_portal);
  prov_server.on("/gen_204", HTTP_GET, send_portal);
  prov_server.on("/redirect", HTTP_GET, send_portal);
  prov_server.on("/success.txt", HTTP_GET, send_portal);
  prov_server.on("/hotspot-detect.html", HTTP_GET, send_portal);
  prov_server.on("/library/test/success.html", HTTP_GET, send_portal);
  prov_server.on("/ncsi.txt", HTTP_GET, send_portal);
  prov_server.on("/connecttest.txt", HTTP_GET, send_portal);
  prov_server.on("/fwlink", HTTP_GET, send_portal);
  prov_server.on("/favicon.ico", HTTP_GET, []() { prov_server.send(204, "text/plain", ""); });

  prov_server.onNotFound([]() {
    prov_server.sendHeader("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0");
    prov_server.sendHeader("Pragma", "no-cache");
    prov_server.sendHeader("Expires", "0");
    prov_server.sendHeader("Location", "/", true);
    prov_server.send(302, "text/plain", "");
  });

  prov_server.begin();
  Serial.print("Provision AP IP: ");
  Serial.println(WiFi.softAPIP());
}

bool ensure_wifi_connected() {
  if (prov_active) {
    return false;
  }
  if (WiFi.status() == WL_CONNECTED) {
    wifi_lost_since_ms = 0;
    return true;
  }

  unsigned long now = millis();
  if (wifi_lost_since_ms == 0) {
    wifi_lost_since_ms = now;
    ws_connected = false;
  }

  if (now - last_wifi_reconnect_ms >= wifi_reconnect_interval_ms) {
    last_wifi_reconnect_ms = now;
    if (have_saved_wifi && saved_wifi_ssid.length() > 0) {
      connect_wifi_sta(saved_wifi_ssid, saved_wifi_pass, wifi_reconnect_timeout_ms);
    } else {
      connect_wifi_sta(String(ssid), String(password), wifi_reconnect_timeout_ms);
    }
  }

  if (WiFi.status() == WL_CONNECTED) {
    wifi_lost_since_ms = 0;
    Serial.println("WiFi reconnected");
    Serial.print("IP address: ");
    Serial.println(WiFi.localIP());
    refresh_server_endpoint(true);
    return true;
  }

  int st = (int)WiFi.status();
  if ((st == WL_NO_SSID_AVAIL || st == WL_CONNECT_FAILED) && (now - wifi_lost_since_ms > 8000)) {
    start_provision_ap();
    return false;
  }
  if (now - wifi_lost_since_ms > wifi_enter_prov_after_ms) {
    start_provision_ap();
    return false;
  }
  return false;
}

#if ENABLE_BLE
const char* ble_service_uuid = "e3b5f9c2-7f2a-4c98-9d72-0a1f1b6b6d01";
const char* ble_cmd_uuid = "4a4f4dd7-6f1f-4e38-a1bd-8a0e46e2f2d1";
#endif
enum CommMode { MODE_WIFI = 0, MODE_BLE = 1 };
CommMode comm_mode = MODE_WIFI;

void i2s_init() {
#if ENABLE_AUDIO
  // 配置 I2S_NUM_0 为内置 DAC/PDM 输出模式（模拟信号），用于驱动 PAM8403
  i2s_config_t config = {};
  config.mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_TX | I2S_MODE_PDM);
  config.sample_rate = 16000;
  config.bits_per_sample = I2S_BITS_PER_SAMPLE_16BIT;
  config.channel_format = I2S_CHANNEL_FMT_ONLY_RIGHT;
  config.communication_format = I2S_COMM_FORMAT_STAND_I2S;
  config.intr_alloc_flags = ESP_INTR_FLAG_LEVEL1;
  config.dma_buf_count = 4;
  config.dma_buf_len = 512;
  config.use_apll = false;
  config.tx_desc_auto_clear = true;
  config.fixed_mclk = 0;
  
  i2s_driver_install(I2S_NUM_0, &config, 0, NULL);
  
  i2s_pin_config_t pins = {};
  pins.bck_io_num = -1;   // PDM TX 模式下，这些时钟信号不需要
  pins.ws_io_num = -1;    // PDM TX 模式下不需要
  pins.data_out_num = i2s_dout; // 这里是 GPIO4(D4)，也就是接 PAM8403 的引脚
  pins.data_in_num = -1;
  i2s_set_pin(I2S_NUM_0, &pins);
  
  i2s_zero_dma_buffer(I2S_NUM_0);
  Serial.println("[AUDIO] I2S_NUM_0 configured for PAM8403 (PDM TX)");
#endif
}

void mic_init() {
#if ENABLE_MIC
  if (enable_button && button_mode_pin == i2s_din) {
    Serial.println("[MIC] disabled: GPIO41 is already reserved for button");
    mic_ready = false;
    return;
  }

  i2s_config_t config = {};
  config.mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX | I2S_MODE_PDM);
  config.sample_rate = mic_sample_rate;
  config.bits_per_sample = I2S_BITS_PER_SAMPLE_16BIT;
  config.channel_format = I2S_CHANNEL_FMT_ONLY_LEFT;
  config.communication_format = I2S_COMM_FORMAT_STAND_I2S;
  config.intr_alloc_flags = 0;
  config.dma_buf_count = 8;
  config.dma_buf_len = 128;
  config.use_apll = false;
  config.tx_desc_auto_clear = false;
  config.fixed_mclk = 0;

  esp_err_t err = i2s_driver_install(i2s_mic_port, &config, 0, NULL);
  if (err != ESP_OK) {
    Serial.printf("[MIC] i2s_driver_install failed: 0x%x\n", err);
    // 强制清理重试机制
    i2s_driver_uninstall(i2s_mic_port);
    delay(100);
    err = i2s_driver_install(i2s_mic_port, &config, 0, NULL);
    if (err != ESP_OK) {
       mic_ready = false;
       return;
    }
  }

  i2s_pin_config_t pins = {};
  pins.bck_io_num = I2S_PIN_NO_CHANGE;
  pins.ws_io_num = i2s_clk;          // PDM CLK
  pins.data_out_num = I2S_PIN_NO_CHANGE;
  pins.data_in_num = i2s_din;        // PDM DATA
  err = i2s_set_pin(i2s_mic_port, &pins);
  if (err != ESP_OK) {
    Serial.printf("[MIC] i2s_set_pin failed: 0x%x\n", err);
    i2s_driver_uninstall(i2s_mic_port);
    mic_ready = false;
    return;
  }

  i2s_zero_dma_buffer(i2s_mic_port);
  mic_ready = true;
  Serial.println("[MIC] ready on I2S_NUM_1 (PDM RX 16kHz mono)");
#endif
}

int read_mic_level() {
#if ENABLE_MIC
  if (!mic_enabled || !mic_ready) {
    return -1;
  }

  int16_t samples[mic_samples_per_read];
  size_t bytes_read = 0;
  esp_err_t err = i2s_read(i2s_mic_port, samples, sizeof(samples), &bytes_read, 20 / portTICK_PERIOD_MS);
  if (err != ESP_OK || bytes_read == 0) {
    return -1;
  }

  size_t count = bytes_read / sizeof(int16_t);
  if (count == 0) {
    return -1;
  }

  uint32_t sum = 0;
  for (size_t i = 0; i < count; i++) {
    sum += abs((int)samples[i]);
  }
  return (int)(sum / count);
#else
  return -1;
#endif
}

bool upload_mic_query() {
#if ENABLE_MIC
  if (!mic_enabled || !mic_ready || mic_busy) {
    return false;
  }
  mic_busy = true;
  const size_t total_samples = (mic_sample_rate * mic_capture_ms) / 1000;
  const size_t total_bytes = total_samples * sizeof(int16_t);
  uint8_t* pcm = (uint8_t*)malloc(total_bytes);
  if (!pcm) {
    Serial.println("[MIC] alloc failed");
    mic_busy = false;
    return false;
  }

  size_t written = 0;
  while (written < total_bytes) {
    size_t chunk_read = 0;
    size_t need = min((size_t)(512), total_bytes - written);
    esp_err_t err = i2s_read(i2s_mic_port, pcm + written, need, &chunk_read, portMAX_DELAY);
    if (err != ESP_OK || chunk_read == 0) {
      break;
    }
    written += chunk_read;
  }

  bool ok = false;
  if (written > 0) {
    HTTPClient http;
    // Fix URL generation
    String base_url = "http://" + active_ws_host + ":" + String(active_ws_port);
    String url = base_url + "/api/device/audio_upload";
    http.begin(url);
    http.addHeader("Content-Type", "application/octet-stream");
    http.addHeader("X-Aiglass-Device", device_id);
    int code = http.POST(pcm, written);
    Serial.printf("[MIC] upload %u bytes, http=%d\n", (unsigned)written, code);
    if (code > 0) {
      String resp = http.getString();
      if (resp.length()) {
        Serial.printf("[MIC] response=%s\n", resp.c_str());
      }
      ok = code >= 200 && code < 300;
    }
    http.end();
  }
  free(pcm);
  mic_busy = false;
  return ok;
#else
  return false;
#endif
}

void play_tone(int freq, int ms) {
#if ENABLE_AUDIO
  const int sample_rate = 16000;
  const int samples = (sample_rate * ms) / 1000;
  const float step = 2.0f * 3.1415926f * (float)freq / (float)sample_rate;
  size_t bytes_written = 0;
  int16_t sample = 0;
  float phase = 0.0f;
  for (int i = 0; i < samples; i++) {
    sample = (int16_t)(sinf(phase) * 12000.0f);
    phase += step;
    i2s_write(I2S_NUM_0, &sample, sizeof(sample), &bytes_written, portMAX_DELAY);
  }
#endif
}

void onWsEvent(WebsocketsEvent event, String data) {
  if (event == WebsocketsEvent::ConnectionOpened) {
    ws_connected = true;
    Serial.println("WS Connected");
  } else if (event == WebsocketsEvent::ConnectionClosed) {
    ws_connected = false;
    Serial.println("WS Disconnected");
  }
}

void onWsMessage(WebsocketsMessage message) {
  if (message.isBinary()) {
#if ENABLE_AUDIO
    // 接收到二进制数据，直接作为 PCM 音频写入 I2S 播放
    size_t len = message.length();
    const char* data = message.c_str();
    size_t bytes_written = 0;
    i2s_write(I2S_NUM_0, data, len, &bytes_written, portMAX_DELAY);
#endif
    return;
  }

  if (!message.isText()) {
    return;
  }
  String s = message.data();
  s.trim();
  if (s.length() == 0) {
    return;
  }
  if (s.startsWith("SET:FPS=")) {
    String v = s.substring(String("SET:FPS=").length());
    handle_command("FPS=" + v);
    return;
  }
  if (s.startsWith("SET:FRAMESIZE=")) {
    String v = s.substring(String("SET:FRAMESIZE=").length());
    v.toUpperCase();
    if (v == "QQVGA") handle_command("FS=0");
    else if (v == "QVGA") handle_command("FS=1");
    else if (v == "VGA") handle_command("FS=2");
    else if (v == "SVGA") handle_command("FS=3");
    else if (v == "XGA") handle_command("FS=4");
    else if (v == "HD") handle_command("FS=5");
    return;
  }
  if (s.startsWith("SET:GRAY=")) {
    String v = s.substring(String("SET:GRAY=").length());
    handle_command("GRAY=" + v);
    return;
  }
  handle_command(s);
}

String build_ws_url(const String& host, uint16_t port, const String& path) {
  return String("ws://") + host + ":" + String(port) + path;
}

String build_http_url(const String& host, uint16_t port, const String& path) {
  return String("http://") + host + ":" + String(port) + path;
}

String json_string_value(const String& payload, const String& key) {
  String token = "\"" + key + "\":\"";
  int start = payload.indexOf(token);
  if (start < 0) {
    return "";
  }
  start += token.length();
  int end = payload.indexOf('"', start);
  if (end < 0) {
    return "";
  }
  String out = payload.substring(start, end);
  out.replace("\\uFF1B", "；");
  out.replace("\\n", " ");
  return out;
}

bool discover_server() {
  discovery_udp.stop();
  if (!discovery_udp.begin(discovery_port)) {
    return false;
  }
  discovery_udp.beginPacket("255.255.255.255", discovery_port);
  discovery_udp.print("AIGLASS_DISCOVER ESP32");
  discovery_udp.endPacket();

  unsigned long start_ms = millis();
  while (millis() - start_ms < 800) {
    int packet_size = discovery_udp.parsePacket();
    if (packet_size > 0) {
      char buffer[160];
      int n = discovery_udp.read(buffer, sizeof(buffer) - 1);
      if (n <= 0) {
        continue;
      }
      buffer[n] = '\0';
      String resp = String(buffer);
      resp.trim();
      if (!resp.startsWith("AIGLASS_SERVER ")) {
        continue;
      }
      int p1 = resp.indexOf(' ');
      int p2 = resp.indexOf(' ', p1 + 1);
      int p3 = resp.indexOf(' ', p2 + 1);
      if (p1 < 0 || p2 < 0 || p3 < 0) {
        continue;
      }
      active_ws_host = resp.substring(p1 + 1, p2);
      active_ws_port = (uint16_t)resp.substring(p2 + 1, p3).toInt();
      active_ws_path = resp.substring(p3 + 1);
      if (!active_ws_path.startsWith("/")) {
        active_ws_path = "/" + active_ws_path;
      }
      active_ws_url = build_ws_url(active_ws_host, active_ws_port, active_ws_path);
      active_http_url = build_http_url(active_ws_host, active_ws_port, http_infer_path);
      Serial.print("Discovered WS URL: ");
      Serial.println(active_ws_url);
      discovery_udp.stop();
      return true;
    }
    delay(20);
  }
  discovery_udp.stop();
  return false;
}

void refresh_server_endpoint(bool force_discovery) {
  if (WiFi.status() != WL_CONNECTED) {
    return;
  }
  if (force_discovery || active_ws_url.length() == 0) {
    if (discover_server()) {
      return;
    }
  }
  active_ws_host = websocket_server_host;
  active_ws_port = websocket_server_port;
  active_ws_path = websocket_path;
  active_ws_url = build_ws_url(active_ws_host, active_ws_port, active_ws_path);
  active_http_url = build_http_url(active_ws_host, active_ws_port, http_infer_path);
}

void apply_camera_settings() {
  sensor_t * s = esp_camera_sensor_get();
  if (s) {
    s->set_special_effect(s, edge_grayscale ? 2 : 0);
    s->set_framesize(s, edge_frame_size);
    s->set_quality(s, edge_jpeg_quality);
  }
}

void handle_command(String cmd) {
  cmd.trim();
  if (cmd.length() == 0) {
    return;
  }
  last_cmd = cmd;
  String up = cmd;
  up.toUpperCase();
  if (up.startsWith("GRAY=")) {
    edge_grayscale = up.substring(5).toInt() > 0;
    apply_camera_settings();
    return;
  }
  if (up.startsWith("FPS=")) {
    int fps = up.substring(4).toInt();
    if (fps <= 0) return;
    edge_min_send_interval_ms = (uint16_t)max(20, 1000 / fps);
    return;
  }
  if (up.startsWith("Q=")) {
    int q = up.substring(2).toInt();
    if (q <= 0) return;
    edge_jpeg_quality = (uint8_t)min(63, max(10, q));
    apply_camera_settings();
    return;
  }
  if (up.startsWith("SENDN=")) {
    int n = up.substring(6).toInt();
    if (n <= 0) return;
    edge_send_every_n = (uint8_t)min(10, max(1, n));
    return;
  }
  if (up.startsWith("LENJUMP=")) {
    int v = up.substring(8).toInt();
    if (v <= 0) return;
    edge_len_delta_threshold = v;
    return;
  }
  if (up.startsWith("FS=")) {
    int v = up.substring(3).toInt();
    if (v == 0) edge_frame_size = FRAMESIZE_QQVGA;
    if (v == 1) edge_frame_size = FRAMESIZE_QVGA;
    if (v == 2) edge_frame_size = FRAMESIZE_VGA;
    if (v == 3) edge_frame_size = FRAMESIZE_SVGA;
    if (v == 4) edge_frame_size = FRAMESIZE_XGA;
    if (v == 5) edge_frame_size = FRAMESIZE_HD;
    apply_camera_settings();
    return;
  }
  if (up == "MIC=1" || up == "MIC=ON") {
#if ENABLE_MIC
    mic_enabled = true;
    Serial.println("[MIC] enabled");
#endif
    return;
  }
  if (up == "MIC=0" || up == "MIC=OFF") {
#if ENABLE_MIC
    mic_enabled = false;
    Serial.println("[MIC] disabled");
#endif
    return;
  }
  if (up == "PROV=1") {
    start_provision_ap();
    return;
  }
  if (up == "FORGETWIFI") {
    clear_wifi();
    start_provision_ap();
    return;
  }
  if (up == "MODE=BLE") {
#if ENABLE_BLE
    comm_mode = MODE_BLE;
    play_tone(880, 120);
#endif
    return;
  }
  if (up == "MODE=WIFI") {
    comm_mode = MODE_WIFI;
    use_http_upload = false;
    play_tone(660, 120);
    return;
  }
  if (up == "MODE=HTTP") {
    comm_mode = MODE_WIFI;
    use_http_upload = true;
    play_tone(660, 120);
    return;
  }
}

#if ENABLE_MQTT
void mqtt_callback(char* topic, byte* payload, unsigned int length) {
  String cmd = "";
  for (unsigned int i = 0; i < length; i++) {
    cmd += (char)payload[i];
  }
  String topic_str = String(topic);
  if (topic_str == mqtt_result_topic) {
    String text = json_string_value(cmd, "text");
    if (text.length() > 0) {
      Serial.print("MQTT Result: ");
      Serial.println(text);
    } else {
      Serial.print("MQTT Result Raw: ");
      Serial.println(cmd);
    }
    return;
  }
  handle_command(cmd);
}
#endif

bool upload_frame_http(camera_fb_t * fb) {
  if (active_http_url.length() == 0) {
    refresh_server_endpoint(false);
  }
  if (active_http_url.length() == 0) {
    return false;
  }
  HTTPClient http;
  http.setConnectTimeout(1200);
  http.setTimeout(2000);
  http.begin(active_http_url);
  http.addHeader("Content-Type", "image/jpeg");
  http.addHeader("X-AIGLASS-Device", device_id);
  int code = http.POST(fb->buf, fb->len);
  bool ok = code > 0 && code < 300;
  if (ok) {
    String resp = http.getString();
    String text = json_string_value(resp, "text");
    if (text.length() > 0) {
      Serial.print("HTTP Result: ");
      Serial.println(text);
    }
  } else {
    Serial.printf("HTTP infer failed: %d\n", code);
  }
  http.end();
  return ok;
}

#if ENABLE_BLE
class BleCmdCallbacks: public BLECharacteristicCallbacks {
  void onWrite(BLECharacteristic* characteristic) override {
    std::string value = characteristic->getValue(); 
    if (value.empty()) {
      return;
    }
    String cmd = "";
    for (size_t i = 0; i < value.size(); i++) {
      cmd += (char)value[i];
    }
    handle_command(cmd);
  }
};
#endif

// Pin definition for XIAO ESP32S3 Sense
#define PWDN_GPIO_NUM     -1
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM     10
#define SIOD_GPIO_NUM     40
#define SIOC_GPIO_NUM     39

#define Y9_GPIO_NUM       48
#define Y8_GPIO_NUM       11
#define Y7_GPIO_NUM       12
#define Y6_GPIO_NUM       14
#define Y5_GPIO_NUM       16
#define Y4_GPIO_NUM       18
#define Y3_GPIO_NUM       17
#define Y2_GPIO_NUM       15
#define VSYNC_GPIO_NUM    38
#define HREF_GPIO_NUM     47
#define PCLK_GPIO_NUM     13

void setup() {
  Serial.begin(115200);
  client.onEvent(onWsEvent);
  client.onMessage(onWsMessage);
  if (enable_button) {
    pinMode(button_mode_pin, INPUT_PULLUP);
  }
  // 等待一段时间，错开 Wi-Fi 连通时的巨大电流峰值
  delay(1500);

  i2s_init(); // 初始化功放扬声器 (I2S0)
  mic_init(); // 初始化 PDM 麦克风 (I2S1)
  
  // 播放开机提示音，测试喇叭是否正常
#if ENABLE_AUDIO
  play_tone(660, 150);
  delay(50);
  play_tone(880, 200);
#endif

  bool force_prov = false;
  if (enable_button) {
    unsigned long t0 = millis();
    unsigned long held_ms = 0;
    while (millis() - t0 < 1200) {
      if (digitalRead(button_mode_pin) == LOW) {
        held_ms += 10;
        if (held_ms >= 900) {
          force_prov = true;
        }
      } else {
        held_ms = 0;
      }
      delay(10);
    }
  }
  
  // Camera Init
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sscb_sda = SIOD_GPIO_NUM;
  config.pin_sscb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;
  
  // 初始配置较高的分辨率，以保证画质和模型识别率
  // 可选：FRAMESIZE_VGA (640x480), FRAMESIZE_SVGA (800x600), FRAMESIZE_XGA (1024x768)
  config.frame_size = edge_frame_size;
  config.jpeg_quality = edge_jpeg_quality;
  config.fb_count = 1;
  // 对于 XIAO ESP32S3，必须要明确设置内存分配方式，因为它的 PSRAM 较大，而且摄像头缓冲区需要连续内存
  config.fb_location = CAMERA_FB_IN_PSRAM;
  config.grab_mode = CAMERA_GRAB_WHEN_EMPTY;

  if(psramFound()){
    config.jpeg_quality = edge_jpeg_quality;
    config.fb_count = 2; // 不要设置太多，XIAO 上 2 个即可
  } else {
    config.frame_size = edge_frame_size;
    config.fb_count = 1;
    config.fb_location = CAMERA_FB_IN_DRAM; // 如果没找到 PSRAM 则降级为内部 RAM
  }
  Serial.printf("PSRAM: %s\n", psramFound() ? "ON" : "OFF");

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed with error 0x%x\n", err);
    return;
  }
  
  sensor_t * s = esp_camera_sensor_get();
  if (s) {
    s->set_special_effect(s, edge_grayscale ? 2 : 0);
    s->set_framesize(s, edge_frame_size);
    // 设置硬件级图像翻转：如果设备横放（平躺），可以尝试在此处直接设置
    s->set_hmirror(s, 1); // 水平镜像
    s->set_vflip(s, 1);   // 垂直翻转
  }

  // WiFi Connect
  if (force_prov) {
    Serial.println("Entering provisioning AP (button long-press)");
    start_provision_ap();
    return;
  }
  bool ok = false;
  have_saved_wifi = load_saved_wifi(saved_wifi_ssid, saved_wifi_pass);
  if (have_saved_wifi) {
    Serial.print("Saved WiFi SSID: ");
    Serial.println(saved_wifi_ssid);
    ok = connect_wifi_sta(saved_wifi_ssid, saved_wifi_pass, 12000);
  }
  if (!ok) {
    Serial.println("Connecting to fallback WiFi...");
    ok = connect_wifi_sta(String(ssid), String(password), 12000);
  }
  if (!ok) {
    Serial.print("WiFi connect failed, status=");
    Serial.println((int)WiFi.status());
    start_provision_ap();
    return;
  }
  Serial.println("WiFi connected");
  Serial.print("IP address: ");
  Serial.println(WiFi.localIP());
  device_id = String((uint32_t)ESP.getEfuseMac(), HEX);
  mqtt_result_topic = String(mqtt_result_prefix) + "/" + device_id + "/result";
  refresh_server_endpoint(true);
#if ENABLE_MQTT
  mqtt_client.setServer(mqtt_server_host, mqtt_server_port);
  mqtt_client.setCallback(mqtt_callback);
#endif
#if ENABLE_BLE
  BLEDevice::init("aiglass-esp32");
  BLEServer *ble_server = BLEDevice::createServer();
  BLEService *ble_service = ble_server->createService(ble_service_uuid);
  BLECharacteristic *ble_cmd = ble_service->createCharacteristic(ble_cmd_uuid, BLECharacteristic::PROPERTY_WRITE);
  ble_cmd->setCallbacks(new BleCmdCallbacks());
  ble_service->start();
  BLEAdvertising *adv = BLEDevice::getAdvertising();
  adv->addServiceUUID(ble_service_uuid);
  adv->start();
#endif
  
  // WS Connect
  if (!use_http_upload) {
    Serial.printf("Connecting to WS Server: %s:%d\n", active_ws_host.c_str(), active_ws_port);
    Serial.printf("Connecting to WS URL: %s\n", active_ws_url.c_str());
    bool connected = client.connect(active_ws_url);
    if(connected) {
      Serial.println("Connected to WS Server!");
      ws_connected = true;
    } else {
      Serial.println("Not Connected to WS Server!");
      ws_connected = false;
    }
  } else {
    Serial.printf("HTTP infer URL: %s\n", active_http_url.c_str());
  }
}

void loop() {
  if (prov_active) {
    prov_dns.processNextRequest();
    prov_server.handleClient();
    delay(2);
    return;
  }
  if (!ensure_wifi_connected()) {
    delay(5);
    return;
  }
#if ENABLE_MIC
  if (mic_enabled && mic_ready) {
    unsigned long now_mic = millis();
    if (now_mic - last_mic_debug_ms >= 400) {
      last_mic_debug_ms = now_mic;
      int mic_level = read_mic_level();
      if (mic_level >= 0) {
        Serial.printf("[MIC] level=%d\n", mic_level);
        if (!mic_busy && mic_level >= mic_voice_threshold && now_mic - last_mic_upload_ms >= mic_upload_cooldown_ms) {
          last_mic_upload_ms = now_mic;
          Serial.println("[MIC] voice trigger -> upload");
          upload_mic_query();
        }
      }
    }
  }
#endif
  if (enable_button) {
    static int last_btn = HIGH;
    static unsigned long last_btn_ms = 0;
    int btn = digitalRead(button_mode_pin);
    unsigned long now_btn = millis();
    if (btn != last_btn && now_btn - last_btn_ms > 50) {
      last_btn_ms = now_btn;
      last_btn = btn;
      if (btn == LOW) {
  #if ENABLE_BLE
        comm_mode = (comm_mode == MODE_WIFI) ? MODE_BLE : MODE_WIFI;
        play_tone(comm_mode == MODE_WIFI ? 660 : 880, 120);
  #else
        comm_mode = MODE_WIFI;
        play_tone(660, 120);
  #endif
      }
    }
  }

  if (!use_http_upload) {
    client.poll();
  }
  if (!use_http_upload && comm_mode == MODE_WIFI && !ws_connected) {
    unsigned long now = millis();
    if (now - last_ws_attempt_ms > 1000) {
      last_ws_attempt_ms = now;
      if (now - last_discovery_ms > 5000 || active_ws_url.length() == 0) {
        last_discovery_ms = now;
        refresh_server_endpoint(true);
      }
      client.connect(active_ws_url);
    }
  }
  
  if (comm_mode != MODE_WIFI) {
    delay(40);
    return;
  }

  static size_t last_len = 0;
  static unsigned long last_send_ms = 0;
  static uint32_t frame_counter = 0;
  frame_counter += 1;

  unsigned long now = millis();
  if (now - last_send_ms < edge_min_send_interval_ms) {
    delay(2);
    return;
  }

  camera_fb_t * fb = esp_camera_fb_get();
  if(!fb) {
    Serial.println("Camera capture failed");
    return;
  }
  size_t fb_len = fb->len;
  
  // Send Frame (Binary)
  bool sent_ok = false;
  if(use_http_upload) {
    if (now - last_http_post_ms >= edge_min_send_interval_ms) {
      sent_ok = upload_frame_http(fb);
      if (sent_ok) {
        last_http_post_ms = now;
        last_send_ms = now;
        last_len = fb->len;
      }
    }
  } else if(ws_connected) {
    bool force_send = (frame_counter % edge_send_every_n) == 0;
    bool len_jump = (last_len == 0) || ((int)fb->len - (int)last_len > edge_len_delta_threshold) || ((int)last_len - (int)fb->len > edge_len_delta_threshold);
    if (now - last_send_ms >= edge_min_send_interval_ms && (force_send || len_jump)) {
      client.sendBinary((const char*)fb->buf, fb->len);
      last_send_ms = now;
      last_len = fb->len;
      sent_ok = true;
    }
  }
  
  esp_camera_fb_return(fb);

#if ENABLE_MQTT
  if (!mqtt_client.connected()) {
    unsigned long now = millis();
    if (now - last_mqtt_attempt_ms > 1500) {
      last_mqtt_attempt_ms = now;
      String client_id = "aiglass-esp32-";
      client_id += String((uint32_t)ESP.getEfuseMac(), HEX);
      if (mqtt_client.connect(client_id.c_str())) {
        mqtt_client.subscribe(mqtt_cmd_topic);
        mqtt_client.subscribe(mqtt_result_topic.c_str());
      }
    }
  } else {
    mqtt_client.loop();
    unsigned long now = millis();
    if (now - last_mqtt_publish_ms > 2000) {
      last_mqtt_publish_ms = now;
      String payload = "{";
      payload += "\"device_id\":\"" + device_id + "\",";
      payload += "\"ip\":\"" + WiFi.localIP().toString() + "\",";
      payload += "\"rssi\":" + String(WiFi.RSSI()) + ",";
      payload += "\"ws\":" + String(ws_connected ? 1 : 0) + ",";
      payload += "\"http\":" + String(use_http_upload ? 1 : 0) + ",";
      payload += "\"len\":" + String(fb_len) + ",";
      payload += "\"cmd\":\"" + last_cmd + "\"";
      payload += "}";
      mqtt_client.publish(mqtt_pub_topic, payload.c_str());
    }
  }
#endif

  delay(2);
}
