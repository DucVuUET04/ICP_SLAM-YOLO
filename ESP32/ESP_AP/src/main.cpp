#include <Arduino.h>
#include <WiFi.h>
#include <esp_wifi.h>


#define DX_PIN 26
#define SX_PIN 27       
#define UART_RX_PIN 16  
#define UART_TX_PIN 17
#define mySerial Serial1

const char* ssid = "ESP32_AP";
const char* password = "12345678";
const int MAX_CLIENTS = 2;

WiFiClient clients[MAX_CLIENTS];
WiFiServer server(80); 

void setup() {
  Serial.begin(9600);
  Serial.println("Bắt đầu khởi tạo AP...");

  WiFi.softAP(ssid, password);
  IPAddress IP = WiFi.softAPIP();

  pinMode(DX_PIN, INPUT);
  pinMode(SX_PIN, INPUT);
  server.begin();
  Serial.println("Khởi tạo server thành công");
  mySerial.begin(9600, SERIAL_8N1, UART_RX_PIN, UART_TX_PIN);
  
}

void handle_uart_loopback() {
  static int lastDxState = LOW;
  int currentDxState = digitalRead(DX_PIN);
  static int lastSxState = LOW;
  int currentSxState = digitalRead(SX_PIN);

    if (currentDxState == HIGH && lastDxState == LOW) {
    Serial.println("Phát hiện chân DX HIGH, bắt đầu gửi dữ liệu...");

    const int MAX_RETRIES = 2;
    int attempt = 0;
    bool success = false;

    while (attempt < MAX_RETRIES && !success) {
      Serial.print("Lần gửi thứ ");
      Serial.println(attempt + 1);

      mySerial.println("DX:0");
      delay(20);

      unsigned long startTime = millis();
      while (!mySerial.available() && millis() - startTime < 1000) {
        // đợi phản hồi tối đa 300ms
      }

      if (mySerial.available()) {
        String response = mySerial.readStringUntil('\n');
        response.trim();

        Serial.print("Phản hồi từ UART: '");
        Serial.print(response);
        Serial.println("'");

        if (response == "DX:0") {
          //mySerial.println("OK");
          Serial.println("Đã nhận : 'DX:0', gửi phản hồi OK.");
          success = true;
        } else {
          //mySerial.println("ERROR");
          Serial.println("Thử lại, phản hồi không chính xác.");
          success = false;
        }
      } else {
        Serial.println("Không có phản hồi UART.");
      }

      attempt++;
      if (!success && attempt < MAX_RETRIES) {
        delay(1000); 
      }
    }

    if (!success) {
      Serial.println("Hết số lần thử, không nhận được phản hồi chính xác.");
    }
  }

  lastDxState = currentDxState;
}
 
void handle_wifi_clients() {
  if (server.hasClient()) {
    bool accepted = false;
    WiFiClient newClient = server.available();
    for (int i = 0; i < MAX_CLIENTS; i++) {
      if (!clients[i] || !clients[i].connected()) {
        if (clients[i]) {
          clients[i].stop();
        }
        clients[i] = newClient;
        accepted = true;
        break;
      }
    }
    if (!accepted) {
      Serial.println("Không thể chấp nhận client mới, đã đủ số lượng.");
      newClient.stop();
    }
  }


  for (int i = 0; i < MAX_CLIENTS; i++) {
    if (clients[i] && clients[i].connected()) {
      if (clients[i].available()) {
        String data = clients[i].readStringUntil('\n');
        data.trim();
        Serial.print("Nhận từ client ");
        Serial.print(clients[i].remoteIP());
        Serial.print(": ");
        Serial.println(data);
      }
    }
  }
}

void loop() {
  handle_wifi_clients();
  handle_uart_loopback();
}