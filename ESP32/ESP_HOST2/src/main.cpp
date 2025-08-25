#include <Arduino.h>
#include<WiFi.h>
#define myserial Serial2

const char* ssid = "ESP32_AP";
const char* password = "12345678";

const char* server_ip= "192.168.4.1";
const int server_port = 80;

int gtri1=0;
int gtri2=0;

String sent2;

void configStaticIP()
{
 
  IPAddress local_IP(172,26,179,199);
  IPAddress gateway(172,26,183,254);
  IPAddress subnet(255,255,248,0);
  WiFi.config(local_IP, gateway, subnet);
 
}

void connectToWiFi() {
    WiFi.begin(ssid, password);

    Serial.print("Connecting to AP: ");
    Serial.println(ssid);

    int retry = 0;
    int max_retries = 10; 

    while (WiFi.status() != WL_CONNECTED && retry < max_retries) {

        Serial.print(".");
        delay(1000);
        retry++;
    }

    Serial.println();

    if (WiFi.status() == WL_CONNECTED) {
        Serial.println("Connect OK!");
        Serial.print("IP Address: ");
        Serial.println(WiFi.localIP());
    } else {
        Serial.println("MIST WiFi, reset ESP ... ");
        delay(1000); 
        ESP.restart();
    }
}


void setup() {
  Serial.begin(9600);
  myserial.begin(9600, SERIAL_8N1, 16, 17); 
  configStaticIP();
  connectToWiFi();
}

void loop() {

  WiFiClient client;
  if(!client.connect(server_ip, server_port)){
    Serial.print("Kết nối thất bại");
    delay(1000);
    return;
  }
  myserial.print(gtri1);
  myserial.print(",");
  myserial.println(gtri2);

  sent2="_______ESP_2:______Địa chỉ MAC: "+WiFi.macAddress()+ "           Cảm biến 3: "+String(gtri1)+"             Cảm biến 4: "+String(gtri2);
  client.println(sent2);
  Serial.println("Đã gửi: "+sent2);
  client.stop();
  gtri1++;
  gtri2++;
  delay(1000);

  
}