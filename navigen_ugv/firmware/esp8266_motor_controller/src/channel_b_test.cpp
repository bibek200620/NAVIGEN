#include <Arduino.h>

#define IN3 D1
#define IN4 D2
#define ENB D7

void setup() {
  pinMode(IN3, OUTPUT);
  pinMode(IN4, OUTPUT);
  pinMode(ENB, OUTPUT);

  digitalWrite(ENB, HIGH);
}

void loop() {
  digitalWrite(IN3, HIGH);
  digitalWrite(IN4, LOW);
  delay(3000);

  digitalWrite(IN3, LOW);
  digitalWrite(IN4, LOW);
  delay(1000);

  digitalWrite(IN3, LOW);
  digitalWrite(IN4, HIGH);
  delay(3000);

  digitalWrite(IN3, LOW);
  digitalWrite(IN4, LOW);
  delay(1000);
}
