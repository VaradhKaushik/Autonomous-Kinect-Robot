#include <Servo.h>

int leftPin = 3;
int rightPin = 5;

int minValue = 1000;
int maxValue = 2000;

Servo left;
Servo right;

void setup() {
  Serial.begin(9600);

  left.attach(leftPin, minValue, maxValue); 
  Serial.println("Left attached");
  right.attach(rightPin, minValue, maxValue);
  Serial.println("Right Attached");

  arm();
  Serial.println("armed");
  delay(5000);
}
void loop() { 
  setBoth(75);
  delay(5000);
  setBoth(0);
  setLeft(50);
  delay(5000);
  setBoth(0);
  setRight(50);
  delay(5000);
  setBoth(0);
  delay(20000);
}

void arm() {
  setBoth(0); //Sets speed variable 
  delay(5000);
}

void setLeft(int speed){
  int angle = map(speed, -100, 100, 0, 180); //Sets servo positions to different speeds 
  left.write(angle);
}

void setRight(int speed) {
  int angle = map(speed, -100, 100, 180, 0);
  right.write(angle);
}

void setBoth(int speed) {
  setLeft(speed);
  setRight(speed);
}
