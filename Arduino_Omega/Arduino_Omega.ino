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
  delay(5000);

  setBoth(75);
  delay(470);
  setBoth(0);

  delay(1000);
  
  turnLeft(259);

  delay(1000);

  setBoth(75);
  delay(1000);
  setBoth(0);
  
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
  int angle = map(speed, -100, 100, 160, 20);
  right.write(angle);
}

void setBoth(int speed) {
  setLeft(speed);
  setRight(speed);
}

void forwardTime(int seconds) {
  setBoth(100);
  delay(seconds*1000);
  setBoth(0);
}

void forwardDistance(int feet) {
  float feetPerSecond = 3;
  forwardTime(int(feet/feetPerSecond));
}

void turnRight(int degrees) {
  int millisecondsPerDegree = 1.95;
  setRight(-50);
  setLeft(50);
  delay(degrees * millisecondsPerDegree);
  setBoth(0);
}

void turnLeft(int degrees) {
  int millisecondsPerDegree = 1.95;
  setRight(50);
  setLeft(-50);
  delay(degrees * millisecondsPerDegree);
  setBoth(0);
}
