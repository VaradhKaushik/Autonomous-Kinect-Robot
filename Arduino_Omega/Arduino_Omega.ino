#include <Servo.h>

int leftPin = 3;
int rightPin = 5;

int minValue = 1000;
int maxValue = 2000;

Servo left;
Servo right;

//int speed = 0;

void setup() {
  left.attach(leftPin, minValue, maxValue); 
  right.attach(rightPin, minValue, maxValue);

  Serial.begin(9600);
}

void loop() {
  int speed; //Implements speed variable
  for(int i = 0; i <= 100; i+=5) {
    Serial.println(i);
    setSpeed(i);
    delay(2000);
  }
  delay(10000);
}

void arm() {
  setSpeed(0); //Sets speed variable 
  delay(1000);
}

void setSpeed(int speed){

  int angle = map(speed, 0, 100, 0, 180); //Sets servo positions to different speeds 
  left.write(angle);
  right.write(angle);

}
