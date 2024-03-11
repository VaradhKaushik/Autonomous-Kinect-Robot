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
}

void loop() {
   int speed; //Implements speed variable
  for(speed = 0; speed <= 70; speed += 5) { //Cycles speed up to 70% power for 1 second

  setSpeed(speed); //Creates variable for speed to be used in in for loop

  delay(1000);

  }

  delay(4000); //Stays on for 4 seconds

  for(speed = 70; speed > 0; speed -= 5) { // Cycles speed down to 0% power for 1 second

  setSpeed(speed); delay(1000);

  }

  setSpeed(0); //Sets speed variable to zero no matter what

  delay(1000); //Turns off for 1 second 
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
