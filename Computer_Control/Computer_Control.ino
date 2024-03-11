#include <Servo.h>

int leftPin = 3;
int rightPin = 5;

int minValue = 1000;
int maxValue = 2000;

Servo left;
Servo right;

const byte buffSize = 40;
char inputBuffer[buffSize];
const char startMarker = '<';
const char endMarker = '>';
byte bytesRecvd = 0;
boolean readInProgress = false;
boolean newDataFromPC = false;
char motor[buffSize];           // which motor (1 = left, 2 = right, 3 = both)
int speed;             // servo angle -100 to 100

void setup() {
  Serial.begin(9600);

  left.attach(leftPin, minValue, maxValue); 
  Serial.println("Left attached");
  right.attach(rightPin, minValue, maxValue);
  Serial.println("Right Attached");

  arm();
  Serial.println("armed");
}

void loop() { 
  if(Serial.available() > 0) {
    char x = Serial.read();
      // the order of these IF clauses is significant
    if (x == endMarker) {
      readInProgress = false;
      newDataFromPC = true;
      inputBuffer[bytesRecvd] = 0;

      Serial.println(inputBuffer);

      // Parse the message
      char * strtokIndx; // this is used by strtok() as an index
      // Motor instruction
      strtokIndx = strtok(inputBuffer,",");      // get the first part - the string
      strcpy(motor, strtokIndx);
      // Speed instruction
      strtokIndx = strtok(NULL, ","); // this continues where the previous call left off
      speed = atoi(strtokIndx);     // convert this part to an integer

      if(speed < -100 || speed > 100) {
        Serial.println("Invalid speed");
      } else if(strcmp(motor, "left") == 0) {
        setLeft(speed);
      } else if(strcmp(motor, "right") == 0) {
        setRight(speed);
      } else if(strcmp(motor, "both") == 0) {
        setBoth(speed);
      } else {
        Serial.println("Invalid motor");
      }
    }

    // Read data in
    if(readInProgress) {
      inputBuffer[bytesRecvd] = x;
      bytesRecvd ++;
      if (bytesRecvd == buffSize) {
        bytesRecvd = buffSize - 1;
      }
    }

    // Should start reading data in
    if (x == startMarker) { 
      bytesRecvd = 0; 
      readInProgress = true;
    }
  }
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
