#include <Servo.h>

int drivePin = 6;
int turnPin = 3;

int minValue = 1000;
int maxValue = 2000;

Servo drive;
Servo turn;

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

  drive.attach(drivePin, minValue, maxValue); 
  Serial.println("Drive attached");
  turn.attach(turnPin, minValue, maxValue);
  Serial.println("Turn Attached");

  //arm();
  Serial.println("armed");

  drive.write(0);
  turn.write(0);
  delay(5000);
  drive.write(90);
  turn.write(90);
  delay(5000);
  drive.write(180);
  turn.write(180);
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
      } else if(strcmp(motor, "forward") == 0) {
        driveGo(speed);
      } else if(strcmp(motor, "reverse") == 0) {
        driveGo(-1*speed);
      } else if(strcmp(motor, "right") == 0) {
        turnRight();
      }else if(strcmp(motor, "left") == 0) {
        turnLeft();
      }else if(strcmp(motor, "straight") == 0) {
        turnStraight();
      }else if(strcmp(motor, "forwardD") == 0) {
        //forwardDistance(speed);
      }else if(strcmp(motor, "forwardT") == 0) {
        //forwardTime(speed);
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
  drive.write(0);
  delay(5000);
}

void driveGo(int speed){
  int angle = map(speed, -100, 100, 0, 180); //Sets servo positions to different speeds 
  drive.write(angle);
}

void forwardTime(int seconds) {
  driveGo(50);
  delay(seconds*1000);
  driveGo(0);
}

void forwardDistance(int feet) {
  float feetPerSecond = 5;
  forwardTime(feet/feetPerSecond);
}

void turnRight() {
  turn.write(180);
}

void turnLeft() {
  turn.write(0);
}

void turnStraight() {
  turn.write(90);
}