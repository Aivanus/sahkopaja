
#include <NewPing.h>

#define maxDistance 400 // Max distance, which is read by sensors, if over 400 cm then maxDistance = 0
// Moottorien pinnien määrittely. driverin käytön takia käytetään motorInterfacetypena 1

// Right front motor
#define dirRightFront 10
#define stepRightFront 11

// Right rear motor
#define dirRightRear 38
#define stepRightRear 39

// Left front motor
#define dirLeftFront 12
#define stepLeftFront 6

// Left rear motor
#define dirLeftRear 32
#define stepLeftRear 33

// One revolution take 1600 microsteps
#define stepsToTake 200
#define motorInterfaceType 1

unsigned long distanceFront, distanceRight, distanceLeft;
int tooClose = 20;


// lukea useampaa sensoria samaan aikaan.
NewPing frontSensor = NewPing(22,23, maxDistance);
NewPing rightSensor = NewPing(24,25, maxDistance);
NewPing leftSensor = NewPing(26,27, maxDistance);

void setup() {
  Serial.begin(115200);

  // Right front motor
  pinMode(stepRightFront, OUTPUT);
  pinMode(dirRightFront, OUTPUT);

  // Right rear motor 
  pinMode(stepRightRear, OUTPUT);
  pinMode(dirRightRear, OUTPUT);
  
  // Left front motor
  pinMode(stepLeftFront, OUTPUT);
  pinMode(dirLeftFront, OUTPUT);

  // Left rear motor
  pinMode(stepLeftRear, OUTPUT);
  pinMode(dirLeftRear, OUTPUT);

  // Directions
  digitalWrite(dirRightFront, HIGH);
  digitalWrite(dirRightRear, HIGH);
  digitalWrite(dirLeftFront, HIGH);
  digitalWrite(dirLeftRear, HIGH);

}

void reverse(){
  digitalWrite(dirRightFront, HIGH);
  digitalWrite(dirRightRear, LOW);
  digitalWrite(dirLeftFront, LOW);
  digitalWrite(dirLeftRear, LOW);
  for (int i = 0; i < stepsToTake; i++) {
    digitalWrite(stepRightFront, HIGH);
    digitalWrite(stepRightRear, HIGH);
    digitalWrite(stepLeftFront, HIGH);
    digitalWrite(stepLeftRear, HIGH);
    delayMicroseconds(950);
    digitalWrite(stepRightFront, LOW);
    digitalWrite(stepRightRear, LOW);
    digitalWrite(stepLeftFront, LOW);
    digitalWrite(stepLeftRear, LOW);
  }
}

void forward(){
  digitalWrite(dirRightFront, LOW);
  digitalWrite(dirRightRear, HIGH);
  digitalWrite(dirLeftFront, HIGH);
  digitalWrite(dirLeftRear, HIGH);
  for (int i = 0; i < stepsToTake; i++) {
    digitalWrite(stepRightFront, HIGH);
    digitalWrite(stepRightRear, HIGH);
    digitalWrite(stepLeftFront, HIGH);
    digitalWrite(stepLeftRear, HIGH);
    delayMicroseconds(950);
    digitalWrite(stepRightFront, LOW);
    digitalWrite(stepRightRear, LOW);
    digitalWrite(stepLeftFront, LOW);
    digitalWrite(stepLeftRear, LOW);
  }
}

void turnRight(){
  digitalWrite(dirRightFront, HIGH);
  digitalWrite(dirRightRear, LOW);
  digitalWrite(dirLeftFront, HIGH);
  digitalWrite(dirLeftRear, HIGH);
  for (int i = 0; i < stepsToTake; i++) {
    digitalWrite(stepRightFront, HIGH);
    digitalWrite(stepRightRear, HIGH);
    digitalWrite(stepLeftFront, HIGH);
    digitalWrite(stepLeftRear, HIGH);
    delayMicroseconds(950);
    digitalWrite(stepRightFront, LOW);
    digitalWrite(stepRightRear, LOW);
    digitalWrite(stepLeftFront, LOW);
    digitalWrite(stepLeftRear, LOW);
  }
}

void turnLeft(){
  digitalWrite(dirRightFront, LOW);
  digitalWrite(dirRightRear, HIGH);
  digitalWrite(dirLeftFront, LOW);
  digitalWrite(dirLeftRear, LOW);
  for (int i = 0; i < stepsToTake; i++) {
    digitalWrite(stepRightFront, HIGH);
    digitalWrite(stepRightRear, HIGH);
    digitalWrite(stepLeftFront, HIGH);
    digitalWrite(stepLeftRear, HIGH);
    delayMicroseconds(950);
    digitalWrite(stepRightFront, LOW);
    digitalWrite(stepRightRear, LOW);
    digitalWrite(stepLeftFront, LOW);
    digitalWrite(stepLeftRear, LOW);
  }
}


void loop() {
  distanceFront = frontSensor.ping_cm();
  distanceRight = rightSensor.ping_cm();
  distanceLeft = leftSensor.ping_cm();
  if (Serial.available() > 0){
    char readChar = Serial.read();
    if(readChar == 'R'){ // Turn right
      turnRight();
    } else if(readChar == 'L'){ // Turn left
      turnLeft();
    } else if(readChar == 'F'){ // Forward 
      if(distanceFront > tooClose || distanceFront == 0){
        forward();  
      } 
    } else if(readChar == 'T'){ // Taaksepäin :D 
        reverse();  
    }  else {
      if(readChar == 'M'){
        tooClose = 100;
      } else {
        tooClose = 30;
      }
      if(distanceRight > 0 && distanceRight < tooClose && (distanceLeft == 0 || distanceLeft > tooClose)){
        turnLeft();
      } else if(distanceLeft > 0 && distanceLeft < tooClose && (distanceRight == 0 || distanceRight > tooClose)){
        turnRight();
      } else if(distanceFront < tooClose && distanceFront > 0){
        reverse();
      } else {
        forward();
      }
     
    } 
  }
  Serial.write('B'); 
}
  

     
