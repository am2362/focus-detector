/*
  Focus Buddy - Arduino Nano Every
  --------------------------------
  The laptop does the "seeing" (webcam + AI) and sends a short message over
  USB about once a second. This sketch turns those messages into lights,
  beeps and text on the MonkMakes 7-segment display.

  Wiring (see the breadboard guide):
    D2        -> resistor -> GREEN LED -> GND
    D3        -> resistor -> RED LED   -> GND
    D4        -> buzzer red wire        (black wire -> GND)
    D1 (TX1)  -> R1 -> middle -> R2 -> R3 -> GND   (three equal resistors)
                 display Rx -> middle   (turns the 5 V signal into ~3.3 V)
    3V3       -> display 3V
    GND       -> display GND

  Messages from the laptop, one per line, 115200 baud. You can also type
  these into the Serial Monitor (set it to "Newline") to test everything
  before you run the Python script:
    F<secs>  focused          e.g. F754 shows 12.34 (12 min 34 s of focus)
    W<secs>  warning          you've looked away a few seconds: red LED blinks
    P        phone spotted    "PHON", red LED, 3 quick beeps
    L        looking away     "EYES", red LED, 2 beeps
    S        eyes closed      "NAP",  red LED, 2 beeps
    A        away from desk   "GONE", red LED, no sound
    C        calibrating      "CAL",  green LED blinks
    Z        paused           "PAUS"
    M0 / M1  buzzer off / on
    T        run the light, beep and display test again
*/

const uint8_t GREEN_LED = 2;
const uint8_t RED_LED   = 3;
const uint8_t BUZZER    = 4;

const unsigned long LINK_TIMEOUT_MS = 3000;  // no message for 3 s = laptop script stopped

// The KPEG260 normally has its own oscillator, so HIGH = beep. If yours only
// clicks during the self-test, it's the plain "transducer" kind: set this to
// true and the Nano will drive it at its 3.5 kHz sweet spot instead.
const bool PASSIVE_BUZZER = false;

char mode = '-';                 // '-' means "waiting for the laptop"
long focusSeconds = 0;
bool buzzerEnabled = true;
unsigned long lastMessageMs = 0;
unsigned long modeStartMs = 0;

char wanted[12] = "----";        // what the display should show
char shown[12]  = "";            // what it is showing right now
unsigned long lastDisplayMs = 0;

char line[16];                   // the message being received from the laptop
uint8_t lineLength = 0;


// ---------------------------------------------------------------- display
// The MonkMakes display listens on Serial1 (TX1 pin) at 9600 baud.
// ","      switches it to buffered mode,
// "    "   pushes the old characters out,
// text     only the last 4 characters are kept ('.' lights a decimal point
//          and doesn't use up a digit),
// ","      shows the new text.
void sendToDisplay(const char *text) {
  Serial1.print(",    ");
  Serial1.print(text);
  Serial1.print(',');
  strcpy(shown, text);
  lastDisplayMs = millis();
}

void setDisplay(const char *text) {
  strncpy(wanted, text, sizeof(wanted) - 1);
  wanted[sizeof(wanted) - 1] = '\0';
}

void updateDisplay() {
  if (strcmp(wanted, shown) == 0) return;          // nothing new to show
  if (millis() - lastDisplayMs < 100) return;      // give the display time between updates
  sendToDisplay(wanted);
}

void showFocusTime(unsigned long seconds) {
  char text[12];
  unsigned int minutes = (seconds / 60) % 10000;   // the display only has 4 digits
  unsigned int secs = seconds % 60;
  if (minutes < 100) {
    snprintf(text, sizeof(text), "%u.%02u", minutes, secs);   // 12.34 = 12 min 34 s
  } else {
    snprintf(text, sizeof(text), "%u", minutes);              // 125 = 125 minutes
  }
  setDisplay(text);
}


// ---------------------------------------------------------------- buzzer
void buzzer(bool on) {
  static bool isOn = false;
  if (on == isOn) return;               // only act when it changes
  isOn = on;
  if (PASSIVE_BUZZER) {
    if (on) tone(BUZZER, 3500); else noTone(BUZZER);
  } else {
    digitalWrite(BUZZER, on ? HIGH : LOW);
  }
}

void updateBuzzer(unsigned long now) {
  uint8_t beeps = 0;
  unsigned long every = 3000;
  if (mode == 'P') {
    beeps = 3;  every = 1500;           // phone: more insistent
  } else if (mode == 'L' || mode == 'S') {
    beeps = 2;  every = 3000;
  }

  bool on = false;
  if (buzzerEnabled && beeps > 0) {
    unsigned long t = (now - modeStartMs) % every;   // where we are in this cycle
    if (t < beeps * 200UL) {
      on = (t % 200) < 100;                          // 100 ms beep, 100 ms gap
    }
  }
  buzzer(on);
}


// ---------------------------------------------------------------- modes
void setMode(char newMode) {
  if (newMode == mode) return;
  mode = newMode;
  modeStartMs = millis();
  Serial.print("mode ");
  Serial.println(mode);
}

void updateOutputs() {
  unsigned long now = millis();
  bool blink = (now / 250) % 2;        // flips every 250 ms
  bool green = false;
  bool red = false;

  switch (mode) {
    case 'F': green = true;   showFocusTime(focusSeconds); break;
    case 'W': red = blink;    showFocusTime(focusSeconds); break;
    case 'P': red = true;     setDisplay("PHON");          break;
    case 'L': red = true;     setDisplay("EYES");          break;
    case 'S': red = true;     setDisplay("NAP");           break;
    case 'A': red = true;     setDisplay("GONE");          break;
    case 'C': green = blink;  setDisplay("CAL");           break;
    case 'Z':                 setDisplay("PAUS");          break;
    default:                  setDisplay("----");          break;  // waiting for laptop
  }

  digitalWrite(GREEN_LED, green ? HIGH : LOW);
  digitalWrite(RED_LED, red ? HIGH : LOW);
  updateBuzzer(now);
}


// ---------------------------------------------------------------- self test
void selfTest() {
  sendToDisplay("8.8.8.8.");                 // every segment on
  digitalWrite(GREEN_LED, HIGH);  delay(400);
  digitalWrite(GREEN_LED, LOW);
  digitalWrite(RED_LED, HIGH);    delay(400);
  digitalWrite(RED_LED, LOW);
  buzzer(true);                   delay(150);
  buzzer(false);
  sendToDisplay("HI");
  delay(600);
}


// ---------------------------------------------------------------- laptop link
void handleMessage(char *msg) {
  char command = toupper(msg[0]);
  long number = atol(msg + 1);
  if (number < 0) number = 0;

  switch (command) {
    case 'F':
    case 'W':
      focusSeconds = number;
      setMode(command);
      break;
    case 'P': case 'L': case 'S': case 'A': case 'C': case 'Z':
      setMode(command);
      break;
    case 'M':
      buzzerEnabled = (number != 0);
      Serial.println(buzzerEnabled ? "buzzer on" : "buzzer off");
      break;
    case 'T':
      selfTest();
      break;
    default:
      return;                               // not one of ours: ignore it
  }
  lastMessageMs = millis();
}

void readLaptop() {
  while (Serial.available() > 0) {
    char c = Serial.read();
    if (c == '\n' || c == '\r') {
      if (lineLength > 0) {
        line[lineLength] = '\0';
        handleMessage(line);
        lineLength = 0;
      }
    } else if (lineLength < sizeof(line) - 1) {
      line[lineLength++] = c;
    }
  }
}


// ---------------------------------------------------------------- main
void setup() {
  pinMode(GREEN_LED, OUTPUT);
  pinMode(RED_LED, OUTPUT);
  pinMode(BUZZER, OUTPUT);

  Serial.begin(115200);       // USB link to the laptop
  Serial1.begin(9600);        // TX1 pin -> voltage divider -> display
  delay(100);                 // the display needs a moment before the first message
  Serial1.print('/');         // clear the display

  selfTest();
  Serial.println("Focus Buddy ready. Try: F754  W754  P  L  S  A  C  Z  M0  M1  T");
}

void loop() {
  readLaptop();
  if (mode != '-' && millis() - lastMessageMs > LINK_TIMEOUT_MS) {
    setMode('-');             // laptop went quiet
  }
  updateOutputs();
  updateDisplay();
}
