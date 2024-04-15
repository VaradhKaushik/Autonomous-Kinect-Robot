import serial

# Assign Arduino's serial port address
#   Windows example
#     usbport = 'COM3'
#   Linux example
#     usbport = '/dev/ttyUSB0'
#   MacOSX example
#     usbport = '/dev/tty.usbserial-FTALLOK2'
def arduino_setup():
    usbport = '/dev/cu.usbserial-1220'
    baud = 9600

    # Set up serial baud rate
    try:
        ser = serial.Serial(usbport, baud, timeout=1)
        return ser
    except:
        print("Failed to open " + usbport + ".\n")
        exit(0)
    print("Serial port " + usbport+ " opened at Baudrate " + str(baud))


def move(motor, speed, ser=None):
    if ser is None:
        ser = arduino_setup()
    if speed < -100 or speed > 100:
        print("Speed must be an integer between -100 and 100.\n")
    elif motor not in ['left', 'right', 'both', 'turnL', 'turnR', 'forwardD', 'forwardT']:
        print("Motor must be 'left', 'right', or 'both'.\n")
    else:
        message = "<" + motor + "," + str(speed) + ">"
        ser.write(bytes(message, 'utf-8'))


if __name__ == '__main__':
    ser = arduino_setup()
    while True:
        a = input("Enter motor: ")
        b = input("Enter speed: ")
        move(a, int(b), ser)
