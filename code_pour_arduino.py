import machine
import time

broche = machine.Pin(2, machine.Pin.OUT)


while True:
    broche.value(1)
    time.sleep(0.5)
    broche.value(0)
    time.sleep(0.5)